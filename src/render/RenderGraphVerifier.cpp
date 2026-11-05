/*
* Copyright (c) 2026, ethereal contributors. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/

// The device-free plan verifier (design 11.1). It shares no logic with the compiler's walk: it replays a plan's
// segments in submission order against a simulated state machine per subresource and per physical resource, with
// happens-before between queues taken only from the plan's fence waits.
//
// Frames: the first replayed frame learns the states that the plan assumes at frame start (an element's state is
// unknown until its first barrier or access), every later frame is checked from the states the previous one left.
// That checks the cyclic assumptions of 6.12. A plan switch replays the previous plan's frame, applies the
// registry's patch (the shared physical resources are moved to the states the new plan expects at their first
// touch, and every queue of the new frame waits for every queue of the old one), then the new plan's frames.

#include "RenderGraphInternal.h"

#include <algorithm>
#include <array>
#include <unordered_map>

namespace donut::render::rdg
{
    namespace
    {
        using nvrhi::CommandQueue;
        using nvrhi::ResourceStates;

        struct Event
        {
            bool valid = false;
            uint32_t queue = 0;
            uint64_t pos = 0;        // per queue, 1-based, increasing across frames
            uint64_t segment = 0;    // global segment instance id
            uint32_t pass = c_None;  // schedule index
            uint32_t frame = 0;
            bool relaxedUav = false; // a UAV write with RelaxedOrder (6.5)
        };

        struct Element
        {
            ResourceStates state = ResourceStates::Unknown;
            bool known = false;      // the state is checked; otherwise the next barrier or access defines it
            bool learned = false;    // the state came from read accesses only and may still widen
            bool splitPending = false;
            ResourceStates splitAfter = ResourceStates::Unknown;
            int8_t owner = -1;
            bool inTransfer = false;
            int8_t transferTo = -1;
            Event release;
            bool initialized = true;
            bool patchPending = false;
            uint32_t occupant = c_None;
            Event lastWrite;
            std::vector<Event> reads;
            Event lastStateChange;
            std::array<uint64_t, 3> lastBarrier = { 0, 0, 0 };
        };

        struct Identity
        {
            std::vector<Element> elements;
            uint32_t planes = 1;
            uint32_t mips = 1;
            uint32_t slices = 1;
            bool transient = false;
            bool hostKept = false;
            uint32_t frame = 0;      // the frame that last touched it
        };

        constexpr uint32_t c_MaxErrors = 64;

        class Verifier
        {
        public:
            explicit Verifier(std::string& report) : m_Report(report) { }

            void RunFrame(const CompiledRenderGraph& plan, bool afterSwitch);
            uint32_t ErrorCount() const { return m_Errors; }
            uint32_t m_Frame = 0;

        private:
            void Error(const std::string& text);
            bool HappensBefore(const Event& x, uint32_t queue) const;
            bool RoundTrip(const Event& x, uint32_t queue) const;
            uint64_t IdentityKey(const CompiledRenderGraph& plan, uint32_t resource) const;
            Identity& GetIdentity(const CompiledRenderGraph& plan, uint32_t resource);
            uint32_t PlaneCount(const CompiledRenderGraph& plan, const PlanResource& resource) const;
            template <typename Fn> void ForEachElement(const CompiledRenderGraph& plan, uint32_t resource,
                                                      const nvrhi::SubresourceRegion& region, Fn&& fn);
            void Barrier(const CompiledRenderGraph& plan, const PlanBarrier& barrier, uint32_t queue, uint32_t sched,
                         const Event& now);
            void Access(const CompiledRenderGraph& plan, const PlanUse& use, PassType type, uint32_t queue, uint32_t sched,
                        const Event& now, bool clearSubStep);
            std::string Where(const CompiledRenderGraph& plan, uint32_t sched) const;
            std::string Name(const CompiledRenderGraph& plan, uint32_t resource) const;

            std::string& m_Report;
            uint32_t m_Errors = 0;
            std::unordered_map<uint64_t, Identity> m_Identities;
            std::array<uint64_t, 3> m_QueuePos = { 0, 0, 0 };
            uint64_t m_SegmentCounter = 0;
            std::array<std::array<uint64_t, 3>, 3> m_Known = {};
            std::vector<std::array<uint64_t, 3>> m_Snapshots = { {} };   // by global segment id; id 0 is "none"
            std::vector<std::vector<uint64_t>> m_FrameSegments;          // [frame][segment] -> global id
            std::array<uint64_t, 3> m_LastSegmentOfQueue = { 0, 0, 0 };
            // Aliasing (6.7), per frame.
            std::vector<uint64_t> m_AliasPos;
            std::vector<uint64_t> m_LastAccessPos;
            std::vector<uint8_t> m_SeenThisFrame;
        };

        void Verifier::Error(const std::string& text)
        {
            if (m_Errors < c_MaxErrors)
            {
                m_Report += "verify: frame " + std::to_string(m_Frame) + ": " + text + "\n";
            }
            else if (m_Errors == c_MaxErrors)
            {
                m_Report += "verify: further violations omitted\n";
            }
            ++m_Errors;
        }

        std::string Verifier::Where(const CompiledRenderGraph& plan, uint32_t sched) const
        {
            if (sched >= plan.m_Passes.size())
                return "the frame";
            return "pass '" + std::string(plan.GetName(plan.m_Passes[sched].nameOffset)) + "' (" + std::to_string(sched) + ")";
        }

        std::string Verifier::Name(const CompiledRenderGraph& plan, uint32_t resource) const
        {
            if (resource >= plan.m_Resources.size())
                return "#" + std::to_string(resource);
            const PlanResource& record = plan.m_Resources[resource];
            std::string name = "'" + std::string(plan.GetName(record.nameOffset)) + "' (#" + std::to_string(resource);
            if (record.kind == ResourceKind::History)
                name += ", layer " + std::to_string(record.historyLayer);
            return name + ")";
        }

        bool Verifier::HappensBefore(const Event& x, uint32_t queue) const
        {
            if (!x.valid)
                return true;
            if (x.queue == queue)
                return true;
            return m_Known[queue][x.queue] >= x.segment;
        }

        // An event on `queue` itself is also ordered, with its memory visible, when the queue has since waited for a
        // segment of another queue that had observed the event's segment: a fence round-trip (signal, wait, signal,
        // wait) is a full memory dependency on D3D12 and Vulkan.
        bool Verifier::RoundTrip(const Event& x, uint32_t queue) const
        {
            for (uint32_t other = 0; other < 3; ++other)
            {
                if (other == queue)
                    continue;
                const uint64_t observed = m_Known[queue][other];
                if (observed && m_Snapshots[observed][queue] >= x.segment)
                    return true;
            }
            return false;
        }

        uint32_t Verifier::PlaneCount(const CompiledRenderGraph& plan, const PlanResource& resource) const
        {
            const BackendProfile& profile = plan.m_Profile;
            return (!resource.isBuffer && resource.hasDepth && resource.hasStencil && profile.planeModel != PlaneModel::Ignored &&
                profile.barrierModel != BarrierModel::Implicit) ? 2u : 1u;
        }

        uint64_t Verifier::IdentityKey(const CompiledRenderGraph& plan, uint32_t resource) const
        {
            const PlanResource& record = plan.m_Resources[resource];
            switch (record.kind)
            {
            case ResourceKind::Persistent:
                return HashCombine(HashCombine(c_HashSeed, 2), record.registryEntry);
            case ResourceKind::History:
            {
                // 5.2: layer k of frame f is physical texture (f - k) mod L.
                const uint32_t layers = std::max<uint32_t>(record.historyLayerCount, 1);
                const uint32_t physical = (m_Frame + layers * 16 - record.historyLayer) % layers;
                return HashCombine(HashCombine(HashCombine(c_HashSeed, 3), record.registryEntry), (uint64_t(layers) << 32) | physical);
            }
            case ResourceKind::Transient:
            case ResourceKind::PassLocal:
                if (!record.extracted && record.physical != c_None)
                    return HashCombine(HashCombine(HashCombine(c_HashSeed, 4), uint64_t(uintptr_t(plan.m_Structure.get()))),
                                       record.physical);
                [[fallthrough]];
            default:
                // Imports and extracted transients: new each frame.
                return HashCombine(HashCombine(HashCombine(c_HashSeed, 1), m_Frame), resource);
            }
        }

        Identity& Verifier::GetIdentity(const CompiledRenderGraph& plan, uint32_t resource)
        {
            const uint64_t key = IdentityKey(plan, resource);
            auto it = m_Identities.find(key);
            if (it != m_Identities.end())
                return it->second;

            const PlanResource& record = plan.m_Resources[resource];
            Identity& identity = m_Identities[key];
            identity.planes = PlaneCount(plan, record);
            identity.mips = std::max(1u, record.mipLevels);
            identity.slices = std::max(1u, record.arraySlices);
            identity.transient = (record.kind == ResourceKind::Transient || record.kind == ResourceKind::PassLocal);
            identity.hostKept = record.kind == ResourceKind::HostKept;
            identity.frame = m_Frame;
            identity.elements.resize(size_t(identity.planes) * identity.mips * identity.slices);
            for (Element& element : identity.elements)
            {
                if (record.kind == ResourceKind::Imported || record.kind == ResourceKind::HostKept)
                {
                    element.state = record.entry;
                    element.known = record.entry != ResourceStates::Unknown;
                }
                else if (record.extracted)
                {
                    element.state = record.exit;
                    element.known = record.exit != ResourceStates::Unknown;
                    element.initialized = false;
                }
                if (identity.transient)
                    element.initialized = false;
            }
            return identity;
        }

        template <typename Fn>
        void Verifier::ForEachElement(const CompiledRenderGraph& plan, uint32_t resource, const nvrhi::SubresourceRegion& region,
                                      Fn&& fn)
        {
            Identity& identity = GetIdentity(plan, resource);
            uint32_t planeBits = 0;
            if (identity.planes == 1)
                planeBits = region.planes != nvrhi::PlaneMask::None ? 1u : 0u;
            else
            {
                if ((region.planes & nvrhi::PlaneMask::Depth) != 0) planeBits |= 1;
                if ((region.planes & nvrhi::PlaneMask::Stencil) != 0) planeBits |= 2;
            }
            const uint32_t mip0 = std::min(region.baseMip, identity.mips);
            const uint32_t mip1 = mip0 + std::min(region.mipCount, identity.mips - mip0);
            const uint32_t slice0 = std::min(region.baseSlice, identity.slices);
            const uint32_t slice1 = slice0 + std::min(region.sliceCount, identity.slices - slice0);
            for (uint32_t plane = 0; plane < identity.planes; ++plane)
            {
                if ((planeBits & (1u << plane)) == 0)
                    continue;
                for (uint32_t mip = mip0; mip < mip1; ++mip)
                {
                    for (uint32_t slice = slice0; slice < slice1; ++slice)
                        fn(identity, identity.elements[(size_t(plane) * identity.mips + mip) * identity.slices + slice]);
                }
            }
        }

        void Verifier::Barrier(const CompiledRenderGraph& plan, const PlanBarrier& barrier, uint32_t queue, uint32_t sched,
                               const Event& now)
        {
            const BackendProfile& profile = plan.m_Profile;
            if (profile.barrierModel == BarrierModel::Implicit)
            {
                Error(Where(plan, sched) + ": a barrier in a plan for a backend without barriers");
                return;
            }

            if (barrier.type == nvrhi::BarrierType::Uav || barrier.type == nvrhi::BarrierType::Aliasing)
            {
                if (barrier.resource == c_None)
                {
                    for (auto& [key, identity] : m_Identities)
                        for (Element& element : identity.elements)
                            element.lastBarrier[queue] = now.pos;
                    return;
                }
                ForEachElement(plan, barrier.resource, nvrhi::SubresourceRegion(), [&](Identity&, Element& element) {
                    element.lastBarrier[queue] = now.pos;
                });
                if (barrier.type == nvrhi::BarrierType::Aliasing && barrier.resource < m_AliasPos.size())
                    m_AliasPos[barrier.resource] = now.pos;
                return;
            }

            const bool release = (barrier.flags & nvrhi::BarrierFlags::QueueRelease) != 0;
            const bool acquire = (barrier.flags & nvrhi::BarrierFlags::QueueAcquire) != 0;
            const bool splitBegin = (barrier.flags & nvrhi::BarrierFlags::SplitBegin) != 0;
            const bool splitEnd = (barrier.flags & nvrhi::BarrierFlags::SplitEnd) != 0;
            const bool discard = (barrier.flags & nvrhi::BarrierFlags::Discard) != 0;
            const CommandQueue commandQueue = CommandQueue(queue);

            // 6.8: legal on the queue that executes it. A release names only its first scope, an acquire its second.
            if ((!acquire && !IsStateLegalOnQueue(barrier.before, commandQueue, profile)) ||
                (!release && !IsStateLegalOnQueue(barrier.after, commandQueue, profile)))
                Error(Where(plan, sched) + ": the transition of " + Name(plan, barrier.resource) + " from " +
                    StatesToString(barrier.before) + " to " + StatesToString(barrier.after) + " is illegal on the " +
                    QueueName(commandQueue) + " queue");

            if ((release || acquire) && !profile.queueOwnershipTransfer)
                Error(Where(plan, sched) + ": an ownership transfer in a profile with one queue family");

            const bool hostKept = plan.m_Resources[barrier.resource].kind == ResourceKind::HostKept;
            bool reported = false;
            ForEachElement(plan, barrier.resource, barrier.region, [&](Identity&, Element& element) {
                auto report = [&](const std::string& text) {
                    if (!reported)
                        Error(Where(plan, sched) + ": " + Name(plan, barrier.resource) + ": " + text);
                    reported = true;
                };

                if (element.splitPending && !splitEnd)
                    report("a barrier on a range whose split transition is still open");

                // The barrier's first scope must follow the accesses it replaces, on every queue.
                if (element.lastWrite.valid && !HappensBefore(element.lastWrite, queue))
                    report("transitions while a write on the " + std::string(QueueName(CommandQueue(element.lastWrite.queue))) +
                        " queue (" + Where(plan, element.lastWrite.pass) + ") may still run: missing fence wait");
                for (const Event& read : element.reads)
                {
                    if (!HappensBefore(read, queue))
                    {
                        report("transitions while a read on the " + std::string(QueueName(CommandQueue(read.queue))) +
                            " queue (" + Where(plan, read.pass) + ") may still run: missing fence wait");
                        break;
                    }
                }
                if (element.lastStateChange.valid && !HappensBefore(element.lastStateChange, queue))
                    report("transitions while an earlier transition on another queue may still run");

                if (acquire)
                {
                    // An owner that is still unknown is the first replayed frame: its release was in the frame before.
                    if (!element.patchPending && profile.queueOwnershipTransfer && (element.owner >= 0 || element.inTransfer))
                    {
                        if (!element.inTransfer || element.transferTo != int8_t(queue))
                            report("an acquire without a matching release to this queue");
                        else if (!HappensBefore(element.release, queue))
                            report("an acquire that may run before its release: missing fence wait");
                    }
                    element.inTransfer = false;
                    element.owner = int8_t(queue);
                }

                if (splitEnd)
                {
                    if (!element.splitPending || element.splitAfter != barrier.after)
                        report("a split end without its begin");
                    element.splitPending = false;
                }
                else if (!acquire)
                {
                    // The state before the barrier is what the range holds (host-kept resources go through the tracker,
                    // which knows it).
                    if (element.known && !element.patchPending && !hostKept && element.state != barrier.before)
                        report("transition from " + StatesToString(barrier.before) + ", but the range is in " +
                            StatesToString(element.state));
                    if (release && profile.queueOwnershipTransfer && element.owner >= 0 && element.owner != int8_t(queue) &&
                        !element.patchPending)
                        report("a release on the " + std::string(QueueName(commandQueue)) + " queue while the " +
                            QueueName(CommandQueue(element.owner)) + " queue owns the range");
                }

                if (splitBegin)
                {
                    element.splitPending = true;
                    element.splitAfter = barrier.after;
                }
                else
                {
                    element.state = barrier.after;
                }
                element.known = true;
                element.learned = false;
                element.patchPending = false;
                if (release)
                {
                    element.inTransfer = true;
                    element.transferTo = int8_t(barrier.otherQueue);
                    element.release = now;
                }
                if (discard)
                {
                    element.initialized = false;
                    element.owner = int8_t(queue);
                }
                if (!element.inTransfer && element.owner < 0)
                    element.owner = int8_t(queue);
                element.lastStateChange = now;
                element.lastBarrier[queue] = now.pos;
            });
        }

        void Verifier::Access(const CompiledRenderGraph& plan, const PlanUse& use, PassType type, uint32_t queue,
                              uint32_t sched, const Event& now, bool clearSubStep)
        {
            const BackendProfile& profile = plan.m_Profile;
            const bool implicit = profile.barrierModel == BarrierModel::Implicit;
            const nvrhi::ShaderType stages = use.stages == nvrhi::ShaderType::None ? DefaultStages(type) : use.stages;
            ResourceStates required = LowerAccess(use.access, stages);
            const bool write = use.isWrite || clearSubStep;
            if (clearSubStep)
            {
                const PlanResource& cleared = plan.m_Resources[use.resource];
                required = ClearState(cleared.isBuffer, cleared.hasDepth, cleared.isRenderTarget, profile);
            }
            const bool fullWrite = clearSubStep || (use.isWrite &&
                (use.flags & (AccessFlags::FullWrite | AccessFlags::Clear)) != AccessFlags::None);
            const PlanResource& resource = plan.m_Resources[use.resource];
            const bool checkHeldState = profile.barrierModel == BarrierModel::D3D12Legacy ||
                profile.barrierModel == BarrierModel::D3D12Enhanced || profile.queueOwnershipTransfer;

            // 6.7: an aliased placed resource starts each frame with an aliasing barrier after the overlapping ones.
            if (!implicit && resource.aliased && use.resource < m_SeenThisFrame.size() && !m_SeenThisFrame[use.resource])
            {
                m_SeenThisFrame[use.resource] = 1;
                if (!m_AliasPos[use.resource])
                    Error(Where(plan, sched) + ": the first use of aliased " + Name(plan, use.resource) +
                        " has no aliasing barrier before it (6.7)");
                const StructurePlan& structure = *plan.m_Structure;
                for (uint32_t other : structure.overlaps[use.resource])
                {
                    if (other < m_LastAccessPos.size() && m_LastAccessPos[other] && m_LastAccessPos[other] > m_AliasPos[use.resource])
                        Error(Where(plan, sched) + ": " + Name(plan, use.resource) + " and " + Name(plan, other) +
                            " share memory and are both live (6.7)");
                }
            }
            if (use.resource < m_LastAccessPos.size())
                m_LastAccessPos[use.resource] = now.pos;

            bool reported = false;
            ForEachElement(plan, use.resource, use.region, [&](Identity& identity, Element& element) {
                auto report = [&](const std::string& text) {
                    if (!reported)
                        Error(Where(plan, sched) + ": " + Name(plan, use.resource) + ": " + text);
                    reported = true;
                };

                if (identity.transient && element.occupant != use.resource)
                {
                    element.occupant = use.resource;
                    element.initialized = false;
                }

                if (element.splitPending)
                    report("an access while a split transition is open");

                if (element.patchPending)
                {
                    // The registry's switch patch moved the range into the state its first use needs.
                    element.state = required;
                    element.known = true;
                    element.patchPending = false;
                    element.owner = int8_t(queue);
                    element.inTransfer = false;
                }

                if (!implicit)
                {
                    if (!element.known)
                    {
                        element.state = required;
                        element.known = true;
                        element.learned = !write;
                    }
                    else if ((element.state & required) != required)
                    {
                        // A state learned from reads may widen with more reads; anything else is a missing barrier.
                        const bool readWiden = element.learned && !write;
                        if (readWiden)
                            element.state = element.state | required;
                        else
                            report(std::string(write ? "writes" : "reads") + " as " + StatesToString(required) +
                                ", but the range is in " + StatesToString(element.state));
                    }
                    if (checkHeldState && !IsStateLegalOnQueue(element.state, CommandQueue(queue), profile))
                        report("the range is in " + StatesToString(element.state) + ", which the " +
                            QueueName(CommandQueue(queue)) + " queue cannot hold (6.8)");
                }

                if (profile.queueOwnershipTransfer)
                {
                    // An ownership transfer preserves contents (6.8). An access that needs none, a full write, takes
                    // the range on its own queue.
                    if (element.inTransfer && !fullWrite)
                        report("an access while the range is being transferred to another queue");
                    else if (element.owner < 0 || fullWrite)
                    {
                        element.owner = int8_t(queue);
                        element.inTransfer = false;
                    }
                    else if (element.owner != int8_t(queue))
                        report("an access on the " + std::string(QueueName(CommandQueue(queue))) + " queue while the " +
                            QueueName(CommandQueue(element.owner)) + " queue owns the range");
                }

                // Contents (6.6).
                if (!element.initialized)
                {
                    if (!write)
                        report("reads undefined contents");
                    else if (!fullWrite)
                        report("a partial write over undefined contents");
                }

                // Hazards (11.1): RAW, WAR, WAW and UAV order, each needs a barrier on one queue or a fence wait across.
                const bool relaxedUav = use.isWrite && (use.access & Access::UAVWrite) != Access::None &&
                    (use.flags & AccessFlags::RelaxedOrder) != AccessFlags::None;
                auto ordered = [&](const Event& earlier, bool sameStateWrite) -> bool {
                    if (!earlier.valid || earlier.pass == sched)
                        return true;
                    if (earlier.queue != queue)
                        return HappensBefore(earlier, queue);
                    if (implicit)
                        return true;
                    if (element.lastBarrier[queue] > earlier.pos || RoundTrip(earlier, queue))
                        return true;
                    // 6.5: consecutive relaxed UAV writes need no barrier between them.
                    if (sameStateWrite && relaxedUav && earlier.relaxedUav)
                        return true;
                    // D3D12 orders writes in one non-UAV state on one queue (render targets, depth, copies).
                    return sameStateWrite && profile.barrierModel != BarrierModel::Vulkan &&
                        (element.state & ResourceStates::UnorderedAccess) == 0;
                };
                if (!ordered(element.lastWrite, write))
                    report(std::string(write ? "write after write" : "read after write") + " (" +
                        Where(plan, element.lastWrite.pass) + ") without a barrier or fence wait");
                if (write)
                {
                    for (const Event& read : element.reads)
                    {
                        if (!ordered(read, false))
                        {
                            report("write after read (" + Where(plan, read.pass) + ") without a barrier or fence wait");
                            break;
                        }
                    }
                }
                if (element.lastStateChange.valid && element.lastStateChange.queue != queue &&
                    !HappensBefore(element.lastStateChange, queue))
                    report("an access that may run before the transition on the " +
                        std::string(QueueName(CommandQueue(element.lastStateChange.queue))) + " queue: missing fence wait");

                if (write)
                {
                    element.lastWrite = now;
                    element.lastWrite.relaxedUav = relaxedUav;
                    element.reads.clear();
                    element.initialized = true;
                    element.learned = false;
                }
                else
                {
                    element.reads.push_back(now);
                }
            });
        }

        void Verifier::RunFrame(const CompiledRenderGraph& plan, bool afterSwitch)
        {
            const uint32_t frame = m_Frame;
            m_FrameSegments.resize(frame + 1);
            m_FrameSegments[frame].assign(plan.m_Segments.size(), 0);

            const uint32_t resourceCount = uint32_t(plan.m_Resources.size());
            m_AliasPos.assign(resourceCount, 0);
            m_LastAccessPos.assign(resourceCount, 0);
            m_SeenThisFrame.assign(resourceCount, 0);

            // Transients hold nothing at frame start; a plan switch patches the shared resources at their first touch.
            for (auto& [key, identity] : m_Identities)
            {
                for (Element& element : identity.elements)
                {
                    if (identity.transient)
                        element.initialized = false;
                    if (afterSwitch)
                        element.patchPending = true;
                }
            }

            const std::array<uint64_t, 3> previousLast = m_LastSegmentOfQueue;
            std::array<bool, 3> firstOfQueue = { true, true, true };

            for (uint32_t s = 0; s < uint32_t(plan.m_Segments.size()); ++s)
            {
                const PlanSegment& segment = plan.m_Segments[s];
                const uint32_t queue = uint32_t(segment.queue);
                const uint64_t id = ++m_SegmentCounter;
                m_FrameSegments[frame][s] = id;
                m_Snapshots.emplace_back();

                for (uint32_t w = 0; w < segment.waitCount; ++w)
                {
                    const SegmentWait& wait = plan.m_SegmentWaits[segment.waitBegin + w];
                    const int64_t targetFrame = int64_t(frame) + wait.frameDelta;
                    if (targetFrame < 0 || (wait.frameDelta < 0 && afterSwitch))
                        continue;   // before the replay, or covered by the switch's full wait
                    const auto& segments = m_FrameSegments[size_t(targetFrame)];
                    const uint64_t target = wait.segment < segments.size() ? segments[wait.segment] : 0;
                    if (!target || target >= id)
                    {
                        Error("segment " + std::to_string(s) + " waits for segment " + std::to_string(wait.segment) +
                            " of frame " + std::to_string(int(wait.frameDelta)) + ", which is not submitted before it");
                        continue;
                    }
                    for (uint32_t k = 0; k < 3; ++k)
                        m_Known[queue][k] = std::max(m_Known[queue][k], m_Snapshots[target][k]);
                    m_Known[queue][uint32_t(wait.queue)] = std::max(m_Known[queue][uint32_t(wait.queue)], target);
                }
                if (afterSwitch && firstOfQueue[queue])
                {
                    for (uint32_t k = 0; k < 3; ++k)
                    {
                        m_Known[queue][k] = std::max(m_Known[queue][k], previousLast[k]);
                        if (previousLast[k])
                            for (uint32_t j = 0; j < 3; ++j)
                                m_Known[queue][j] = std::max(m_Known[queue][j], m_Snapshots[previousLast[k]][j]);
                    }
                }
                firstOfQueue[queue] = false;
                m_Known[queue][queue] = id;

                for (uint32_t sched = segment.firstPass, counted = 0; counted < segment.passCount && sched < plan.m_Passes.size(); ++sched)
                {
                    const PlanPass& pass = plan.m_Passes[sched];
                    if (!pass.enabled || pass.segment != s)
                        continue;
                    ++counted;
                    if (pass.queue != segment.queue)
                        Error(Where(plan, sched) + ": recorded on the " + QueueName(pass.queue) + " queue in a " +
                            QueueName(segment.queue) + " segment");

                    auto event = [&]() {
                        Event e;
                        e.valid = true;
                        e.queue = queue;
                        e.pos = ++m_QueuePos[queue];
                        e.segment = id;
                        e.pass = sched;
                        e.frame = frame;
                        return e;
                    };
                    auto batch = [&](uint32_t slot) {
                        for (uint32_t b = 0; b < pass.batchCount[slot]; ++b)
                            Barrier(plan, plan.m_Barriers[pass.batchBegin[slot] + b], queue, sched, event());
                    };

                    batch(uint32_t(BatchSlot::Prologue));
                    if (plan.m_Profile.barrierModel != BarrierModel::Implicit)
                    {
                        for (uint32_t u = pass.useBegin; u < pass.useBegin + pass.useCount; ++u)
                        {
                            if (plan.m_Uses[u].hasClear)
                                Access(plan, plan.m_Uses[u], pass.type, queue, sched, event(), true);
                        }
                    }
                    batch(uint32_t(BatchSlot::PostClear));
                    const Event body = event();
                    for (uint32_t u = pass.useBegin; u < pass.useBegin + pass.useCount; ++u)
                        Access(plan, plan.m_Uses[u], pass.type, queue, sched, body, false);
                    batch(uint32_t(BatchSlot::Epilogue));
                }

                // 6.4: no split is left open at the end of a command list.
                for (auto& [key, identity] : m_Identities)
                {
                    for (const Element& element : identity.elements)
                    {
                        if (element.splitPending && element.lastStateChange.segment == id)
                        {
                            Error("segment " + std::to_string(s) + " ends with an open split transition");
                            goto splitsChecked;
                        }
                    }
                }
            splitsChecked:
                m_Snapshots[id] = m_Known[queue];
                m_LastSegmentOfQueue[queue] = id;
            }

            // Imports end in their declared exit states.
            for (uint32_t r = 0; r < resourceCount; ++r)
            {
                const PlanResource& resource = plan.m_Resources[r];
                const bool tracked = resource.kind == ResourceKind::Imported || resource.extracted;
                if (!tracked || resource.firstPass == c_None || resource.exit == ResourceStates::Unknown ||
                    plan.m_Profile.barrierModel == BarrierModel::Implicit)
                    continue;
                bool reported = false;
                ForEachElement(plan, r, nvrhi::SubresourceRegion(), [&](Identity&, Element& element) {
                    if (!reported && element.known && element.state != resource.exit)
                    {
                        Error(Name(plan, r) + " ends the frame in " + StatesToString(element.state) +
                            ", but its declared exit state is " + StatesToString(resource.exit));
                        reported = true;
                    }
                });
            }

            // Aliased resources never overlap in lifetime and bytes (6.7).
            if (plan.m_Structure)
            {
                const StructurePlan& structure = *plan.m_Structure;
                for (uint32_t r = 0; r < resourceCount && r < structure.overlaps.size(); ++r)
                {
                    const PlanResource& a = plan.m_Resources[r];
                    if (a.firstPass == c_None)
                        continue;
                    for (uint32_t o : structure.overlaps[r])
                    {
                        if (o <= r || o >= resourceCount)
                            continue;
                        const PlanResource& b = plan.m_Resources[o];
                        if (b.firstPass == c_None)
                            continue;
                        if (!(a.lastPass < b.firstPass || b.lastPass < a.firstPass))
                            Error(Name(plan, r) + " and " + Name(plan, o) + " share heap bytes and their lifetimes overlap");
                    }
                }
            }
            ++m_Frame;
        }
    }

    nvrhi::FRESULT VerifyCompiledPlan(const CompiledRenderGraph& plan, const CompiledRenderGraph* previous, std::string& report)
    {
        Verifier verifier(report);
        uint32_t maxLayers = 2;
        for (const PlanResource& resource : plan.m_Resources)
            maxLayers = std::max<uint32_t>(maxLayers, resource.historyLayerCount);

        if (previous)
        {
            verifier.RunFrame(*previous, false);
            verifier.RunFrame(plan, true);
            verifier.RunFrame(plan, false);
        }
        else
        {
            // The first frame learns the frame-start states; the rest check them, through a whole history rotation.
            const uint32_t frames = std::max<uint32_t>(2, maxLayers + 1);
            for (uint32_t f = 0; f < frames; ++f)
                verifier.RunFrame(plan, false);
        }
        return verifier.ErrorCount() ? nvrhi::FE_GENERIC_ERROR : nvrhi::FS_OK;
    }
}

namespace donut::render
{
    nvrhi::FRESULT VerifyPlan(ICompiledRenderGraph* plan, IRenderGraph* graph, ICompiledRenderGraph* previous,
                              nvrhi::string* report) noexcept
    {
        if (!plan || !graph)
            return nvrhi::FE_INVALID_ARGS;
        try
        {
            const rdg::CompiledRenderGraph* concrete = static_cast<const rdg::CompiledRenderGraph*>(plan);
            const rdg::CompiledRenderGraph* before = static_cast<const rdg::CompiledRenderGraph*>(previous);
            std::string text;
            nvrhi::FRESULT result = nvrhi::FS_OK;
            if (concrete->m_StructuralHash != graph->GetStructuralHash())
            {
                text = "verify: the graph no longer holds the frame the plan was compiled from\n";
                result = nvrhi::FE_INVALID_ARGS;
            }
            else
            {
                result = rdg::VerifyCompiledPlan(*concrete, before, text);
            }
            if (report)
                *report = text.c_str();
            return result;
        }
        catch (...)
        {
            return nvrhi::FE_OUT_OF_MEMORY;
        }
    }
}
