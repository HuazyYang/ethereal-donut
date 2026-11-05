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

// Phase 6 of the compiler: the walk of design section 6.
//
// The walk runs per state object (a physical resource: a pooled slot, a placed or dedicated resource, or an import)
// and per unit of its subresource space (6.2). For each unit it builds the steps (one per pass that touches the
// unit), groups consecutive reads into read epochs and partitions each epoch through the backend lattice (6.3),
// then places one transition before each group and each write, with the waits the queues need (6.8). The frame is
// cyclic for transients, persistent resources and histories: the first step's predecessor is the last step of the
// previous frame (6.12), on the same object, or for a history layer on the layer before it.

#include "RenderGraphCompile.h"

#include <algorithm>

namespace donut::render::rdg
{
    namespace
    {
        using nvrhi::CommandQueue;
        using nvrhi::ResourceStates;

        constexpr uint8_t c_Prologue = uint8_t(BatchSlot::Prologue);
        constexpr uint8_t c_PostClear = uint8_t(BatchSlot::PostClear);
        constexpr uint8_t c_Epilogue = uint8_t(BatchSlot::Epilogue);

        // An access that a barrier waits for or that needs a barrier's state.
        struct Acc
        {
            uint32_t sched;
            CommandQueue queue;
            int8_t frameDelta;
            nvrhi::ShaderType stages;
        };

        struct Step
        {
            uint32_t sched;
            CommandQueue queue;
            uint32_t resource;          // the occupant
            ResourceStates state;
            nvrhi::ShaderType stages;
            uint8_t slot;               // where the step's incoming barriers go
            bool write;
            bool fullWrite;
            bool relaxed;
            bool firstOfOccupant;
            bool clearSubStep;
        };

        struct Item
        {
            uint32_t begin;             // steps [begin, end)
            uint32_t end;
            bool write;
            ResourceStates state;
        };

        struct Prev
        {
            ResourceStates state = ResourceStates::Unknown;
            std::vector<Acc> accesses;
            bool entry = true;          // no access to wait for: an import's entry, or no use at all
            bool isWrite = false;
            bool uavWrite = false;
            bool uavAny = false;
            bool relaxed = false;
            uint32_t resource = c_None;
        };

        struct Object
        {
            std::vector<uint32_t> occupants;
            std::vector<uint32_t> uses;   // indices into CompileState::uses, in schedule order
            std::vector<uint32_t> useSched;
            uint32_t record = c_None;     // the first occupant
            bool cyclic = false;
            bool transientLike = false;
            uint64_t size = 0;
        };

        class TransitionBuilder
        {
        public:
            explicit TransitionBuilder(CompileState& state) : m_State(state), m_Profile(state.profile) { }

            void Run();

        private:
            void BuildObjects();
            void BuildQueueTables();
            void ProcessGroup(const std::vector<uint32_t>& group);
            std::vector<UnitKey> ComputeUnits(const std::vector<uint32_t>& group);
            uint32_t UsePlaneBits(const PlanUse& use, const ResourceRecord& record) const;
            bool Touches(const PlanUse& use, const ResourceRecord& record, const UnitKey& unit) const;
            void BuildSteps(const Object& object, const UnitKey& unit, std::vector<Step>& steps);
            void BuildItems(const std::vector<Step>& steps, bool isTexture, std::vector<Item>& items);
            bool GroupLegal(ResourceStates state, uint32_t queueMask, CommandQueue firstQueue) const;
            Prev MakePrev(const std::vector<Step>& steps, const Item& item, int8_t frameDelta) const;
            // Returns the unit's frame-start state.
            ResourceStates Walk(const Object& object, const UnitKey& unit, const std::vector<Step>& steps,
                                const std::vector<Item>& items, Prev prev, bool hasPredecessor);

            bool Legal(ResourceStates state, CommandQueue queue) const { return IsStateLegalOnQueue(state, queue, m_Profile); }
            void AddWait(uint32_t waiter, uint32_t producer, int8_t frameDelta);
            void CrossQueueWaits(const Prev& prev, const std::vector<Step>& steps, const Item& item);
            void Emit(uint32_t sched, uint8_t slot, CommandQueue queue, const PlanBarrier& barrier, const UnitKey& unit,
                      uint32_t pairId = c_None);
            uint32_t LatestGraphics(uint32_t low, uint32_t high) const;
            uint32_t LastGraphics() const;
            uint32_t LivePassesBetween(uint32_t a, uint32_t b, CommandQueue queue) const;

            struct PlaceArgs
            {
                ResourceStates before;
                ResourceStates after;
                nvrhi::BarrierFlags flags;
                BarrierReason reason;
                uint32_t resource;         // the occupant that needs `after`
                uint32_t resourceBefore;   // the occupant that held `before` (a wrapped transition names it)
                const UnitKey* unit;
                const Prev* prev;
                const std::vector<Step>* steps;
                const Item* item;          // consumers: the item's steps
                bool isTexture;
                uint64_t size;
            };
            // Places a transition before the item's first step (6.4, 6.8). Returns false on an error. `wrapped` is
            // set when the transition went to the end of the previous frame.
            bool Place(const PlaceArgs& args, bool& wrapped);
            void PlaceExit(const Prev& prev, ResourceStates exit, uint32_t resource, const UnitKey& unit);
            void EmitTransfer(const Prev& prev, CommandQueue from, ResourceStates before, ResourceStates after,
                              const Step& consumer, uint32_t resource, const UnitKey& unit);
            CommandQueue OwnerQueue(const Prev& prev) const;

            static nvrhi::ShaderType StagesOf(const std::vector<Acc>& accesses);

            CompileState& m_State;
            const BackendProfile& m_Profile;
            std::vector<Object> m_Objects;
            std::vector<uint32_t> m_ObjectOf;               // per resource
            std::vector<uint32_t> m_GraphicsLive;           // live graphics schedule indices, ascending
            std::vector<std::array<uint32_t, 3>> m_QueuePrefix;  // per schedule index: live passes before it, per queue
        };

        nvrhi::ShaderType TransitionBuilder::StagesOf(const std::vector<Acc>& accesses)
        {
            nvrhi::ShaderType stages = nvrhi::ShaderType::None;
            for (const Acc& access : accesses)
                stages = stages | access.stages;
            return stages;
        }

        void TransitionBuilder::BuildQueueTables()
        {
            const uint32_t count = uint32_t(m_State.schedule.size());
            m_QueuePrefix.assign(count + 1, { 0, 0, 0 });
            m_GraphicsLive.clear();
            for (uint32_t s = 0; s < count; ++s)
            {
                m_QueuePrefix[s + 1] = m_QueuePrefix[s];
                if (!m_State.scheduledLive[s])
                    continue;
                const CommandQueue queue = m_State.schedQueue[s];
                ++m_QueuePrefix[s + 1][uint32_t(queue)];
                if (queue == CommandQueue::Graphics)
                    m_GraphicsLive.push_back(s);
            }
        }

        uint32_t TransitionBuilder::LivePassesBetween(uint32_t a, uint32_t b, CommandQueue queue) const
        {
            if (b <= a + 1)
                return 0;
            return m_QueuePrefix[b][uint32_t(queue)] - m_QueuePrefix[a + 1][uint32_t(queue)];
        }

        uint32_t TransitionBuilder::LatestGraphics(uint32_t low, uint32_t high) const
        {
            // The latest live graphics pass g with low <= g < high.
            auto it = std::lower_bound(m_GraphicsLive.begin(), m_GraphicsLive.end(), high);
            if (it == m_GraphicsLive.begin())
                return c_None;
            --it;
            return *it >= low ? *it : c_None;
        }

        uint32_t TransitionBuilder::LastGraphics() const
        {
            return m_GraphicsLive.empty() ? c_None : m_GraphicsLive.back();
        }

        void TransitionBuilder::AddWait(uint32_t waiter, uint32_t producer, int8_t frameDelta)
        {
            if (frameDelta == 0 && producer >= waiter)
                return;
            m_State.waits.push_back(WaitEdge{ waiter, producer, frameDelta });
        }

        void TransitionBuilder::Emit(uint32_t sched, uint8_t slot, CommandQueue queue, const PlanBarrier& barrier,
                                     const UnitKey& unit, uint32_t pairId)
        {
            m_State.emitted.push_back(EmittedBarrier{ sched, slot, queue, barrier, unit, pairId });
        }

        void TransitionBuilder::BuildObjects()
        {
            const RenderGraph& graph = m_State.graph;
            const StructurePlan& structure = *m_State.structure;
            const uint32_t resourceCount = graph.ResourceCount();

            m_ObjectOf.assign(resourceCount, c_None);
            std::vector<uint32_t> objectOfPhysical(structure.physicals.size(), c_None);

            // Occupants in order of first use, so that a pooled slot's steps follow its occupants.
            std::vector<uint32_t> order(resourceCount);
            for (uint32_t r = 0; r < resourceCount; ++r)
                order[r] = r;
            std::stable_sort(order.begin(), order.end(), [this](uint32_t a, uint32_t b) {
                return m_State.first[a] < m_State.first[b];
            });

            for (uint32_t r : order)
            {
                const ResourceRecord& record = graph.GetResource(r);
                const uint32_t physical = structure.resourcePhysical[r];
                uint32_t object = c_None;
                if (physical != c_None)
                    object = objectOfPhysical[physical];
                if (object == c_None)
                {
                    object = uint32_t(m_Objects.size());
                    m_Objects.emplace_back();
                    Object& created = m_Objects.back();
                    created.record = r;
                    const bool transient = record.kind == ResourceKind::Transient || record.kind == ResourceKind::PassLocal;
                    created.transientLike = transient;
                    created.cyclic = (transient && !record.extracted) || record.kind == ResourceKind::Persistent ||
                        record.kind == ResourceKind::History;
                    uint64_t alignment = 0;
                    EstimateResourceSize(record, created.size, alignment);
                    if (physical != c_None)
                        objectOfPhysical[physical] = object;
                }
                m_Objects[object].occupants.push_back(r);
                m_ObjectOf[r] = object;
            }

            for (uint32_t u = 0; u < uint32_t(m_State.uses.size()); ++u)
            {
                // uses are stored per schedule index, in schedule order
                (void)u;
            }
            for (uint32_t s = 0; s < uint32_t(m_State.schedule.size()); ++s)
            {
                for (uint32_t u = m_State.useBegin[s]; u < m_State.useBegin[s] + m_State.useCount[s]; ++u)
                {
                    Object& object = m_Objects[m_ObjectOf[m_State.uses[u].resource]];
                    object.uses.push_back(u);
                    object.useSched.push_back(s);
                }
            }
        }

        uint32_t TransitionBuilder::UsePlaneBits(const PlanUse& use, const ResourceRecord& record) const
        {
            if (UnitPlaneCount(record, m_Profile) == 1)
                return use.region.planes != nvrhi::PlaneMask::None ? 1u : 0u;
            uint32_t bits = 0;
            if ((use.region.planes & nvrhi::PlaneMask::Depth) != 0)
                bits |= 1;
            if ((use.region.planes & nvrhi::PlaneMask::Stencil) != 0)
                bits |= 2;
            return bits;
        }

        bool TransitionBuilder::Touches(const PlanUse& use, const ResourceRecord& record, const UnitKey& unit) const
        {
            const uint32_t planes = UsePlaneBits(use, record);
            if (unit.plane == c_WholeUnit)
                return planes != 0;
            if ((planes & (1u << unit.plane)) == 0)
                return false;
            if (unit.mip < use.region.baseMip || unit.mip >= use.region.baseMip + use.region.mipCount)
                return false;
            return unit.slice0 >= use.region.baseSlice && unit.slice1 <= use.region.baseSlice + use.region.sliceCount;
        }

        std::vector<UnitKey> TransitionBuilder::ComputeUnits(const std::vector<uint32_t>& group)
        {
            const ResourceRecord& record = m_State.graph.GetResource(m_Objects[group[0]].record);
            const uint32_t planeCount = UnitPlaneCount(record, m_Profile);
            const uint32_t mips = MipCount(record);
            const uint32_t slices = SliceCount(record);
            const uint32_t allPlanes = planeCount == 2 ? 3u : 1u;

            bool whole = true;
            for (uint32_t object : group)
            {
                for (uint32_t u : m_Objects[object].uses)
                {
                    const PlanUse& use = m_State.uses[u];
                    if (use.region.baseMip != 0 || use.region.mipCount != mips || use.region.baseSlice != 0 ||
                        use.region.sliceCount != slices || UsePlaneBits(use, record) != allPlanes)
                    {
                        whole = false;
                        break;
                    }
                }
                if (!whole)
                    break;
            }

            std::vector<UnitKey> units;
            if (whole)
            {
                units.push_back(UnitKey{ c_WholeUnit, 0, 0, slices });
                return units;
            }

            std::vector<uint32_t> breaks;
            for (uint32_t plane = 0; plane < planeCount; ++plane)
            {
                for (uint32_t mip = 0; mip < mips; ++mip)
                {
                    breaks.clear();
                    breaks.push_back(0);
                    breaks.push_back(slices);
                    for (uint32_t object : group)
                    {
                        for (uint32_t u : m_Objects[object].uses)
                        {
                            const PlanUse& use = m_State.uses[u];
                            if ((UsePlaneBits(use, record) & (1u << plane)) == 0)
                                continue;
                            if (mip < use.region.baseMip || mip >= use.region.baseMip + use.region.mipCount)
                                continue;
                            breaks.push_back(use.region.baseSlice);
                            breaks.push_back(use.region.baseSlice + use.region.sliceCount);
                        }
                    }
                    std::sort(breaks.begin(), breaks.end());
                    breaks.erase(std::unique(breaks.begin(), breaks.end()), breaks.end());
                    for (size_t b = 0; b + 1 < breaks.size(); ++b)
                        units.push_back(UnitKey{ uint8_t(plane), mip, breaks[b], breaks[b + 1] });
                }
            }
            return units;
        }

        void TransitionBuilder::BuildSteps(const Object& object, const UnitKey& unit, std::vector<Step>& steps)
        {
            steps.clear();
            const ResourceRecord& record = m_State.graph.GetResource(object.record);
            const bool implicit = m_State.IsImplicit();
            uint32_t previousResource = c_None;

            size_t k = 0;
            while (k < object.uses.size())
            {
                const uint32_t sched = object.useSched[k];
                size_t end = k;
                while (end < object.uses.size() && object.useSched[end] == sched)
                    ++end;

                Step step{};
                bool any = false;
                bool clear = false;
                bool hasUavWrite = false;
                bool allUavWritesRelaxed = true;
                Access access = Access::None;
                Access clearAccess = Access::None;
                for (size_t i = k; i < end; ++i)
                {
                    const uint32_t u = object.uses[i];
                    const PlanUse& use = m_State.uses[u];
                    // Overlapping declarations of one pass were checked against the declared regions (phase 1); on a
                    // profile that ignores planes, disjoint planes share a unit and their states combine.
                    if (!Touches(use, record, unit))
                        continue;
                    if (!any)
                        step.resource = use.resource;
                    any = true;
                    access = access | use.access;
                    step.state = step.state | m_State.useState[u];
                    step.stages = step.stages | m_State.useStages[u];
                    step.write = step.write || use.isWrite;
                    step.fullWrite = step.fullWrite || (use.isWrite && ((use.flags & (AccessFlags::FullWrite | AccessFlags::Clear)) != AccessFlags::None));
                    if (use.hasClear)
                    {
                        clear = true;
                        clearAccess = clearAccess | use.access;
                    }
                    if ((use.access & Access::UAVWrite) != Access::None)
                    {
                        hasUavWrite = true;
                        allUavWritesRelaxed = allUavWritesRelaxed && (use.flags & AccessFlags::RelaxedOrder) != AccessFlags::None;
                    }
                }
                k = end;
                if (!any)
                    continue;

                step.sched = sched;
                step.queue = m_State.schedQueue[sched];
                step.slot = c_Prologue;
                step.relaxed = hasUavWrite && allUavWritesRelaxed;
                step.firstOfOccupant = step.resource != previousResource;
                previousResource = step.resource;

                if (clear && !implicit)
                {
                    // 6.9: the prologue moves the range into the clear state, the clear is recorded, and a post-clear
                    // batch moves it into the pass's state.
                    Step clearStep = step;
                    clearStep.state = ClearState(record.isBuffer, record.hasDepth, !record.isBuffer && record.texture.isRenderTarget, m_Profile);
                    clearStep.stages = nvrhi::ShaderType::None;
                    clearStep.write = true;
                    clearStep.fullWrite = true;
                    clearStep.relaxed = false;
                    clearStep.clearSubStep = true;
                    steps.push_back(clearStep);
                    step.slot = c_PostClear;
                    step.firstOfOccupant = false;
                    step.write = true;
                }
                steps.push_back(step);
            }
        }

        bool TransitionBuilder::GroupLegal(ResourceStates state, uint32_t queueMask, CommandQueue firstQueue) const
        {
            if (m_State.IsImplicit())
                return true;
            // Vulkan layouts are not queue-restricted; with one queue family only the transition into the group names
            // the state, on the first reader's queue. With separate families the acquire halves do too (nvrhi derives
            // a depth read's stages from the state), so the state must be legal on every queue then, as on D3D12.
            if (m_Profile.barrierModel == BarrierModel::Vulkan && !m_Profile.queueOwnershipTransfer)
                return Legal(state, firstQueue);
            for (uint32_t q = 0; q < 3; ++q)
            {
                if ((queueMask & (1u << q)) && !Legal(state, CommandQueue(q)))
                    return false;
            }
            return true;
        }

        void TransitionBuilder::BuildItems(const std::vector<Step>& steps, bool isTexture, std::vector<Item>& items)
        {
            items.clear();
            const uint32_t n = uint32_t(steps.size());
            uint32_t i = 0;
            while (i < n)
            {
                if (steps[i].write)
                {
                    items.push_back(Item{ i, i + 1, true, steps[i].state });
                    ++i;
                    continue;
                }
                uint32_t j = i;
                while (j < n && !steps[j].write && steps[j].resource == steps[i].resource)
                    ++j;

                // The epoch [i, j): fewest compatible groups, in schedule order (6.3).
                uint32_t groupBegin = i;
                ResourceStates groupState = steps[i].state;
                uint32_t groupQueues = 1u << uint32_t(steps[i].queue);
                for (uint32_t k = i + 1; k < j; ++k)
                {
                    const CombineResult combined = CombineReadStates(groupState, steps[k].state, isTexture, m_Profile);
                    const uint32_t queues = groupQueues | (1u << uint32_t(steps[k].queue));
                    if (combined.compatible && combined.extraCost <= m_Profile.lattice.transition &&
                        GroupLegal(combined.state, queues, steps[groupBegin].queue))
                    {
                        groupState = combined.state;
                        groupQueues = queues;
                        continue;
                    }
                    items.push_back(Item{ groupBegin, k, false, groupState });
                    groupBegin = k;
                    groupState = steps[k].state;
                    groupQueues = 1u << uint32_t(steps[k].queue);
                }
                items.push_back(Item{ groupBegin, j, false, groupState });
                i = j;
            }
        }

        Prev TransitionBuilder::MakePrev(const std::vector<Step>& steps, const Item& item, int8_t frameDelta) const
        {
            Prev prev;
            prev.entry = false;
            prev.state = item.state;
            prev.isWrite = item.write;
            prev.resource = steps[item.begin].resource;
            prev.uavAny = (item.state & ResourceStates::UnorderedAccess) != 0;
            prev.uavWrite = item.write && prev.uavAny;
            prev.relaxed = item.write && steps[item.begin].relaxed;
            for (uint32_t k = item.begin; k < item.end; ++k)
                prev.accesses.push_back(Acc{ steps[k].sched, steps[k].queue, frameDelta, steps[k].stages });
            return prev;
        }

        CommandQueue TransitionBuilder::OwnerQueue(const Prev& prev) const
        {
            uint32_t best = 0;
            int8_t bestDelta = -2;
            CommandQueue queue = CommandQueue::Graphics;
            for (const Acc& access : prev.accesses)
            {
                if (access.frameDelta > bestDelta || (access.frameDelta == bestDelta && access.sched >= best))
                {
                    best = access.sched;
                    bestDelta = access.frameDelta;
                    queue = access.queue;
                }
            }
            return queue;
        }

        void TransitionBuilder::CrossQueueWaits(const Prev& prev, const std::vector<Step>& steps, const Item& item)
        {
            for (uint32_t k = item.begin; k < item.end; ++k)
            {
                for (const Acc& access : prev.accesses)
                {
                    if (access.queue != steps[k].queue)
                        AddWait(steps[k].sched, access.sched, access.frameDelta);
                }
            }
        }

        bool TransitionBuilder::Place(const PlaceArgs& args, bool& wrapped)
        {
            wrapped = false;
            const std::vector<Step>& steps = *args.steps;
            const Item& item = *args.item;
            const Step& first = steps[item.begin];
            const Prev& prev = *args.prev;

            uint32_t sched = c_None;
            uint8_t slot = c_Prologue;
            CommandQueue queue = first.queue;
            uint32_t named = args.resource;

            if (Legal(args.before, first.queue) && Legal(args.after, first.queue))
            {
                sched = first.sched;
                slot = first.slot;
                queue = first.queue;
            }
            else
            {
                // 6.8: a transition that the consumer's queue cannot perform moves to the graphics queue, after
                // every access it waits for and before the consumer.
                uint32_t low = 0;
                bool anyThisFrame = false;
                for (const Acc& access : prev.accesses)
                {
                    if (access.frameDelta == 0)
                    {
                        low = std::max(low, access.sched);
                        anyThisFrame = true;
                    }
                }
                const uint32_t graphics = LatestGraphics(low, first.sched);
                if (graphics != c_None && Legal(args.before, CommandQueue::Graphics) && Legal(args.after, CommandQueue::Graphics))
                {
                    sched = graphics;
                    slot = c_Epilogue;
                    queue = CommandQueue::Graphics;
                }
                else if (!anyThisFrame && !prev.accesses.empty())
                {
                    // 6.12: the hand-off goes to the end of the previous frame, after the last access, on a queue that
                    // can perform it.
                    const Acc* last = &prev.accesses[0];
                    for (const Acc& access : prev.accesses)
                    {
                        if (access.sched > last->sched)
                            last = &access;
                    }
                    if (Legal(args.before, last->queue) && Legal(args.after, last->queue))
                    {
                        sched = last->sched;
                        slot = c_Epilogue;
                        queue = last->queue;
                        wrapped = true;
                        named = args.resourceBefore;
                    }
                }
            }

            if (sched == c_None)
            {
                m_State.Error("no queue can perform the transition of " + m_State.ResourceName(args.resource) + " from " +
                    StatesToString(args.before) + " to " + StatesToString(args.after) + " before " +
                    m_State.PassName(m_State.schedule[first.sched]) + " (6.8)");
                return false;
            }

            nvrhi::ShaderType consumerStages = nvrhi::ShaderType::None;
            for (uint32_t k = item.begin; k < item.end; ++k)
                consumerStages = consumerStages | steps[k].stages;

            PlanBarrier barrier;
            barrier.type = nvrhi::BarrierType::Transition;
            barrier.flags = args.flags;
            barrier.reason = args.reason;
            barrier.resource = named;
            barrier.before = args.before;
            barrier.after = args.after;
            barrier.stagesBefore = StagesOf(prev.accesses);
            barrier.stagesAfter = consumerStages;

            for (const Acc& access : prev.accesses)
            {
                if (access.queue != queue)
                    AddWait(sched, access.sched, wrapped ? int8_t(0) : access.frameDelta);
            }
            for (uint32_t k = item.begin; k < item.end; ++k)
            {
                if (steps[k].queue != queue)
                    AddWait(steps[k].sched, sched, wrapped ? int8_t(-1) : int8_t(0));
            }

            // 6.4: split when the begin and end points are far enough apart on one queue.
            bool split = false;
            uint32_t begin = 0;
            if (!wrapped && slot == c_Prologue && m_Profile.splitBarriers && !prev.accesses.empty())
            {
                bool sameQueueThisFrame = true;
                for (const Acc& access : prev.accesses)
                {
                    if (access.frameDelta != 0 || access.queue != queue)
                        sameQueueThisFrame = false;
                    else
                        begin = std::max(begin, access.sched);
                }
                split = sameQueueThisFrame && LivePassesBetween(begin, sched, queue) >= m_Profile.lattice.splitMinGap;
                if (split && m_Profile.barrierModel == BarrierModel::Vulkan)
                {
                    if (m_Profile.lattice.splitLayoutTransitionsOnly &&
                        (!args.isTexture || !ChangesVulkanLayout(args.before, args.after)))
                        split = false;
                    if (args.size < m_Profile.lattice.splitMinBytes)
                        split = false;
                }
            }

            if (split)
            {
                const uint32_t pairId = uint32_t(m_State.splits.size());
                m_State.splits.push_back(SplitPair{ begin, sched, false });
                PlanBarrier beginHalf = barrier;
                beginHalf.flags = beginHalf.flags | nvrhi::BarrierFlags::SplitBegin;
                PlanBarrier endHalf = barrier;
                endHalf.flags = endHalf.flags | nvrhi::BarrierFlags::SplitEnd;
                Emit(begin, c_Epilogue, queue, beginHalf, *args.unit, pairId);
                Emit(sched, slot, queue, endHalf, *args.unit, pairId);
            }
            else
            {
                Emit(sched, slot, queue, barrier, *args.unit);
            }
            ++m_State.transitionCount;
            return true;
        }

        void TransitionBuilder::EmitTransfer(const Prev& prev, CommandQueue from, ResourceStates before,
                                             ResourceStates after, const Step& consumer, uint32_t resource,
                                             const UnitKey& unit)
        {
            const Acc* last = nullptr;
            for (const Acc& access : prev.accesses)
            {
                if (access.queue == from && (!last || access.frameDelta > last->frameDelta ||
                    (access.frameDelta == last->frameDelta && access.sched > last->sched)))
                    last = &access;
            }
            if (!last)
                return;

            if (!Legal(before, from) || !Legal(after, consumer.queue))
            {
                m_State.Error("the ownership transfer of " + m_State.ResourceName(resource) + " from the " +
                    QueueName(from) + " queue to the " + QueueName(consumer.queue) + " queue names a state that one of "
                    "them cannot hold (" + StatesToString(before) + " to " + StatesToString(after) + ")");
                return;
            }

            nvrhi::ShaderType fromStages = nvrhi::ShaderType::None;
            for (const Acc& access : prev.accesses)
            {
                if (access.queue == from)
                    fromStages = fromStages | access.stages;
            }

            // 6.8: the release half in the producer's epilogue, the acquire half in the consumer's prologue; the
            // layout change is folded into the pair.
            PlanBarrier release;
            release.type = nvrhi::BarrierType::Transition;
            release.flags = nvrhi::BarrierFlags::QueueRelease;
            release.reason = BarrierReason::QueueTransfer;
            release.otherQueue = consumer.queue;
            release.resource = prev.resource != c_None ? prev.resource : resource;
            release.before = before;
            release.after = after;
            release.stagesBefore = fromStages;
            Emit(last->sched, c_Epilogue, from, release, unit);

            PlanBarrier acquire = release;
            acquire.flags = nvrhi::BarrierFlags::QueueAcquire;
            acquire.otherQueue = from;
            acquire.resource = resource;
            acquire.stagesBefore = nvrhi::ShaderType::None;
            acquire.stagesAfter = consumer.stages;
            Emit(consumer.sched, consumer.slot, consumer.queue, acquire, unit);

            AddWait(consumer.sched, last->sched, last->frameDelta);
            for (const Acc& access : prev.accesses)
            {
                if (access.queue != from && access.queue != consumer.queue)
                    AddWait(consumer.sched, access.sched, access.frameDelta);
            }
            ++m_State.transferCount;
        }

        void TransitionBuilder::PlaceExit(const Prev& prev, ResourceStates exit, uint32_t resource, const UnitKey& unit)
        {
            uint32_t sched = c_None;
            CommandQueue queue = CommandQueue::Graphics;
            const Acc* last = nullptr;
            for (const Acc& access : prev.accesses)
            {
                if (!last || access.sched > last->sched)
                    last = &access;
            }
            if (last && Legal(prev.state, last->queue) && Legal(exit, last->queue))
            {
                sched = last->sched;
                queue = last->queue;
            }
            else
            {
                const uint32_t graphics = LastGraphics();
                if (graphics != c_None && (!last || graphics >= last->sched) &&
                    Legal(prev.state, CommandQueue::Graphics) && Legal(exit, CommandQueue::Graphics))
                {
                    sched = graphics;
                    queue = CommandQueue::Graphics;
                }
            }
            if (sched == c_None)
            {
                m_State.Error("no queue can move " + m_State.ResourceName(resource) + " to its exit state " +
                    StatesToString(exit) + " at the end of the frame");
                return;
            }

            PlanBarrier barrier;
            barrier.type = nvrhi::BarrierType::Transition;
            barrier.reason = BarrierReason::ExitState;
            barrier.resource = resource;
            barrier.before = prev.state;
            barrier.after = exit;
            barrier.stagesBefore = StagesOf(prev.accesses);
            Emit(sched, c_Epilogue, queue, barrier, unit);
            for (const Acc& access : prev.accesses)
            {
                if (access.queue != queue)
                    AddWait(sched, access.sched, 0);
            }
            ++m_State.transitionCount;
        }

        static bool HasWriteState(ResourceStates state) noexcept
        {
            const ResourceStates writes = ResourceStates::UnorderedAccess | ResourceStates::RenderTarget |
                ResourceStates::DepthWrite | ResourceStates::CopyDest | ResourceStates::ResolveDest |
                ResourceStates::StreamOut | ResourceStates::AccelStructWrite;
            return (state & writes) != 0;
        }

        static bool IsLegacyDiscardable(ResourceStates state) noexcept
        {
            return state == ResourceStates::RenderTarget || state == ResourceStates::DepthWrite ||
                state == ResourceStates::UnorderedAccess;
        }

        ResourceStates TransitionBuilder::Walk(const Object& object, const UnitKey& unit, const std::vector<Step>& steps,
                                               const std::vector<Item>& items, Prev prev, bool hasPredecessor)
        {
            const RenderGraph& graph = m_State.graph;
            const ResourceRecord& record = graph.GetResource(object.record);
            const bool implicit = m_State.IsImplicit();
            const bool isTexture = !record.isBuffer;
            const bool frameStartWrap = object.cyclic && hasPredecessor;
            bool firstItem = true;
            ResourceStates entry = prev.state;

            for (const Item& item : items)
            {
                const Step& first = steps[item.begin];
                const ResourceRecord& occupant = graph.GetResource(first.resource);
                const bool discard = object.transientLike && first.firstOfOccupant;

                if (discard)
                {
                    if (!item.write)
                        m_State.Error(m_State.PassName(m_State.schedule[first.sched]) + " reads " +
                            m_State.ResourceName(first.resource) + " before any pass writes it (6.6)");
                    else if (!first.fullWrite)
                        m_State.Error(m_State.PassName(m_State.schedule[first.sched]) + ": the first access of transient " +
                            m_State.ResourceName(first.resource) + " must be a FullWrite or a Clear (6.6)");
                }

                const bool hazard = item.write || prev.isWrite;
                if (implicit)
                {
                    if (hazard && !prev.entry)
                        CrossQueueWaits(prev, steps, item);
                    prev = MakePrev(steps, item, 0);
                    firstItem = false;
                    continue;
                }

                BarrierReason reason;
                if (discard)
                    reason = BarrierReason::Discard;
                else if (firstItem && frameStartWrap && (occupant.kind == ResourceKind::Persistent || occupant.kind == ResourceKind::History))
                    reason = BarrierReason::Persistent;
                else if (item.write)
                    reason = prev.isWrite || prev.entry ? BarrierReason::WAW : BarrierReason::WAR;
                else
                    reason = prev.isWrite || prev.entry ? BarrierReason::RAW : BarrierReason::ReadGroupChange;

                bool discardFlag = discard && (m_Profile.barrierModel != BarrierModel::D3D12Legacy || IsLegacyDiscardable(item.state));
                const CommandQueue owner = prev.accesses.empty() ? first.queue : OwnerQueue(prev);
                const bool transfer = m_Profile.queueOwnershipTransfer && !discard && !prev.entry &&
                    !prev.accesses.empty() && owner != first.queue;

                // Whether a barrier placed below already orders the hazard. A legacy D3D12 transition between equal
                // states records nothing (nvrhi drops it and only discards), so it covers no hazard.
                bool placed = false;
                const bool stateChange = prev.state != item.state;
                if (transfer)
                {
                    EmitTransfer(prev, owner, prev.state, item.state, first, first.resource, unit);
                    placed = true;
                }
                else if (stateChange || discardFlag)
                {
                    PlaceArgs args{};
                    args.before = prev.state;
                    args.after = item.state;
                    args.flags = discardFlag ? nvrhi::BarrierFlags::Discard : nvrhi::BarrierFlags::None;
                    args.reason = reason;
                    args.resource = first.resource;
                    args.resourceBefore = prev.resource != c_None ? prev.resource : first.resource;
                    args.unit = &unit;
                    args.prev = &prev;
                    args.steps = &steps;
                    args.item = &item;
                    args.isTexture = isTexture;
                    args.size = object.size;
                    bool wrapped = false;
                    if (Place(args, wrapped) && wrapped)
                    {
                        // The hand-off happens at the end of the previous frame: the frame starts in the new state.
                        entry = item.state;
                    }
                    placed = stateChange || m_Profile.barrierModel != BarrierModel::D3D12Legacy;
                }

                if (!placed && hazard && !prev.entry)
                {
                    // Same state: the hazards that a state change would have covered (6.5).
                    bool sameQueuePrev = false;
                    for (const Acc& access : prev.accesses)
                        sameQueuePrev = sameQueuePrev || access.queue == first.queue;
                    if (sameQueuePrev)
                    {
                        if ((item.state & ResourceStates::UnorderedAccess) != 0 && prev.uavAny)
                        {
                            const bool relaxedRun = prev.isWrite && item.write && prev.relaxed && first.relaxed;
                            if (!relaxedRun)
                            {
                                PlanBarrier uav;
                                uav.type = nvrhi::BarrierType::Uav;
                                uav.reason = prev.isWrite ? BarrierReason::UavOrder : BarrierReason::WAR;
                                uav.resource = first.resource;
                                Emit(first.sched, first.slot, first.queue, uav, unit);
                            }
                        }
                        else if (m_Profile.barrierModel == BarrierModel::Vulkan && prev.isWrite && item.write)
                        {
                            // Vulkan orders nothing implicitly between two writes in one layout.
                            PlanBarrier same;
                            same.type = nvrhi::BarrierType::Transition;
                            same.reason = BarrierReason::WAW;
                            same.resource = first.resource;
                            same.before = prev.state;
                            same.after = item.state;
                            same.stagesBefore = StagesOf(prev.accesses);
                            same.stagesAfter = first.stages;
                            Emit(first.sched, first.slot, first.queue, same, unit);
                            ++m_State.transitionCount;
                        }
                    }
                    CrossQueueWaits(prev, steps, item);
                }
                else if (!placed && prev.entry && firstItem && !object.cyclic && !object.transientLike &&
                         m_Profile.barrierModel == BarrierModel::Vulkan && (item.write || HasWriteState(prev.state)))
                {
                    // An import first used in its entry state: its previous user's accesses come from another
                    // submission, which Vulkan does not order (D3D12 orders and flushes between ExecuteCommandLists
                    // calls). A same-state barrier from the entry state's stages makes them a dependency.
                    PlanBarrier same;
                    same.type = nvrhi::BarrierType::Transition;
                    same.reason = item.write ? BarrierReason::WAW : BarrierReason::RAW;
                    same.resource = first.resource;
                    same.before = prev.state;
                    same.after = item.state;
                    same.stagesAfter = first.stages;
                    Emit(first.sched, first.slot, first.queue, same, unit);
                    ++m_State.transitionCount;
                }

                // Readers of one group on queues of different families hand the resource over between them.
                if (!item.write && m_Profile.queueOwnershipTransfer)
                {
                    for (uint32_t k = item.begin + 1; k < item.end; ++k)
                    {
                        if (steps[k].queue == steps[k - 1].queue)
                            continue;
                        Prev readers;
                        readers.entry = false;
                        readers.state = item.state;
                        readers.resource = steps[k - 1].resource;
                        for (uint32_t r = item.begin; r < k; ++r)
                            readers.accesses.push_back(Acc{ steps[r].sched, steps[r].queue, 0, steps[r].stages });
                        EmitTransfer(readers, steps[k - 1].queue, item.state, item.state, steps[k], steps[k].resource, unit);
                    }
                }

                prev = MakePrev(steps, item, 0);
                firstItem = false;
            }

            if (implicit)
                return entry;

            // Imports and extracted transients end the frame in their declared exit state.
            if (!object.cyclic && !items.empty())
            {
                const ResourceRecord& last = graph.GetResource(steps[items.back().begin].resource);
                const ResourceStates exit = last.exit;
                if (exit != ResourceStates::Unknown && exit != prev.state)
                    PlaceExit(prev, exit, steps[items.back().begin].resource, unit);
            }
            return entry;
        }

        void TransitionBuilder::ProcessGroup(const std::vector<uint32_t>& group)
        {
            const RenderGraph& graph = m_State.graph;
            const ResourceRecord& record = graph.GetResource(m_Objects[group[0]].record);
            const bool isTexture = !record.isBuffer;
            const std::vector<UnitKey> units = ComputeUnits(group);

            std::vector<std::vector<Step>> steps(group.size());
            std::vector<std::vector<Item>> items(group.size());

            for (const UnitKey& unit : units)
            {
                for (size_t g = 0; g < group.size(); ++g)
                {
                    BuildSteps(m_Objects[group[g]], unit, steps[g]);
                    BuildItems(steps[g], isTexture, items[g]);
                }

                for (size_t g = 0; g < group.size(); ++g)
                {
                    const Object& object = m_Objects[group[g]];
                    if (items[g].empty())
                        continue;
                    const ResourceRecord& first = graph.GetResource(object.record);

                    Prev prev;
                    bool hasPredecessor = false;
                    if (object.cyclic)
                    {
                        // The predecessor: this object's last item in the previous frame, or for history layer k the
                        // most recent earlier frame's layer that used the same physical texture (5.2).
                        size_t source = g;
                        const size_t layers = group.size();
                        for (size_t back = 1; back <= layers; ++back)
                        {
                            const size_t candidate = (g + layers - back % layers) % layers;
                            if (!items[candidate].empty())
                            {
                                source = candidate;
                                break;
                            }
                        }
                        prev = MakePrev(steps[source], items[source].back(), -1);
                        hasPredecessor = true;
                    }
                    else
                    {
                        prev.entry = true;
                        prev.state = first.entry;
                        if (first.kind == ResourceKind::Transient && first.extracted)
                            prev.state = first.exit;
                    }

                    // PlanResource::entry is per resource: the first unit's frame-start state. The verifier does not
                    // rely on it (its first replayed frame learns every state).
                    const ResourceStates entry = Walk(object, unit, steps[g], items[g], prev, hasPredecessor);
                    if (&unit == &units[0])
                    {
                        for (uint32_t occupant : object.occupants)
                            m_State.entryState[occupant] = entry;
                    }
                    if (!m_State.IsImplicit())
                        m_State.entryUnits.push_back(CompileState::EntryUnit{ object.record, unit, entry });
                }
            }
        }

        void TransitionBuilder::Run()
        {
            CompileState& state = m_State;
            const RenderGraph& graph = state.graph;
            state.entryState.assign(graph.ResourceCount(), ResourceStates::Unknown);
            for (uint32_t r = 0; r < graph.ResourceCount(); ++r)
            {
                const ResourceRecord& record = graph.GetResource(r);
                if (record.kind == ResourceKind::Imported || record.kind == ResourceKind::HostKept)
                    state.entryState[r] = record.entry;
            }

            BuildQueueTables();
            BuildObjects();

            // Groups: one object each, except the layers of one history, which form a ring.
            std::vector<uint8_t> done(m_Objects.size(), 0);
            std::vector<uint32_t> group;
            for (uint32_t o = 0; o < uint32_t(m_Objects.size()); ++o)
            {
                if (done[o] || m_Objects[o].uses.empty())
                    continue;
                group.clear();
                const ResourceRecord& record = graph.GetResource(m_Objects[o].record);
                if (record.kind == ResourceKind::History)
                {
                    group.assign(record.historyLayerCount, c_None);
                    for (uint32_t p = 0; p < uint32_t(m_Objects.size()); ++p)
                    {
                        const ResourceRecord& other = graph.GetResource(m_Objects[p].record);
                        if (other.kind == ResourceKind::History && other.registryEntry == record.registryEntry &&
                            other.historyLayer < group.size())
                            group[other.historyLayer] = p;
                    }
                    bool complete = true;
                    for (uint32_t member : group)
                        complete = complete && member != c_None;
                    if (!complete)
                    {
                        group.clear();
                        group.push_back(o);
                    }
                }
                else
                {
                    group.push_back(o);
                }
                for (uint32_t member : group)
                    done[member] = 1;
                ProcessGroup(group);
            }

            // Aliasing barriers (6.7) at each placed resource's first use.
            if (!state.IsImplicit() && (state.options.debugFlags & DebugFlags::SkipAliasingBarriers) == DebugFlags::None)
            {
                const StructurePlan& structure = *state.structure;
                for (uint32_t r = 0; r < graph.ResourceCount(); ++r)
                {
                    if (!structure.resourceAliased[r] || state.first[r] == c_None)
                        continue;
                    PlanBarrier alias;
                    alias.type = nvrhi::BarrierType::Aliasing;
                    alias.reason = BarrierReason::Alias;
                    alias.resource = r;
                    alias.aliasBefore = structure.overlaps[r].size() == 1 ? structure.overlaps[r][0] : c_None;
                    Emit(state.first[r], c_Prologue, state.schedQueue[state.first[r]], alias, UnitKey{ c_WholeUnit, 0, 0, 0 });
                }
            }

            // 11.4 BarrierAfterEveryPass: a barrier on every UAV access after every pass.
            if (!state.IsImplicit() && (state.options.debugFlags & DebugFlags::BarrierAfterEveryPass) != DebugFlags::None)
            {
                for (uint32_t s = 0; s < uint32_t(state.schedule.size()); ++s)
                {
                    if (!state.scheduledLive[s])
                        continue;
                    PlanBarrier full;
                    full.type = nvrhi::BarrierType::Uav;
                    full.reason = BarrierReason::Debug;
                    full.resource = c_None;
                    Emit(s, c_Epilogue, state.schedQueue[s], full, UnitKey{ c_WholeUnit, 0, 0, 0 });
                }
            }

            // Load and store operations of the attachments (6.9).
            const uint32_t schedCount = uint32_t(state.schedule.size());
            state.load.assign(schedCount, {});
            state.store.assign(schedCount, {});
            state.clearMask.assign(schedCount, 0);
            for (uint32_t s = 0; s < schedCount; ++s)
            {
                state.load[s].fill(LoadOp::Load);
                state.store[s].fill(StoreOp::Store);
                for (uint32_t u = state.useBegin[s]; u < state.useBegin[s] + state.useCount[s]; ++u)
                {
                    const PlanUse& use = state.uses[u];
                    if (use.attachment == c_NoAttachment)
                        continue;
                    const ResourceRecord& record = graph.GetResource(use.resource);
                    const bool transient = (record.kind == ResourceKind::Transient || record.kind == ResourceKind::PassLocal) &&
                        !record.extracted;
                    if (use.hasClear)
                    {
                        state.load[s][use.attachment] = LoadOp::Clear;
                        state.clearMask[s] |= 1u << use.attachment;
                    }
                    else if (transient && state.first[use.resource] == s)
                        state.load[s][use.attachment] = LoadOp::DontCare;
                    if (transient && state.last[use.resource] == s)
                        state.store[s][use.attachment] = StoreOp::DontCare;
                }
            }
        }
    }

    void RunTransitions(CompileState& state)
    {
        TransitionBuilder builder(state);
        builder.Run();
    }
}
