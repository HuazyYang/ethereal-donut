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

// Observers (aggregate docs/ecs-design.md 6.6; plan step E1.4). Module-private (com-interface-abi.md section 6).
//
// The storage reports every change through the Observe* seams of WorldImpl, in the order it applies them. A report
// names rows of one chunk: [firstRow, firstRow + count), and for a Sparse component the dense values
// [firstDense, firstDense + count) of its set, which line up with the rows. Delivery follows the reports:
// - A removal (OnRemove, OnUnmatch) is delivered at once, inside the report, while its rows and values are in place:
//   the storage destroys or moves them after the report returns (EcsWorld.h: "the batch still reads it").
// - An addition or a set (OnAdd, OnMatch, OnSet) is held, so that the reports after it can extend it: the next rows of
//   the same chunk with the same event and archetypes (and the next dense values). The held batch is delivered before
//   anything else is delivered, before a removal moves rows of its archetype or values of its Sparse set, at the start
//   of a batch operation that writes values without reporting them first or copies values (ObserveValueAccess), and
//   by FlushObservers. So every batch reads its rows as their reports left them, and the observers see the changes in
//   the order they were made. Nothing is queued: at most one batch is held, so delivery never allocates.
// - Per batch, the observers that take it run in registration order. Filters are per archetype.
// Two benign consequences, by design: a callback that reads world state outside its batch (other entities, other
// components) may see later changes that nobody observes and that were made before the held batch was delivered; and
// within one batch operation that names an entity twice (a coalesced Add run [e1, e1], or a Set run [e0, e0]), the
// first callback already shows the value the second one wrote, because the operation writes it before (or without)
// reporting it. Both callbacks still come, in order, for a Set.
//
// The cascade (FlushObservers): observers record into one of two command buffers; playing one back (the observers
// that it triggers record into the other) is a round. After c_MaxObserverRounds rounds the commands left are dropped
// and the call fails with FE_GENERIC_ERROR.

#pragma once

#include "EcsWorldImpl.h"

#include <nvrhi/core/autoptr.h>

namespace donut::engine::ecs
{
    class CommandBufferImpl;
    class QueryImpl;

    // A cascade stops when commands remain after this many playbacks of the observers' commands (EcsWorld.h).
    constexpr uint32_t c_MaxObserverRounds = 64;

    enum class ReportKind : uint8_t
    {
        None,
        RowsAdded,        // the rows are complete in `archetype`; they came from `other` (c_NoArchetype: created)
        RowsRemoving,     // the rows leave `archetype` for `other` (c_NoArchetype: destroyed)
        SparseAdded,      // `component` (Sparse) was added to the entity of the row
        SparseRemoving,   // `component` (Sparse) is about to be destroyed
        Set,              // `component` was written by ICommandBuffer::SetComponents
    };

    struct ObserverReport
    {
        ReportKind kind = ReportKind::None;
        uint8_t reserved[3] = {};
        ComponentId component = ComponentId::Invalid;   // SparseAdded, SparseRemoving, Set
        uint32_t archetype = 0;
        uint32_t chunk = 0;
        uint32_t firstRow = 0;
        uint32_t count = 0;
        uint32_t other = c_NoArchetype;                  // RowsAdded, RowsRemoving
        uint32_t firstDense = 0;   // a Sparse component's first value; firstRow for every other report
    };

    // One registration.
    struct Observer
    {
        ObserverId id = ObserverId::Invalid;
        ObserverEvent event = ObserverEvent::OnAdd;
        Access access = Access::Read;
        uint8_t reserved[2] = {};
        ComponentId component = ComponentId::Invalid;   // Invalid for OnMatch, OnUnmatch
        const ComponentType* type = nullptr;            // the observed component's type
        const SparseSet* sparse = nullptr;              // a Sparse observed component's set
        nvrhi::AutoPtr<IObserver> callback;
        nvrhi::AutoPtr<IQuery> filterReference;         // QueryWorldReference::NotHeld; null without filter terms
        QueryImpl* filter = nullptr;                    // the same object
        Vector<QueryTerm> dataTerms;                    // the filter's data terms in term order: the batch's columns
        nvrhi::string name;
    };

    class ObserverRegistry
    {
    public:
        struct Counters
        {
            uint64_t deliveries = 0;        // IObserver::OnEntities calls
            uint64_t batches = 0;           // batches that some observer took
            uint64_t extended = 0;          // reports that extended the held batch
            uint64_t rounds = 0;            // playbacks of the observers' commands
            uint64_t cascadeFailures = 0;   // cascades stopped after c_MaxObserverRounds rounds
        };

        explicit ObserverRegistry(WorldImpl& world) noexcept;
        ~ObserverRegistry();
        ObserverRegistry(const ObserverRegistry&) = delete;
        ObserverRegistry& operator=(const ObserverRegistry&) = delete;
        // The two command buffers; FE_OUT_OF_MEMORY.
        nvrhi::FRESULT Initialize() noexcept;

        // IWorld::AddObserver and RemoveObserver after their phase and observer checks.
        nvrhi::FRESULT Add(const ObserverDesc& desc, IObserver* observer, ObserverId& outId) noexcept;
        nvrhi::FRESULT Remove(ObserverId id) noexcept;
        uint32_t GetObserverCount() const noexcept { return uint32_t(m_Observers.size()); }

        // The seams (EcsWorldImpl.h).
        void RowsAdded(uint32_t archetype, uint32_t chunk, uint32_t firstRow, uint32_t count,
                       uint32_t fromArchetype) noexcept;
        void RowsRemoving(uint32_t archetype, uint32_t chunk, uint32_t firstRow, uint32_t count,
                          uint32_t toArchetype) noexcept;
        void Sparse(ComponentId component, Entity entity, bool added) noexcept;
        void Set(ComponentId component, Entity entity) noexcept;
        void DeliverHeld() noexcept;
        // Delivers the held batch and plays the observers' commands until none remain. Returns the first failure of
        // an observer or a playback since the last call, or FE_GENERIC_ERROR when the cascade does not settle.
        nvrhi::FRESULT RunCascade() noexcept;

        const Counters& GetCounters() const noexcept { return m_Counters; }
        const ObserverReport& GetHeld() const noexcept { return m_Held; }
        // The buffer the observers record into now.
        ICommandBuffer* GetRecordingBuffer() const noexcept;

    private:
        nvrhi::FRESULT Validate(const ObserverDesc& desc, IObserver* observer) noexcept;
        nvrhi::FRESULT ValidateFilter(const ObserverDesc& desc) noexcept;
        bool FilterMatches(const Observer& observer, const Archetype& archetype) const noexcept;
        bool Takes(const Observer& observer, const ObserverReport& report) const noexcept;
        // takers[i]: whether m_Observers[i] takes the report (Takes runs once per observer and report); true for any.
        bool CollectTakers(const ObserverReport& report, Vector<uint8_t>& takers) const noexcept;
        // An OnAdd, OnMatch or OnSet report: extends the held batch, or delivers it and holds this one.
        void Hold(const ObserverReport& report) noexcept;
        // Every observer that takes the report, in registration order.
        void Deliver(const ObserverReport& report, const Vector<uint8_t>& takers) noexcept;
        void Call(const Observer& observer, const ObserverReport& report) noexcept;
        // One round: the seam to E1.3's playback.
        nvrhi::FRESULT PlayRound(CommandBufferImpl* buffer, bool firstRound, uint64_t& outCommands) noexcept;
        static void DestroyObserver(Observer* observer) noexcept;

        WorldImpl& m_World;
        Vector<Observer*> m_Observers;     // registration order, which is id order
        uint32_t m_NextId = 0;
        ObserverReport m_Held;
        Vector<uint8_t> m_HeldTakers;      // CollectTakers of the held batch; one per observer
        Vector<uint8_t> m_ReportTakers;    // ... of the report at hand
        Vector<void*> m_Columns;           // a batch's columns; capacity for the widest observer
        nvrhi::AutoPtr<ICommandBuffer> m_BufferReferences[2];   // CommandBufferWorldReference::NotHeld
        CommandBufferImpl* m_Buffers[2] = {};
        uint32_t m_Recording = 0;          // the buffer observers record into
        nvrhi::FRESULT m_Result = nvrhi::FS_OK;
        Counters m_Counters;
    };
}
