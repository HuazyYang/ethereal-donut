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

// Observers (aggregate docs/ecs-design.md 6.6, 11.3; plan step E1.4). See EcsObserverImpl.h.

#include "EcsObserverImpl.h"
#include "EcsCommandBufferImpl.h"
#include "EcsQueryImpl.h"

#include <algorithm>
#include <new>

namespace donut::engine::ecs
{
    namespace
    {
        void KeepFirstFailure(nvrhi::FRESULT& result, nvrhi::FRESULT next) noexcept
        {
            if (NVRHI_SUCCEEDED(result) && NVRHI_FAILED(next))
                result = next;
        }

        const char* NameOf(const ComponentType& type) noexcept
        {
            return type.desc.name ? type.desc.name : "(unnamed)";
        }

        const char* EventName(ObserverEvent event) noexcept
        {
            switch (event)
            {
            case ObserverEvent::OnAdd: return "OnAdd";
            case ObserverEvent::OnRemove: return "OnRemove";
            case ObserverEvent::OnSet: return "OnSet";
            case ObserverEvent::OnMatch: return "OnMatch";
            case ObserverEvent::OnUnmatch: return "OnUnmatch";
            default: return "?";
            }
        }

        bool IsMatchEvent(ObserverEvent event) noexcept
        {
            return event == ObserverEvent::OnMatch || event == ObserverEvent::OnUnmatch;
        }

        // A batch column: the value of the batch's first row, the one value of a Shared or Chunk component, or null
        // for a tag.
        void* ColumnAt(const Archetype& archetype, uint32_t column, uint32_t chunk, uint32_t row) noexcept
        {
            const ArchetypeColumn& c = archetype.columns[column];
            switch (c.kind)
            {
            case StorageKind::Data:
            case StorageKind::Buffer:
                return static_cast<uint8_t*>(archetype.ColumnData(chunk, column)) + size_t(row) * c.stride;
            case StorageKind::Shared:
            case StorageKind::Chunk:
                return archetype.ColumnData(chunk, column);
            default:
                return nullptr;
            }
        }

        // Whether `next` continues `held`: the same event on the next rows of the same chunk, from the same archetype
        // (or of the same component), with the next dense values.
        bool Extends(const ObserverReport& held, const ObserverReport& next) noexcept
        {
            return held.kind == next.kind && held.component == next.component && held.archetype == next.archetype &&
                   held.chunk == next.chunk && held.other == next.other &&
                   held.firstRow + held.count == next.firstRow && held.firstDense + held.count == next.firstDense;
        }
    }

    // ---------------------------------------------------------------------------------------------- lifetime

    ObserverRegistry::ObserverRegistry(WorldImpl& world) noexcept
        : m_World(world)
    {
    }

    ObserverRegistry::~ObserverRegistry()
    {
        // Each record is taken out before its references go: an observer's destructor may run user code.
        while (!m_Observers.empty())
        {
            Observer* observer = m_Observers.back();
            m_Observers.pop_back();
            DestroyObserver(observer);
        }
        // Not held, so their release leaves the world alone; their values die with the component types' hooks.
        m_Buffers[0] = m_Buffers[1] = nullptr;
        m_BufferReferences[0] = nullptr;
        m_BufferReferences[1] = nullptr;
    }

    nvrhi::FRESULT ObserverRegistry::Initialize() noexcept
    {
        // The batch columns of observers without filter data terms; never indexed past their count, but not null.
        m_Columns.reserve(1);
        for (uint32_t i = 0; i < 2; ++i)
        {
            const nvrhi::FRESULT result = CreateCommandBufferObject(m_World, CommandBufferWorldReference::NotHeld,
                                                                    m_BufferReferences[i].ReleaseAndGetAddressOf());
            if (NVRHI_FAILED(result))
                return result;
            m_Buffers[i] = donut::query_cast<CommandBufferImpl>(m_BufferReferences[i].Get());
            NVRHI_ASSERT(m_Buffers[i] != nullptr);
        }
        return nvrhi::FS_OK;
    }

    void ObserverRegistry::DestroyObserver(Observer* observer) noexcept
    {
        observer->~Observer();
        HeapFree(observer);
    }

    void DestroyObserverRegistry(ObserverRegistry* registry) noexcept
    {
        if (!registry)
            return;
        registry->~ObserverRegistry();
        HeapFree(registry);
    }

    ICommandBuffer* ObserverRegistry::GetRecordingBuffer() const noexcept
    {
        return m_Buffers[m_Recording];
    }

    // ------------------------------------------------------------------------------------------ registration

    nvrhi::FRESULT ObserverRegistry::Validate(const ObserverDesc& desc, IObserver* observer) noexcept
    {
        const char* name = desc.debugName ? desc.debugName : "";
        auto fail = [&](const char* reason) noexcept
        {
            m_World.ReportValidation("AddObserver", "observer '%s': %s", name, reason);
            return nvrhi::FE_INVALID_ARGS;
        };
        if (!observer)
            return fail("the IObserver is null");
        if (uint8_t(desc.event) > uint8_t(ObserverEvent::OnUnmatch))
            return fail("unknown event");
        if (uint8_t(desc.access) > uint8_t(Access::Write))
            return fail("unknown access");
        if (desc.filterTermCount != 0 && !desc.filterTerms)
            return fail("filterTermCount without filterTerms");
        if (IsMatchEvent(desc.event))
        {
            if (desc.component != ComponentId::Invalid)
                return fail("OnMatch and OnUnmatch observe the filter terms: the component must be Invalid");
            if (desc.filterTermCount == 0)
                return fail("OnMatch and OnUnmatch need filter terms (the query whose membership is observed)");
            return nvrhi::FS_OK;
        }
        const ComponentType* type = m_World.GetType(desc.component);
        if (!type || type->kind == StorageKind::Resource)
            return fail("the observed component is not a registered component (resources are not on entities)");
        if (desc.event == ObserverEvent::OnSet &&
            (type->kind == StorageKind::Buffer || type->kind == StorageKind::Chunk))
            return fail("OnSet on a Buffer or Chunk component: ICommandBuffer::SetComponents never sets them");
        if (desc.access == Access::Write && type->kind == StorageKind::Shared)
            return fail("Shared values are read-only (4.4); Write access is refused");
        // A zero-initialised ObserverDesc has Access::Filter: for a component with data it would hand out no column.
        if (desc.access == Access::Filter && type->desc.size != 0)
            return fail("Access::Filter gives no column: observe a component with data with Read or Write");
        return nvrhi::FS_OK;
    }

    nvrhi::FRESULT ObserverRegistry::ValidateFilter(const ObserverDesc& desc) noexcept
    {
        // After CreateQueryObject accepted the terms: what observers add. A filter is matched per archetype, as the
        // structural reports see entities, so terms that act per row or per version are refused.
        const char* name = desc.debugName ? desc.debugName : "";
        for (uint32_t i = 0; i < desc.filterTermCount; ++i)
        {
            const QueryTerm& term = desc.filterTerms[i];
            const ComponentType& type = *m_World.GetType(term.component);
            const char* reason = nullptr;
            if (type.kind == StorageKind::Sparse)
                reason = "Sparse components are per row, not per archetype: they cannot filter observers";
            else if (term.op == TermOp::Changed || term.op == TermOp::Added)
                reason = "Changed and Added compare versions, which observers do not see";
            else if (term.op == TermOp::Enabled || term.op == TermOp::Disabled)
                reason = "enabled bits change without a structural report: Enabled and Disabled cannot filter "
                         "observers";
            else if (!IsMatchEvent(desc.event) && term.op == TermOp::Without && term.component == desc.component)
                reason = "the filter excludes the observed component: the observer could never run";
            if (reason)
            {
                m_World.ReportValidation("AddObserver", "observer '%s', filter term %u (%s): %s", name, i,
                                         NameOf(type), reason);
                return nvrhi::FE_INVALID_ARGS;
            }
        }
        return nvrhi::FS_OK;
    }

    nvrhi::FRESULT ObserverRegistry::Add(const ObserverDesc& desc, IObserver* observer, ObserverId& outId) noexcept
    {
        outId = ObserverId::Invalid;
        nvrhi::FRESULT result = Validate(desc, observer);
        if (NVRHI_FAILED(result))
            return result;
        if (m_NextId == uint32_t(ObserverId::Invalid))
        {
            m_World.ReportValidation("AddObserver", "the world has handed out every observer id");
            return nvrhi::FE_OUT_OF_MEMORY;
        }

        nvrhi::AutoPtr<IQuery> filter;
        if (desc.filterTermCount != 0)
        {
            QueryDesc queryDesc{};
            queryDesc.terms = desc.filterTerms;
            queryDesc.termCount = desc.filterTermCount;
            queryDesc.debugName = desc.debugName;
            // Not held: the world owns its observers, so the filter must not keep the world alive (E1.2).
            result = CreateQueryObject(m_World, queryDesc, QueryWorldReference::NotHeld,
                                       filter.ReleaseAndGetAddressOf());
            if (NVRHI_FAILED(result))
                return result;
            result = ValidateFilter(desc);
            if (NVRHI_FAILED(result))
                return result;
        }

        void* memory = HeapAllocate(sizeof(Observer), alignof(Observer));
        if (!memory)
            return nvrhi::FE_OUT_OF_MEMORY;
        Observer* record = new (memory) Observer();
        record->id = ObserverId(m_NextId);
        record->event = desc.event;
        record->access = desc.access;
        record->name = desc.debugName ? desc.debugName : "";
        record->callback = observer;
        if (!IsMatchEvent(desc.event))
        {
            record->component = desc.component;
            record->type = m_World.GetType(desc.component);
            record->sparse = record->type->kind == StorageKind::Sparse ? m_World.GetSparseSet(desc.component) : nullptr;
        }
        if (filter)
        {
            record->filter = donut::query_cast<QueryImpl>(filter.Get());
            NVRHI_ASSERT(record->filter != nullptr);
            record->filterReference = filter;
            for (uint32_t i = 0; i < desc.filterTermCount; ++i)
                if (IsDataTerm(desc.filterTerms[i]))
                    record->dataTerms.push_back(desc.filterTerms[i]);
        }

        // Registration allocates what delivery needs, so that delivery never does.
        const size_t columns = record->dataTerms.size() + 1;
        if (m_Columns.capacity() < columns)
            m_Columns.reserve(columns);
        m_Columns.resize(m_Columns.capacity(), nullptr);
        m_Observers.push_back(record);
        m_HeldTakers.resize(m_Observers.size(), 0);
        m_ReportTakers.resize(m_Observers.size(), 0);
        ++m_NextId;
        m_World.m_ObserverCount = GetObserverCount();
        outId = record->id;
        return nvrhi::FS_OK;
    }

    nvrhi::FRESULT ObserverRegistry::Remove(ObserverId id) noexcept
    {
        auto found = std::lower_bound(m_Observers.begin(), m_Observers.end(), id,
            [](const Observer* observer, ObserverId value) { return uint32_t(observer->id) < uint32_t(value); });
        if (found == m_Observers.end() || (*found)->id != id)
        {
            m_World.ReportValidation("RemoveObserver", "observer %u is not registered (never added, or removed)",
                                     uint32_t(id));
            return nvrhi::FE_INVALID_ARGS;
        }
        Observer* observer = *found;
        m_Observers.erase(found);
        m_HeldTakers.resize(m_Observers.size());   // nothing is held between calls
        m_ReportTakers.resize(m_Observers.size());
        m_World.m_ObserverCount = GetObserverCount();
        DestroyObserver(observer);   // last: its references may run user code
        return nvrhi::FS_OK;
    }

    // ---------------------------------------------------------------------------------------------- matching

    bool ObserverRegistry::FilterMatches(const Observer& observer, const Archetype& archetype) const noexcept
    {
        return !observer.filter || observer.filter->Matches(archetype);
    }

    bool ObserverRegistry::Takes(const Observer& observer, const ObserverReport& report) const noexcept
    {
        const Archetype& archetype = m_World.GetArchetypeStorage(report.archetype);
        switch (report.kind)
        {
        case ReportKind::RowsAdded:
        case ReportKind::RowsRemoving:
        {
            // A move adds and removes exactly the components that its two archetypes do not share.
            const bool added = report.kind == ReportKind::RowsAdded;
            const Archetype* other =
                report.other == c_NoArchetype ? nullptr : &m_World.GetArchetypeStorage(report.other);
            if (observer.event == (added ? ObserverEvent::OnAdd : ObserverEvent::OnRemove))
                return !observer.sparse && archetype.FindColumn(observer.component) != c_NoColumn &&
                       (!other || other->FindColumn(observer.component) == c_NoColumn) &&
                       FilterMatches(observer, archetype);
            if (observer.event == (added ? ObserverEvent::OnMatch : ObserverEvent::OnUnmatch))
                return observer.filter->Matches(archetype) && (!other || !observer.filter->Matches(*other));
            return false;
        }
        case ReportKind::SparseAdded:
            return observer.event == ObserverEvent::OnAdd && observer.component == report.component &&
                   FilterMatches(observer, archetype);
        case ReportKind::SparseRemoving:
            return observer.event == ObserverEvent::OnRemove && observer.component == report.component &&
                   FilterMatches(observer, archetype);
        case ReportKind::Set:
            return observer.event == ObserverEvent::OnSet && observer.component == report.component &&
                   FilterMatches(observer, archetype);
        default:
            return false;
        }
    }

    bool ObserverRegistry::CollectTakers(const ObserverReport& report, Vector<uint8_t>& takers) const noexcept
    {
        bool any = false;
        for (size_t i = 0; i < m_Observers.size(); ++i)
        {
            const bool takes = Takes(*m_Observers[i], report);
            takers[i] = uint8_t(takes);
            any = any || takes;
        }
        return any;
    }

    // ---------------------------------------------------------------------------------------------- delivery

    void ObserverRegistry::Call(const Observer& observer, const ObserverReport& report) noexcept
    {
        const Archetype& archetype = m_World.GetArchetypeStorage(report.archetype);
        const uint64_t version = m_World.GetVersion();
        size_t n = 0;
        if (!IsMatchEvent(observer.event))
        {
            void* column = nullptr;
            if (observer.access != Access::Filter)
            {
                if (observer.sparse)
                {
                    if (observer.type->desc.size != 0)
                        column = observer.sparse->Value(report.firstDense);
                }
                else
                {
                    const uint32_t c = archetype.FindColumn(observer.component);
                    column = ColumnAt(archetype, c, report.chunk, report.firstRow);
                    if (observer.access == Access::Write)   // what the observer writes is a change (4.5)
                        StampRowWrites(archetype, report.chunk, c, report.firstRow, report.count, version);
                }
            }
            m_Columns[n++] = column;
        }
        for (const QueryTerm& term : observer.dataTerms)
        {
            const uint32_t c = archetype.FindColumn(term.component);
            void* column = nullptr;
            if (c != c_NoColumn)
            {
                column = ColumnAt(archetype, c, report.chunk, report.firstRow);
                if (term.access == Access::Write)
                    StampRowWrites(archetype, report.chunk, c, report.firstRow, report.count, version);
            }
            m_Columns[n++] = column;
        }

        EntityBatch batch{};
        batch.entities = archetype.Entities(report.chunk) + report.firstRow;
        batch.columns = m_Columns.data();
        batch.count = report.count;
        batch.archetypeIndex = report.archetype;
        ++m_World.m_ObserverDepth;
        const nvrhi::FRESULT result = observer.callback->OnEntities(observer.event, batch, m_Buffers[m_Recording]);
        --m_World.m_ObserverDepth;
        ++m_World.m_ObserverBatches;
        ++m_Counters.deliveries;
        if (NVRHI_FAILED(result))
        {
            m_World.ReportValidation("observer", "'%s' (%s) returned 0x%08x", observer.name.c_str(),
                                     EventName(observer.event), uint32_t(result));
            KeepFirstFailure(m_Result, result);
        }
    }

    void ObserverRegistry::Deliver(const ObserverReport& report, const Vector<uint8_t>& takers) noexcept
    {
        ++m_Counters.batches;
        // Nothing can add or remove an observer while one runs (CanChangeStructure), so the list and the flags stand.
        for (size_t i = 0; i < m_Observers.size(); ++i)
            if (takers[i])
                Call(*m_Observers[i], report);
    }

    void ObserverRegistry::DeliverHeld() noexcept
    {
        if (m_Held.kind == ReportKind::None)
            return;
        const ObserverReport held = m_Held;
        m_Held.kind = ReportKind::None;
        Deliver(held, m_HeldTakers);
    }

    // Only a strict continuation extends the held batch. A report of a row the batch holds already (the same entity set
    // twice in one coalesced run) delivers the held batch and is held itself, so both callbacks come, in order; the
    // first shows the newer value too, because a set writes before it reports (as the first OnAdd of a coalesced Add
    // run [e1, e1] shows the value the second one wrote: see EcsObserverImpl.h).
    void ObserverRegistry::Hold(const ObserverReport& report) noexcept
    {
        if (Extends(m_Held, report))
        {
            m_Held.count += report.count;
            ++m_Counters.extended;
            return;
        }
        if (!CollectTakers(report, m_ReportTakers))
            return;
        DeliverHeld();
        m_Held = report;
        m_HeldTakers.swap(m_ReportTakers);   // same size, no allocation
    }

    // ------------------------------------------------------------------------------------------------ reports

    void ObserverRegistry::RowsAdded(uint32_t archetype, uint32_t chunk, uint32_t firstRow, uint32_t count,
                                     uint32_t fromArchetype) noexcept
    {
        NVRHI_ASSERT(m_World.m_ObserverDepth == 0);
        if (count == 0)
            return;
        ObserverReport report;
        report.kind = ReportKind::RowsAdded;
        report.archetype = archetype;
        report.chunk = chunk;
        report.firstRow = firstRow;
        report.count = count;
        report.other = fromArchetype;
        report.firstDense = firstRow;
        Hold(report);
    }

    void ObserverRegistry::RowsRemoving(uint32_t archetype, uint32_t chunk, uint32_t firstRow, uint32_t count,
                                        uint32_t toArchetype) noexcept
    {
        NVRHI_ASSERT(m_World.m_ObserverDepth == 0);
        if (count == 0)
            return;
        ObserverReport report;
        report.kind = ReportKind::RowsRemoving;
        report.archetype = archetype;
        report.chunk = chunk;
        report.firstRow = firstRow;
        report.count = count;
        report.other = toArchetype;
        report.firstDense = firstRow;
        if (!CollectTakers(report, m_ReportTakers))
        {
            // The removal moves the archetype's last row into the hole: a held batch of this archetype goes first.
            if (m_Held.kind != ReportKind::None && m_Held.archetype == archetype)
                DeliverHeld();
            return;
        }
        DeliverHeld();   // with m_HeldTakers: m_ReportTakers stands
        Deliver(report, m_ReportTakers);
    }

    void ObserverRegistry::Sparse(ComponentId component, Entity entity, bool added) noexcept
    {
        NVRHI_ASSERT(m_World.m_ObserverDepth == 0);
        const EntityRecord* record = m_World.FindAlive(entity);
        const SparseSet* set = m_World.GetSparseSet(component);
        NVRHI_ASSERT(record && set);
        if (!record || !set)
            return;
        ObserverReport report;
        report.kind = added ? ReportKind::SparseAdded : ReportKind::SparseRemoving;
        report.component = component;
        report.archetype = record->archetype;
        report.chunk = record->chunk;
        report.firstRow = record->row;
        report.count = 1;
        report.firstDense = set->Find(GetEntityIndex(entity));
        if (added)
        {
            Hold(report);
            return;
        }
        if (!CollectTakers(report, m_ReportTakers))
        {
            // The erase moves the set's last value into the hole: a held batch of this set's values goes first.
            if (m_Held.kind != ReportKind::None && m_Held.component == component)
                DeliverHeld();
            return;
        }
        DeliverHeld();
        Deliver(report, m_ReportTakers);
    }

    void ObserverRegistry::Set(ComponentId component, Entity entity) noexcept
    {
        NVRHI_ASSERT(m_World.m_ObserverDepth == 0);
        const EntityRecord* record = m_World.FindAlive(entity);
        NVRHI_ASSERT(record != nullptr);
        if (!record)
            return;
        ObserverReport report;
        report.kind = ReportKind::Set;
        report.component = component;
        report.archetype = record->archetype;
        report.chunk = record->chunk;
        report.firstRow = record->row;
        report.count = 1;
        const SparseSet* set = m_World.GetSparseSet(component);
        report.firstDense = set ? set->Find(GetEntityIndex(entity)) : report.firstRow;
        Hold(report);
    }

    // ------------------------------------------------------------------------------------------------ cascade

    nvrhi::FRESULT ObserverRegistry::PlayRound(CommandBufferImpl* buffer, bool firstRound,
                                               uint64_t& outCommands) noexcept
    {
        // The observers' placeholders resolve within one cascade, through a table of their own
        // (EcsCommandBufferImpl.h): flushes of immediate calls never grow IWorld::Playback's table, and
        // ResolvePlaceholder never answers for them.
        PlaceholderTable& placeholders = m_World.GetCommandPlayback().observerPlaceholders;
        if (firstRound)
            placeholders.Clear();
        return PlayCommandBuffers(m_World, &buffer, 1, m_World.GetVersion(), placeholders, outCommands);
    }

    nvrhi::FRESULT ObserverRegistry::RunCascade() noexcept
    {
        NVRHI_ASSERT(m_World.m_ObserverDepth == 0);
        DeliverHeld();
        uint32_t rounds = 0;
        while (m_Buffers[m_Recording]->GetCommandCount() != 0)
        {
            CommandBufferImpl* commands = m_Buffers[m_Recording];
            if (rounds == c_MaxObserverRounds)
            {
                m_World.ReportValidation("observers", "the cascade still has %u commands after %u rounds; they are "
                                         "dropped (an observer keeps triggering itself?)", commands->GetCommandCount(),
                                         c_MaxObserverRounds);
                commands->Clear();
                ++m_Counters.cascadeFailures;
                KeepFirstFailure(m_Result, nvrhi::FE_GENERIC_ERROR);
                break;
            }
            m_Recording ^= 1u;   // the observers this round triggers record into the other buffer
            uint64_t played = 0;
            KeepFirstFailure(m_Result, PlayRound(commands, rounds == 0, played));
            m_World.m_CommandsPlayedBack += played;
            ++rounds;
            ++m_Counters.rounds;
            DeliverHeld();
        }
        const nvrhi::FRESULT result = m_Result;
        m_Result = nvrhi::FS_OK;
        return result;
    }

    // ------------------------------------------------------------------------------------------ the world

    void WorldImpl::ReportRowsAdded(uint32_t archetype, uint32_t chunk, uint32_t firstRow, uint32_t count,
                                    uint32_t fromArchetype) noexcept
    {
        m_ObserverRegistry->RowsAdded(archetype, chunk, firstRow, count, fromArchetype);
    }

    void WorldImpl::ReportRowsRemoving(uint32_t archetype, uint32_t chunk, uint32_t firstRow, uint32_t count,
                                       uint32_t toArchetype) noexcept
    {
        m_ObserverRegistry->RowsRemoving(archetype, chunk, firstRow, count, toArchetype);
    }

    void WorldImpl::ReportSparse(ComponentId component, Entity entity, bool added) noexcept
    {
        m_ObserverRegistry->Sparse(component, entity, added);
    }

    void WorldImpl::ReportSet(ComponentId component, Entity entity) noexcept
    {
        m_ObserverRegistry->Set(component, entity);
    }

    void WorldImpl::DeliverHeldBatch() noexcept
    {
        m_ObserverRegistry->DeliverHeld();
    }

    nvrhi::FRESULT WorldImpl::RunObserverCascade() noexcept
    {
        return m_ObserverRegistry->RunCascade();
    }

    nvrhi::FRESULT WorldImpl::AddObserver(const ObserverDesc& desc, IObserver* observer, ObserverId& outId) noexcept
    {
        outId = ObserverId::Invalid;
        if (!CanChangeStructure())
        {
            ReportValidation("AddObserver", "observers are added and removed outside phases (or by an exclusive "
                             "system) and outside observers");
            return nvrhi::FE_INVALID_ARGS;
        }
        if (!m_ObserverRegistry)
        {
            void* memory = HeapAllocate(sizeof(ObserverRegistry), alignof(ObserverRegistry));
            if (!memory)
                return nvrhi::FE_OUT_OF_MEMORY;
            ObserverRegistry* registry = new (memory) ObserverRegistry(*this);
            const nvrhi::FRESULT result = registry->Initialize();
            if (NVRHI_FAILED(result))
            {
                DestroyObserverRegistry(registry);
                return result;
            }
            m_ObserverRegistry = registry;
        }
        return m_ObserverRegistry->Add(desc, observer, outId);
    }

    nvrhi::FRESULT WorldImpl::RemoveObserver(ObserverId observer) noexcept
    {
        if (!CanChangeStructure())
        {
            ReportValidation("RemoveObserver", "observers are added and removed outside phases (or by an exclusive "
                             "system) and outside observers");
            return nvrhi::FE_INVALID_ARGS;
        }
        if (!m_ObserverRegistry)
        {
            ReportValidation("RemoveObserver", "observer %u is not registered (never added, or removed)",
                             uint32_t(observer));
            return nvrhi::FE_INVALID_ARGS;
        }
        return m_ObserverRegistry->Remove(observer);
    }
}
