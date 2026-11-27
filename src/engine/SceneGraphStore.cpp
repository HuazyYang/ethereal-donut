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

// The scene graph over the world (plan steps E3, E4): storage, structural operations, the TransformPropagation and
// Bounds systems, the leaf observer, and the mesh-instance columns and slots. See SceneGraphStore.h for the rules.

#include "SceneGraphStore.h"
#include "EcsSchedulerImpl.h"
#include "SharedJobSystem.h"   // the job system that graphs share unless SetJobSystem gives them another

#include <donut/core/log.h>
#include <donut/core/query_cast.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <exception>
#include <cstring>
#include <limits>
#include <new>
#include <unordered_map>
#include <unordered_set>
#if defined(_M_X64) || defined(__x86_64__)
#include <xmmintrin.h>
#endif
#include <cstdlib>
#include <thread>

namespace donut::engine
{
    namespace
    {
        // The stores with a Defer scope open (SceneGraphStore::IsAnyDeferring; plan step E7, part I7).
        std::atomic<uint32_t> g_DeferringStores{ 0 };

        void EnterDeferring() noexcept
        {
            g_DeferringStores.fetch_add(1, std::memory_order_acq_rel);
        }

        void LeaveDeferring() noexcept
        {
            g_DeferringStores.fetch_sub(1, std::memory_order_acq_rel);
        }
    }

    // ------------------------------------------------------------------------------ reference hooks (D14)

    SceneNode::SceneNode(const SceneNode& other) noexcept
        : object(other.object)
    {
        if (object)
            object->AddRef();
    }

    SceneNode& SceneNode::operator=(const SceneNode& other) noexcept
    {
        if (other.object)
            other.object->AddRef();
        if (object)
            object->Release();
        object = other.object;
        return *this;
    }

    SceneNode& SceneNode::operator=(SceneNode&& other) noexcept
    {
        if (this != &other)
        {
            if (object)
                object->Release();
            object = other.object;
            other.object = nullptr;
        }
        return *this;
    }

    SceneNode::~SceneNode()
    {
        if (object)
            object->Release();
    }

    LeafObject::LeafObject(const LeafObject& other) noexcept
        : object(other.object)
    {
        if (object)
            object->AddRef();
    }

    LeafObject& LeafObject::operator=(const LeafObject& other) noexcept
    {
        if (other.object)
            other.object->AddRef();
        if (object)
            object->Release();
        object = other.object;
        return *this;
    }

    LeafObject& LeafObject::operator=(LeafObject&& other) noexcept
    {
        if (this != &other)
        {
            if (object)
                object->Release();
            object = other.object;
            other.object = nullptr;
        }
        return *this;
    }

    LeafObject::~LeafObject()
    {
        if (object)
            object->Release();
    }

    namespace
    {
        using namespace ecs;

        constexpr uint32_t c_None = ~0u;
        // ComposeWorld: local transforms on the stack, and the pieces of a longer chain.
        constexpr uint32_t c_ComposeBlock = 64;
        // What a leaf's row has beside the class's components, and loses when the leaf goes and none comes.
        constexpr SceneGraphStore::Slot c_BoundsSlots[] = { SceneGraphStore::SlotLocalBounds,
                                                            SceneGraphStore::SlotWorldBounds,
                                                            SceneGraphStore::SlotChunkBounds };

        // The local transform as the old SceneGraphNode::UpdateLocalTransform composed it; the trace goldens pin the
        // order of the operations bit for bit. Inlined by force: propagation's loop is the hot caller and must not
        // depend on there being another one (ComposeWorld).
#if defined(_MSC_VER)
        __forceinline
#endif
        dm::daffine3 ComposeLocal(const LocalTransform& local) noexcept
        {
            dm::daffine3 transform = dm::scaling(local.scaling);
            transform *= local.rotation.toAffine();
            transform *= dm::translation(local.translation);
            return transform;
        }

        // scaling * rotation * translation of a pose: the rows of dquat::toMatrix, scaled. Written out, without the
        // products with zeros of the general form and without toMatrix, so that the compiler's choice of what it inlines
        // in propagation does not depend on this.
        inline dm::daffine3 ComposePose(const dm::double3& t, const dm::dquat& q, const dm::double3& s) noexcept
        {
            return dm::daffine3(
                (1.0 - 2.0 * (q.y * q.y + q.z * q.z)) * s.x, (2.0 * (q.x * q.y + q.z * q.w)) * s.x,
                (2.0 * (q.x * q.z - q.y * q.w)) * s.x,
                (2.0 * (q.x * q.y - q.z * q.w)) * s.y, (1.0 - 2.0 * (q.x * q.x + q.z * q.z)) * s.y,
                (2.0 * (q.y * q.z + q.x * q.w)) * s.y,
                (2.0 * (q.x * q.z + q.y * q.w)) * s.z, (2.0 * (q.y * q.z - q.x * q.w)) * s.z,
                (1.0 - 2.0 * (q.x * q.x + q.y * q.y)) * s.z,
                t.x, t.y, t.z);
        }

        // The world transform between two poses (8.4), with + - * / sqrt only, so that all builds agree: the previous
        // pose at 0 and the last one at 1, bit for bit; between them linear in the translation, and in the rotation
        // linear on the short side and normalised.
        inline dm::daffine3 InterpolatePose(const SimPosePrevious& previous, const SimPose& pose, const SimScale& scale,
                                            double alpha) noexcept
        {
            if (alpha == 0.0)
                return ComposePose(previous.translation, previous.rotation, scale.scaling);
            if (alpha == 1.0)
                return ComposePose(pose.translation, pose.rotation, scale.scaling);
            const double rest = 1.0 - alpha;
            const dm::double3 translation = previous.translation * rest + pose.translation * alpha;
            const double side = dm::dot(previous.rotation, pose.rotation) < 0.0 ? -alpha : alpha;
            dm::dquat rotation = previous.rotation * rest + pose.rotation * side;
            const double length = std::sqrt(dm::dot(rotation, rotation));
            if (length > 0.0)
                rotation = rotation / length;
            return ComposePose(translation, rotation, scale.scaling);
        }

        // A matrix as scaling * rotation * translation: the rows of the linear part are the scaled axes. (dm's
        // decomposeAffine takes the scale from the columns, and half of the digits of a small rotation go.)
        struct Decomposition
        {
            LocalTransform transform;   // a mirrored matrix gets a negative scaling.x
            bool mirrored;
            bool degenerate;            // an axis without length, or a number that is not finite: no rotation
            double shear;               // the largest cosine between two axes
        };
        Decomposition Decompose(const dm::daffine3& matrix) noexcept
        {
            Decomposition result{ { matrix.m_translation, dm::dquat::identity(), dm::double3(1.0) }, false, false, 0.0 };
            dm::double3 axes[3] = { matrix.m_linear.row0, matrix.m_linear.row1, matrix.m_linear.row2 };
            for (int i = 0; i < 3; ++i)
            {
                const double length = std::sqrt(dm::dot(axes[i], axes[i]));
                result.transform.scaling[i] = length;
                if (!(length > 0.0) || !std::isfinite(length) || !std::isfinite(matrix.m_translation[i]))
                    result.degenerate = true;
                else
                    axes[i] /= length;
            }
            if (result.degenerate)
                return result;
            result.shear = std::max({ std::abs(dm::dot(axes[0], axes[1])), std::abs(dm::dot(axes[1], axes[2])),
                                      std::abs(dm::dot(axes[2], axes[0])) });
            if (dm::dot(dm::cross(axes[0], axes[1]), axes[2]) < 0.0)
            {
                result.mirrored = true;
                result.transform.scaling.x = -result.transform.scaling.x;
                axes[0] = -axes[0];
            }
            // the quaternion by its largest part (Shepperd), so that the small ones keep their digits
            const dm::double3& a = axes[0];
            const dm::double3& b = axes[1];
            const dm::double3& c = axes[2];
            const double trace = a.x + b.y + c.z;
            dm::dquat q;
            if (trace > 0.0)
                q = dm::dquat(1.0 + trace, b.z - c.y, c.x - a.z, a.y - b.x);
            else if (a.x >= b.y && a.x >= c.z)
                q = dm::dquat(b.z - c.y, 1.0 + a.x - b.y - c.z, a.y + b.x, c.x + a.z);
            else if (b.y >= c.z)
                q = dm::dquat(c.x - a.z, a.y + b.x, 1.0 - a.x + b.y - c.z, b.z + c.y);
            else
                q = dm::dquat(a.y - b.x, c.x + a.z, b.z + c.y, 1.0 - a.x - b.y + c.z);
            result.transform.rotation = q / std::sqrt(dm::dot(q, q));
            return result;
        }
        // The shear that a decomposition may lose: above what float rotations and their products leave.
        constexpr double c_MaxShear = 1e-5;
        // Whether the axes of `a` are those of `b`, within as much. No number that is not finite is.
        bool SameAxes(const dm::daffine3& a, const dm::daffine3& b) noexcept
        {
            bool same = true;
            for (int i = 0; i < 3; ++i)
            {
                const dm::double3 axis = b.m_linear[i];
                const dm::double3 difference = a.m_linear[i] - axis;
                same = same && dm::dot(difference, difference) <= c_MaxShear * c_MaxShear * dm::dot(axis, axis);
            }
            return same;
        }

        // `world` relative to `parent`: what times `parent` gives `world`. When `parent` has no inverse, `world` itself;
        // then `outInverse` tells.
        dm::daffine3 RelativeTo(const dm::daffine3& world, const dm::daffine3& parent, bool& outInverse) noexcept
        {
            const dm::double3& a = parent.m_linear.row0;
            const dm::double3& b = parent.m_linear.row1;
            const dm::double3& c = parent.m_linear.row2;
            const dm::double3 columns[3] = { dm::cross(b, c), dm::cross(c, a), dm::cross(a, b) };
            const double determinant = dm::dot(a, columns[0]);
            auto divide = [&](const dm::double3& v) noexcept
            {
                return dm::double3(dm::dot(v, columns[0]), dm::dot(v, columns[1]), dm::dot(v, columns[2])) / determinant;
            };
            const dm::daffine3 result(divide(world.m_linear.row0), divide(world.m_linear.row1),
                                      divide(world.m_linear.row2), divide(world.m_translation - parent.m_translation));
            bool finite = true;
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                    finite = finite && std::isfinite(result.m_linear[i][j]);
                finite = finite && std::isfinite(result.m_translation[i]);
            }
            outInverse = finite;
            return finite ? result : world;
        }

        bool IsFinite(const LocalTransform& t) noexcept
        {
            bool finite = std::isfinite(t.rotation.w) && std::isfinite(t.rotation.x) && std::isfinite(t.rotation.y) &&
                          std::isfinite(t.rotation.z);
            for (int i = 0; i < 3; ++i)
                finite = finite && std::isfinite(t.translation[i]) && std::isfinite(t.scaling[i]);
            return finite;
        }

        // The parts of a driven node's transform in its parent's space and in the world. The two functions are inverse to
        // each other, up to rounding, below every parent, one that shears the result included: what the getters return
        // gives the same pose when it is set again, and a request for one part leaves the others alone. Where a scaling,
        // a rotation and a translation can express the product (the quotient) of the matrices, they are its parts; the
        // mirror of a mirrored parent, which Decompose takes along x, goes into scaling.x and the rotation. A parent
        // without an inverse counts as none, as in RelativeTo; so does one that gives a number that is not finite.
        struct ParentParts
        {
            bool usable = false;
            bool mirrored = false;
            bool conformal = false;   // a turn and one scale (and a mirror): the parts are the matrix's, exactly enough
            dm::dquat rotation = dm::dquat::identity();

            explicit ParentParts(const dm::daffine3& parent) noexcept
            {
                RelativeTo(dm::daffine3::identity(), parent, usable);
                const Decomposition parts = Decompose(parent);
                usable = usable && !parts.degenerate;
                mirrored = parts.mirrored;
                rotation = parts.transform.rotation;
                const dm::double3 lengths = dm::abs(parts.transform.scaling);
                const double largest = std::max({ lengths.x, lengths.y, lengths.z });
                conformal = usable && parts.shear <= c_MaxShear &&
                            largest - std::min({ lengths.x, lengths.y, lengths.z }) <= c_MaxShear * largest;
            }
            // x -> -x, rotation, x -> -x
            dm::dquat Mirror(const dm::dquat& q) const noexcept
            {
                return mirrored ? dm::dquat(q.w, q.x, -q.y, -q.z) : q;
            }
        };
        // The lengths that `parent` gives to the axes of the local rotation `q`.
        dm::double3 AxisLengths(const dm::dquat& q, const dm::daffine3& parent) noexcept
        {
            const dm::daffine3 axes = ComposePose(dm::double3(0.0), q, dm::double3(1.0)) * parent;
            const dm::double3x3& m = axes.m_linear;
            return dm::double3(std::sqrt(dm::dot(m.row0, m.row0)), std::sqrt(dm::dot(m.row1, m.row1)),
                               std::sqrt(dm::dot(m.row2, m.row2)));
        }
        // `outCounts`: the parent counts; `outConformal`: its parts are the matrix's (ParentParts).
        LocalTransform PartsInParent(const LocalTransform& world, const dm::daffine3& parent, bool& outCounts,
                                     bool* outConformal = nullptr) noexcept
        {
            const ParentParts p(parent);
            outCounts = p.usable;
            if (outConformal)
                *outConformal = p.conformal;
            if (!p.usable)
                return world;
            bool inverse;
            LocalTransform local;
            local.translation = RelativeTo(dm::translation(world.translation), parent, inverse).m_translation;
            local.rotation = p.Mirror(dm::conjugate(p.rotation) * world.rotation);
            local.scaling = world.scaling / AxisLengths(local.rotation, parent);
            if (p.mirrored)
                local.scaling.x = -local.scaling.x;
            outCounts = inverse && IsFinite(local);
            return outCounts ? local : world;
        }
        LocalTransform PartsInWorld(const LocalTransform& local, const dm::daffine3& parent) noexcept
        {
            const ParentParts p(parent);
            if (!p.usable)
                return local;
            LocalTransform world;
            world.translation = parent.transformPoint(local.translation);
            world.rotation = p.rotation * p.Mirror(local.rotation);
            world.scaling = local.scaling * AxisLengths(local.rotation, parent);
            if (p.mirrored)
                world.scaling.x = -world.scaling.x;
            return IsFinite(world) ? world : local;
        }

        // The threads that can take ranges: IJobSystem::GetThreadCount also counts the external slots (64 in the shared
        // system), which are mostly idle; the hardware bounds the workers that can run at once.
        uint32_t WorkingThreads(IJobSystem* jobs) noexcept
        {
            static const uint32_t s_Hardware = std::max(1u, std::thread::hardware_concurrency());
            return jobs ? std::max(1u, std::min(jobs->GetThreadCount(), s_Hardware)) : 1u;
        }

        nvrhi::FLONG ReferenceCount(nvrhi::IObject* object) noexcept
        {
            object->AddRef();
            return object->Release();
        }

        uint8_t* ColumnBase(const Archetype& archetype, uint32_t chunk, uint8_t column) noexcept
        {
            return archetype.chunks[chunk].memory + archetype.columns[column].offset;
        }

        // Walks over subgraphs are chains of cache misses; the passes after them fetch a few nodes ahead.
        constexpr size_t c_PrefetchDistance = 8;
        inline void Prefetch(const void* address) noexcept
        {
#if defined(_M_X64) || defined(__x86_64__)
            _mm_prefetch(static_cast<const char*>(address), _MM_HINT_T0);
#elif defined(__GNUC__) || defined(__clang__)
            __builtin_prefetch(address);
#else
            (void)address;
#endif
        }

        // Records are recycled (OrphanRecord::Create): an attach gives one back per node, and the next detach or
        // import takes them, without a heap call each. The pool keeps at most c_MaxPooledRecords; it is never
        // destroyed, so that nodes destroyed during static destruction can still give theirs back.
        constexpr size_t c_MaxPooledRecords = size_t(1) << 16;
        struct RecordPool
        {
            std::mutex mutex;
            std::vector<OrphanRecord*> records;
        };
        RecordPool& GetRecordPool() noexcept
        {
            static RecordPool* const s_Pool = new RecordPool();
            return *s_Pool;
        }

        // The graph's observer on LeafObject (design 5.5): it fills a mesh instance's columns (asset tables, slots)
        // and maintains the leaf vectors through the virtual RegisterLeaf and UnregisterLeaf, in creation and
        // destruction order.
        class LeafObserver final : public nvrhi::ObjectImpl<IObserver>
        {
        public:
            NVRHI_BEGIN_INTERFACE_TABLE_INLINE(LeafObserver)
            NVRHI_IMPLEMENTS_INTERFACE(IObserver)
            NVRHI_END_INTERFACE_TABLE()

            using Register = void (*)(SceneGraphStore* store, Entity entity, SceneGraphLeaf* leaf, bool added);

            LeafObserver(SceneGraphStore* store, Register registerLeaf) noexcept
                : m_Store(store), m_Register(registerLeaf) { }

            // The world may outlive the store (someone holds it), and this observer with it: it then does nothing.
            void Disconnect() noexcept { m_Store = nullptr; }

            nvrhi::FRESULT OnEntities(ObserverEvent event, const EntityBatch& batch, ICommandBuffer*) noexcept override
            {
                if (!m_Store)
                    return nvrhi::FS_OK;
                const LeafObject* leaves = static_cast<const LeafObject*>(batch.columns[0]);
                const bool added = event == ObserverEvent::OnAdd;
                for (uint32_t i = 0; i < batch.count; ++i)
                {
                    if (!leaves[i].object)
                        continue;
                    // RegisterLeaf overrides and the OnMeshAdded callbacks may throw; this boundary is noexcept
                    try
                    {
                        m_Register(m_Store, batch.entities[i], leaves[i].object, added);
                    }
                    catch (const std::exception& error)
                    {
                        log::warning("SceneGraph: %s threw '%s'; the leaf is skipped",
                                     added ? "RegisterLeaf" : "UnregisterLeaf", error.what());
                    }
                    catch (...)
                    {
                        log::warning("SceneGraph: %s threw; the leaf is skipped",
                                     added ? "RegisterLeaf" : "UnregisterLeaf");
                    }
                }
                return nvrhi::FS_OK;
            }

        private:
            SceneGraphStore* m_Store;   // not a reference: the store owns the world, which owns this observer
            Register m_Register;
        };

        // A process-wide serial per store (SceneGraphStore::GetSerial).
        std::atomic<uint64_t> g_StoreSerial{ 0 };

        // The flags of an instance of `mesh` (null: none) whose content flags are `content`.
        RenderFlags MakeRenderFlags(const MeshInfo* mesh, SceneContentFlags content) noexcept
        {
            RenderFlags flags{ 0, uint32_t(content) };
            if (mesh)
            {
                // as Scene::UpdateInstance set InstanceData::flags
                if (mesh->type == MeshType::CurveDisjointOrthogonalTriangleStrips)
                    flags.instanceFlags |= c_InstanceFlagCurveDisjointOrthogonalTriangleStrips;
                else if (mesh->type == MeshType::CurveLinearSweptSpheres)
                    flags.instanceFlags |= c_InstanceFlagCurveLinearSweptSpheres;
            }
            return flags;
        }

        // A leaf's: its content flags through the virtual call, which a subclass may override.
        RenderFlags MakeRenderFlags(MeshInstance* instance) noexcept
        {
            return MakeRenderFlags(instance->GetMesh(), instance->GetContentFlags());
        }

        // The graph's observer of runtime-only mesh instances (E7.1), registered three times by SceneGraphStore's
        // constructor: OnMatch and OnUnmatch of the membership {RuntimeMeshInstance, GpuInstanceSlot, no SceneNode}
        // take and release the slots and table references (SceneGraphStore::OnRuntimeRows; the batch's columns are the
        // filter's data terms), and OnAdd of the tag on rows without GpuInstanceSlot adds the column
        // (SceneGraphStore::AddRuntimeSlotColumn), which makes them members.
        class RuntimeObserver final : public nvrhi::ObjectImpl<IObserver>
        {
        public:
            NVRHI_BEGIN_INTERFACE_TABLE_INLINE(RuntimeObserver)
            NVRHI_IMPLEMENTS_INTERFACE(IObserver)
            NVRHI_END_INTERFACE_TABLE()

            explicit RuntimeObserver(SceneGraphStore* store) noexcept : m_Store(store) { }

            // As LeafObserver's: the world may outlive the store, and this observer with it; it then does nothing.
            void Disconnect() noexcept { m_Store = nullptr; }

            nvrhi::FRESULT OnEntities(ObserverEvent event, const EntityBatch& batch,
                                      ICommandBuffer* commands) noexcept override
            {
                if (!m_Store || batch.count == 0)
                    return nvrhi::FS_OK;
                if (event == ObserverEvent::OnAdd)
                    m_Store->AddRuntimeSlotColumn(batch, commands);
                else
                    m_Store->OnRuntimeRows(batch, event == ObserverEvent::OnMatch);
                return nvrhi::FS_OK;
            }

        private:
            SceneGraphStore* m_Store;
        };
    }

    SceneContentFlags GetMeshContentFlags(const MeshInfo* mesh) noexcept
    {
        if (!mesh)
            return SceneContentFlags::None;
        SceneContentFlags flags = SceneContentFlags::None;
        for (const auto& geometry : mesh->geometries)
        {
            if (!geometry || !geometry->material)
                continue;
            switch (geometry->material->domain)   // NOLINT(clang-diagnostic-switch-enum)
            {
            case MaterialDomain::Opaque:
                flags |= SceneContentFlags::OpaqueMeshes;
                break;
            case MaterialDomain::AlphaTested:
                flags |= SceneContentFlags::AlphaTestedMeshes;
                break;
            default:
                flags |= SceneContentFlags::BlendedMeshes;
                break;
            }
        }
        return flags;
    }

    OrphanRecord* OrphanRecord::Create()
    {
        void* memory = nullptr;
        {
            RecordPool& pool = GetRecordPool();
            std::lock_guard<std::mutex> lock(pool.mutex);
            if (!pool.records.empty())
            {
                memory = pool.records.back();
                pool.records.pop_back();
            }
        }
        if (!memory)
            memory = ::operator new(sizeof(OrphanRecord));
        return new (memory) OrphanRecord();
    }

    void OrphanRecord::Destroy(OrphanRecord* record) noexcept
    {
        if (record)
            Destroy(&record, 1);
    }

    void OrphanRecord::Destroy(OrphanRecord* const* records, size_t count) noexcept
    {
        // destroyed first: releasing children may give their records back too
        for (size_t i = 0; i < count; ++i)
            records[i]->~OrphanRecord();
        RecordPool& pool = GetRecordPool();
        size_t kept = 0;
        {
            std::lock_guard<std::mutex> lock(pool.mutex);
            kept = std::min(count, c_MaxPooledRecords - std::min(c_MaxPooledRecords, pool.records.size()));
            pool.records.insert(pool.records.end(), records, records + kept);
        }
        for (size_t i = kept; i < count; ++i)
            ::operator delete(records[i]);
    }

    // ------------------------------------------------------------------------------- leaf records (E6.3)

    namespace
    {
        // The value operations of a column by its descriptor, as EcsWorld.cpp's by its registered type: a null hook is
        // the trivial operation.
        void ConstructValue(const ComponentTypeDesc& desc, void* value) noexcept
        {
            if (desc.construct)
                desc.construct(value, 1);
            else
                std::memset(value, 0, desc.size);
        }
        void DestructValue(const ComponentTypeDesc& desc, void* value) noexcept
        {
            if (desc.destruct)
                desc.destruct(value, 1);
        }
        void RelocateValue(const ComponentTypeDesc& desc, void* dst, void* src) noexcept
        {
            if (desc.relocate)
                desc.relocate(dst, src, 1);
            else
                std::memcpy(dst, src, desc.size);
        }

        // Where the row has a plain column of this GUID; null where it has none.
        void* RowValue(WorldImpl& world, const Archetype& archetype, uint32_t chunk, uint32_t row,
                       const nvrhi::GUID& uuid, uint32_t* outColumn = nullptr) noexcept
        {
            const ComponentId id = world.FindComponent(uuid);
            const uint32_t column = id == ComponentId::Invalid ? c_NoColumn : archetype.FindColumn(id);
            if (column == c_NoColumn || archetype.columns[column].kind != StorageKind::Data)
                return nullptr;
            if (outColumn)
                *outColumn = column;
            return static_cast<uint8_t*>(archetype.ColumnData(chunk, column)) +
                size_t(row) * archetype.columns[column].stride;
        }

        // Freed blocks by size class (64 << class bytes), linked through LeafRecord::m_NextFree, so that giving one back
        // allocates nothing. Bounded, never destroyed, like the pool of orphan records. Larger blocks (a row of these
        // columns would not fit a chunk either) are not kept.
        constexpr uint32_t c_LeafRecordClasses = 11;   // 64 bytes to 64 KiB
        constexpr size_t c_MaxPooledLeafRecordBytes = size_t(16) << 20;
        struct LeafRecordPool
        {
            std::mutex mutex;
            LeafRecord* free[c_LeafRecordClasses] = {};
            size_t bytes = 0;
        };
        LeafRecordPool& GetLeafRecordPool() noexcept
        {
            static LeafRecordPool* const s_Pool = new LeafRecordPool();
            return *s_Pool;
        }
    }

    bool IsPlainColumn(const ComponentTypeDesc& desc) noexcept
    {
        constexpr ComponentFlags c_NotPlain = ComponentFlags::Tag | ComponentFlags::Sparse | ComponentFlags::Shared |
            ComponentFlags::Chunk | ComponentFlags::Buffer | ComponentFlags::Resource;
        return desc.size != 0 && !(desc.flags & c_NotPlain);
    }

    std::atomic<int64_t> LeafRecord::s_LiveCount{ 0 };

    template <typename Visit>
    void LeafRecord::ForEachValue(Visit&& visit) noexcept
    {
        size_t offset = sizeof(LeafRecord);
        for (uint32_t c = 0; c < m_ColumnCount; ++c)
        {
            const ComponentTypeDesc& desc = *m_Columns[c];
            if (!IsPlainColumn(desc))
                continue;
            offset = (offset + desc.alignment - 1) & ~size_t(desc.alignment - 1);
            visit(desc, static_cast<void*>(reinterpret_cast<uint8_t*>(this) + offset));
            offset += desc.size;
        }
    }

    LeafRecord* LeafRecord::Allocate(const LeafTypeDesc& type) noexcept
    {
        size_t bytes = sizeof(LeafRecord);
        for (uint32_t c = 0; c < type.columnCount; ++c)
        {
            const ComponentTypeDesc& desc = *type.columns[c];
            if (!IsPlainColumn(desc))
                continue;
            assert(desc.alignment <= c_ColumnAlignment);   // no world accepts more
            bytes = ((bytes + desc.alignment - 1) & ~size_t(desc.alignment - 1)) + desc.size;
        }
        uint32_t sizeClass = 0;
        while (sizeClass < c_LeafRecordClasses && (size_t(64) << sizeClass) < bytes)
            ++sizeClass;

        void* memory = nullptr;
        if (sizeClass < c_LeafRecordClasses)
        {
            bytes = size_t(64) << sizeClass;
            LeafRecordPool& pool = GetLeafRecordPool();
            std::lock_guard<std::mutex> lock(pool.mutex);
            if (LeafRecord* record = pool.free[sizeClass])
            {
                pool.free[sizeClass] = record->m_NextFree;
                pool.bytes -= bytes;
                memory = record;
            }
        }
        if (!memory)
            memory = HeapAllocate(bytes, c_ColumnAlignment);
        if (!memory)
            return nullptr;
        s_LiveCount.fetch_add(1, std::memory_order_relaxed);
        LeafRecord* record = new (memory) LeafRecord();
        record->m_Columns = type.columns;
        record->m_ColumnCount = type.columnCount;
        record->m_SizeClass = sizeClass;
        return record;
    }

    void LeafRecord::Free(LeafRecord* record) noexcept
    {
        s_LiveCount.fetch_sub(1, std::memory_order_relaxed);
        if (record->m_SizeClass < c_LeafRecordClasses)
        {
            const size_t bytes = size_t(64) << record->m_SizeClass;
            LeafRecordPool& pool = GetLeafRecordPool();
            std::lock_guard<std::mutex> lock(pool.mutex);
            if (pool.bytes + bytes <= c_MaxPooledLeafRecordBytes)
            {
                record->m_NextFree = pool.free[record->m_SizeClass];
                pool.free[record->m_SizeClass] = record;
                pool.bytes += bytes;
                return;
            }
        }
        HeapFree(record);
    }

    LeafRecord* LeafRecord::Create(const LeafTypeDesc& type) noexcept
    {
        LeafRecord* record = Allocate(type);
        if (record)
            record->ForEachValue(ConstructValue);
        return record;
    }

    LeafRecord* LeafRecord::CreateInRow(const LeafTypeDesc& type, SceneGraphStore* store, Entity entity) noexcept
    {
        LeafRecord* record = Allocate(type);
        if (record)
        {
            record->m_Store = store;
            record->m_Entity = entity;
        }
        return record;
    }

    void LeafRecord::Destroy(LeafRecord* record) noexcept
    {
        if (!record)
            return;
        if (!record->IsInRow())
            record->ForEachValue(DestructValue);
        Free(record);
    }

    void* LeafRecord::Find(const nvrhi::GUID& uuid) noexcept
    {
        assert(!IsInRow());
        void* found = nullptr;
        ForEachValue([&](const ComponentTypeDesc& desc, void* value) noexcept
        {
            if (desc.uuid == uuid)
                found = value;
        });
        return found;
    }

    bool LeafRecord::HasColumnsOf(const LeafTypeDesc& type) const noexcept
    {
        auto covers = [](const ComponentTypeDesc* const* columns, uint32_t count,
                         const ComponentTypeDesc& column) noexcept
        {
            for (uint32_t c = 0; c < count; ++c)
            {
                if (columns[c]->uuid == column.uuid)
                    return true;
            }
            return false;
        };
        for (uint32_t c = 0; c < type.columnCount; ++c)
        {
            if (!covers(m_Columns, m_ColumnCount, *type.columns[c]))
                return false;
        }
        for (uint32_t c = 0; c < m_ColumnCount; ++c)
        {
            if (!covers(type.columns, type.columnCount, *m_Columns[c]))
                return false;
        }
        return true;
    }

    void LeafRecord::MoveIntoRow(LeafRecord* record, WorldImpl& world, const Archetype& archetype, uint32_t chunk,
                                 uint32_t row) noexcept
    {
        record->ForEachValue([&](const ComponentTypeDesc& desc, void* value) noexcept
        {
            if (void* target = RowValue(world, archetype, chunk, row, desc.uuid))
            {
                DestructValue(desc, target);   // the default value the row was made with
                RelocateValue(desc, target, value);
            }
            else
                DestructValue(desc, value);
        });
        Free(record);   // no value is left in it
    }

    void LeafRecord::TakeFromRow(WorldImpl& world, const Archetype& archetype, uint32_t chunk, uint32_t row) noexcept
    {
        assert(IsInRow());
        ForEachValue([&](const ComponentTypeDesc& desc, void* value) noexcept
        {
            if (void* source = RowValue(world, archetype, chunk, row, desc.uuid))
            {
                RelocateValue(desc, value, source);
                ConstructValue(desc, source);   // the row destroys it
            }
            else
                ConstructValue(desc, value);
        });
        m_Store = nullptr;
        m_Entity = Entity::Null;
    }

    namespace
    {
        // The subgraph rule for NaN: a box with a NaN anywhere counts as the infinite box (it is never culled, and a
        // union with it is the same in any order), in the pass, its summaries, the Bounds system's chunk union
        // (ChunkWorldBounds) and the lazy path.
        inline bool HasNan(const dm::box3& box) noexcept
        {
            return box.m_mins.x != box.m_mins.x || box.m_mins.y != box.m_mins.y || box.m_mins.z != box.m_mins.z ||
                   box.m_maxs.x != box.m_maxs.x || box.m_maxs.y != box.m_maxs.y || box.m_maxs.z != box.m_maxs.z;
        }

        inline dm::box3 InfiniteBox() noexcept
        {
            const float infinity = std::numeric_limits<float>::infinity();
            return dm::box3(dm::float3(-infinity), dm::float3(infinity));
        }

        inline dm::box3 NanToInfinite(const dm::box3& box) noexcept
        {
            return HasNan(box) ? InfiniteBox() : box;
        }
    }

    struct SceneGraphStore::PropagationContext
    {
        SceneGraphStore* store;
        uint32_t level;
        uint64_t since;
        uint64_t version;
        std::atomic<bool> wrote{ false };
        std::atomic<uint64_t> chunks{ 0 };
        std::atomic<uint64_t> rows{ 0 };
    };

    struct SceneGraphStore::InterpolationContext
    {
        const SceneGraphStore* store;
        uint64_t version;
        uint32_t changedDomains;                      // a bit per domain whose alpha changed since the last run
        double alphas[c_MaxSimulationDomains + 1];    // the last one is that of rows of no domain
        std::atomic<uint64_t> chunks{ 0 };
    };

    struct SceneGraphStore::TeleportContext
    {
        const SceneGraphStore* store;
        uint64_t version;
        std::atomic<uint64_t> resolved{ 0 };
    };

    struct SceneGraphStore::BoundsContext
    {
        SceneGraphStore* store;
        uint64_t version;
        bool subgraph;   // an incremental subgraph pass is prepared: summaries and marks of the chunks written
        bool childless;  // E7.4 remedy 4 (SceneGraph::GetChildlessBoundsPath at this run)
        std::atomic<uint64_t> chunks{ 0 };
        std::atomic<uint64_t> childlessChunks{ 0 };
    };

    // ---------------------------------------------------------------------------------------- the store

    SceneGraphStore::SceneGraphStore(SceneGraph* graph) noexcept
        : m_Graph(graph)
    {
        WorldDesc desc;
        desc.debugName = "SceneGraph";
#ifndef NDEBUG
        desc.validation = true;
#endif
        if (NVRHI_FAILED(CreateWorld(desc, &m_World)))
        {
            log::warning("SceneGraph: cannot create the world");
            std::abort();
        }
        m_Impl = GetWorldImpl(m_World);
        // chunk pages come from the process-wide slab cache: a graph made after another was dropped reuses its memory
        m_Impl->ShareChunkPages();

        m_Ids[SlotNode] = RegisterComponent<SceneNode>(m_World);
        m_Ids[SlotLinks] = RegisterComponent<SceneLinks>(m_World);
        m_Ids[SlotParent] = RegisterComponent<Parent>(m_World);
        // E7.5: the transforms get row versions while SceneGraph::SetRowTrackedTransforms is on (off by default). The
        // flag is the world's storage option only: ids, layouts, snapshots and hashes are the same either way.
        const bool rowTracked = SceneGraph::GetRowTrackedTransforms();
        auto registerTransform = [this, rowTracked](ComponentTypeDesc desc) noexcept
        {
            if (rowTracked)
                desc.flags |= ComponentFlags::RowTracked;
            ComponentId id = ComponentId::Invalid;
            m_World->RegisterComponent(desc, id);
            return id;
        };
        m_Ids[SlotLocal] = registerTransform(ComponentTraits<LocalTransform>::Desc());
        // E7.4 remedy 3: whether Static rows are honoured (off by default); the tag is looked up, never registered here,
        // so that the world's types and ids are the same either way.
        m_StaticTagHonoured = SceneGraph::GetStaticTag();
        m_Ids[SlotLocalMatrix] = RegisterComponent<LocalMatrix>(m_World);
        m_Ids[SlotWorld] = registerTransform(ComponentTraits<WorldTransform>::Desc());
        m_Ids[SlotLeaf] = RegisterComponent<LeafObject>(m_World);
        m_Ids[SlotLocalBounds] = RegisterComponent<LocalBounds>(m_World);
        m_Ids[SlotWorldBounds] = RegisterComponent<WorldBounds>(m_World);
        m_Ids[SlotChunkBounds] = RegisterComponent<ChunkWorldBounds>(m_World);
        m_Ids[SlotMeshRef] = RegisterComponent<MeshRef>(m_World);
        m_Ids[SlotRenderFlags] = RegisterComponent<RenderFlags>(m_World);
        m_Ids[SlotInstance] = RegisterComponent<GpuInstanceSlot>(m_World);
        m_DepthId = RegisterComponent<HierarchyDepth>(m_World);
        m_Ids[SlotDriven] = RegisterComponent<SimulationDriven>(m_World);
        m_Ids[SlotSimPose] = RegisterComponent<SimPose>(m_World);
        m_Ids[SlotSimScale] = RegisterComponent<SimScale>(m_World);
        m_SimPreviousId = RegisterComponent<SimPosePrevious>(m_World);
        m_SimBodyId = RegisterComponent<SimBodyRef>(m_World);
        m_SimTeleportId = RegisterComponent<SimTeleport>(m_World);
        m_Serial = ++g_StoreSerial;

        m_LeafObserver = MAKE_RC_OBJ_PTR(LeafObserver, this,
            [](SceneGraphStore* store, Entity entity, SceneGraphLeaf* leaf, bool added)
            {
                store->OnLeaf(entity, leaf, added);
            });
        ObserverDesc observer{};
        observer.debugName = "SceneGraph leaves";
        observer.component = m_Ids[SlotLeaf];
        observer.access = Access::Read;
        ObserverId id = ObserverId::Invalid;
        observer.event = ObserverEvent::OnAdd;
        m_World->AddObserver(observer, m_LeafObserver, id);
        observer.event = ObserverEvent::OnRemove;
        m_World->AddObserver(observer, m_LeafObserver, id);

        m_Jobs = GetSharedJobSystem();
        Systems systems;
        CreateSystems(m_Jobs, systems);   // what cannot be made is warned about; the graph lives with the rest
        TakeSystems(systems);

        // Runtime-only mesh instances (E7.1): the tag, registered after every type that the constructor registers
        // (the store's and the scheduler's), so that those ids stay as they were; its observers after the leaf
        // observer. Membership is {RuntimeMeshInstance, GpuInstanceSlot, MeshRef, no SceneNode} (a node's row with the
        // tag is not one): a row holds slots and a reference exactly while it is a member, so whatever ends membership
        // (destruction, the tag, the slot column or MeshRef removed, a SceneNode added) releases them while MeshRef is
        // still readable, and a row that never became one (destroyed before the slot column was added) holds nothing.
        // The batch's columns: GpuInstanceSlot, MeshRef, and at OnMatch RenderFlags, LocalBounds and Parent, each null
        // where the row lacks it (Optional).
        m_RuntimeTag = RegisterComponent<RuntimeMeshInstance>(m_World);
        m_RuntimeObserver = MAKE_RC_OBJ_PTR(RuntimeObserver, this);
        const QueryTerm memberTerms[] = {
            { m_RuntimeTag, TermOp::All, Access::Filter, {} },
            { m_Ids[SlotNode], TermOp::Without, Access::Filter, {} },
            { m_Ids[SlotInstance], TermOp::All, Access::Write, {} },
            { m_Ids[SlotMeshRef], TermOp::All, Access::Read, {} },
            { m_Ids[SlotRenderFlags], TermOp::Optional, Access::Write, {} },
            { m_Ids[SlotLocalBounds], TermOp::Optional, Access::Write, {} },
            { m_Ids[SlotParent], TermOp::Optional, Access::Read, {} },
        };
        ObserverDesc runtime{};
        runtime.debugName = "SceneGraph runtime-only mesh instances";
        runtime.component = ComponentId::Invalid;
        runtime.event = ObserverEvent::OnMatch;
        runtime.filterTerms = memberTerms;
        runtime.filterTermCount = uint32_t(std::size(memberTerms));
        bool added = m_RuntimeTag != ComponentId::Invalid &&
                     NVRHI_SUCCEEDED(m_World->AddObserver(runtime, m_RuntimeObserver, id));
        runtime.event = ObserverEvent::OnUnmatch;
        runtime.filterTermCount = 4;   // the same membership; GpuInstanceSlot and MeshRef
        added = added && NVRHI_SUCCEEDED(m_World->AddObserver(runtime, m_RuntimeObserver, id));
        // Rows that get the tag: without the slot column (a snapshot's: GpuInstanceSlot is NoSerialize) they get the
        // column; without MeshRef they are reported (they never become members). The batch's columns: the tag's (null),
        // GpuInstanceSlot and MeshRef, each null where the row lacks it.
        const QueryTerm slotlessTerms[] = {
            { m_Ids[SlotNode], TermOp::Without, Access::Filter, {} },
            { m_Ids[SlotInstance], TermOp::Optional, Access::Read, {} },
            { m_Ids[SlotMeshRef], TermOp::Optional, Access::Read, {} },
        };
        runtime.debugName = "SceneGraph runtime-only mesh instances added";
        runtime.component = m_RuntimeTag;
        runtime.access = Access::Filter;
        runtime.event = ObserverEvent::OnAdd;
        runtime.filterTerms = slotlessTerms;
        runtime.filterTermCount = uint32_t(std::size(slotlessTerms));
        added = added && NVRHI_SUCCEEDED(m_World->AddObserver(runtime, m_RuntimeObserver, id));
        if (!added)
            log::warning("SceneGraph: cannot add the observers of runtime-only mesh instances");

        // Donut's read-only list for inspectors (E7.7): registered last, after the store's own types, the scheduler's
        // and the runtime tag, so that their ids stay as they were. NoSerialize: snapshots and hashes leave it out.
        InspectorReadOnlyComponents readOnly{};
        const ComponentId readOnlyIds[] = {
            m_Ids[SlotLinks], m_Ids[SlotParent], m_DepthId,                       // the hierarchy
            m_Ids[SlotLocalMatrix], m_Ids[SlotWorld], m_Ids[SlotLocalBounds],     // derived: systems overwrite them
            m_Ids[SlotWorldBounds], m_Ids[SlotChunkBounds],
            m_Ids[SlotInstance], m_Ids[SlotMeshRef], m_Ids[SlotRenderFlags],     // slots and references
            m_SimBodyId,
            m_Ids[SlotNode], m_Ids[SlotLeaf],                                     // hooked
            // the simulation's: the domain derives SimPosePrevious; a SimTeleport request goes through RequestTeleport
            // (normalised, checked, its space and enabled bit set; backlog I3-R1, I3-R2)
            m_SimPreviousId, m_SimTeleportId,
        };
        static_assert(std::size(readOnlyIds) <= c_MaxInspectorReadOnlyComponents, "the read-only list");
        for (const ComponentId component : readOnlyIds)
            if (component != ComponentId::Invalid)
                readOnly.components[readOnly.count++] = component;
        ComponentId readOnlyId = ComponentId::Invalid;
        if (NVRHI_FAILED(m_World->RegisterResource(ComponentTraits<InspectorReadOnlyComponents>::Desc(), &readOnly,
                                                   readOnlyId)))
            log::warning("SceneGraph: cannot register the inspector's read-only list");
        // ChangesScope makes its levels on first use (not here: the constructor's last allocations are reported ones)
    }

    void SceneGraphStore::TakeSystems(const Systems& systems) noexcept
    {
        m_Scheduler = systems.scheduler;
        m_InterpolationPhase = systems.interpolation;
        m_TeleportPhase = systems.teleport;
        ++m_SchedulerGeneration;
        m_JointsMarkedVersion = 0;   // the new propagation has not run: its first run treats every chunk as changed
    }

    bool SceneGraphStore::CreateSystems(IJobSystem* jobs, Systems& out) noexcept
    {
        if (!jobs || NVRHI_FAILED(CreateScheduler(m_World, jobs, &out.scheduler)))
        {
            log::warning("SceneGraph: cannot create the scheduler");
            return false;
        }
        IScheduler* scheduler = out.scheduler;
        bool complete = true;

        const ComponentAccess propagationAccess[] = {
            { m_Ids[SlotLocal], Access::Read, {} },
            { m_Ids[SlotLocalMatrix], Access::Write, {} },
            { m_Ids[SlotParent], Access::Read, {} },
            { m_DepthId, Access::Read, {} },
            { m_Ids[SlotWorld], Access::Write, {} },
        };
        SystemDesc propagation{};
        propagation.name = "SceneGraph.TransformPropagation";
        propagation.accesses = propagationAccess;
        propagation.accessCount = uint32_t(std::size(propagationAccess));
        propagation.phase = Phase::TransformPropagation;
        propagation.flags = SystemFlags::MainThread;   // it splits its levels with ParallelFor itself

        const ComponentAccess boundsAccess[] = {
            { m_Ids[SlotWorld], Access::Read, {} },
            { m_Ids[SlotLocalBounds], Access::Write, {} },   // read again from leaves with dynamic bounds
            { m_Ids[SlotLeaf], Access::Read, {} },
            { m_Ids[SlotWorldBounds], Access::Write, {} },
            { m_Ids[SlotChunkBounds], Access::Write, {} },
        };
        SystemDesc bounds{};
        bounds.name = "SceneGraph.Bounds";
        bounds.accesses = boundsAccess;
        bounds.accessCount = uint32_t(std::size(boundsAccess));
        bounds.phase = Phase::Bounds;
        bounds.flags = SystemFlags::MainThread;

        // The interpolation of driven entities (8.4) has a phase of its own: the domains' scatter systems register later
        // than it, so in PostSimulation the pair would run in registration order and be reported as ambiguous (6.4).
        m_FixedStepId = FindComponent<FixedStepState>(m_World);   // the scheduler registered it
        if (NVRHI_FAILED(scheduler->AddPhase("SimulationInterpolation", Phase::PostSimulation, out.interpolation)))
            out.interpolation = c_NoPhase;
        const ComponentAccess interpolationAccess[] = {
            { m_Ids[SlotSimPose], Access::Read, {} },
            { m_SimPreviousId, Access::Read, {} },
            { m_Ids[SlotSimScale], Access::Read, {} },
            { m_SimBodyId, Access::Read, {} },
            { m_FixedStepId, Access::Read, {} },
            { m_Ids[SlotWorld], Access::Write, {} },
        };
        SystemDesc interpolation{};
        interpolation.name = "SceneGraph.SimulationInterpolation";
        interpolation.accesses = interpolationAccess;
        interpolation.accessCount = uint32_t(std::size(interpolationAccess));
        interpolation.phase = out.interpolation;

        // So has the resolving of teleport requests: in BeginFrame itself it would be ambiguous with every system of the
        // application that moves nodes there, and after it it sees what they wrote.
        if (NVRHI_FAILED(scheduler->AddPhase("SimulationTeleports", Phase::BeginFrame, out.teleport)))
            out.teleport = c_NoPhase;
        const ComponentAccess teleportAccess[] = {
            { m_SimTeleportId, Access::Write, {} },
            { m_Ids[SlotParent], Access::Read, {} },
            { m_Ids[SlotLocal], Access::Read, {} },
            { m_Ids[SlotSimPose], Access::Read, {} },
            { m_Ids[SlotSimScale], Access::Read, {} },
            { m_Ids[SlotWorld], Access::Read, {} },    // of a driven ancestor without a pose
        };
        SystemDesc teleports{};
        teleports.name = "SceneGraph.SimulationTeleports";
        teleports.accesses = teleportAccess;
        teleports.accessCount = uint32_t(std::size(teleportAccess));
        teleports.phase = out.teleport;

        SceneGraphStore* self = this;
        auto add = [&](const SystemDesc& desc, auto update) noexcept
        {
            nvrhi::AutoPtr<ISystem> system;
            SystemId id = SystemId::Invalid;
            if (desc.phase != c_NoPhase && NVRHI_SUCCEEDED(CreateLambdaSystem(desc, update, &system)) &&
                NVRHI_SUCCEEDED(scheduler->AddSystem(system, id)))
                return;
            log::warning("SceneGraph: cannot add the system '%s'", desc.name);
            complete = false;
        };
        add(propagation, [self](ISystemContext* context) noexcept { return self->RunPropagation(context); });
        add(bounds, [self](ISystemContext* context) noexcept { return self->RunBounds(context); });
        add(interpolation, [self](ISystemContext* context) noexcept { return self->RunInterpolation(context); });
        add(teleports, [self](ISystemContext* context) noexcept { return self->RunTeleports(context); });
        if (NVRHI_FAILED(scheduler->Build()))
        {
            log::warning("SceneGraph: the scheduler's phases do not build");
            complete = false;
        }
        return complete;
    }

    void SceneGraphStore::SetJobSystem(IJobSystem* jobSystem) noexcept
    {
        nvrhi::AutoPtr<IJobSystem> jobs = jobSystem ? jobSystem : GetSharedJobSystem();
        if (jobs == m_Jobs)
            return;
        // Refused before anything changes: inside a phase no scheduler can be made, and the graph must keep its own.
        if (m_World->IsInPhase() || !m_Impl->CanChangeStructure())
        {
            log::warning("SceneGraph::SetJobSystem: inside a phase or a leaf callback; nothing changed");
            return;
        }
        if (GetSimulationDomainCount(m_Scheduler) != 0)
        {
            log::warning("SceneGraph::SetJobSystem: the new scheduler would not have the simulation domains; nothing "
                         "changed (remove them first)");
            return;
        }
        Systems systems;
        if (!CreateSystems(jobs, systems))
        {
            log::warning("SceneGraph::SetJobSystem: the new scheduler is not complete; nothing changed");
            return;
        }
        m_Jobs = jobs;
        TakeSystems(systems);
        // the new systems have not run: their first run treats every chunk as changed
    }

    SceneGraphStore::~SceneGraphStore()
    {
        // Lifetime (5.3): every node whose object is referenced elsewhere is detached into records first, so that it
        // outlives the graph as the root of its own subgraph, as a referenced child outlived its parent before. The
        // world holds one reference to each node; the graph has released its root already. One pre-order pass from
        // each root (a node without a Parent): a referenced node takes its subgraph along, so the walk skips it.
        // A Defer scope still open (E7.8): what it kept goes first, without the deferred reconciliation.
        if (m_DeferReads.exchange(0, std::memory_order_acq_rel) != 0)
            LeaveDeferring();
        m_ReconcileDeferred = false;
        ReleaseDeferred();
        SyncArchetypes();
        {
            ScratchScope scope(*this);
            std::vector<Entity> roots;
            const uint32_t archetypeCount = m_Impl->GetArchetypeStorageCount();
            for (uint32_t a = 0; a < archetypeCount; ++a)
            {
                const Archetype& archetype = m_Impl->GetArchetypeStorage(a);
                if (ColumnIndex(archetype, a, SlotNode) == c_Absent ||
                    ColumnIndex(archetype, a, SlotParent) != c_Absent)
                    continue;
                for (uint32_t c = 0; c < archetype.GetChunkCount(); ++c)
                    roots.insert(roots.end(), archetype.Entities(c), archetype.Entities(c) + archetype.chunks[c].count);
            }
            std::sort(roots.begin(), roots.end());
            std::vector<Entity> tops;
            for (Entity root : roots)
            {
                Entity entity = root;
                while (entity != Entity::Null)
                {
                    SceneGraphNode* node = NodeOf(entity);
                    const SceneLinks* links = Get<SceneLinks>(entity, SlotLinks);
                    const bool referenced = node && ReferenceCount(node) > 1;
                    if (referenced)
                        tops.push_back(entity);
                    if (!referenced && links && links->firstChild != Entity::Null)
                    {
                        entity = links->firstChild;
                        continue;
                    }
                    // the next sibling, or that of the nearest ancestor that has one, inside the root's subgraph
                    for (;;)
                    {
                        if (entity == root)
                        {
                            entity = Entity::Null;
                            break;
                        }
                        const SceneLinks* current = Get<SceneLinks>(entity, SlotLinks);
                        if (current && current->nextSibling != Entity::Null)
                        {
                            entity = current->nextSibling;
                            break;
                        }
                        const Parent* parent = Get<Parent>(entity, SlotParent);
                        entity = parent ? parent->entity : root;
                    }
                }
            }
            // after the walk: moving a subgraph into records keeps the rows, but the nodes no longer name them
            for (Entity top : tops)
            {
                GatherSubgraph(top);
                MoveToRecords(false);
            }
        }

        // Nothing is delivered while the world is torn down: leaves are not unregistered, as before.
        m_Scheduler.Reset();
        static_cast<LeafObserver*>(m_LeafObserver.Get())->Disconnect();
        m_LeafObserver.Reset();
        // Runtime-only rows (E7.1) keep their slots and MeshRef; the tables they name go with the graph, whole.
        static_cast<RuntimeObserver*>(m_RuntimeObserver.Get())->Disconnect();
        m_RuntimeObserver.Reset();
        // A world that someone else holds lives on with its rows: its nodes and leaves are in no graph all the same.
        const bool worldIsHeld = ReferenceCount(m_World.Get()) > 1;
        if (m_AnyLeafRecord)
            TakeLeafValuesAtTeardown(worldIsHeld);
        if (worldIsHeld)
        {
            const uint32_t archetypeCount = m_Impl->GetArchetypeStorageCount();
            for (uint32_t a = 0; a < archetypeCount; ++a)
            {
                const Archetype& archetype = m_Impl->GetArchetypeStorage(a);
                const uint8_t column = ColumnIndex(archetype, a, SlotNode);
                const uint8_t leafColumn = ColumnIndex(archetype, a, SlotLeaf);
                if (column == c_Absent)
                    continue;
                for (uint32_t c = 0; c < archetype.GetChunkCount(); ++c)
                {
                    const SceneNode* nodes = reinterpret_cast<const SceneNode*>(ColumnBase(archetype, c, column));
                    const LeafObject* leaves = leafColumn == c_Absent ? nullptr :
                        reinterpret_cast<const LeafObject*>(ColumnBase(archetype, c, leafColumn));
                    for (uint32_t r = 0; r < archetype.chunks[c].count; ++r)
                    {
                        SceneGraphNode* node = nodes[r].object;
                        if (!node || node->m_Store != this)
                            continue;
                        node->m_Store = nullptr;
                        node->m_Entity = Entity::Null;
                        ForgetRow(node);
                        if (leaves && leaves[r].object && leaves[r].object->GetNode() == node)
                            leaves[r].object->m_Node.Reset();   // the node has no leaf any more
                    }
                }
            }
        }
        m_Impl = nullptr;
        m_World.Reset();
        m_Jobs.Reset();
    }

    // ------------------------------------------------------------------------------------------ rows

    void SceneGraphStore::SyncArchetypes() noexcept
    {
        const uint32_t count = m_Impl->GetArchetypeStorageCount();
        for (uint32_t a = uint32_t(m_ArchetypeColumns.size()); a < count; ++a)
        {
            const Archetype& archetype = m_Impl->GetArchetypeStorage(a);
            std::array<uint8_t, SlotCount> columns;
            for (uint32_t s = 0; s < SlotCount; ++s)
            {
                assert(archetype.FindColumn(m_Ids[s]) == c_NoColumn || archetype.FindColumn(m_Ids[s]) < c_Absent);
                columns[s] = LookUpColumn(archetype, Slot(s));
            }
            m_ArchetypeColumns.push_back(columns);
        }
    }

#if defined(_MSC_VER)
    __declspec(noinline)
#endif
    uint8_t SceneGraphStore::LookUpColumn(const Archetype& archetype, Slot slot) const noexcept
    {
        if (slot == SlotLocal && archetype.FindColumn(m_Ids[SlotDriven]) != c_NoColumn)
            return c_Absent;
        const uint32_t column = archetype.FindColumn(m_Ids[slot]);
        return column == c_NoColumn ? c_Absent : uint8_t(column);
    }

#if defined(_MSC_VER)
    __declspec(noinline)
#endif
    bool SceneGraphStore::LocateNodeSlow(const SceneGraphNode* node, RowRef& out) const noexcept
    {
        return Locate(node->m_Entity, out);
    }

    SceneGraphNode* SceneGraphStore::NodeOfSlow(Entity entity) const noexcept
    {
        const SceneNode* node = Get<SceneNode>(entity, SlotNode);
        return node ? node->object : nullptr;
    }

    uint32_t SceneGraphStore::DepthOf(Entity entity) const noexcept
    {
        RowRef row;
        if (!Locate(entity, row))
            return 0;
        const uint32_t column = row.archetype->FindColumn(m_DepthId);
        if (column == c_NoColumn)
            return 0;
        uint32_t depth = 0;
        std::memcpy(&depth, row.archetype->sharedValues + row.archetype->columns[column].offset, sizeof(depth));
        return depth;
    }

    bool SceneGraphStore::IsSeparate(ComponentId id) const noexcept
    {
        const StorageKind kind = m_Impl->GetType(id)->kind;
        return kind == StorageKind::Sparse || kind == StorageKind::Shared;
    }

    uint32_t SceneGraphStore::ResolveLeafType(SceneGraphLeaf* leaf)
    {
        const LeafTypeDesc& desc = leaf->GetTypeDesc();
        for (uint32_t i = 0; i < uint32_t(m_LeafTypes.size()); ++i)
        {
            if (m_LeafTypes[i].desc == &desc)
                return i;
        }
        m_LeafTypeTags.push_back(m_Ids[SlotLeaf]);
        LeafType type{ &desc, uint32_t(m_LeafTypeTags.size()), 0, 0, false };
        // The mesh-instance tags and columns say what a row's leaf is: the leaf observer, InvalidateContent and the
        // render service's queries cast the leaf of a row with them (MeshRef, RenderFlags and GpuInstanceSlot, or
        // MeshInstanceLeaf: MeshInstance; SkinnedMeshInstanceLeaf: SkinnedMeshInstance). A class that is not of their
        // class gets none of them (backlog E72-R4), with one warning per class and graph. The class is known here: a
        // descriptor is its class's alone.
        const bool meshInstance = query_cast<MeshInstance>(leaf) != nullptr;
        const bool skinnedInstance = query_cast<SkinnedMeshInstance>(leaf) != nullptr;
        bool refused = false;
        for (uint32_t t = 0; t < desc.tagCount; ++t)
        {
            const nvrhi::GUID& tag = desc.tags[t]->uuid;
            if ((!meshInstance && tag == ComponentTraits<MeshInstanceLeaf>::Desc().uuid) ||
                (!skinnedInstance && tag == ComponentTraits<SkinnedMeshInstanceLeaf>::Desc().uuid))
            {
                refused = true;
                continue;
            }
            ComponentId id = ComponentId::Invalid;
            if (NVRHI_FAILED(m_World->RegisterComponent(*desc.tags[t], id)))
            {
                log::warning("SceneGraph: cannot register the leaf tag %s", desc.tags[t]->name);
                continue;
            }
            if (desc.tags[t]->uuid == ComponentTraits<SkinnedMeshReferenceLeaf>::Desc().uuid)
                m_JointTag = id;
            m_LeafTypeTags.push_back(id);
            ++type.tagCount;
        }
        if (desc.dynamicBounds)
        {
            const ComponentId id = RegisterComponent<DynamicBoundsLeaf>(m_World);
            if (id != ComponentId::Invalid)
            {
                m_DynamicBoundsTag = id;
                m_LeafTypeTags.push_back(id);
                ++type.tagCount;
            }
            else
                log::warning("SceneGraph: cannot register the dynamic-bounds tag");
        }
        // The class's columns (D19): with leafRecord, a row takes the leaf's values; else zero until something fills them
        // (the mesh-instance columns: OnLeaf).
        m_AnyLeafRecord |= desc.leafRecord;
        for (uint32_t c = 0; c < desc.columnCount; ++c)
        {
            const nvrhi::GUID& uuid = desc.columns[c]->uuid;
            if (!meshInstance && (uuid == ComponentTraits<MeshRef>::Desc().uuid ||
                                  uuid == ComponentTraits<RenderFlags>::Desc().uuid ||
                                  uuid == ComponentTraits<GpuInstanceSlot>::Desc().uuid))
            {
                refused = true;
                continue;
            }
            ComponentId id = ComponentId::Invalid;
            if (NVRHI_FAILED(m_World->RegisterComponent(*desc.columns[c], id)))
            {
                log::warning("SceneGraph: cannot register the leaf column %s", desc.columns[c]->name);
                continue;
            }
            m_LeafTypeTags.push_back(id);
            ++type.columnCount;
        }
        if (refused)
        {
            const nvrhi::GUID& id = desc.classId;
            log::warning("SceneGraph: the leaf class {%08x-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x} declares mesh-"
                         "instance tags or columns of a class it is not (MeshInstanceLeaf, SkinnedMeshInstanceLeaf, "
                         "MeshRef, RenderFlags, GpuInstanceSlot); its rows get none of them (one warning per class and "
                         "graph)", id.Data1, id.Data2, id.Data3, id.Data4[0], id.Data4[1], id.Data4[2], id.Data4[3],
                         id.Data4[4], id.Data4[5], id.Data4[6], id.Data4[7]);
        }
        for (uint32_t t = 0; t < type.tagCount + type.columnCount; ++t)
            type.separate |= IsSeparate(m_LeafTypeTags[type.firstTag + t]);
        for (Slot slot : c_BoundsSlots)
            m_LeafTypeTags.push_back(m_Ids[slot]);
        m_LeafTypes.push_back(type);
        return uint32_t(m_LeafTypes.size() - 1);
    }

    // --------------------------------------------------------------------------------- leaf columns (9.2)

    bool SceneGraphStore::DeclaresColumn(const LeafTypeDesc& type, const nvrhi::GUID& uuid) noexcept
    {
        for (uint32_t c = 0; c < type.columnCount; ++c)
        {
            if (type.columns[c]->uuid == uuid)
                return IsPlainColumn(*type.columns[c]);
        }
        return false;
    }

    void* SceneGraphStore::LeafColumn(const SceneGraphLeaf* leaf, const ComponentTypeDesc& type, bool write) noexcept
    {
        // The record first: it exists exactly while no row is the values' home, or names the row they are leaving.
        SceneGraphStore* store = nullptr;
        Entity entity = Entity::Null;
        if (LeafRecord* record = leaf->m_LeafRecord)
        {
            if (!record->IsInRow())
                return record->Find(type.uuid);
            store = record->GetStore();
            entity = record->GetEntity();
        }
        else
        {
            store = StoreOf(leaf);
            if (!store)
                return nullptr;
            entity = leaf->GetNode()->m_Entity;
        }
        RowRef row;
        if (!DeclaresColumn(leaf->GetTypeDesc(), type.uuid) || !store->Locate(entity, row))
            return nullptr;
        const LeafObject* object = static_cast<const LeafObject*>(store->Column(row, SlotLeaf));
        if (!object || object->object != leaf)
            return nullptr;   // the row of another leaf: this one is not in yet, or not any more
        uint32_t column = c_NoColumn;
        void* value = RowValue(*store->m_Impl, *row.archetype, row.chunk, row.row, type.uuid, &column);
        if (value && write)
            StampRowWrite(*row.archetype, row.chunk, column, row.row, store->m_Impl->GetStampVersion());
        return value;
    }

    void SceneGraphStore::BeginLeavingRow(SceneGraphLeaf* leaf, const SceneGraphNode* node, Entity entity) noexcept
    {
        // A leaf that two nodes hold (no API makes one, SetLeaf of an attached leaf does) belongs to the node it names:
        // its values in the other's row are not its own.
        const LeafTypeDesc& type = leaf->GetTypeDesc();
        if (!type.leafRecord || leaf->m_LeafRecord || leaf->GetNode() != node)
            return;
        leaf->m_LeafRecord = LeafRecord::CreateInRow(type, this, entity);
        if (!leaf->m_LeafRecord)
            log::warning("SceneGraph: out of memory for a leaf's record; its column values are lost");
    }

    void SceneGraphStore::OnRowRemoving(void* context, Entity entity, const Archetype& archetype, uint32_t chunk,
                                        uint32_t row) noexcept
    {
        SceneGraphStore* store = static_cast<SceneGraphStore*>(context);
        const uint32_t column = archetype.FindColumn(store->m_Ids[SlotLeaf]);
        if (column == c_NoColumn)
            return;
        const LeafObject& leaf = static_cast<const LeafObject*>(archetype.ColumnData(chunk, column))[row];
        LeafRecord* record = leaf.object ? leaf.object->m_LeafRecord : nullptr;
        if (record && record->GetStore() == store && record->GetEntity() == entity)
            record->TakeFromRow(*store->m_Impl, archetype, chunk, row);
    }

    void SceneGraphStore::OnRowsAdded(void* context, uint32_t count) noexcept
    {
        const LeafMove& move = *static_cast<const LeafMove*>(context);
        SceneGraphStore* store = move.store;
        for (uint32_t i = 0; i < count; ++i)
        {
            SceneGraphLeaf* leaf = move.nodes[i].leaf;
            LeafRecord* record = leaf ? leaf->m_LeafRecord : nullptr;
            RowRef row;
            if (!record || leaf->GetNode() != move.nodes[i].node || !store->Locate(move.entities[i], row))
                continue;   // a leaf without a record: the row's default values
            // A record is made once and does not grow: it was made for this class, or (in a constructor of a base class)
            // for one with other columns.
            assert(record->HasColumnsOf(leaf->GetTypeDesc()) &&
                   "a leaf's first Edit ran in a constructor whose class has other columns");
            if (record->IsInRow())
            {
                // attached by a callback of its removal from another graph: it takes its values along now
                const Entity leaving = record->GetEntity();
                record->GetStore()->FinishLeavingRows(&leaving, 1);
                if (record->IsInRow())   // the row is gone already (cannot happen while its removal runs)
                {
                    leaf->m_LeafRecord = nullptr;
                    LeafRecord::Destroy(record);
                    continue;
                }
            }
            leaf->m_LeafRecord = nullptr;
            LeafRecord::MoveIntoRow(record, *store->m_Impl, *row.archetype, row.chunk, row.row);
        }
    }

    void SceneGraphStore::FinishLeavingRows(const Entity* entities, size_t count) noexcept
    {
        for (size_t i = 0; i < count; ++i)
        {
            RowRef row;
            if (Locate(entities[i], row))
                OnRowRemoving(this, entities[i], *row.archetype, row.chunk, row.row);
        }
    }

    void SceneGraphStore::FlushObservers(nvrhi::FRESULT& result) noexcept
    {
        const nvrhi::FRESULT flushed = m_Impl->FlushObservers();
        if (NVRHI_SUCCEEDED(result))
            result = flushed;
    }

    void SceneGraphStore::WarnObservers(const char* operation, nvrhi::FRESULT result) noexcept
    {
        if (NVRHI_FAILED(result))
            log::warning("SceneGraph::%s: an observer failed, or the observers' commands did not settle (%d); the call "
                         "stands", operation, int(result));
    }

    nvrhi::FRESULT SceneGraphStore::DestroyRows(const Entity* entities, size_t count, const char* operation) noexcept
    {
        const WorldImpl::RowHooks hooks = LeavingHooks();
        const nvrhi::FRESULT result = m_Impl->DestroyEntitiesInternal(entities, uint32_t(count),
            m_Impl->GetStampVersion(), m_AnyLeafRecord ? &hooks : nullptr);
        if (NVRHI_FAILED(result) && m_AnyLeafRecord)
            FinishLeavingRows(entities, count);
        nvrhi::FRESULT observers = nvrhi::FS_OK;
        FlushObservers(observers);
        WarnObservers(operation, observers);
        return result;
    }

    // The values of leaves that outlive the store move into their records: leaves that someone else references, and all
    // of them when the world itself lives on. The others go with their rows.
    void SceneGraphStore::TakeLeafValuesAtTeardown(bool worldIsHeld) noexcept
    {
        const uint32_t archetypeCount = m_Impl->GetArchetypeStorageCount();
        for (uint32_t a = 0; a < archetypeCount; ++a)
        {
            const Archetype& archetype = m_Impl->GetArchetypeStorage(a);
            const uint8_t column = ColumnIndex(archetype, a, SlotLeaf);
            if (column == c_Absent)
                continue;
            for (uint32_t c = 0; c < archetype.GetChunkCount(); ++c)
            {
                const LeafObject* leaves = reinterpret_cast<const LeafObject*>(ColumnBase(archetype, c, column));
                for (uint32_t r = 0; r < archetype.chunks[c].count; ++r)
                {
                    SceneGraphLeaf* leaf = leaves[r].object;
                    if (!leaf)
                        continue;
                    const Entity entity = archetype.Entities(c)[r];
                    // the leaves of the nodes detached above name their rows already (MoveToRecords)
                    const SceneGraphNode* node = leaf->GetNode();
                    if ((worldIsHeld || ReferenceCount(leaf) > 1) && node && node->m_Store == this &&
                        node->m_Entity == entity)
                        BeginLeavingRow(leaf, node, entity);
                    OnRowRemoving(this, entity, archetype, c, r);
                }
            }
        }
    }

    // ------------------------------------------------------------------------------------- orphan links

    nvrhi::AutoPtr<SceneGraphNode> SceneGraphStore::UnlinkOrphan(SceneGraphNode* child)
    {
        OrphanRecord* record = child ? child->m_Record : nullptr;
        if (!record || !record->parent)
            return nullptr;
        OrphanRecord& parent = *record->parent->m_Record;

        SceneGraphNode* prev = record->prevSibling;
        nvrhi::AutoPtr<SceneGraphNode>& link = prev ? prev->m_Record->nextSibling : parent.firstChild;
        assert(link.Get() == child);

        // take the list's reference to the child, and close the gap with the child's next sibling
        nvrhi::AutoPtr<SceneGraphNode> removed = std::move(link);
        link = std::move(record->nextSibling);
        if (link)
            link->m_Record->prevSibling = prev;
        else
            parent.lastChild = prev;

        record->parent = nullptr;
        record->prevSibling = nullptr;
        return removed;
    }

    void SceneGraphStore::AppendOrphan(SceneGraphNode* parent, SceneGraphNode* child)
    {
        assert(parent && child && !parent->m_Store && !child->m_Store);
#ifndef NDEBUG
        // linking a node under itself or under one of its descendants would make a cycle
        for (SceneGraphNode* ancestor = parent; ancestor;
             ancestor = ancestor->m_Record ? ancestor->m_Record->parent : nullptr)
            assert(ancestor != child);
#endif
        // A node is linked into at most one child list: a node that moves brings the reference its old list held,
        // any other node gets a new one.
        nvrhi::AutoPtr<SceneGraphNode> link = UnlinkOrphan(child);
        if (!link)
            link = child;

        OrphanRecord& record = child->EnsureRecord();
        OrphanRecord& parentRecord = parent->EnsureRecord();
        assert(!record.nextSibling && !record.prevSibling);
        record.parent = parent;
        record.prevSibling = parentRecord.lastChild;
        (parentRecord.lastChild ? parentRecord.lastChild->m_Record->nextSibling : parentRecord.firstChild) =
            std::move(link);
        parentRecord.lastChild = child;
    }

    // ------------------------------------------------------------------------------- attach (5.3)

    namespace
    {
        // Scratch per thread and nesting depth, kept for its capacity.
        constexpr uint32_t c_ScratchDepth = 4;
        thread_local uint32_t t_ScratchDepth = 0;
    }

    namespace
    {
        // Set when the thread's scratch is destroyed (thread exit, or the main thread's before static destruction): a
        // graph torn down after that gets scratch of its own. Trivially destructible, so it stays readable then.
        thread_local bool t_ScratchGone = false;
        // The same for RunDeepTasks's walk stack.
        thread_local bool t_DeepStackGone = false;
    }

    SceneGraphStore::ScratchScope::ScratchScope(SceneGraphStore& s) noexcept
        : store(s), previous(s.m_S)
    {
        if (t_ScratchGone || t_ScratchDepth >= c_ScratchDepth)
        {
            owned = new Scratch();
            store.m_S = owned;
            return;
        }
        struct Holder
        {
            Scratch scratch[c_ScratchDepth];
            ~Holder() { t_ScratchGone = true; }
        };
        thread_local Holder holder;
        store.m_S = &holder.scratch[t_ScratchDepth];
        ++t_ScratchDepth;
    }

    SceneGraphStore::ScratchScope::~ScratchScope()
    {
        if (owned)
            delete owned;
        else
            --t_ScratchDepth;
        store.m_S = previous;
    }

    bool SceneGraphStore::CanChangeStructure(const char* operation) const noexcept
    {
        if (m_Impl->CanChangeStructure())
            return true;
        // design 5.6: the scene-graph API changes structure outside phases; observers cannot change it either
        log::warning("SceneGraph::%s: the graph's world takes no structural change now (inside a phase or a leaf "
                     "callback); nothing changed", operation);
        return false;
    }

    nvrhi::FRESULT SceneGraphStore::RunReadOnlyPhase(Phase phase) noexcept
    {
        if (!m_Scheduler)
            return nvrhi::FE_INVALID_ARGS;
        // A system that declares a write (or runs alone) may change anything: the phase is then an ordinary one.
        const bool writers = ecs::PhaseDeclaresWrites(m_Scheduler, phase);
#ifndef NDEBUG
        if (writers)
            log::warning("SceneGraph::RunReadOnlyPhase: a system of phase %u declares a write or is exclusive; the next "
                         "Refresh runs its phases", uint32_t(phase));
#endif
        const uint64_t version = m_World->GetVersion();
        const uint64_t structuralVersion = m_World->GetStructuralVersion();
        ecs::SchedulerCounters before{};
        const bool counted = ecs::GetSchedulerCounters(m_Scheduler, before);
        const nvrhi::FRESULT result = m_Scheduler->RunPhase(phase);
        if (NVRHI_FAILED(result) || writers || !counted)
            return result;
        ecs::SchedulerCounters after{};
        ecs::GetSchedulerCounters(m_Scheduler, after);
        // Current before the phase (nothing ran on the world since the last Refresh), and the phase changed nothing:
        // no command played back, no structural change. Its versions are then not a change.
        if (m_RefreshedVersion == version && m_RefreshedStructuralVersion == structuralVersion &&
            m_World->GetStructuralVersion() == structuralVersion && after.playbacks == before.playbacks)
            m_RefreshedVersion = m_World->GetVersion();
        return result;
    }

    void SceneGraphStore::WarnRenderRead(const char* operation) const noexcept
    {
        // Open point 12 of the E5 sub-plan, beside the phase check of design 5.6: the render service's snapshot points
        // at this graph's leaves and assets until its EndFrame. A warning, not an error (log::error breaks into the
        // debugger); the write applies, so Debug and Release behave alike.
        // A Defer scope (E7.8) allows writes: what they release goes to the retire list.
        if (m_RenderReads != 0 && m_RenderReadQuiet == 0 && !IsDeferring())
            log::warning("SceneGraph::%s while a frame records (between RenderService::Extract and EndFrame): the "
                         "frame's snapshot may point at what this changes; the write applies", operation);
    }

    // ------------------------------------------------------------------- deferred releases (E7.8)

    uint32_t SceneGraphStore::BeginRenderRead(SceneGraph::RenderReadMode mode) noexcept
    {
        const uint32_t depth = m_RenderReads++;
        if (depth >= 64)
        {
            if (mode == SceneGraph::RenderReadMode::Defer)
                log::warning("SceneGraph::BeginRenderRead: a Defer scope nested %u deep acts as a Warn scope",
                             depth + 1);
            return 0;
        }
        if (++m_LastRenderReadScope == 0)
            m_LastRenderReadScope = 1;
        m_RenderReadScopes[depth] = m_LastRenderReadScope;
        if (mode == SceneGraph::RenderReadMode::Defer)
        {
            m_RenderReadModes |= uint64_t(1) << depth;
            if (m_DeferReads.fetch_add(1, std::memory_order_acq_rel) == 0)
                EnterDeferring();
        }
        return m_LastRenderReadScope;
    }

    void SceneGraphStore::EndRenderRead() noexcept
    {
        if (m_RenderReads != 0)
            CloseRenderRead(m_RenderReads - 1);
    }

    // The scope BeginRenderRead returned, wherever it is: the scopes opened after it stay open, with their modes and
    // identities (backlog I6-N3). 0 (a scope beyond the 64th has no identity) closes the innermost.
    void SceneGraphStore::EndRenderRead(uint32_t scope) noexcept
    {
        if (scope == 0)
        {
            EndRenderRead();
            return;
        }
        for (uint32_t i = std::min(m_RenderReads, 64u); i-- > 0;)
        {
            if (m_RenderReadScopes[i] == scope)
            {
                CloseRenderRead(i);
                return;
            }
        }
        log::warning("SceneGraph::EndRenderRead: the scope is not open (an EndRenderRead without its BeginRenderRead "
                     "closed it already); nothing closed");
    }

    void SceneGraphStore::CloseRenderRead(uint32_t index) noexcept
    {
        bool defer = false;
        if (index < 64)
        {
            defer = ((m_RenderReadModes >> index) & 1) != 0;
            // the scopes above move one down; one beyond the 64th that comes to the 64th has no identity (0) and acts
            // as Warn, as it did
            const uint64_t below = m_RenderReadModes & ((uint64_t(1) << index) - 1);
            const uint64_t above = index + 1 < 64 ? (m_RenderReadModes >> (index + 1)) << index : 0;
            m_RenderReadModes = below | above;
            for (uint32_t i = index; i + 1 < 64; ++i)
                m_RenderReadScopes[i] = m_RenderReadScopes[i + 1];
            m_RenderReadScopes[63] = 0;
        }
        --m_RenderReads;
        // The last Defer scope: what it deferred runs now, on this thread (after the pipelined frame's join).
        if (defer && m_DeferReads.fetch_sub(1, std::memory_order_acq_rel) == 1)
        {
            LeaveDeferring();
            ReleaseDeferred();
        }
    }

    bool SceneGraphStore::IsAnyDeferring() noexcept
    {
        return g_DeferringStores.load(std::memory_order_acquire) != 0;
    }

    void SceneGraphStore::Retire(nvrhi::IObject* object) noexcept
    {
        if (!object || !IsDeferring())
            return;
        try
        {
            m_Retired.push_back(object);   // grows to its high-water mark, then allocates nothing
        }
        catch (...)
        {
            log::warning("SceneGraph: out of memory; an object released while a pipelined frame records is not kept "
                         "until its EndRenderRead");
            return;
        }
        object->AddRef();
    }

    void SceneGraphStore::ReconcileOrDefer()
    {
        if (IsDeferring())
        {
            m_ReconcileDeferred = true;
            return;
        }
        ReconcileMeshes();
        m_ReconcileDeferred = false;
    }

    void SceneGraphStore::ReleaseDeferred() noexcept
    {
        // The mesh table first: it may release materials and meshes, which go at once now (no Defer scope is open).
        if (m_ReconcileDeferred)
        {
            try
            {
                ReconcileMeshes();
                m_ReconcileDeferred = false;
            }
            catch (const std::exception& error)
            {
                log::warning("SceneGraph::EndRenderRead: the deferred mesh reconciliation failed ('%s'); the next "
                             "Refresh runs it", error.what());
            }
            catch (...)
            {
                log::warning("SceneGraph::EndRenderRead: the deferred mesh reconciliation failed; the next Refresh "
                             "runs it");
            }
        }
        // In retirement order. A release may destroy an object whose destructor releases others (a node its record's
        // children and leaf); none of them comes back here, since no Defer scope is open.
        const size_t count = m_Retired.size();
        for (size_t i = 0; i < count; ++i)
        {
            nvrhi::IObject* object = m_Retired[i];
            m_Retired[i] = nullptr;
            object->Release();
        }
        m_Retired.clear();   // capacity kept
    }

    // The leaf classes and local boxes of the gathered nodes. A separate pass, so that the leaves, which the walk does
    // not touch, can be fetched ahead.
    void SceneGraphStore::ResolveLeaves()
    {
        std::vector<NewNode>& nodes = m_S->newNodes;
        const size_t n = nodes.size();
        for (size_t i = 0; i < n; ++i)
        {
            if (i + c_PrefetchDistance < n)
                Prefetch(nodes[i + c_PrefetchDistance].leaf);
            NewNode& entry = nodes[i];
            if (entry.leaf)
            {
                entry.leafType = ResolveLeafType(entry.leaf);
                entry.localBounds = entry.leaf->GetLocalBoundingBox();
            }
        }
    }

    nvrhi::AutoPtr<SceneGraphNode> SceneGraphStore::AttachOrphan(SceneGraphNode* parent, SceneGraphNode* child)
    {
        assert(!child->m_Store);
        if (!CanChangeStructure("Attach"))
            return nullptr;
        m_World->AdvanceVersion();   // what this stamps is newer than any version read before it (GpuScene)
        ScratchScope scope(*this);

        // The reference that the subgraph's root brings: its old orphan parent's link, or a new one.
        nvrhi::AutoPtr<SceneGraphNode> incoming = UnlinkOrphan(child);
        if (!incoming)
            incoming = child;

        // The subgraph in pre-order, from the records' links. The references move into the world: each link's to the
        // node it names, the record's to its leaf.
        std::vector<NewNode>& nodes = m_S->newNodes;
        nodes.clear();
        bool anyLocal = false;
        uint32_t parentIndex = c_None;
        SceneGraphNode* node = child;
        while (node)
        {
            OrphanRecord* record = node->m_Record;
            const uint32_t index = uint32_t(nodes.size());
            NewNode entry{ node, record, nullptr, parentIndex, c_None, dm::box3::empty(), false };
            SceneGraphNode* firstChild = nullptr;
            if (record)
            {
                entry.leaf = record->leaf.Detach();
                entry.hasLocal = record->hasLocal;
                anyLocal |= record->hasLocal;
                firstChild = record->firstChild.Get();
            }
            nodes.push_back(entry);
            if (firstChild)
            {
                parentIndex = index;
                node = firstChild;
                continue;
            }
            for (;;)
            {
                if (node == child)
                {
                    node = nullptr;
                    break;
                }
                OrphanRecord& current = *node->m_Record;
                if (current.nextSibling)
                {
                    node = current.nextSibling.Get();
                    break;
                }
                node = current.parent;
                parentIndex = nodes[parentIndex].parent;
            }
        }
        for (const NewNode& entry : nodes)
        {
            if (entry.record)
            {
                (void)entry.record->firstChild.Detach();
                (void)entry.record->nextSibling.Detach();
            }
        }
        (void)incoming.Detach();
        ResolveLeaves();

        nvrhi::AutoPtr<SceneGraphNode> result(child);
        const Entity parentEntity = parent ? parent->m_Entity : Entity::Null;
        const bool created = CreateNodes(parentEntity, parent ? DepthOf(parentEntity) : 0);

        m_PendingStructure = true;
        m_StaleWorldTransforms = true;
        m_PendingTransform |= anyLocal;
        BumpEpoch();
        return created ? result : nullptr;
    }

    nvrhi::AutoPtr<SceneGraphNode> SceneGraphStore::AttachCopy(SceneGraphNode* parent, SceneGraphNode* source,
                                                               SceneGraphStore* sourceStore)
    {
        if (!CanChangeStructure("Attach"))
            return nullptr;
        m_World->AdvanceVersion();
        ScratchScope scope(*this);

        // The source subgraph as it is before the call (the parent may lie inside it), copied into new nodes in no
        // graph, with records that hold the local transforms and cloned leaves; then attached like any orphans.
        std::vector<NewNode>& nodes = m_S->newNodes;
        nodes.clear();
        std::unordered_map<SceneGraphNode*, uint32_t> nodeMap;
        ParentWorld parentWorld;
        bool anyLocal = false;
        uint32_t parentIndex = c_None;
        SceneGraphWalker walker(source);
        while (walker)
        {
            SceneGraphNode* original = walker.Get();
            const uint32_t index = uint32_t(nodes.size());
            SceneGraphNode* copy = MAKE_RC_OBJ_PTR(SceneGraphNode).Detach();   // its reference goes to the world
            copy->m_Name = original->m_Name;
            OrphanRecord& record = copy->EnsureRecord();

            RowRef row;
            if (sourceStore->Locate(original->m_Entity, row))
            {
                if (const auto* local = static_cast<const LocalTransform*>(sourceStore->Column(row, SlotLocal)))
                {
                    record.local = *local;
                    record.hasLocal = true;
                    anyLocal = true;
                }
                else if (sourceStore->IsDriven(row))   // the copy is authored, at the simulated pose
                {
                    record.local = sourceStore->SimulatedLocal(row, parentWorld, "Attach");
                    record.hasLocal = true;
                    anyLocal = true;
                }
            }
            SceneGraphLeaf* clone = nullptr;
            if (SceneGraphLeaf* leaf = original->GetLeaf())
            {
                clone = leaf->Clone().Detach();
                clone->m_Node = nvrhi::WeakPtr<SceneGraphNode>(copy);
            }
            nodes.push_back(NewNode{ copy, &record, clone, parentIndex, c_None, dm::box3::empty(), record.hasLocal });
            nodeMap[original] = index;

            int deltaDepth = walker.Next(true);
            if (deltaDepth > 0)
                parentIndex = index;
            else
            {
                while (deltaDepth++ < 0)
                    parentIndex = nodes[parentIndex].parent;
            }
        }

        // Point the cloned animations, skins and joint references at the new nodes.
        auto mapped = [&](SceneGraphNode* original) -> const NewNode*
        {
            auto it = nodeMap.find(original);
            return it != nodeMap.end() ? &nodes[it->second] : nullptr;
        };
        for (const NewNode& entry : nodes)
        {
            if (auto animation = query_cast<SceneGraphAnimation>(entry.leaf))
            {
                for (const auto& channel : animation->GetChannels())
                {
                    if (const NewNode* target = mapped(channel->GetTargetNode()))
                        channel->SetTargetNode(target->node);
                }
            }
            else if (auto skinnedInstance = query_cast<SkinnedMeshInstance>(entry.leaf))
            {
                for (auto& joint : skinnedInstance->joints)
                {
                    auto jointNode = joint.node.Lock();
                    if (const NewNode* target = mapped(jointNode.Get()))
                        joint.node = target->node;
                }
            }
            else if (auto meshReference = query_cast<SkinnedMeshReference>(entry.leaf))
            {
                auto instance = meshReference->m_Instance.Lock();
                if (instance)
                {
                    if (const NewNode* target = mapped(instance->GetNode()))
                        meshReference->m_Instance = query_cast<SkinnedMeshInstance>(target->leaf);
                    else
                        meshReference->m_Instance.Reset();
                }
            }
        }
        ResolveLeaves();

        nvrhi::AutoPtr<SceneGraphNode> result(nodes.front().node);
        const Entity parentEntity = parent ? parent->m_Entity : Entity::Null;
        const bool created = CreateNodes(parentEntity, parent ? DepthOf(parentEntity) : 0);

        m_PendingStructure = true;
        m_StaleWorldTransforms = true;
        m_PendingTransform |= anyLocal;
        BumpEpoch();
        return created ? result : nullptr;
    }

    // Creates the entities of the gathered nodes in pre-order, one CreateEntities call per run of nodes with the same
    // components, and links the first under `parentEntity`. Every node hands its reference and its leaf's to the world;
    // its record is read, then given back to the pool.
    std::atomic<int32_t> SceneGraphStore::s_FailCreateCountdown{ -1 };
    std::atomic<int32_t> SceneGraphStore::s_FailSubgraphCountdown{ -1 };
    std::atomic<uint64_t> SceneGraphStore::s_SubgraphRedos{ 0 };
    std::atomic<uint32_t> SceneGraphStore::s_SerialSubgraphRows{ 4096 };

    SceneContentFlags SceneGraphStore::SafeContentFlags(const SceneGraphLeaf* leaf) noexcept
    {
        try
        {
            return leaf->GetContentFlags();
        }
        catch (...)
        {
            log::warning("SceneGraph: a leaf's GetContentFlags threw; taken as None");
            return SceneContentFlags::None;
        }
    }

    dm::box3 SceneGraphStore::SafeLocalBoundingBox(SceneGraphLeaf* leaf, const dm::box3& fallback) noexcept
    {
        try
        {
            return leaf->GetLocalBoundingBox();
        }
        catch (...)
        {
            log::warning("SceneGraph: a leaf's GetLocalBoundingBox threw; its box stays as it was");
            return fallback;
        }
    }

    bool SceneGraphStore::CreateNodes(Entity parentEntity, uint32_t parentDepth)
    {
        Scratch& S = *m_S;
        const std::vector<NewNode>& nodes = S.newNodes;
        const uint32_t n = uint32_t(nodes.size());
        S.entities.resize(n);
        if (NVRHI_FAILED(m_World->ReserveEntities(n, S.entities.data())))
        {
            log::warning("SceneGraph: cannot reserve %u entities", n);
            std::abort();
        }
        // NodeOf's table covers the new ids before any row exists (a growth that throws leaves nothing half made)
        uint32_t highest = 0;
        for (Entity entity : S.entities)
            highest = std::max(highest, GetEntityIndex(entity));
        if (highest >= m_NodeOfIndex.size())
            m_NodeOfIndex.resize(std::max<size_t>(size_t(highest) + 1, m_NodeOfIndex.size() * 2));

        // The links between the new nodes, by index, and their depth buckets.
        S.firstChild.assign(n, c_None);
        S.lastChild.assign(n, c_None);
        S.prevSibling.assign(n, c_None);
        S.nextSibling.assign(n, c_None);
        S.depth.resize(n);
        S.depth[0] = parentEntity != Entity::Null ? std::min(parentDepth + 1, c_MaxHierarchyDepth) : 0;
        for (uint32_t i = 1; i < n; ++i)
        {
            const uint32_t p = nodes[i].parent;
            if (S.lastChild[p] == c_None)
                S.firstChild[p] = i;
            else
            {
                S.nextSibling[S.lastChild[p]] = i;
                S.prevSibling[i] = S.lastChild[p];
            }
            S.lastChild[p] = i;
            S.depth[i] = std::min(S.depth[p] + 1, c_MaxHierarchyDepth);
        }
        Entity rootPrev = Entity::Null;
        if (parentEntity != Entity::Null)
        {
            if (const SceneLinks* links = Get<SceneLinks>(parentEntity, SlotLinks))
                rootPrev = links->lastChild;
        }
        auto entityAt = [&](uint32_t index) noexcept { return index == c_None ? Entity::Null : S.entities[index]; };

        // The handles name their entities before the rows exist, so that the leaf observer finds them; a node reads
        // its record until its batch is created. A cache filled in another graph is not this graph's.
        for (uint32_t i = 0; i < n; ++i)
        {
            nodes[i].node->m_Store = this;
            nodes[i].node->m_Entity = S.entities[i];
            nodes[i].node->m_CacheEpoch.store(0, std::memory_order_relaxed);
        }

        static const OrphanRecord s_Defaults;
        nvrhi::FRESULT observers = nvrhi::FS_OK;   // they undo no attach
        uint32_t first = 0;
        while (first < n)
        {
            const bool hasParent = first > 0 || parentEntity != Entity::Null;
            const NewNode& head = nodes[first];
            uint32_t end = first + 1;
            while (hasParent && end < n && nodes[end].hasLocal == head.hasLocal && nodes[end].leafType == head.leafType)
                ++end;
            const uint32_t count = end - first;
            const bool hasLeaf = head.leafType != c_None;

            // Value arrays, 16-byte aligned, in one block.
            enum { ValueNode, ValueLinks, ValueDepth, ValueWorld, ValueParent, ValueLocal, ValueLeaf, ValueLocalBounds,
                   ValueWorldBounds, ValueCount };
            size_t offsets[ValueCount] = {};
            size_t total = 0;
            auto reserve = [&](uint32_t slot, size_t elementSize)
            {
                offsets[slot] = total;
                total += (elementSize * count + 15) & ~size_t(15);
            };
            reserve(ValueNode, sizeof(SceneNode));
            reserve(ValueLinks, sizeof(SceneLinks));
            reserve(ValueDepth, sizeof(HierarchyDepth));
            reserve(ValueWorld, sizeof(WorldTransform));
            if (hasParent)
                reserve(ValueParent, sizeof(Parent));
            if (head.hasLocal)
                reserve(ValueLocal, sizeof(LocalTransform));
            if (hasLeaf)
            {
                reserve(ValueLeaf, sizeof(LeafObject));
                reserve(ValueLocalBounds, sizeof(LocalBounds));
            }
            if (S.valueBytes.size() < total + 16)
                S.valueBytes.resize(total + 16);
            uint8_t* base = reinterpret_cast<uint8_t*>(
                (reinterpret_cast<uintptr_t>(S.valueBytes.data()) + 15) & ~uintptr_t(15));

            SceneNode* sceneNodes = reinterpret_cast<SceneNode*>(base + offsets[ValueNode]);
            SceneLinks* links = reinterpret_cast<SceneLinks*>(base + offsets[ValueLinks]);
            HierarchyDepth* depths = reinterpret_cast<HierarchyDepth*>(base + offsets[ValueDepth]);
            WorldTransform* worlds = reinterpret_cast<WorldTransform*>(base + offsets[ValueWorld]);
            Parent* parents = reinterpret_cast<Parent*>(base + offsets[ValueParent]);
            LocalTransform* locals = reinterpret_cast<LocalTransform*>(base + offsets[ValueLocal]);
            LeafObject* leaves = reinterpret_cast<LeafObject*>(base + offsets[ValueLeaf]);
            LocalBounds* localBounds = reinterpret_cast<LocalBounds*>(base + offsets[ValueLocalBounds]);
            for (uint32_t k = 0; k < count; ++k)
            {
                const uint32_t i = first + k;
                if (i + c_PrefetchDistance < n && nodes[i + c_PrefetchDistance].record)
                {
                    const char* ahead = reinterpret_cast<const char*>(nodes[i + c_PrefetchDistance].record);
                    for (size_t line = 0; line < sizeof(OrphanRecord); line += 64)
                        Prefetch(ahead + line);
                }
                const NewNode& entry = nodes[i];
                const OrphanRecord& record = entry.record ? *entry.record : s_Defaults;
                new (sceneNodes + k) SceneNode(entry.node);
                links[k] = SceneLinks{ entityAt(S.firstChild[i]), entityAt(S.lastChild[i]),
                                       i == 0 ? rootPrev : entityAt(S.prevSibling[i]),
                                       i == 0 ? Entity::Null : entityAt(S.nextSibling[i]) };
                depths[k] = HierarchyDepth{ S.depth[i] };
                new (worlds + k) WorldTransform(record.world);
                if (hasParent)
                    parents[k] = Parent{ i == 0 ? parentEntity : S.entities[entry.parent] };
                if (head.hasLocal)
                    new (locals + k) LocalTransform(record.local);
                if (hasLeaf)
                {
                    new (leaves + k) LeafObject(entry.leaf);
                    new (localBounds + k) LocalBounds{ entry.localBounds };
                }
            }

            S.runComponents.clear();
            S.runValues.clear();
            auto add = [&](ComponentId id, const void* values)
            {
                S.runComponents.push_back(id);
                S.runValues.push_back(values);
            };
            add(m_Ids[SlotNode], sceneNodes);
            add(m_Ids[SlotLinks], links);
            add(m_DepthId, depths);
            add(m_Ids[SlotWorld], worlds);
            if (hasParent)
                add(m_Ids[SlotParent], parents);
            if (head.hasLocal)
            {
                add(m_Ids[SlotLocal], locals);
                add(m_Ids[SlotLocalMatrix], nullptr);   // composed by the next propagation (the rows are new)
            }
            if (hasLeaf)
            {
                add(m_Ids[SlotLeaf], leaves);
                add(m_Ids[SlotLocalBounds], localBounds);
                add(m_Ids[SlotWorldBounds], nullptr);   // written by the Bounds system before anything reads it
                add(m_Ids[SlotChunkBounds], nullptr);
                const LeafType& type = m_LeafTypes[head.leafType];
                for (uint32_t t = 0; t < type.tagCount + type.columnCount; ++t)
                    add(m_LeafTypeTags[type.firstTag + t], nullptr);   // columns: filled by the leaf observer
            }

            if (s_FailCreateCountdown.load(std::memory_order_relaxed) >= 0 &&
                s_FailCreateCountdown.fetch_sub(1, std::memory_order_relaxed) == 0)
                add(ComponentId(0x7ffffff0u), nullptr);   // test seam: the world refuses the call and drops the values

            // The batch's nodes read their rows from now on (the leaf observer of the batch reads them): their records
            // leave them now and go back to the pool after the call. Their references and their leaves' stay held
            // until the rows hold theirs, so that a refused call cannot release the last one.
            for (uint32_t i = first; i < end; ++i)
            {
                nodes[i].node->m_Record = nullptr;
                nodes[i].node->AddRef();
                if (nodes[i].leaf)
                    nodes[i].leaf->AddRef();
            }

            // The values of leaves with records move in when the rows exist, before the observers run.
            LeafMove move{ this, nodes.data() + first, S.entities.data() + first };
            const WorldImpl::RowHooks hooks{ &move, &SceneGraphStore::OnRowsAdded, nullptr };
            const bool leafRecord = hasLeaf && m_LeafTypes[head.leafType].desc->leafRecord;

            EntityCreateDesc desc{};
            desc.components = S.runComponents.data();
            desc.values = S.runValues.data();
            desc.reservedEntities = S.entities.data() + first;
            desc.componentCount = uint32_t(S.runComponents.size());
            desc.entityCount = count;
            desc.transfer = ValueTransfer::Relocate;   // the world takes the references and the values
            const nvrhi::FRESULT result = m_Impl->CreateEntitiesInternal(desc, nullptr, m_Impl->GetStampVersion(),
                                                                         leafRecord ? &hooks : nullptr);
            FlushObservers(observers);
            ++m_Counters.createCalls;
            if (NVRHI_FAILED(result))
            {
                WarnObservers("Attach", observers);
                log::warning("SceneGraph: cannot create %u entities (%d); the nodes from there on stay in no graph",
                             count, int(result));
                AbandonNodes(first);
                if (first > 0 && parentEntity != Entity::Null)
                    LinkUnder(parentEntity, S.entities[0]);
                SyncArchetypes();
                return false;
            }

            // the rows hold the references now; the nodes remember where their rows are
            S.records.clear();
            for (uint32_t i = first; i < end; ++i)
            {
                if (nodes[i].record)
                    S.records.push_back(nodes[i].record);
                if (nodes[i].leaf)
                    nodes[i].leaf->Release();
                RowRef row;
                if (Locate(S.entities[i], row))
                    RememberRow(nodes[i].node, row);
                m_NodeOfIndex[GetEntityIndex(S.entities[i])] = NodeOfEntry{ S.entities[i], nodes[i].node };
                nodes[i].node->Release();
            }
            OrphanRecord::Destroy(S.records.data(), S.records.size());
            first = end;
        }

        if (parentEntity != Entity::Null)
            LinkUnder(parentEntity, S.entities[0]);
        SyncArchetypes();
        WarnObservers("Attach", observers);
        return true;
    }

    // A refused CreateEntities call (out of memory): the batch from `first` lost the references it gave the world,
    // and holds the ones taken before the call; the nodes after it never gave theirs. Every node from `first` on goes
    // back to its record (links to the others are lost) with its leaf, its reserved id is released, and the reference
    // that would have gone to the world is released: a node nothing else holds goes, as with a Detach nobody keeps.
    void SceneGraphStore::AbandonNodes(uint32_t first)
    {
        Scratch& S = *m_S;
        std::vector<NewNode>& nodes = S.newNodes;
        const uint32_t n = uint32_t(nodes.size());
        if (m_AnyLeafRecord)
        {
            for (uint32_t i = first; i < n; ++i)
            {
                RowRef row;   // of a batch that the world made in part: its leaves' values are in the rows
                if (nodes[i].leaf && Locate(S.entities[i], row))
                    BeginLeavingRow(nodes[i].leaf, nodes[i].node, S.entities[i]);
            }
        }
        if (NVRHI_FAILED(DestroyRows(S.entities.data() + first, n - first, "Attach")))
            log::warning("SceneGraph: cannot release the reserved ids of a failed attach");

        // The created nodes' links to abandoned ones go: in pre-order, a node's created children are a prefix of its
        // children, so its last child becomes the last created one.
        for (uint32_t i = 0; i < first; ++i)
        {
            SceneLinks* links = Get<SceneLinks>(S.entities[i], SlotLinks);
            if (!links)
                continue;
            uint32_t lastCreated = c_None;
            for (uint32_t child = S.firstChild[i]; child != c_None && child < first; child = S.nextSibling[child])
                lastCreated = child;
            if (lastCreated == c_None)
            {
                links->firstChild = Entity::Null;
                links->lastChild = Entity::Null;
            }
            else
                links->lastChild = S.entities[lastCreated];
            if (i != 0 && S.nextSibling[i] != c_None && S.nextSibling[i] >= first)
                links->nextSibling = Entity::Null;
        }
        for (uint32_t i = first; i < n; ++i)
        {
            NewNode& entry = nodes[i];
            SceneGraphNode* node = entry.node;
            node->m_Store = nullptr;
            node->m_Entity = Entity::Null;
            ForgetRow(node);
            OrphanRecord* record = entry.record ? entry.record : OrphanRecord::Create();
            record->parent = nullptr;
            record->prevSibling = nullptr;
            record->lastChild = nullptr;
            record->leaf.Attach(entry.leaf);   // the batch's held reference, or the one that never left
            node->m_Record = record;
            node->Release();
        }
    }

    // Appends `child`, whose SceneLinks already name the parent's old last child as its previous sibling.
    void SceneGraphStore::LinkUnder(Entity parentEntity, Entity child)
    {
        RowRef parentRow;
        if (!Locate(parentEntity, parentRow))
            return;
        SceneLinks* parentLinks = static_cast<SceneLinks*>(Column(parentRow, SlotLinks));
        if (parentLinks->lastChild != Entity::Null)
        {
            RowRef lastRow;
            if (Locate(parentLinks->lastChild, lastRow))
            {
                static_cast<SceneLinks*>(Column(lastRow, SlotLinks))->nextSibling = child;
                Stamp(lastRow, SlotLinks);
            }
        }
        else
            parentLinks->firstChild = child;
        parentLinks->lastChild = child;
        Stamp(parentRow, SlotLinks);
    }

    // ------------------------------------------------------------------------------- detach (5.3)

    // The subgraph of `root` in pre-order: m_S->entities, m_S->parents (indices) and m_S->nodes.
    void SceneGraphStore::GatherSubgraph(Entity root)
    {
        m_S->entities.clear();
        m_S->parents.clear();
        m_S->nodes.clear();
        uint32_t parentIndex = c_None;
        Entity entity = root;
        while (entity != Entity::Null)
        {
            const uint32_t index = uint32_t(m_S->entities.size());
            m_S->entities.push_back(entity);
            m_S->parents.push_back(parentIndex);
            m_S->nodes.push_back(NodeOf(entity));
            const SceneLinks* links = Get<SceneLinks>(entity, SlotLinks);
            if (links && links->firstChild != Entity::Null)
            {
                parentIndex = index;
                entity = links->firstChild;
                continue;
            }
            for (;;)
            {
                if (entity == root)
                {
                    entity = Entity::Null;
                    break;
                }
                const SceneLinks* current = Get<SceneLinks>(entity, SlotLinks);
                if (current && current->nextSibling != Entity::Null)
                {
                    entity = current->nextSibling;
                    break;
                }
                entity = m_S->entities[parentIndex];
                parentIndex = m_S->parents[parentIndex];
            }
        }
    }

    // Moves the gathered subgraph's rows into records, in pre-order: the children links of the records take references,
    // the root's caller keeps its own. With `unlinkRoot`, the root also leaves its parent's child list.
    void SceneGraphStore::MoveToRecords(bool unlinkRoot)
    {
        const uint32_t n = uint32_t(m_S->entities.size());
        if (unlinkRoot)
        {
            RowRef rootRow;
            if (Locate(m_S->entities[0], rootRow))
            {
                const SceneLinks rootLinks = *static_cast<const SceneLinks*>(Column(rootRow, SlotLinks));
                const Parent* parent = static_cast<const Parent*>(Column(rootRow, SlotParent));
                RowRef parentRow;
                if (parent && Locate(parent->entity, parentRow))
                {
                    SceneLinks* parentLinks = static_cast<SceneLinks*>(Column(parentRow, SlotLinks));
                    RowRef siblingRow;
                    if (rootLinks.prevSibling != Entity::Null && Locate(rootLinks.prevSibling, siblingRow))
                    {
                        static_cast<SceneLinks*>(Column(siblingRow, SlotLinks))->nextSibling = rootLinks.nextSibling;
                        Stamp(siblingRow, SlotLinks);
                    }
                    else
                        parentLinks->firstChild = rootLinks.nextSibling;
                    if (rootLinks.nextSibling != Entity::Null && Locate(rootLinks.nextSibling, siblingRow))
                    {
                        static_cast<SceneLinks*>(Column(siblingRow, SlotLinks))->prevSibling = rootLinks.prevSibling;
                        Stamp(siblingRow, SlotLinks);
                    }
                    else
                        parentLinks->lastChild = rootLinks.prevSibling;
                    Stamp(parentRow, SlotLinks);
                }
            }
        }

        ParentWorld parentWorld;
        for (uint32_t i = 0; i < n; ++i)
        {
            SceneGraphNode* node = m_S->nodes[i];
            OrphanRecord* record = OrphanRecord::Create();
            RowRef row;
            if (Locate(m_S->entities[i], row))
            {
                if (const LocalTransform* local = static_cast<const LocalTransform*>(Column(row, SlotLocal)))
                {
                    record->local = *local;
                    record->hasLocal = true;
                }
                else if (IsDriven(row))   // the simulation leaves it: authored from here on, at the simulated pose
                {
                    record->local = SimulatedLocal(row, parentWorld, "Detach");
                    record->hasLocal = true;
                }
                if (const WorldTransform* world = static_cast<const WorldTransform*>(Column(row, SlotWorld)))
                    record->world = *world;
                if (const LeafObject* leaf = static_cast<const LeafObject*>(Column(row, SlotLeaf)))
                {
                    record->leaf = leaf->object;
                    if (m_AnyLeafRecord && leaf->object)
                        BeginLeavingRow(leaf->object, node, m_S->entities[i]);
                }
            }
            if (i > 0)
            {
                SceneGraphNode* parent = m_S->nodes[m_S->parents[i]];
                OrphanRecord& parentRecord = *parent->m_Record;
                record->parent = parent;
                record->prevSibling = parentRecord.lastChild;
                (parentRecord.lastChild ? parentRecord.lastChild->m_Record->nextSibling : parentRecord.firstChild) =
                    node;
                parentRecord.lastChild = node;
            }
            const uint32_t index = GetEntityIndex(m_S->entities[i]);
            if (index < m_NodeOfIndex.size())
                m_NodeOfIndex[index] = NodeOfEntry{};
            node->m_Record = record;
            node->m_Store = nullptr;
            node->m_Entity = Entity::Null;
            node->m_CacheEpoch.store(0, std::memory_order_relaxed);
            ForgetRow(node);
        }
    }

    nvrhi::AutoPtr<SceneGraphNode> SceneGraphStore::Detach(SceneGraphNode* node)
    {
        assert(node->m_Store == this);
        nvrhi::AutoPtr<SceneGraphNode> result(node);
        if (!CanChangeStructure("Detach"))
            return result;
        m_World->AdvanceVersion();
        ScratchScope scope(*this);

        GatherSubgraph(node->m_Entity);
        MoveToRecords(true);
        // While a Defer scope is open (E7.8), the nodes that leave the world are kept until its end: the caller may drop
        // the subgraph before the open frame has recorded (their leaves are kept by the leaf observer's OnRemove).
        if (IsDeferring())
        {
            for (SceneGraphNode* gathered : m_S->nodes)
                Retire(gathered);
        }

        // The leaf observer unregisters the leaves in pre-order; the graph removes them from its vectors in one pass.
        m_Graph->m_DeferLeafRemoval = true;
        if (NVRHI_FAILED(DestroyRows(m_S->entities.data(), m_S->entities.size(), "Detach")))
            log::warning("SceneGraph: cannot destroy the entities of a detached subgraph");
        m_Graph->m_DeferLeafRemoval = false;
        m_Graph->CompactLeafVectors();

        SyncArchetypes();
        m_PendingStructure = true;
        m_StaleWorldTransforms = true;
        BumpEpoch();
        return result;
    }

    // ------------------------------------------------------------------------------- node writes (5.2)

    void SceneGraphStore::SetLeaf(SceneGraphNode* node, SceneGraphLeaf* leaf)
    {
        if (!CanChangeStructure("SetLeaf"))
            return;
        m_World->AdvanceVersion();
        const Entity entity = node->m_Entity;
        RowRef row;
        if (!Locate(entity, row))
            return;
        const LeafObject* current = static_cast<const LeafObject*>(Column(row, SlotLeaf));
        nvrhi::AutoPtr<SceneGraphLeaf> old = current ? current->object : nullptr;
        const uint32_t oldType = old ? ResolveLeafType(old) : c_None;
        const uint32_t newType = leaf ? ResolveLeafType(leaf) : c_None;

        // The old leaf leaves first (its OnRemove unregisters it), then the new one comes (OnAdd registers it), as the
        // old SetLeaf unregistered and registered. Each is one move of the row, with the class's tags and columns, also
        // between leaves of one class: an observer on a class tag sees the complete row of every leaf that comes or goes
        // (E6.3). Sparse and Shared components cannot take part: they come before the row and go after it.
        // The lists first (they may allocate), and whether the world can make both rows: until then nothing has changed.
        ScratchScope scope(*this);
        std::vector<WorldImpl::NewValue>& added = m_S->newValues;
        std::vector<ComponentId>& separate = m_S->separateIds;
        WorldImpl::ComponentChange leaving, coming;
        added.clear();
        separate.clear();
        LocalBounds bounds{ dm::box3::empty() };
        const WorldBounds empty{ dm::box3::empty() };
        LeafObject value;
        if (old)
        {
            const LeafType& type = m_LeafTypes[oldType];
            leaving.removed = m_LeafTypeTags.data() + type.firstTag - 1;
            leaving.removedCount = 1 + type.tagCount + type.columnCount + (leaf ? 0 : uint32_t(std::size(c_BoundsSlots)));
            if (type.separate)
            {
                std::vector<ComponentId>& removed = m_S->runComponents;
                removed.clear();
                for (uint32_t i = 0; i < leaving.removedCount; ++i)
                {
                    if (!IsSeparate(leaving.removed[i]))
                        removed.push_back(leaving.removed[i]);
                }
                leaving.removed = removed.data();
                leaving.removedCount = uint32_t(removed.size());
            }
        }
        if (leaf)
        {
            auto add = [&](ComponentId id, const void* initial)
            {
                added.push_back(WorldImpl::NewValue{ id, initial, ValueTransfer::Copy });
            };
            if (!old)
            {
                add(m_Ids[SlotWorldBounds], &empty);
                add(m_Ids[SlotChunkBounds], nullptr);
            }
            const LeafType& type = m_LeafTypes[newType];
            for (uint32_t t = 0; t < type.tagCount + type.columnCount; ++t)
            {
                if (!type.separate || !IsSeparate(m_LeafTypeTags[type.firstTag + t]))
                    add(m_LeafTypeTags[type.firstTag + t], nullptr);   // the leaf's record (OnRowsAdded), or the leaf
            }                                                         // observer, fills a column
            add(m_Ids[SlotLocalBounds], &bounds);
            add(m_Ids[SlotLeaf], &value);
            separate.reserve(type.separate ? type.tagCount + type.columnCount : 0);
            coming.added = added.data();
            coming.addedCount = uint32_t(added.size());
        }
        nvrhi::FRESULT fits = old ? m_Impl->PrepareChange(row.archetypeIndex, leaving) : nvrhi::FS_OK;
        if (leaf && NVRHI_SUCCEEDED(fits))
            fits = m_Impl->PrepareChange(old ? leaving.target : row.archetypeIndex, coming);
        if (NVRHI_FAILED(fits))
        {
            log::warning("SceneGraph::SetLeaf: the world cannot make the node's new row (%d); nothing changed",
                         int(fits));
            return;
        }

        nvrhi::FRESULT observers = nvrhi::FS_OK;
        if (old)
        {
            if (m_AnyLeafRecord)
                BeginLeavingRow(old, node, entity);
            old->m_Node.Reset();
            const WorldImpl::RowHooks hooks = LeavingHooks();
            const nvrhi::FRESULT result = m_Impl->ChangeComponentsInternal(entity, leaving, m_Impl->GetStampVersion(),
                                                                           m_AnyLeafRecord ? &hooks : nullptr);
            if (NVRHI_FAILED(result))
            {
                log::warning("SceneGraph::SetLeaf: the leaf cannot leave its row (%d); nothing changed", int(result));
                if (m_AnyLeafRecord && old->m_LeafRecord && old->m_LeafRecord->IsInRow())
                {
                    LeafRecord::Destroy(old->m_LeafRecord);   // the row keeps the values
                    old->m_LeafRecord = nullptr;
                }
                old->m_Node = nvrhi::WeakPtr<SceneGraphNode>(node);
                return;
            }
            const LeafType type = m_LeafTypes[oldType];
            for (uint32_t t = 0; type.separate && t < type.tagCount + type.columnCount; ++t)
            {
                if (IsSeparate(m_LeafTypeTags[type.firstTag + t]))
                    m_World->RemoveComponent(entity, m_LeafTypeTags[type.firstTag + t]);
            }
            FlushObservers(observers);
        }
        if (leaf)
        {
            // When the leaf does not come after all (out of memory for the second move, or its box throws): the node has
            // no leaf, and nothing of either stays in its row.
            auto abandon = [&]() noexcept
            {
                leaf->m_Node.Reset();
                for (ComponentId id : separate)
                    m_World->RemoveComponent(entity, id);
                if (!old)
                    return;
                const LeafType& type = m_LeafTypes[oldType];
                WorldImpl::ComponentChange rest;
                rest.removed = m_LeafTypeTags.data() + type.firstTag + type.tagCount + type.columnCount;
                rest.removedCount = uint32_t(std::size(c_BoundsSlots));
                if (NVRHI_FAILED(m_Impl->ChangeComponentsInternal(entity, rest, m_Impl->GetStampVersion(), nullptr)))
                {
                    m_World->AddComponent(entity, m_Ids[SlotLocalBounds], &bounds);   // empty, as no leaf's
                    m_World->AddComponent(entity, m_Ids[SlotWorldBounds], &empty);
                }
                FlushObservers(observers);
            };
            leaf->m_Node = nvrhi::WeakPtr<SceneGraphNode>(node);
            try
            {
                bounds.bounds = leaf->GetLocalBoundingBox();
            }
            catch (...)
            {
                abandon();
                SyncArchetypes();
                ForgetRow(node);
                m_PendingStructure = true;
                m_StaleWorldTransforms = true;
                BumpEpoch();
                throw;
            }
            leaf->AddRef();
            value.object = leaf;   // holds that reference; the row takes a copy
            const LeafType type = m_LeafTypes[newType];
            for (uint32_t t = 0; type.separate && t < type.tagCount + type.columnCount; ++t)
            {
                const ComponentId id = m_LeafTypeTags[type.firstTag + t];
                if (!IsSeparate(id) || m_World->HasComponent(entity, id))
                    continue;
                if (NVRHI_SUCCEEDED(m_World->AddComponent(entity, id, nullptr)))
                    separate.push_back(id);
            }
            NewNode entry{ node, nullptr, leaf, c_None, newType, dm::box3::empty(), false };
            LeafMove move{ this, &entry, &entity };
            const WorldImpl::RowHooks hooks{ &move, &SceneGraphStore::OnRowsAdded, nullptr };
            const nvrhi::FRESULT result = m_Impl->ChangeComponentsInternal(entity, coming, m_Impl->GetStampVersion(),
                                                                           type.desc->leafRecord ? &hooks : nullptr);
            if (NVRHI_FAILED(result))
            {
                log::warning("SceneGraph::SetLeaf: the new leaf cannot come into the row (%d); the node has no leaf",
                             int(result));
                bounds.bounds = dm::box3::empty();
                abandon();
            }
            FlushObservers(observers);
        }
        WarnObservers("SetLeaf", observers);

        SyncArchetypes();
        if (Locate(entity, row))
            RememberRow(node, row);   // its row moved to another archetype
        m_PendingStructure = true;
        m_StaleWorldTransforms = true;
        BumpEpoch();
    }

    void SceneGraphStore::SetTransform(SceneGraphNode* node, const dm::double3* translation, const dm::dquat* rotation,
                                       const dm::double3* scaling)
    {
        assert(!m_World->IsInPhase());   // design 5.6: the API writes outside phases
        RowRef row;
        if (!LocateNode(node, row))
            return;
        RememberRow(node, row);
        // the hot path of the API (design 10.1): one column lookup, the write, the stamp
        const uint8_t column = ColumnIndex(*row.archetype, row.archetypeIndex, SlotLocal);
        if (column != c_Absent)
        {
#ifndef NDEBUG
            // E7.4 remedy 3: a Static row does not move (refused in Debug builds; Release builds write, and propagation
            // does not take it). Only where propagation leaves the row out (backlog I5-R2): not in the deep bucket,
            // which ignores the tag, nor before the run that takes the row, whose write is its initial value. A row
            // without a LocalTransform is not refused: its first transform adds the column, a structural change that
            // the next propagation takes as a row that joined, in both builds.
            if (IsSettledStaticRow(row))
            {
                log::warning("SceneGraph: SetTransform of node '%s' is refused: its row is Static (SceneComponents.h)",
                             node->GetName().c_str());
                return;
            }
#endif
            const ArchetypeColumn& c = row.archetype->columns[column];
            uint8_t* memory = row.archetype->chunks[row.chunk].memory;
            LocalTransform* local = reinterpret_cast<LocalTransform*>(memory + c.offset + size_t(row.row) * c.stride);
            if (scaling) local->scaling = *scaling;
            if (rotation) local->rotation = *rotation;
            if (translation) local->translation = *translation;
            StampRowWrite(*row.archetype, row.chunk, column, row.row, m_Impl->GetStampVersion());
        }
        else
        {
            if (IsDriven(row))
            {
                RequestTeleport(node, translation, rotation, scaling);   // no edit: the pose is the domain's
                return;
            }
            // the first transform of a node adds the component (a structural change, outside phases)
            if (!CanChangeStructure("SetTransform"))
                return;
            m_World->AdvanceVersion();
            static const OrphanRecord s_Defaults;
            LocalTransform value = s_Defaults.local;
            if (scaling) value.scaling = *scaling;
            if (rotation) value.rotation = *rotation;
            if (translation) value.translation = *translation;
            m_World->AddComponent(node->m_Entity, m_Ids[SlotLocal], &value);
            m_World->AddComponent(node->m_Entity, m_Ids[SlotLocalMatrix], nullptr);
            SyncArchetypes();
            if (Locate(node->m_Entity, row))
                RememberRow(node, row);   // its row moved to another archetype
        }
        m_PendingTransform = true;
        m_StaleWorldTransforms = true;
    }

    void SceneGraphStore::InvalidateContent(SceneGraphNode* node)
    {
        RowRef row;
        if (Locate(node->m_Entity, row))
        {
            const LeafObject* leaf = static_cast<const LeafObject*>(Column(row, SlotLeaf));
            LocalBounds* bounds = static_cast<LocalBounds*>(Column(row, SlotLocalBounds));
            if (leaf && leaf->object && bounds)
            {
                m_World->AdvanceVersion();   // the stamps below are newer than what GpuScene read
                bounds->bounds = leaf->object->GetLocalBoundingBox();
                Stamp(row, SlotLocalBounds);
                Stamp(row, SlotLeaf);   // consumers that want to know see the leaf column changed (5.4)
                // a mesh instance's material domains may have changed with its content
                if (RenderFlags* flags = static_cast<RenderFlags*>(Column(row, SlotRenderFlags)))
                {
                    *flags = MakeRenderFlags(static_cast<MeshInstance*>(leaf->object));
                    Stamp(row, SlotRenderFlags);
                }
                m_PendingContent = true;   // the next Refresh runs Bounds for it
                ++m_ContentChanges;        // E7.4 remedy 3: Static archetypes' bounds are read again
                BumpEpoch();
            }
        }
        // a node without a leaf has no content: nothing changed
    }

    // ------------------------------------------------------------------------- mesh instances (step E4)

    const GpuInstanceSlot* SceneGraphStore::InstanceSlotOf(const SceneGraphLeaf* leaf) noexcept
    {
        const SceneGraphNode* node = leaf->GetNode();
        if (!node || !node->m_Store)
            return nullptr;
        const GpuInstanceSlot* slot = node->m_Store->GetOf<GpuInstanceSlot>(node, SlotInstance);
        return slot && slot->instance != c_NoInstanceSlot ? slot : nullptr;
    }

    void SceneGraphStore::OnLeaf(Entity entity, SceneGraphLeaf* leaf, bool added)
    {
        // The asset tables and slots change first, and SceneGraph's callbacks run after them (they may throw), then
        // the virtual RegisterLeaf or UnregisterLeaf: as RegisterLeaf raised OnMeshAdded before it added the leaf to
        // its vectors.
        if (!added)
            Retire(leaf);   // while a Defer scope is open (E7.8): the open frame may name the leaf
        {
            ChangesScope scope(*this);
            if (!scope.Get())
                throw std::bad_alloc();   // the leaf observer reports it
            MeshTable::Changes& changes = *scope.Get();
            RowRef row;
            if (Locate(entity, row) && ColumnIndex(*row.archetype, row.archetypeIndex, SlotMeshRef) != c_Absent)
            {
                // Only MeshInstance classes declare these columns (MeshInstance::GetTypeDesc, inherited by subclasses).
                assert(query_cast<MeshInstance>(leaf));
                if (added)
                    AcquireInstance(row, static_cast<MeshInstance*>(leaf), changes);
                else
                    ReleaseInstance(row, changes);
            }
            RaiseAssetCallbacks(changes);
        }
        if (added)
            m_Graph->RegisterLeaf(leaf);
        else
            m_Graph->UnregisterLeaf(leaf);
    }

    std::atomic<int32_t> SceneGraphStore::s_FailAcquireCountdown{ -1 };

    void SceneGraphStore::AcquireInstance(const RowRef& row, MeshInstance* instance, MeshTable::Changes& changes)
    {
        // Until every slot is allocated, the row holds no slot (c_NoInstanceSlot): an exception from the table (out of
        // memory) leaves nothing for ReleaseInstance to free.
        // The same for the mesh reference: a row copied through the world (Instantiate) arrives with its source's
        // MeshId, a reference it never took.
        GpuInstanceSlot* slot = static_cast<GpuInstanceSlot*>(Column(row, SlotInstance));
        if (slot)
            *slot = GpuInstanceSlot{};
        MeshRef* ref = static_cast<MeshRef*>(Column(row, SlotMeshRef));
        ref->mesh = MeshId::Invalid;
        if (s_FailAcquireCountdown.load(std::memory_order_relaxed) >= 0 &&
            s_FailAcquireCountdown.fetch_sub(1, std::memory_order_relaxed) == 0)
            throw std::bad_alloc();   // test seam (FailAcquireAfter)

        SceneGraph& graph = *m_Graph;
        const MeshId mesh = graph.m_Meshes.Acquire(instance->GetMesh(), graph.m_Materials, changes);
        graph.m_MaxGeometryCountPerMesh = std::max(graph.m_MaxGeometryCountPerMesh, changes.geometriesAdded);
        const uint32_t geometryCount =
            mesh != MeshId::Invalid ? graph.m_Meshes.GetMeshRow(MeshTable::SlotOf(mesh)).geometryCount : 0;

        // Written without a stamp: the row was just created or moved here, so its chunk's added version is newer than
        // anything a reader saw.
        ref->mesh = mesh;
        if (RenderFlags* flags = static_cast<RenderFlags*>(Column(row, SlotRenderFlags)))
            *flags = MakeRenderFlags(instance);
        if (slot)
        {
            const uint32_t first = m_GeometryInstanceSlots.Allocate(geometryCount);
            slot->instance = m_InstanceSlots.Allocate(1);
            slot->firstGeometryInstance = first;
            slot->geometryCount = geometryCount;
        }
    }

    void SceneGraphStore::ReleaseInstance(const RowRef& row, MeshTable::Changes& changes)
    {
        GpuInstanceSlot* slot = static_cast<GpuInstanceSlot*>(Column(row, SlotInstance));
        if (slot && slot->instance != c_NoInstanceSlot)
        {
            m_InstanceSlots.Free(slot->instance, 1);
            m_GeometryInstanceSlots.Free(slot->firstGeometryInstance, slot->geometryCount);
        }
        if (slot)
            *slot = GpuInstanceSlot{};   // the getters report -1 from here on (UnregisterLeaf overrides see that)
        MeshRef* ref = static_cast<MeshRef*>(Column(row, SlotMeshRef));
        const MeshId mesh = ref->mesh;
        ref->mesh = MeshId::Invalid;
        m_Graph->m_Meshes.Release(mesh, m_Graph->m_Materials, changes);
    }

    void SceneGraphStore::RaiseAssetCallbacks(const MeshTable::Changes& changes)
    {
        const SceneGraph& graph = *m_Graph;
        for (const MeshTable::Change& change : changes.events)
        {
            if (change.mesh)
            {
                const auto& callback = change.added ? graph.OnMeshAdded : graph.OnMeshRemoved;
                if (callback)
                    callback(change.mesh);
            }
            else if (change.material)
            {
                const auto& callback = change.added ? graph.OnMaterialAdded : graph.OnMaterialRemoved;
                if (callback)
                    callback(change.material);
            }
        }
    }

    // ----------------------------------------------------------- runtime-only mesh instances (5.8; E7.1)

    namespace
    {
        // A callback of the asset tables may throw; the runtime calls are noexcept, as the leaf observer is.
        template <typename Raise>
        void RaiseQuietly(const char* operation, Raise&& raise) noexcept
        {
            try
            {
                raise();
            }
            catch (const std::exception& error)
            {
                log::warning("SceneGraph: %s: an asset callback threw '%s'", operation, error.what());
            }
            catch (...)
            {
                log::warning("SceneGraph: %s: an asset callback threw", operation);
            }
        }
    }

    SceneGraphStore::ChangesScope::ChangesScope(SceneGraphStore& store) noexcept
        : m_Store(store)
    {
        const uint32_t depth = store.m_ChangesDepth++;
        if (depth >= store.m_ChangesLevels.size() || !store.m_ChangesLevels[depth])
        {
            try
            {
                if (depth >= store.m_ChangesLevels.size())
                    store.m_ChangesLevels.resize(size_t(depth) + 1);
                store.m_ChangesLevels[depth] = std::make_unique<MeshTable::Changes>();
            }
            catch (...)
            {
                return;   // Get() is null
            }
        }
        m_Changes = store.m_ChangesLevels[depth].get();
        m_Changes->Clear();
    }

    SceneGraphStore::ChangesScope::~ChangesScope()
    {
        if (m_Changes)
        {
            // While a Defer scope is open (E7.8) the removed assets go to the retire list: the open frame may name them.
            if (m_Store.IsDeferring())
            {
                for (const MeshTable::Change& change : m_Changes->events)
                {
                    if (change.added)
                        continue;
                    if (change.mesh)
                        m_Store.Retire(change.mesh.Get());
                    else if (change.material)
                        m_Store.Retire(change.material.Get());
                }
            }
            m_Changes->Clear();   // the events hold their objects: released here, after the callbacks
        }
        --m_Store.m_ChangesDepth;
    }

    MeshId SceneGraphStore::AcquireRuntimeMesh(MeshInfo* mesh) noexcept
    {
        if (!mesh)
        {
            log::warning("SceneGraph::AcquireRuntimeMesh: the mesh is null");
            return MeshId::Invalid;
        }
        if (!CanChangeStructure("AcquireRuntimeMesh"))
            return MeshId::Invalid;
        SceneGraph& graph = *m_Graph;
        MeshId id = MeshId::Invalid;
        try
        {
            // The holds first: a new row takes the lowest free slot, at most the end, so nothing can fail after the
            // reference is taken.
            if (m_RuntimeMeshHolds.size() <= graph.m_Meshes.GetSlotEnd())
                m_RuntimeMeshHolds.resize(size_t(graph.m_Meshes.GetSlotEnd()) + 1, 0);
        }
        catch (...)
        {
            log::warning("SceneGraph::AcquireRuntimeMesh: out of memory; no reference taken");
            return MeshId::Invalid;
        }
        ChangesScope scope(*this);
        MeshTable::Changes* changes = scope.Get();
        try
        {
            if (!changes)
                throw std::bad_alloc();
            id = graph.m_Meshes.Acquire(mesh, graph.m_Materials, *changes);
        }
        catch (...)
        {
            log::warning("SceneGraph::AcquireRuntimeMesh: out of memory; no reference taken");
            return MeshId::Invalid;
        }
        ++m_RuntimeMeshHolds[MeshTable::SlotOf(id)];
        graph.m_MaxGeometryCountPerMesh = std::max(graph.m_MaxGeometryCountPerMesh, changes->geometriesAdded);
        RaiseQuietly("AcquireRuntimeMesh", [&]() { RaiseAssetCallbacks(*changes); });
        return id;
    }

    void SceneGraphStore::ReleaseRuntimeMesh(MeshId mesh) noexcept
    {
        if (!CanChangeStructure("ReleaseRuntimeMesh"))
            return;
        SceneGraph& graph = *m_Graph;
        const uint32_t slot = MeshTable::SlotOf(mesh);
        if (!graph.m_Meshes.Get(mesh) || slot >= m_RuntimeMeshHolds.size() || m_RuntimeMeshHolds[slot] == 0)
        {
            log::warning("SceneGraph::ReleaseRuntimeMesh: the caller holds no reference to this mesh id; nothing "
                         "released");
            return;
        }
        ChangesScope scope(*this);
        if (!scope.Get())
        {
            log::warning("SceneGraph::ReleaseRuntimeMesh: out of memory; nothing released");
            return;
        }
        --m_RuntimeMeshHolds[slot];
        RaiseQuietly("ReleaseRuntimeMesh", [&]()
        {
            graph.m_Meshes.Release(mesh, graph.m_Materials, *scope.Get());
            RaiseAssetCallbacks(*scope.Get());
        });
    }

    uint32_t SceneGraphStore::GetRuntimeMeshHolds(MeshId mesh) const noexcept
    {
        const uint32_t slot = MeshTable::SlotOf(mesh);
        return m_Graph->m_Meshes.Get(mesh) && slot < m_RuntimeMeshHolds.size() ? m_RuntimeMeshHolds[slot] : 0;
    }

    uint32_t SceneGraphStore::GetRuntimeMeshInstanceComponents(ComponentId* out, uint32_t capacity,
                                                               bool simulationDriven) const noexcept
    {
        // The order of SceneComponents.h's row layout.
        const ComponentId ids[SceneGraph::c_MaxRuntimeMeshInstanceComponents] = {
            m_RuntimeTag, m_Ids[SlotLocal], m_Ids[SlotWorld], m_Ids[SlotLocalBounds], m_Ids[SlotWorldBounds],
            m_Ids[SlotChunkBounds], m_Ids[SlotMeshRef], m_Ids[SlotRenderFlags], m_Ids[SlotInstance],
            m_Ids[SlotDriven], m_Ids[SlotSimPose], m_SimPreviousId, m_Ids[SlotSimScale], m_SimBodyId,
        };
        const uint32_t count = simulationDriven ? SceneGraph::c_MaxRuntimeMeshInstanceComponents : 9;
        if (out)
        {
            for (uint32_t i = 0; i < std::min(count, capacity); ++i)
                out[i] = ids[i];
        }
        return count;
    }

    // Per batch (a run of rows of one chunk, in delivery order) of rows that became members: references in bulk per run
    // of rows that name one mesh, then each row's slots, geometry-instance range first, as AcquireInstance allocates
    // them. A row holds a reference exactly while it holds an instance slot; one that names no mesh of the table
    // (Invalid, released, another graph's that is not one of this table) gets none, an empty LocalBounds, and is never
    // drawn. Copies (Instantiate, another world's rows) arrive with their source's slot values, so every row starts
    // without one. Of rows that stop being members: the slots and references of those that hold them. MeshRef is read
    // at both: a row's MeshRef must not change in between (the inspector's read-only list has it).
    void SceneGraphStore::OnRuntimeRows(const EntityBatch& batch, bool added) noexcept
    {
        GpuInstanceSlot* slots = static_cast<GpuInstanceSlot*>(batch.columns[0]);
        const MeshRef* refs = static_cast<const MeshRef*>(batch.columns[1]);
        SceneGraph& graph = *m_Graph;
        MeshTable& meshes = graph.m_Meshes;
        const uint32_t count = batch.count;
        if (added)
        {
            for (uint32_t r = 0; r < count; ++r)
                slots[r] = GpuInstanceSlot{};
            RenderFlags* flags = static_cast<RenderFlags*>(batch.columns[2]);
            LocalBounds* bounds = static_cast<LocalBounds*>(batch.columns[3]);
            if (batch.columns[4])
                log::warning("SceneGraph: %u runtime-only mesh instances have a Parent, which is not supported; they "
                             "are roots", count);
            uint32_t unknown = 0;
            uint32_t overflowed = 0;
            for (uint32_t first = 0; first < count;)
            {
                const MeshId id = refs[first].mesh;
                uint32_t end = first + 1;
                while (end < count && refs[end].mesh == id)
                    ++end;
                const MeshInfo* mesh = meshes.Get(id);
                if (!mesh || !meshes.AddReferences(id, end - first))
                {
                    // no mesh, no box: such rows add nothing to their chunk's bounds
                    for (uint32_t r = first; !mesh && bounds && r < end; ++r)
                        bounds[r].bounds = dm::box3::empty();
                    (mesh ? overflowed : unknown) += end - first;
                    first = end;
                    continue;
                }
                const uint32_t geometryCount = meshes.GetMeshRow(MeshTable::SlotOf(id)).geometryCount;
                const RenderFlags meshFlags = MakeRenderFlags(mesh, GetMeshContentFlags(mesh));
                for (uint32_t r = first; r < end; ++r)
                {
                    // Allocate takes no memory: nothing below can fail
                    slots[r].firstGeometryInstance = m_GeometryInstanceSlots.Allocate(geometryCount);
                    slots[r].instance = m_InstanceSlots.Allocate(1);
                    slots[r].geometryCount = geometryCount;
                    if (flags)
                        flags[r] = meshFlags;
                    if (bounds)
                        bounds[r].bounds = mesh->objectSpaceBounds;
                }
                m_RuntimeRows += end - first;
                first = end;
            }
            if (unknown)
                log::warning("SceneGraph: %u runtime-only mesh instances name no mesh of the graph (not from "
                             "AcquireRuntimeMesh, released, or of another graph); they get no slots and are not drawn",
                             unknown);
            if (overflowed)
                log::warning("SceneGraph: %u runtime-only mesh instances would overflow their mesh's reference count; "
                             "they get no slots and are not drawn", overflowed);
            return;
        }

        ChangesScope scope(*this);
        if (!scope.Get())
        {
            log::warning("SceneGraph: out of memory; the slots and references of %u runtime-only mesh instances are "
                         "not released", count);
            return;
        }
        MeshTable::Changes& changes = *scope.Get();
        RaiseQuietly("runtime-only mesh instances removed", [&]()
        {
            // Runs of held rows with one mesh: one ReleaseReferences each.
            MeshId run = MeshId::Invalid;
            uint32_t runCount = 0;
            for (uint32_t r = 0; r < count; ++r)
            {
                if (slots[r].instance == c_NoInstanceSlot)
                    continue;
                if (runCount != 0 && refs[r].mesh != run)
                {
                    meshes.ReleaseReferences(run, runCount, graph.m_Materials, changes);
                    runCount = 0;
                }
                m_InstanceSlots.Free(slots[r].instance, 1);
                m_GeometryInstanceSlots.Free(slots[r].firstGeometryInstance, slots[r].geometryCount);
                slots[r] = GpuInstanceSlot{};
                --m_RuntimeRows;
                run = refs[r].mesh;
                ++runCount;
            }
            if (runCount != 0)
                meshes.ReleaseReferences(run, runCount, graph.m_Materials, changes);
            RaiseAssetCallbacks(changes);
        });
    }

    // Rows that got the tag: those without MeshRef are reported (they never become members). Those without
    // GpuInstanceSlot (a snapshot's: it is NoSerialize; or made without it) get the column through the observer's
    // commands, which play back before the structural call returns; the column makes them members, and OnRuntimeRows
    // gives them their slots then. A row destroyed before that playback never became a member and holds nothing; the
    // command finds it dead and fails (the structural call or the scheduler's run still succeeds; a world with
    // validation logs a warning per row): the command buffer has no form that skips dead entities.
    void SceneGraphStore::AddRuntimeSlotColumn(const EntityBatch& batch, ICommandBuffer* commands) noexcept
    {
        if (!batch.columns[2])
            log::warning("SceneGraph: %u runtime-only mesh instances lack MeshRef "
                         "(SceneGraph::GetRuntimeMeshInstanceComponents); they get no slots and are not drawn",
                         batch.count);
        if (!batch.columns[1] && commands)
            commands->AddComponents(batch.entities, batch.count, m_Ids[SlotInstance], nullptr, 0);
    }

    void SceneGraphStore::ReconcileMeshes()
    {
        SceneGraph& graph = *m_Graph;
        MeshTable& meshes = graph.m_Meshes;
        ChangesScope scope(*this);   // a callback may change the tables again (AcquireRuntimeMesh): a level of its own
        if (!scope.Get())
            throw std::bad_alloc();
        MeshTable::Changes& changes = *scope.Get();
        std::vector<uint8_t> reranged;   // per mesh slot; empty while none was
        const uint32_t slotEnd = meshes.GetSlotEnd();
        for (uint32_t slot = 0; slot < slotEnd; ++slot)
        {
            if (meshes.GetRow(slot).object && meshes.Reconcile(slot, graph.m_Materials, changes))
            {
                reranged.resize(slotEnd, 0);
                reranged[slot] = 1;
            }
        }
        // A material shared with another graph gets this graph's slot written back (Reconcile does it for meshes and
        // geometries): the last graph refreshed wins, as with the old renumbering. GpuScene does not read the objects'
        // slots; other consumers (MaterialConstants::materialID of shaders without bindless) do.
        graph.m_Materials.WriteSlots();
        if (!reranged.empty())
        {
            // The instances of a mesh with a new geometry count get geometry-instance ranges of that count; their
            // chunks are stamped, so that GpuScene writes them again (as it writes every row after a layout change).
            m_World->AdvanceVersion();
            ++m_LayoutSerial;
            SyncArchetypes();
            const uint32_t archetypeCount = m_Impl->GetArchetypeStorageCount();
            for (uint32_t a = 0; a < archetypeCount; ++a)
            {
                const Archetype& archetype = m_Impl->GetArchetypeStorage(a);
                const uint8_t refColumn = ColumnIndex(archetype, a, SlotMeshRef);
                const uint8_t slotColumn = ColumnIndex(archetype, a, SlotInstance);
                if (refColumn == c_Absent || slotColumn == c_Absent)
                    continue;
                for (uint32_t c = 0; c < archetype.GetChunkCount(); ++c)
                {
                    const MeshRef* refs = reinterpret_cast<const MeshRef*>(ColumnBase(archetype, c, refColumn));
                    GpuInstanceSlot* slots = reinterpret_cast<GpuInstanceSlot*>(ColumnBase(archetype, c, slotColumn));
                    bool wrote = false;
                    for (uint32_t r = 0; r < archetype.chunks[c].count; ++r)
                    {
                        const uint32_t mesh = MeshTable::SlotOf(refs[r].mesh);
                        if (refs[r].mesh == MeshId::Invalid || mesh >= reranged.size() || !reranged[mesh] ||
                            slots[r].instance == c_NoInstanceSlot)
                            continue;
                        const uint32_t count = meshes.GetMeshRow(mesh).geometryCount;
                        m_GeometryInstanceSlots.Free(slots[r].firstGeometryInstance, slots[r].geometryCount);
                        slots[r].firstGeometryInstance = m_GeometryInstanceSlots.Allocate(count);
                        slots[r].geometryCount = count;
                        wrote = true;
                    }
                    if (wrote)
                        StampVersion(archetype.Versions(c) + slotColumn, m_Impl->GetStampVersion());
                }
            }
        }
        // Callbacks after the tables are consistent; an exception from one is reported, as the leaf observer does.
        try
        {
            RaiseAssetCallbacks(changes);
        }
        catch (const std::exception& error)
        {
            log::warning("SceneGraph::Refresh: an asset callback threw '%s'", error.what());
        }
        catch (...)
        {
            log::warning("SceneGraph::Refresh: an asset callback threw");
        }
    }

    // ------------------------------------------------------------------------------------- refresh

    bool SceneGraphStore::Refresh(uint32_t frameIndex)
    {
        ++m_RefreshCalls;
        SyncArchetypes();
        const bool structure = m_PendingStructure;
        const bool moved = m_PendingTransform;
        // Nothing changed through the API (structure, transforms, content), no structural change and no phase ran on
        // the world since: nothing to do. Writes into the world outside phases are not seen here (SceneGraph.h).
        const bool unchanged = !structure && !moved && !m_PendingContent &&
                               m_World->GetStructuralVersion() == m_RefreshedStructuralVersion &&
                               m_World->GetVersion() == m_RefreshedVersion;
        // Leaves with dynamic bounds change their boxes by themselves: then only the Bounds phase runs (and the
        // incremental subgraph pass for what it changed).
        const bool dynamicOnly = unchanged && HasDynamicBoundsRows();
        if (unchanged && !dynamicOnly)
        {
            m_LastRefreshMoved = false;
            ++m_Counters.skippedRefreshes;
            return false;
        }

        // a Refresh that a system of another's phases calls is refused below, and must not change that one's marks
        const uint32_t outerFrameIndex = m_FrameIndex;
        const bool outerMarks = m_MarkJoints;
        m_FrameIndex = frameIndex;
        PrepareSubgraphs();
        m_MarkJoints = true;
        const nvrhi::FRESULT result = !m_Scheduler ? nvrhi::FE_INVALID_ARGS
            : dynamicOnly ? m_Scheduler->RunPhases(Phase::Bounds, Phase::Bounds)
                          : m_Scheduler->RunPhases(Phase::TransformPropagation, Phase::Bounds);
        m_MarkJoints = outerMarks;
        if (NVRHI_FAILED(result))
        {
            m_FrameIndex = outerFrameIndex;
            m_SubgraphPrepared = false;
            m_SubgraphForceFull = true;
            // the changes stay pending for the next Refresh
            log::warning("SceneGraph::Refresh: the phases did not run (no scheduler, or inside a phase); the changes "
                         "stay pending");
            return false;
        }
        ++m_Counters.refreshes;

        m_PendingStructure = false;
        m_PendingTransform = false;
        m_PendingContent = false;
        m_LastRefreshMoved = moved;
        m_RefreshedStructuralVersion = m_World->GetStructuralVersion();
        m_RefreshedVersion = m_World->GetVersion();
        // The subgraph values of the nodes with children, computed now, in parallel (5.7): the getters read them
        // without a lock. The epoch only moves with structural changes, SetLeaf and InvalidateContent; when the pass
        // cannot run (out of memory), it moves here, and the getters fall back to the on-demand caches.
        RefreshRowHints();
        const uint64_t passes = m_Counters.subgraphPasses;
        UpdateSubgraphs();
        if (m_Counters.subgraphPasses == passes)
        {
            // the pass did not complete: the values it had are stale now (the rows moved), and what it wrote this time
            // is stamped with the epoch that moves below; the next pass is a full one
            m_SubgraphForceFull = true;
            m_SubgraphEpoch.store(~0ull, std::memory_order_release);
        }
        if (!IsSubgraphCurrent())
            BumpEpoch();
        return structure;
    }

    nvrhi::FRESULT SceneGraphStore::Simulate(const FrameTime& time) noexcept
    {
        if (!m_Scheduler || m_TeleportPhase == c_NoPhase || m_InterpolationPhase == c_NoPhase)
        {
            log::warning("SceneGraph::Simulate: the graph has no scheduler with the simulation phases; nothing ran");
            return nvrhi::FE_INVALID_ARGS;
        }
        // Refused before the time advances: SetFrameTime takes it in an observer.
        if (m_World->IsInPhase() || !m_Impl->CanChangeStructure())
        {
            log::warning("SceneGraph::Simulate: inside a phase or a leaf callback; nothing ran");
            return nvrhi::FE_INVALID_ARGS;
        }
        SyncArchetypes();
        if (m_StaleWorldTransforms)
        {
            const nvrhi::FRESULT result = m_Scheduler->RunPhase(Phase::TransformPropagation);
            if (NVRHI_FAILED(result))
                return result;
        }
        m_Scheduler->SetFrameTime(time);
        return m_Scheduler->RunPhases(Phase::BeginFrame, m_InterpolationPhase);
    }

    void SceneGraphStore::UpdateSystemLists() noexcept
    {
        const uint32_t count = m_Impl->GetArchetypeStorageCount();
        // E7.4 remedy 3: an archetype with Static is made after the tag is registered, so the id is known for it here
        const ComponentId staticTag = m_StaticTagHonoured && m_SystemArchetypesSeen < count ? ResolveStaticTag()
                                                                                           : ComponentId::Invalid;
        for (uint32_t a = m_SystemArchetypesSeen; a < count; ++a)
        {
            const Archetype& archetype = m_Impl->GetArchetypeStorage(a);
            auto column = [&](ComponentId id) -> uint8_t
            {
                const uint32_t c = id == ComponentId::Invalid ? c_NoColumn : archetype.FindColumn(id);
                return c == c_NoColumn ? c_Absent : uint8_t(c);
            };
            const uint8_t world = column(m_Ids[SlotWorld]);
            const uint8_t local = column(m_Ids[SlotLocal]);
            const bool driven = column(m_Ids[SlotDriven]) != c_Absent;
            // Runtime-only mesh instances are roots (E7.1, open point 3): a Parent or HierarchyDepth that such a row
            // carries is not read (the observer warns about a Parent). A node's row is never one: the tag on it is
            // ignored here and by the observers, with a warning per archetype.
            const bool tagged = m_RuntimeTag != ComponentId::Invalid && column(m_RuntimeTag) != c_Absent;
            const bool runtime = tagged && column(m_Ids[SlotNode]) == c_Absent;
            if (tagged && !runtime)
                log::warning("SceneGraph: RuntimeMeshInstance on a node's rows is not supported; it is ignored");
            uint8_t level = c_Absent;
            if (world != c_Absent && archetype.components.size() > 0)
            {
                uint32_t depth = 0;
                const uint8_t depthColumn = column(m_DepthId);
                if (depthColumn != c_Absent && !runtime)
                    std::memcpy(&depth, archetype.sharedValues + archetype.columns[depthColumn].offset, sizeof(depth));
                level = uint8_t(std::min(depth, c_MaxHierarchyDepth));
                m_ArchetypeLevelSlot.resize(a + 1, c_None);
                m_ArchetypeLevelSlot[a] = uint32_t(m_Levels[level].size());
                const bool staticRows = staticTag != ComponentId::Invalid && column(staticTag) != c_Absent && !driven;
                LevelArchetype entry{ a, local, column(m_Ids[SlotLocalMatrix]),
                                      runtime ? c_Absent : column(m_Ids[SlotParent]), world, driven, runtime };
                entry.staticRows = staticRows && level != c_MaxHierarchyDepth;
                m_Levels[level].push_back(entry);

                const uint8_t localBounds = column(m_Ids[SlotLocalBounds]);
                const uint8_t worldBounds = column(m_Ids[SlotWorldBounds]);
                const uint8_t chunkBounds = column(m_Ids[SlotChunkBounds]);
                const bool dynamicBounds =
                    m_DynamicBoundsTag != ComponentId::Invalid && column(m_DynamicBoundsTag) != c_Absent;
                if (localBounds != c_Absent && worldBounds != c_Absent && chunkBounds != c_Absent)
                {
                    BoundsArchetype boundsEntry{ a, world, localBounds, worldBounds, chunkBounds,
                                                 dynamicBounds ? column(m_Ids[SlotLeaf]) : c_Absent };
                    // the level entry's rule (not in the deep bucket, which follows every move: backlog I5-R1)
                    boundsEntry.staticRows = entry.staticRows && !dynamicBounds;
                    m_BoundsArchetypes.push_back(boundsEntry);
                }
            }
            m_ArchetypeLevel.push_back(level);

            const uint8_t leaf = column(m_Ids[SlotLeaf]);
            const uint8_t moved = driven ? world : local;
            if (column(m_JointTag) != c_Absent && moved != c_Absent && leaf != c_Absent)
                m_JointArchetypes.push_back(JointArchetype{ a, moved, leaf });

            const DrivenArchetype poses{ a, column(m_Ids[SlotSimPose]), column(m_SimPreviousId),
                                         column(m_Ids[SlotSimScale]), column(m_SimBodyId), world,
                                         column(m_SimTeleportId), runtime ? c_Absent : column(m_Ids[SlotParent]) };
            if (driven && poses.pose != c_Absent && poses.previous != c_Absent && poses.scale != c_Absent &&
                poses.body != c_Absent && poses.world != c_Absent)
                m_DrivenArchetypes.push_back(poses);
        }
        m_SystemArchetypesSeen = count;
    }

    // ------------------------------------------------------------------ Static rows (4.7; E7.4 remedy 3)

    ComponentId SceneGraphStore::ResolveStaticTag() noexcept
    {
        if (m_StaticTag == ComponentId::Invalid)
            m_StaticTag = m_World->FindComponent(ComponentTraits<Static>::Desc().uuid);
        return m_StaticTag;
    }

    const SceneGraphStore::LevelArchetype* SceneGraphStore::LevelOf(uint32_t archetypeIndex) const noexcept
    {
        if (archetypeIndex >= m_ArchetypeLevel.size() || m_ArchetypeLevel[archetypeIndex] == c_Absent)
            return nullptr;
        return &m_Levels[m_ArchetypeLevel[archetypeIndex]][m_ArchetypeLevelSlot[archetypeIndex]];
    }

    // A row that propagation leaves out: its archetype's level entry is Static (honoured, not driven, not in the deep
    // bucket) and the last propagation run took its chunk (its Added version is not newer). Such a row keeps the world
    // that run gave it. A row whose chunk rows joined since, or whose archetype is not classified yet (no propagation
    // ran since it was made), is taken whole by the next run, from its LocalTransform.
    bool SceneGraphStore::IsSettledStaticRow(const RowRef& row) const noexcept
    {
        if (!m_StaticTagHonoured)
            return false;
        const LevelArchetype* level = LevelOf(row.archetypeIndex);
        return level && level->staticRows &&
               row.archetype->Versions(row.chunk)[row.archetype->AddedVersionSlot()] <= m_PropagatedVersion;
    }

    // A Static archetype gets tasks only when rows joined or left it since the run that last took it: then the chunks
    // that rows joined (their Added version), whole, from their parents' worlds as they are. Its LocalTransform and its
    // parents are not followed; in Debug builds a write of its LocalTransform is reported. A run that takes it counts
    // in `takes`, so that Bounds takes it again too (rows that joined after propagation and before Bounds got boxes of
    // worlds that were not propagated yet: backlog I5-R3).
    void SceneGraphStore::QueueStaticLevel(LevelArchetype& level, uint32_t index, const Archetype& archetype,
                                           uint64_t since)
    {
#ifndef NDEBUG
        WarnStaticWrites(level, archetype, since);
#endif
        if (archetype.entityCount == level.settledCount && archetype.rowMoves == level.settledMoves)
        {
            ++m_Counters.staticSkips;
            return;
        }
        level.settledCount = archetype.entityCount;
        level.settledMoves = archetype.rowMoves;
        ++level.takes;
        for (uint32_t c = 0; c < archetype.GetChunkCount(); ++c)
        {
            if (archetype.chunks[c].count != 0 && archetype.Versions(c)[archetype.AddedVersionSlot()] > since)
                m_Tasks.push_back(Task{ index, c, true, nullptr });
        }
    }

    void SceneGraphStore::WarnStaticWrites(const LevelArchetype& level, const Archetype& archetype,
                                           uint64_t since) const
    {
        if (level.local == c_Absent)
            return;
        uint32_t written = 0;
        for (uint32_t c = 0; c < archetype.GetChunkCount(); ++c)
        {
            const uint64_t* versions = archetype.Versions(c);
            if (archetype.chunks[c].count != 0 && versions[level.local] > since &&
                versions[archetype.AddedVersionSlot()] <= since)
                ++written;
        }
        if (written != 0)
            log::warning("SceneGraph: the LocalTransform of Static rows was written in %u chunk(s); refused: Static rows "
                         "do not move (SceneComponents.h)", written);
    }

    // ------------------------------------------------------------------- TransformPropagation (4.6, D5)

    // Whether the parent's chunk was written since the last run (by this run, or a driven one by the interpolation), and
    // its world.
    bool SceneGraphStore::ParentWritten(Entity parent, uint64_t since, const dm::daffine3** outWorld,
                                        bool readVersion) const noexcept
    {
        const EntityRecord* record = m_Impl->FindAlive(parent);
        if (!record)
        {
            *outWorld = nullptr;
            return false;
        }
        const Archetype& archetype = m_Impl->GetArchetypeStorage(record->archetype);
        const uint8_t column = ColumnIndex(archetype, record->archetype, SlotWorld);
        if (column == c_Absent)
        {
            *outWorld = nullptr;
            return false;
        }
        *outWorld = &reinterpret_cast<const WorldTransform*>(ColumnBase(archetype, record->chunk, column))[record->row]
            .transform;
        // a shallower level's chunk: finished before this level started, so a plain read
        if (!readVersion || archetype.Versions(record->chunk)[column] <= since)
            return false;
        // E7.5: with row versions, only a parent whose own row was written
        const uint64_t* rows = ChangedRowVersions(archetype, record->chunk, column);
        return !rows || rows[record->row] > since;
    }

    // world = local * parent world for a row with a local transform, a bit copy of the parent's world without one
    // (no multiplication: no rounding, signed zeros kept); local, or the identity, for a row without a parent.
    // With `compose`, the local transform is composed again (its chunk's LocalTransform changed) and cached in
    // LocalMatrix; otherwise the cache is current.
    void SceneGraphStore::PropagateRow(const LevelArchetype& level, const Archetype& archetype, uint32_t chunk,
                                       uint32_t row, const dm::daffine3* parentWorld, bool compose) const noexcept
    {
        WorldTransform* worlds = reinterpret_cast<WorldTransform*>(ColumnBase(archetype, chunk, level.world));
        if (level.local == c_Absent)
        {
            worlds[row].transform = parentWorld ? *parentWorld : dm::daffine3::identity();
            return;
        }
        LocalMatrix* matrices = level.matrix != c_Absent
            ? reinterpret_cast<LocalMatrix*>(ColumnBase(archetype, chunk, level.matrix)) : nullptr;
        dm::daffine3 localTransform;
        if (compose || !matrices)
        {
            const LocalTransform* locals =
                reinterpret_cast<const LocalTransform*>(ColumnBase(archetype, chunk, level.local));
            localTransform = ComposeLocal(locals[row]);
            if (matrices)
                matrices[row].transform = localTransform;
        }
        else
            localTransform = matrices[row].transform;
        worlds[row].transform = parentWorld ? localTransform * *parentWorld : localTransform;
    }

    void SceneGraphStore::RunPropagationTasks(void* context, uint32_t begin, uint32_t end, uint32_t) noexcept
    {
        PropagationContext& ctx = *static_cast<PropagationContext*>(context);
        const SceneGraphStore& store = *ctx.store;
        const std::vector<LevelArchetype>& levels = store.m_Levels[ctx.level];
        uint64_t chunks = 0;
        uint64_t rows = 0;
        for (uint32_t t = begin; t < end; ++t)
        {
            const Task& task = store.m_Tasks[t];
            const LevelArchetype& level = levels[task.archetype];
            const Archetype& archetype = store.m_Impl->GetArchetypeStorage(level.archetype);
            const uint32_t count = archetype.chunks[task.chunk].count;
            const Parent* parents = level.parent != c_Absent
                ? reinterpret_cast<const Parent*>(ColumnBase(archetype, task.chunk, level.parent)) : nullptr;

            // E7.5: with row versions of WorldTransform, the rows written are stamped; a chunk-only write of it is
            // spread first, so that it stays seen. With row versions of LocalTransform (task.localRows), a chunk that
            // changed only there composes only its rows that changed.
            uint64_t* worldRows = archetype.RowVersions(task.chunk, level.world);
            if (worldRows)
                SpreadChunkWrite(archetype, task.chunk, level.world);
            uint32_t written = 0;
            bool composed = false;
            Entity cachedParent = Entity::Null;
            const dm::daffine3* parentWorld = nullptr;
            bool parentWritten = false;
            for (uint32_t r = 0; r < count; ++r)
            {
                const bool compose = task.whole && (!task.localRows || task.localRows[r] > ctx.since);
                if (parents)
                {
                    const Entity parent = parents[r].entity;
                    if (parent != cachedParent)
                    {
                        cachedParent = parent;
                        parentWritten = store.ParentWritten(parent, ctx.since, &parentWorld, true);
                    }
                    if (!parentWorld || (!compose && !parentWritten))
                        continue;
                }
                else if (!compose)
                    continue;
                store.PropagateRow(level, archetype, task.chunk, r, parents ? parentWorld : nullptr, compose);
                if (worldRows)
                    StampVersion(worldRows + r, ctx.version);
                composed = composed || compose;
                ++written;
            }
            if (written)
            {
                if (composed && level.matrix != c_Absent)
                    StampVersion(archetype.Versions(task.chunk) + level.matrix, ctx.version);
                if (worldRows)
                    StampVersion(worldRows - 1, ctx.version);
                StampVersion(archetype.Versions(task.chunk) + level.world, ctx.version);
                ++chunks;
                rows += written;
            }
        }
        if (chunks)
        {
            ctx.wrote.store(true, std::memory_order_relaxed);
            ctx.chunks.fetch_add(chunks, std::memory_order_relaxed);
            ctx.rows.fetch_add(rows, std::memory_order_relaxed);
        }
    }

    nvrhi::FRESULT SceneGraphStore::RunPropagation(ISystemContext* context) noexcept
    {
        const uint64_t since = context->GetLastRunVersion();
        const uint64_t version = context->GetVersion();
        IJobSystem* jobs = context->GetJobSystem();
        UpdateSystemLists();

        // Level by level: a level reads the worlds that the previous one finished. A chunk is computed whole when its
        // own local transforms, parents or rows changed since the last run; otherwise only its rows whose parent's
        // chunk was written since then, and only when anything of the level above was. A driven archetype gets no task
        // (its worlds are the interpolation's); a chunk of it that is newer than the last run was written.
        bool levelAboveWrote = false;
        for (uint32_t d = 0; d < c_MaxHierarchyDepth; ++d)
        {
            std::vector<LevelArchetype>& levels = m_Levels[d];
            bool drivenWritten = false;
            m_Tasks.clear();
            for (uint32_t i = 0; i < uint32_t(levels.size()); ++i)
            {
                LevelArchetype& level = levels[i];
                const Archetype& archetype = m_Impl->GetArchetypeStorage(level.archetype);
                if (level.driven)
                {
                    drivenWritten = drivenWritten || DrivenWritten(level, archetype, since);
                    continue;
                }
                if (level.staticRows)
                {
                    QueueStaticLevel(level, i, archetype, since);
                    continue;
                }
                for (uint32_t c = 0; c < archetype.GetChunkCount(); ++c)
                {
                    if (archetype.chunks[c].count == 0)
                        continue;
                    const uint64_t* versions = archetype.Versions(c);
                    const bool whole = (level.local != c_Absent && versions[level.local] > since) ||
                                       (level.parent != c_Absent && versions[level.parent] > since) ||
                                       versions[archetype.AddedVersionSlot()] > since;
                    if (whole || levelAboveWrote)
                        m_Tasks.push_back(Task{ i, c, whole, whole ? LocalRowsChanged(level, archetype, c, since)
                                                                   : nullptr });
                }
            }
            if (m_Tasks.empty())
            {
                levelAboveWrote = drivenWritten;
                continue;
            }

            PropagationContext ctx;
            ctx.store = this;
            ctx.level = d;
            ctx.since = since;
            ctx.version = version;
            const uint32_t taskCount = uint32_t(m_Tasks.size());
            const uint32_t grain = std::max(1u, taskCount / (WorkingThreads(jobs) * 4));
            if (!jobs || grain >= taskCount)
                RunPropagationTasks(&ctx, 0, taskCount, 0);
            else
                jobs->ParallelFor(taskCount, grain, &RunPropagationTasks, &ctx);
            levelAboveWrote = drivenWritten || ctx.wrote.load(std::memory_order_relaxed);
            m_Counters.propagatedChunks += ctx.chunks.load(std::memory_order_relaxed);
            m_Counters.propagatedRows += ctx.rows.load(std::memory_order_relaxed);
        }
        PropagateDeepBucket(jobs, since, version, levelAboveWrote);
        m_PropagatedVersion = version;    // the chunks added before this run are taken (IsSettledStaticRow)
        m_StaleWorldTransforms = false;   // every edit so far is in the world transforms
        if (m_MarkJoints)
        {
            MarkSkinnedJoints(m_JointsMarkedVersion);
            m_JointsMarkedVersion = version;
        }
        return nvrhi::FS_OK;
    }

    // The last bucket holds every entity at depth c_MaxHierarchyDepth or deeper, so a row's parent may be in it too.
    // Its entries (rows whose parent is in a shallower bucket) root independent subtrees, which run in parallel, each
    // depth first through SceneLinks, so that a parent is always computed before its children. Rows that no entry
    // reaches (entities with a Parent but no SceneLinks) take a serial pass that resolves each row's parent first.
    // Marks per entity index record which rows were visited and written; each is written by the one thread that owns
    // its row.
    struct SceneGraphStore::DeepContext
    {
        SceneGraphStore* store;
        uint64_t since;
        uint64_t version;
        std::atomic<uint64_t> rows{ 0 };
        std::atomic<uint64_t> written{ 0 };
    };

    namespace
    {
        constexpr uint8_t c_DeepVisited = 1;
        constexpr uint8_t c_DeepWritten = 2;
    }

    const SceneGraphStore::LevelArchetype* SceneGraphStore::DeepLevel(uint32_t archetypeIndex) const noexcept
    {
        if (archetypeIndex >= m_ArchetypeLevel.size() || m_ArchetypeLevel[archetypeIndex] != c_MaxHierarchyDepth)
            return nullptr;
        return &m_Levels[c_MaxHierarchyDepth][m_ArchetypeLevelSlot[archetypeIndex]];
    }

    const uint64_t* SceneGraphStore::LocalRowsChanged(const LevelArchetype& level, const Archetype& archetype,
                                                      uint32_t chunk, uint64_t since) noexcept
    {
        if (level.local == c_Absent)
            return nullptr;
        const uint64_t* versions = archetype.Versions(chunk);
        if ((level.parent != c_Absent && versions[level.parent] > since) ||
            versions[archetype.AddedVersionSlot()] > since)
            return nullptr;
        return ChangedRowVersions(archetype, chunk, level.local);
    }

    bool SceneGraphStore::DeepChunkDirty(const LevelArchetype& level, const Archetype& archetype, uint32_t chunk,
                                         uint64_t since) noexcept
    {
        const uint64_t* versions = archetype.Versions(chunk);
        return (level.local != c_Absent && versions[level.local] > since) ||
               (level.parent != c_Absent && versions[level.parent] > since) ||
               versions[archetype.AddedVersionSlot()] > since;
    }

    // Whether the interpolation (or a structural change) wrote a chunk of a driven archetype since the last run.
    bool SceneGraphStore::DrivenWritten(const LevelArchetype& level, const Archetype& archetype, uint64_t since) noexcept
    {
        for (uint32_t c = 0; c < archetype.GetChunkCount(); ++c)
        {
            if (archetype.chunks[c].count != 0 && archetype.Versions(c)[level.world] > since)
                return true;
        }
        return false;
    }

    // One row of the last bucket whose parent is final: computed when its chunk changed or its parent was written. A
    // driven row is not computed; propagation stamps no chunk of its archetype, so its chunk's version can be read here.
    bool SceneGraphStore::PropagateDeepRow(Entity entity, bool parentInBucket, uint64_t since,
                                           uint64_t version) noexcept
    {
        const uint32_t index = GetEntityIndex(entity);
        const EntityRecord& record = m_Impl->GetEntityRecord(index);
        const LevelArchetype* level = DeepLevel(record.archetype);
        m_DeepMarks[index] |= c_DeepVisited;
        if (!level)
            return false;
        const Archetype& archetype = m_Impl->GetArchetypeStorage(record.archetype);
        if (level->driven)
        {
            if (archetype.Versions(record.chunk)[level->world] > since)
                m_DeepMarks[index] |= c_DeepWritten;
            return false;
        }
        if (level->parent == c_Absent)
            return false;
        const Entity parent =
            reinterpret_cast<const Parent*>(ColumnBase(archetype, record.chunk, level->parent))[record.row].entity;
        // A parent in the bucket is in this subtree, computed on this thread before its children: its mark says whether
        // it was written; its chunk's version may be stamped by another subtree at this moment, so it is not read.
        const dm::daffine3* parentWorld = nullptr;
        bool parentWritten = ParentWritten(parent, since, &parentWorld, !parentInBucket);
        if (parentInBucket)
            parentWritten = (m_DeepMarks[GetEntityIndex(parent)] & c_DeepWritten) != 0;
        bool dirty = DeepChunkDirty(*level, archetype, record.chunk, since);
        if (dirty)
        {
            // E7.5: a chunk that changed only in LocalTransform's row versions composes only its rows that changed
            const uint64_t* localRows = LocalRowsChanged(*level, archetype, record.chunk, since);
            dirty = !localRows || localRows[record.row] > since;
        }
        if (!parentWorld || (!parentWritten && !dirty))
            return false;
        PropagateRow(*level, archetype, record.chunk, record.row, parentWorld, dirty);
        uint64_t* versions = archetype.Versions(record.chunk);
        if (dirty && level->matrix != c_Absent)
            StampVersion(versions + level->matrix, version);
        StampRowWrite(archetype, record.chunk, level->world, record.row, version);
        m_DeepMarks[index] |= c_DeepWritten;
        return true;
    }

    void SceneGraphStore::RunDeepTasks(void* context, uint32_t begin, uint32_t end, uint32_t) noexcept
    {
        DeepContext& ctx = *static_cast<DeepContext*>(context);
        SceneGraphStore& store = *ctx.store;
        // Per thread, kept for its capacity. A range can run on a thread whose thread_locals are destroyed already
        // (exit destroys the main thread's before the atexit handlers, which may refresh a graph): it then uses a
        // stack of its own, like ScratchScope.
        struct StackHolder
        {
            std::vector<Entity> entities;
            ~StackHolder() { t_DeepStackGone = true; }
        };
        std::unique_ptr<std::vector<Entity>> ownStack;   // built only then (a Debug vector allocates when built)
        std::vector<Entity>* stackPointer = nullptr;
        if (!t_DeepStackGone)
        {
            thread_local StackHolder holder;
            stackPointer = &holder.entities;
        }
        else
        {
            ownStack = std::make_unique<std::vector<Entity>>();
            stackPointer = ownStack.get();
        }
        std::vector<Entity>& stack = *stackPointer;
        uint64_t rows = 0;
        uint64_t written = 0;
        for (uint32_t e = begin; e < end; ++e)
        {
            const Entity entry = store.m_DeepEntries[e];
            stack.clear();
            stack.push_back(entry);
            while (!stack.empty())
            {
                const Entity entity = stack.back();
                stack.pop_back();
                written += store.PropagateDeepRow(entity, entity != entry, ctx.since, ctx.version) ? 1 : 0;
                ++rows;
                const SceneLinks* links = store.Get<SceneLinks>(entity, SlotLinks);
                for (Entity child = links ? links->firstChild : Entity::Null; child != Entity::Null;)
                {
                    stack.push_back(child);
                    const SceneLinks* childLinks = store.Get<SceneLinks>(child, SlotLinks);
                    child = childLinks ? childLinks->nextSibling : Entity::Null;
                }
            }
        }
        ctx.rows.fetch_add(rows, std::memory_order_relaxed);
        ctx.written.fetch_add(written, std::memory_order_relaxed);
    }

    void SceneGraphStore::PropagateDeepBucket(IJobSystem* jobs, uint64_t since, uint64_t version, bool parentLevelWrote)
    {
        const std::vector<LevelArchetype>& levels = m_Levels[c_MaxHierarchyDepth];
        if (levels.empty())
            return;
        bool any = parentLevelWrote;
        for (const LevelArchetype& level : levels)
        {
            const Archetype& archetype = m_Impl->GetArchetypeStorage(level.archetype);
            if (level.driven)
            {
                any = any || DrivenWritten(level, archetype, since);
                continue;
            }
            for (uint32_t c = 0; c < archetype.GetChunkCount() && !any; ++c)
                any = archetype.chunks[c].count != 0 && DeepChunkDirty(level, archetype, c, since);
        }
        if (!any)
            return;

        if (m_DeepMarks.size() < m_Impl->GetEntityTableSize())
            m_DeepMarks.resize(m_Impl->GetEntityTableSize(), 0);

        // The entries, in storage order.
        m_DeepEntries.clear();
        for (const LevelArchetype& level : levels)
        {
            const Archetype& archetype = m_Impl->GetArchetypeStorage(level.archetype);
            if (level.parent == c_Absent)
                continue;
            for (uint32_t c = 0; c < archetype.GetChunkCount(); ++c)
            {
                const Entity* entities = archetype.Entities(c);
                const Parent* parents = reinterpret_cast<const Parent*>(ColumnBase(archetype, c, level.parent));
                for (uint32_t r = 0; r < archetype.chunks[c].count; ++r)
                {
                    const EntityRecord* parentRecord = m_Impl->FindAlive(parents[r].entity);
                    if (parentRecord && !DeepLevel(parentRecord->archetype))
                        m_DeepEntries.push_back(entities[r]);
                }
            }
        }

        DeepContext ctx;
        ctx.store = this;
        ctx.since = since;
        ctx.version = version;
        const uint32_t entryCount = uint32_t(m_DeepEntries.size());
        const uint32_t grain = std::max(1u, entryCount / (WorkingThreads(jobs) * 4));
        if (entryCount != 0)
        {
            if (!jobs || grain >= entryCount)
                RunDeepTasks(&ctx, 0, entryCount, 0);
            else
                jobs->ParallelFor(entryCount, grain, &RunDeepTasks, &ctx);
        }
        m_Counters.deepRows += ctx.rows.load(std::memory_order_relaxed);
        m_Counters.propagatedRows += ctx.written.load(std::memory_order_relaxed);

        // The rows that no entry reached, serially: each resolves its parent first (an explicit stack).
        for (const LevelArchetype& start : levels)
        {
            const Archetype& startArchetype = m_Impl->GetArchetypeStorage(start.archetype);
            for (uint32_t c = 0; c < startArchetype.GetChunkCount(); ++c)
            {
                const Entity* entities = startArchetype.Entities(c);
                for (uint32_t r = 0; r < startArchetype.chunks[c].count; ++r)
                {
                    if (m_DeepMarks[GetEntityIndex(entities[r])] & c_DeepVisited)
                        continue;
                    m_DeepStack.clear();
                    m_DeepStack.push_back(entities[r]);
                    while (!m_DeepStack.empty())
                    {
                        const Entity entity = m_DeepStack.back();
                        const uint32_t index = GetEntityIndex(entity);
                        if (m_DeepMarks[index] & c_DeepVisited)
                        {
                            m_DeepStack.pop_back();
                            continue;
                        }
                        const EntityRecord& record = m_Impl->GetEntityRecord(index);
                        const LevelArchetype* level = DeepLevel(record.archetype);
                        bool parentInBucket = false;
                        if (level && level->parent != c_Absent)
                        {
                            const Archetype& archetype = m_Impl->GetArchetypeStorage(record.archetype);
                            const Entity parent = reinterpret_cast<const Parent*>(
                                ColumnBase(archetype, record.chunk, level->parent))[record.row].entity;
                            const EntityRecord* parentRecord = m_Impl->FindAlive(parent);
                            parentInBucket = parentRecord && DeepLevel(parentRecord->archetype);
                            if (parentInBucket && !(m_DeepMarks[GetEntityIndex(parent)] & c_DeepVisited))
                            {
                                m_DeepStack.push_back(parent);
                                continue;
                            }
                        }
                        m_Counters.propagatedRows += PropagateDeepRow(entity, parentInBucket, since, version) ? 1 : 0;
                        ++m_Counters.deepRows;
                        m_DeepStack.pop_back();
                    }
                }
            }
        }

        for (const LevelArchetype& level : levels)
        {
            const Archetype& archetype = m_Impl->GetArchetypeStorage(level.archetype);
            for (uint32_t c = 0; c < archetype.GetChunkCount(); ++c)
            {
                const Entity* entities = archetype.Entities(c);
                for (uint32_t r = 0; r < archetype.chunks[c].count; ++r)
                    m_DeepMarks[GetEntityIndex(entities[r])] = 0;
            }
        }
    }

    // SkinnedMeshInstance::GetLastUpdateFrameIndex: the frame at which a joint node's own local transform changed (the
    // world transform of a driven joint, which has no local one behind it). Per chunk (D9), so a joint that shares a
    // chunk with one that moved counts as moved; that only adds a skinning pass.
    void SceneGraphStore::MarkSkinnedJoints(uint64_t since)
    {
        for (const JointArchetype& joints : m_JointArchetypes)
        {
            const Archetype& archetype = m_Impl->GetArchetypeStorage(joints.archetype);
            for (uint32_t c = 0; c < archetype.GetChunkCount(); ++c)
            {
                const uint64_t* versions = archetype.Versions(c);
                if (versions[joints.moved] <= since && versions[archetype.AddedVersionSlot()] <= since)
                    continue;
                const LeafObject* leaves = reinterpret_cast<const LeafObject*>(ColumnBase(archetype, c, joints.leaf));
                for (uint32_t r = 0; r < archetype.chunks[c].count; ++r)
                {
                    if (auto reference = query_cast<SkinnedMeshReference>(leaves[r].object))
                    {
                        if (auto instance = reference->m_Instance.Lock())
                            instance->m_LastUpdateFrameIndex = m_FrameIndex;
                    }
                }
            }
        }
    }

    // ----------------------------------------------------------------- simulation-driven nodes (4.7, 8.4; E6.2a)

    dm::daffine3 SceneGraphStore::SimulatedPose(const RowRef& row) const noexcept
    {
        const SimPose* pose = static_cast<const SimPose*>(Column(row, SlotSimPose));
        const SimScale* scale = static_cast<const SimScale*>(Column(row, SlotSimScale));
        if (pose && scale)
            return ComposePose(pose->translation, pose->rotation, scale->scaling);
        const WorldTransform* world = static_cast<const WorldTransform*>(Column(row, SlotWorld));
        return world ? world->transform : dm::daffine3::identity();
    }

    LocalTransform SceneGraphStore::SimulatedParts(const RowRef& row) const noexcept
    {
        const SimPose* pose = static_cast<const SimPose*>(Column(row, SlotSimPose));
        const SimScale* scale = static_cast<const SimScale*>(Column(row, SlotSimScale));
        if (pose && scale)
            return LocalTransform{ pose->translation, pose->rotation, scale->scaling };
        return Decompose(SimulatedPose(row)).transform;
    }

    // A settled Static row (E7.4 remedy 3) ends the walk like a driven one: propagation keeps its world, whatever its
    // ancestors did since, and composes the rows below it from that world (backlog I5-R4).
    bool SceneGraphStore::EndsComposition(const RowRef& row, dm::daffine3& outWorld) const noexcept
    {
        if (IsDriven(row))
        {
            outWorld = SimulatedPose(row);
            return true;
        }
        if (IsSettledStaticRow(row))
        {
            const WorldTransform* world = static_cast<const WorldTransform*>(Column(row, SlotWorld));
            outWorld = world ? world->transform : dm::daffine3::identity();
            return true;
        }
        return false;
    }

    dm::daffine3 SceneGraphStore::ComposeWorld(Entity entity) const noexcept
    {
        // the local transforms from the entity up, without the driven (or settled Static) row that ends the walk
        const LocalTransform* locals[c_ComposeBlock];
        uint32_t count = 0;
        dm::daffine3 world = dm::daffine3::identity();
        bool hasParent = false;   // the topmost of the locals multiplies a world above it
        const Entity bottom = entity;
        RowRef row;
        while (Locate(entity, row))
        {
            if (EndsComposition(row, world))
            {
                hasParent = true;
                break;
            }
            const LocalTransform* local = static_cast<const LocalTransform*>(Column(row, SlotLocal));
            if (local && count == c_ComposeBlock)
                return ComposeLongChain(bottom);
            if (local)
                locals[count++] = local;
            hasParent = !local;   // of the rows so far; a row without a local transform takes its parent's world
            const Parent* parent = static_cast<const Parent*>(Column(row, SlotParent));
            if (!parent)
                break;
            entity = parent->entity;
        }
        // as PropagateRow, bit for bit
        for (uint32_t i = count; i-- > 0; hasParent = true)
            world = hasParent ? ComposeLocal(*locals[i]) * world : ComposeLocal(*locals[i]);
        return world;
    }

    // The links lead upwards and the products start at the top: the chain is cut into at most c_ComposeBlock pieces, the
    // topmost piece first, and so on, until a piece fits the block of locals. Every cut reads the rows of its piece once;
    // what is left to do is a stack of at most c_ComposeBlock - 1 pieces per size of piece.
#if defined(_MSC_VER)
    __declspec(noinline)   // out of ComposeWorld's frame
#endif
    dm::daffine3 SceneGraphStore::ComposeLongChain(Entity entity) const noexcept
    {
        struct Piece
        {
            Entity bottom;
            uint32_t rows;   // from it upwards
        };
        constexpr uint32_t c_MaxPieces = (c_ComposeBlock - 1) * 5 + 1;   // 2^32 rows are cut five times
        Piece pieces[c_MaxPieces];
        uint32_t pieceCount = 0;

        dm::daffine3 world = dm::daffine3::identity();
        bool hasParent = false;
        RowRef row;
        auto above = [&](Entity at) noexcept
        {
            const Parent* parent = Locate(at, row) ? static_cast<const Parent*>(Column(row, SlotParent)) : nullptr;
            return parent ? parent->entity : Entity::Null;
        };
        uint32_t rows = 0;
        for (Entity at = entity; Locate(at, row);)
        {
            if (EndsComposition(row, world))
            {
                hasParent = true;
                break;
            }
            ++rows;
            const Parent* parent = static_cast<const Parent*>(Column(row, SlotParent));
            if (!parent)
                break;
            at = parent->entity;
        }
        pieces[pieceCount++] = Piece{ entity, rows };
        while (pieceCount != 0)
        {
            const Piece piece = pieces[--pieceCount];
            Entity at = piece.bottom;
            if (piece.rows <= c_ComposeBlock)
            {
                const LocalTransform* locals[c_ComposeBlock];
                for (uint32_t i = 0; i < piece.rows; ++i, at = above(at))
                    locals[i] = Get<LocalTransform>(at, SlotLocal);
                // as PropagateRow, bit for bit: a row without a local transform takes its parent's world
                for (uint32_t i = piece.rows; i-- > 0; hasParent = true)
                {
                    if (locals[i])
                        world = hasParent ? ComposeLocal(*locals[i]) * world : ComposeLocal(*locals[i]);
                }
                continue;
            }
            const uint32_t size = (piece.rows - 1) / c_ComposeBlock + 1;
            for (uint32_t done = 0; done < piece.rows; done += size)
            {
                assert(pieceCount < c_MaxPieces);
                pieces[pieceCount++] = Piece{ at, std::min(size, piece.rows - done) };
                for (uint32_t i = 0; i < size && done + i + 1 < piece.rows; ++i)
                    at = above(at);
            }
        }
        return world;
    }

    LocalTransform SceneGraphStore::SimulatedLocal(const RowRef& row, ParentWorld& cache, const char* operation) const
    {
        LocalTransform local = SimulatedParts(row);
        bool counts = true;
        bool conformal = true;   // the parts give the pose: no need to compare the matrices
        if (const Parent* parent = static_cast<const Parent*>(Column(row, SlotParent)))
        {
            if (cache.parent != parent->entity)
            {
                cache.parent = parent->entity;
                cache.world = ComposeWorld(parent->entity);
            }
            local = PartsInParent(local, cache.world, counts, &conformal);
        }
        const bool same = conformal ? IsFinite(local)
                                    : SameAxes(ComposeLocal(local) * cache.world, SimulatedPose(row));
        if (!counts || !same)
            log::warning("SceneGraph::%s: no local transform gives a simulated node its pose (its parent's scale shears "
                         "it or has no inverse, or a number is not finite); %s is taken", operation,
                         counts ? "the nearest" : "its pose in the world");
        return local;
    }

    bool SceneGraphStore::GetDrivenLocal(const SceneGraphNode* node, dm::daffine3& outMatrix) const noexcept
    {
        RowRef row;
        if (!LocateNode(node, row) || !IsDriven(row))
            return false;
        outMatrix = SimulatedPose(row);
        bool inverse;
        if (const Parent* parent = static_cast<const Parent*>(Column(row, SlotParent)))
            outMatrix = RelativeTo(outMatrix, ComposeWorld(parent->entity), inverse);
        return true;
    }

    bool SceneGraphStore::GetDrivenLocal(const SceneGraphNode* node, LocalTransform& out) const noexcept
    {
        RowRef row;
        if (!LocateNode(node, row) || !IsDriven(row))
            return false;
        out = InParentOf(row, SimulatedParts(row));
        return true;
    }

    LocalTransform SceneGraphStore::InParentOf(const RowRef& row, const LocalTransform& world) const noexcept
    {
        const Parent* parent = static_cast<const Parent*>(Column(row, SlotParent));
        bool counts;
        return parent ? PartsInParent(world, ComposeWorld(parent->entity), counts) : world;
    }

    bool SceneGraphStore::ToDrivenLocal(const SceneGraphNode* node, const dm::daffine3& world,
                                        LocalTransform& out) noexcept
    {
        const SceneGraphStore* store = node->m_Record ? nullptr : node->m_Store;
        RowRef row;
        if (!store || !store->LocateNode(node, row) || !store->IsDriven(row))
            return false;
        const Decomposition parts = Decompose(world);
        if (parts.degenerate)
            return false;
        out = store->InParentOf(row, parts.transform);
        return true;
    }

#if defined(_MSC_VER)
    __declspec(noinline)   // out of SetTransform's hot path
#endif
    void SceneGraphStore::RequestTeleport(const SceneGraphNode* node, const dm::double3* translation,
                                          const dm::dquat* rotation, const dm::double3* scaling) noexcept
    {
        RowRef row;
        if ((!translation && !rotation && !scaling) || !LocateNode(node, row))
            return;
        const uint32_t column = row.archetype->FindColumn(m_SimTeleportId);
        LocalTransform given{ dm::double3(0.0), dm::dquat::identity(), dm::double3(1.0) };
        if (translation) given.translation = *translation;
        if (scaling) given.scaling = *scaling;
        double length = 1.0;
        if (rotation)
        {
            length = std::sqrt(dm::dot(*rotation, *rotation));
            given.rotation = *rotation / length;   // engines want it normalised
        }
        if (column == c_NoColumn || !IsFinite(given) || !std::isfinite(length))
        {
            log::warning("SceneGraph::SetTransform: %s; nothing changed", column == c_NoColumn
                         ? "the row of a node that a simulation drives has no SimTeleport"
                         : "a simulation drives the node, and no body has such a pose (a number is not finite, or the "
                           "rotation is 0)");
            return;
        }
        SimTeleport& request =
            reinterpret_cast<SimTeleport*>(ColumnBase(*row.archetype, row.chunk, uint8_t(column)))[row.row];
        uint64_t* enabled = row.archetype->EnabledMask(row.chunk, column);
        if (!translation || !rotation || !scaling)
        {
            // the parts that are not given: of the pending request, else where the simulation has the node now
            const bool pending = GetMaskBit(enabled, row.row);
            LocalTransform rest = pending ? LocalTransform{ request.translation, request.rotation, request.scaling }
                                          : SimulatedParts(row);
            if (!pending || request.space == SimTeleportSpace::World)
                rest = InParentOf(row, rest);
            if (!translation) given.translation = rest.translation;
            if (!rotation) given.rotation = rest.rotation;
            if (!scaling) given.scaling = rest.scaling;
        }
        request = SimTeleport{ given.translation, given.rotation, given.scaling, SimTeleportSpace::Parent, 0 };
        SetMaskBit(enabled, row.row, true);
        StampVersion(row.archetype->Versions(row.chunk) + column, m_Impl->GetStampVersion());
    }

    nvrhi::FRESULT SceneGraphStore::SetSimulationDriven(SceneGraphNode* node, bool driven)
    {
        if (!CanChangeStructure("SetSimulationDriven"))
            return nvrhi::FE_INVALID_ARGS;
        RowRef row;
        if (!Locate(node->m_Entity, row))
            return nvrhi::FE_INVALID_ARGS;
        if (IsDriven(row) == driven)
            return nvrhi::FS_OK;
        return driven ? BindToSimulation(node) : UnbindFromSimulation(node, row);
    }

    nvrhi::FRESULT SceneGraphStore::ChangeDriven(SceneGraphNode* node, WorldImpl::ComponentChange& change,
                                                 const WorldImpl::RowHooks* hooks)
    {
        const Entity entity = node->m_Entity;
        m_World->AdvanceVersion();   // what the move stamps is newer than any version read before it (GpuScene)
        const nvrhi::FRESULT result =
            m_Impl->ChangeComponentsInternal(entity, change, m_Impl->GetStampVersion(), hooks);
        if (NVRHI_FAILED(result))
        {
            log::warning("SceneGraph::SetSimulationDriven: the world cannot make the node's new row (%d); nothing "
                         "changed", int(result));
            return result;
        }
        nvrhi::FRESULT observers = nvrhi::FS_OK;   // they undo nothing
        FlushObservers(observers);
        WarnObservers("SetSimulationDriven", observers);
        SyncArchetypes();
        RowRef row;
        if (Locate(entity, row))
            RememberRow(node, row);   // its row moved to another archetype
        return nvrhi::FS_OK;
    }

    nvrhi::FRESULT SceneGraphStore::BindToSimulation(SceneGraphNode* node)
    {
        // Composed, whatever the WorldTransform column holds: it lacks the edits that no propagation has processed (the
        // node's own are not propagated any more; a node that is made, placed and bound before the first Refresh starts
        // at its place), and after a simulation frame it is older than its driven ancestors until the next Refresh. Over
        // their simulated poses, so that the engine's bodies are of one time and unbinding gives the local transform
        // back.
        const Entity entity = node->m_Entity;
        const WorldTransform world{ ComposeWorld(entity) };
        const Decomposition parts = Decompose(world.transform);
        if (parts.degenerate || parts.mirrored || parts.shear > c_MaxShear)
        {
            log::warning("SceneGraph::SetSimulationDriven: the node's world transform is no pose with a scale (%s); "
                         "nothing changed", parts.degenerate ? "a scale is 0, or it is not finite"
                                            : parts.mirrored ? "it mirrors" : "it shears");
            return nvrhi::FE_INVALID_ARGS;
        }
        const SimPose pose{ parts.transform.translation, parts.transform.rotation };
        const SimPosePrevious previous{ pose.translation, pose.rotation };
        const SimScale scale{ parts.transform.scaling };
        const WorldImpl::NewValue added[] = {
            { m_Ids[SlotDriven], nullptr, ValueTransfer::Copy },
            { m_Ids[SlotSimPose], &pose, ValueTransfer::Copy },
            { m_SimPreviousId, &previous, ValueTransfer::Copy },
            { m_Ids[SlotSimScale], &scale, ValueTransfer::Copy },
            { m_SimBodyId, nullptr, ValueTransfer::Copy },       // of no domain
            { m_SimTeleportId, nullptr, ValueTransfer::Copy },
            { m_Ids[SlotWorld], &world, ValueTransfer::Copy },   // the row has it: overwritten
        };
        WorldImpl::ComponentChange change;
        change.added = added;
        change.addedCount = uint32_t(std::size(added));
        // A row starts with every component enabled: without this, OnEntitiesAdded would see a pending request.
        struct Bound
        {
            SceneGraphStore* store;
            Entity entity;
        };
        Bound bound{ this, entity };
        const WorldImpl::RowHooks hooks{ &bound, [](void* context, uint32_t) noexcept
        {
            const Bound& b = *static_cast<const Bound*>(context);
            RowRef row;
            if (!b.store->Locate(b.entity, row))
                return;
            const uint32_t column = row.archetype->FindColumn(b.store->m_SimTeleportId);
            SetMaskBit(row.archetype->EnabledMask(row.chunk, column), row.row, false);
        }, nullptr };
        return ChangeDriven(node, change, &hooks);
    }

    nvrhi::FRESULT SceneGraphStore::UnbindFromSimulation(SceneGraphNode* node, const RowRef& boundRow)
    {
        ParentWorld parentWorld;
        const LocalTransform local = SimulatedLocal(boundRow, parentWorld, "SetSimulationDriven");
        const ComponentId removed[] = { m_Ids[SlotDriven], m_Ids[SlotSimPose], m_SimPreviousId, m_Ids[SlotSimScale],
                                        m_SimBodyId, m_SimTeleportId };
        const WorldImpl::NewValue added[] = {
            { m_Ids[SlotLocal], &local, ValueTransfer::Copy },
            { m_Ids[SlotLocalMatrix], nullptr, ValueTransfer::Copy },   // composed by the next propagation
        };
        WorldImpl::ComponentChange change;
        change.removed = removed;
        change.removedCount = uint32_t(std::size(removed));
        change.added = added;
        change.addedCount = uint32_t(std::size(added));
        const nvrhi::FRESULT result = ChangeDriven(node, change, nullptr);
        if (NVRHI_SUCCEEDED(result))
        {
            m_PendingTransform = true;
            m_StaleWorldTransforms = true;
        }
        return result;
    }

    // Every request is resolved from the rows as they are: from other rows it reads poses and local transforms, which no
    // request changes, so the order of the rows does not matter. A pending request of an ancestor does not count.
    void SceneGraphStore::RunTeleportTasks(void* context, uint32_t begin, uint32_t end, uint32_t) noexcept
    {
        TeleportContext& ctx = *static_cast<TeleportContext*>(context);
        const SceneGraphStore& store = *ctx.store;
        uint64_t resolved = 0;
        for (uint32_t t = begin; t < end; ++t)
        {
            const Task& task = store.m_Tasks[t];
            const DrivenArchetype& driven = store.m_DrivenArchetypes[task.archetype];
            const Archetype& archetype = store.m_Impl->GetArchetypeStorage(driven.archetype);
            SimTeleport* requests =
                reinterpret_cast<SimTeleport*>(ColumnBase(archetype, task.chunk, driven.teleport));
            const uint64_t* enabled = archetype.EnabledMask(task.chunk, driven.teleport);
            const Parent* parents = driven.parent != c_Absent
                ? reinterpret_cast<const Parent*>(ColumnBase(archetype, task.chunk, driven.parent)) : nullptr;
            ParentWorld above;
            uint64_t written = 0;
            for (uint32_t r = 0; r < archetype.chunks[task.chunk].count; ++r)
            {
                SimTeleport& request = requests[r];
                if (!GetMaskBit(enabled, r) || request.space != SimTeleportSpace::Parent)
                    continue;
                if (parents)
                {
                    if (above.parent != parents[r].entity)
                    {
                        above.parent = parents[r].entity;
                        above.world = store.ComposeWorld(above.parent);
                    }
                    const LocalTransform world =
                        PartsInWorld({ request.translation, request.rotation, request.scaling }, above.world);
                    request.translation = world.translation;
                    request.rotation = world.rotation;
                    request.scaling = world.scaling;
                }
                request.space = SimTeleportSpace::World;
                ++written;
            }
            if (written)
                StampVersion(archetype.Versions(task.chunk) + driven.teleport, ctx.version);
            resolved += written;
        }
        ctx.resolved.fetch_add(resolved, std::memory_order_relaxed);
    }

    nvrhi::FRESULT SceneGraphStore::RunTeleports(ISystemContext* context) noexcept
    {
        IJobSystem* jobs = context->GetJobSystem();
        UpdateSystemLists();
        m_Tasks.clear();
        for (uint32_t i = 0; i < uint32_t(m_DrivenArchetypes.size()); ++i)
        {
            const DrivenArchetype& driven = m_DrivenArchetypes[i];
            if (driven.teleport == c_Absent)
                continue;
            const Archetype& archetype = m_Impl->GetArchetypeStorage(driven.archetype);
            for (uint32_t c = 0; c < archetype.GetChunkCount(); ++c)
            {
                // by the bits, not by the column's version: a pending request may be older than the last run
                const uint32_t count = archetype.chunks[c].count;
                const uint64_t* enabled = archetype.EnabledMask(c, driven.teleport);
                uint64_t any = 0;
                for (uint32_t word = 0; word * 64 < count; ++word)
                    any |= enabled[word];
                if (any == 0)
                    continue;
                try
                {
                    m_Tasks.push_back(Task{ i, c, false });
                }
                catch (...)
                {
                    return nvrhi::FE_OUT_OF_MEMORY;   // none is resolved: the domains leave them to the next run
                }
            }
        }
        if (m_Tasks.empty())
            return nvrhi::FS_OK;
        TeleportContext ctx;
        ctx.store = this;
        ctx.version = context->GetVersion();
        const uint32_t taskCount = uint32_t(m_Tasks.size());
        const uint32_t grain = std::max(1u, taskCount / (WorkingThreads(jobs) * 4));
        if (!jobs || grain >= taskCount)
            RunTeleportTasks(&ctx, 0, taskCount, 0);
        else
            jobs->ParallelFor(taskCount, grain, &RunTeleportTasks, &ctx);
        m_Counters.resolvedTeleports += ctx.resolved.load(std::memory_order_relaxed);
        return nvrhi::FS_OK;
    }

    void SceneGraphStore::RunInterpolationTasks(void* context, uint32_t begin, uint32_t end, uint32_t) noexcept
    {
        InterpolationContext& ctx = *static_cast<InterpolationContext*>(context);
        const SceneGraphStore& store = *ctx.store;
        uint64_t chunks = 0;
        for (uint32_t t = begin; t < end; ++t)
        {
            const Task& task = store.m_Tasks[t];
            const DrivenArchetype& driven = store.m_DrivenArchetypes[task.archetype];
            const Archetype& archetype = store.m_Impl->GetArchetypeStorage(driven.archetype);
            const uint32_t count = archetype.chunks[task.chunk].count;
            const SimBodyRef* bodies =
                reinterpret_cast<const SimBodyRef*>(ColumnBase(archetype, task.chunk, driven.body));
            auto domainOf = [&](uint32_t row) noexcept { return std::min(bodies[row].domain, c_MaxSimulationDomains); };
            if (!task.whole)
            {
                // the poses are as they were: only for rows of a domain whose alpha changed
                uint32_t domains = 0;
                for (uint32_t r = 0; r < count; ++r)
                    domains |= 1u << domainOf(r);
                if ((domains & ctx.changedDomains) == 0)
                    continue;
            }
            const SimPose* poses = reinterpret_cast<const SimPose*>(ColumnBase(archetype, task.chunk, driven.pose));
            const SimPosePrevious* previous =
                reinterpret_cast<const SimPosePrevious*>(ColumnBase(archetype, task.chunk, driven.previous));
            const SimScale* scales = reinterpret_cast<const SimScale*>(ColumnBase(archetype, task.chunk, driven.scale));
            WorldTransform* worlds = reinterpret_cast<WorldTransform*>(ColumnBase(archetype, task.chunk, driven.world));
            for (uint32_t r = 0; r < count; ++r)
                worlds[r].transform = InterpolatePose(previous[r], poses[r], scales[r], ctx.alphas[domainOf(r)]);
            StampRowWrites(archetype, task.chunk, driven.world, 0, count, ctx.version);
            ++chunks;
        }
        ctx.chunks.fetch_add(chunks, std::memory_order_relaxed);
    }

    nvrhi::FRESULT SceneGraphStore::RunInterpolation(ISystemContext* context) noexcept
    {
        const uint64_t since = context->GetLastRunVersion();
        IJobSystem* jobs = context->GetJobSystem();
        UpdateSystemLists();
        const FixedStepState* state = static_cast<const FixedStepState*>(m_World->GetResource(m_FixedStepId));
        if (!state)
            return nvrhi::FE_INVALID_ARGS;

        InterpolationContext ctx;
        ctx.store = this;
        ctx.version = context->GetVersion();
        ctx.changedDomains = 0;
        for (uint32_t d = 0; d < c_MaxSimulationDomains; ++d)
        {
            // Below one once the planned steps were taken. Without a domain (it was removed, too): the last pose.
            const FixedStepEntry& entry = state->domains[d];
            ctx.alphas[d] = entry.fixedTimeStep > 0.0 ? std::min(entry.accumulator / entry.fixedTimeStep, 1.0) : 1.0;
            if (ctx.alphas[d] != m_LastAlphas[d])
                ctx.changedDomains |= 1u << d;
            m_LastAlphas[d] = ctx.alphas[d];
        }
        ctx.alphas[c_MaxSimulationDomains] = 1.0;

        m_Tasks.clear();
        for (uint32_t i = 0; i < uint32_t(m_DrivenArchetypes.size()); ++i)
        {
            const DrivenArchetype& driven = m_DrivenArchetypes[i];
            const Archetype& archetype = m_Impl->GetArchetypeStorage(driven.archetype);
            for (uint32_t c = 0; c < archetype.GetChunkCount(); ++c)
            {
                if (archetype.chunks[c].count == 0)
                    continue;
                const uint64_t* versions = archetype.Versions(c);
                const bool whole = versions[driven.pose] > since || versions[driven.previous] > since ||
                                   versions[driven.scale] > since || versions[driven.body] > since;
                if (whole || ctx.changedDomains != 0)
                    m_Tasks.push_back(Task{ i, c, whole });
            }
        }
        if (m_Tasks.empty())
            return nvrhi::FS_OK;
        const uint32_t taskCount = uint32_t(m_Tasks.size());
        const uint32_t grain = std::max(1u, taskCount / (WorkingThreads(jobs) * 4));
        if (!jobs || grain >= taskCount)
            RunInterpolationTasks(&ctx, 0, taskCount, 0);
        else
            jobs->ParallelFor(taskCount, grain, &RunInterpolationTasks, &ctx);
        m_Counters.interpolatedChunks += ctx.chunks.load(std::memory_order_relaxed);
        return nvrhi::FS_OK;
    }

    // ------------------------------------------------------------------------------------ Bounds (4.7)

    void SceneGraphStore::RunBoundsTasks(void* context, uint32_t begin, uint32_t end, uint32_t) noexcept
    {
        BoundsContext& ctx = *static_cast<BoundsContext*>(context);
        const SceneGraphStore& store = *ctx.store;
        uint64_t childless = 0;
        for (uint32_t t = begin; t < end; ++t)
        {
            const Task& task = store.m_Tasks[t];
            const BoundsArchetype& bounds = store.m_BoundsArchetypes[task.archetype];
            const Archetype& archetype = store.m_Impl->GetArchetypeStorage(bounds.archetype);
            const uint32_t count = archetype.chunks[task.chunk].count;
            const WorldTransform* worlds =
                reinterpret_cast<const WorldTransform*>(ColumnBase(archetype, task.chunk, bounds.world));
            LocalBounds* locals =
                reinterpret_cast<LocalBounds*>(ColumnBase(archetype, task.chunk, bounds.localBounds));
            uint64_t* versions = archetype.Versions(task.chunk);

            // Leaves with dynamic bounds: their boxes again. A chunk queued only for them (task.whole false) is written
            // only when one of its boxes changed.
            if (bounds.leaf != c_Absent)
            {
                const LeafObject* leaves =
                    reinterpret_cast<const LeafObject*>(ColumnBase(archetype, task.chunk, bounds.leaf));
                bool changed = false;
                for (uint32_t r = 0; r < count; ++r)
                {
                    if (!leaves[r].object)
                        continue;
                    const dm::box3 box = SafeLocalBoundingBox(leaves[r].object, locals[r].bounds);
                    if (std::memcmp(&box, &locals[r].bounds, sizeof(box)) != 0)
                    {
                        locals[r].bounds = box;
                        changed = true;
                    }
                }
                if (changed)
                    StampVersion(versions + bounds.localBounds, ctx.version);
                else if (!task.whole)
                    continue;
            }

            WorldBounds* results =
                reinterpret_cast<WorldBounds*>(ColumnBase(archetype, task.chunk, bounds.worldBounds));
            dm::box3 chunkBounds = dm::box3::empty();
            bool nan = false;
            for (uint32_t r = 0; r < count; ++r)
            {
                const dm::box3& local = locals[r].bounds;
                const dm::box3 world = local.isempty() ? dm::box3::empty() : local * dm::affine3(worlds[r].transform);
                results[r].bounds = world;
                chunkBounds |= world;
                // a NaN would fall out of the union (dm::min keeps the other operand): such a chunk is never culled
                nan = nan || HasNan(world);
            }
            if (nan)
                chunkBounds = InfiniteBox();
            reinterpret_cast<ChunkWorldBounds*>(ColumnBase(archetype, task.chunk, bounds.chunkBounds))->bounds =
                chunkBounds;
            StampVersion(versions + bounds.worldBounds, ctx.version);
            StampVersion(versions + bounds.chunkBounds, ctx.version);
            if (ctx.subgraph &&
                ctx.store->SubgraphFromBounds(archetype, bounds.archetype, task.chunk, chunkBounds, ctx.childless))
                ++childless;
        }
        ctx.chunks.fetch_add(end - begin, std::memory_order_relaxed);
        if (childless)
            ctx.childlessChunks.fetch_add(childless, std::memory_order_relaxed);
    }

    nvrhi::FRESULT SceneGraphStore::RunBounds(ISystemContext* context) noexcept
    {
        const uint64_t since = context->GetLastRunVersion();
        IJobSystem* jobs = context->GetJobSystem();
        UpdateSystemLists();
        if (!m_SubgraphPrepared)
        {
            // run by someone else than Refresh: no marks for what it changes, so the pass's values are stale until a
            // full pass, and the getters fall back to the on-demand caches (a new epoch) until then
            m_SubgraphForceFull = true;
            BumpEpoch();
        }
        m_Tasks.clear();
        for (uint32_t i = 0; i < uint32_t(m_BoundsArchetypes.size()); ++i)
        {
            BoundsArchetype& bounds = m_BoundsArchetypes[i];
            const Archetype& archetype = m_Impl->GetArchetypeStorage(bounds.archetype);
            if (bounds.staticRows)
            {
                // E7.4 remedy 3: a Static archetype is read again only when rows joined or left it, a leaf's content
                // was invalidated, or propagation took it since (rows that joined between the two systems: backlog
                // I5-R3); then by the rule below
                const LevelArchetype* level = LevelOf(bounds.archetype);
                const uint64_t takes = level ? level->takes : 0;
                if (archetype.entityCount == bounds.settledCount && archetype.rowMoves == bounds.settledMoves &&
                    m_ContentChanges == bounds.settledContent && takes == bounds.settledTakes)
                {
                    ++m_Counters.staticSkips;
                    continue;
                }
                bounds.settledCount = archetype.entityCount;
                bounds.settledMoves = archetype.rowMoves;
                bounds.settledContent = m_ContentChanges;
                bounds.settledTakes = takes;
            }
            for (uint32_t c = 0; c < archetype.GetChunkCount(); ++c)
            {
                if (archetype.chunks[c].count == 0)
                    continue;
                const uint64_t* versions = archetype.Versions(c);
                const bool changed = versions[bounds.world] > since || versions[bounds.localBounds] > since ||
                                     versions[archetype.AddedVersionSlot()] > since;
                // a chunk of leaves with dynamic bounds is read at every run
                if (changed || bounds.leaf != c_Absent)
                    m_Tasks.push_back(Task{ i, c, changed });
            }
        }
        if (m_Tasks.empty())
            return nvrhi::FS_OK;

        BoundsContext ctx;
        ctx.store = this;
        ctx.version = context->GetVersion();
        ctx.subgraph = m_SubgraphPrepared && !m_SubgraphFull;
        ctx.childless = SceneGraph::GetChildlessBoundsPath();
        const uint32_t taskCount = uint32_t(m_Tasks.size());
        const uint32_t grain = std::max(1u, taskCount / (WorkingThreads(jobs) * 4));
        if (!jobs || grain >= taskCount)
            RunBoundsTasks(&ctx, 0, taskCount, 0);
        else
            jobs->ParallelFor(taskCount, grain, &RunBoundsTasks, &ctx);
        m_Counters.boundsChunks += ctx.chunks.load(std::memory_order_relaxed);
        m_Counters.childlessChunks += ctx.childlessChunks.load(std::memory_order_relaxed);
        return nvrhi::FS_OK;
    }

    // ---------------------------------------------------------------------- subgraph caches (5.7)

    void SceneGraphStore::ComputeSubgraph(const SceneGraphNode* node, std::vector<CacheFrame>& stack, uint64_t epoch,
                                          bool publish, dm::box3& outBounds, SceneContentFlags& outLeafContent,
                                          SceneContentFlags& outContent)
    {
        // The node's own part: its leaf's box in world space (an empty box adds nothing) and its content flags.
        auto start = [](const SceneGraphNode* current)
        {
            CacheFrame frame{ current, current->GetFirstChild(), dm::box3::empty(), SceneContentFlags::None,
                              SceneContentFlags::None };
            // An attached node: from its row, as Refresh's pass reads it (LocalBounds, not the leaf's box now), so that
            // the two paths give the same values.
            RowRef row;
            if (current->m_Store && !current->m_Record && current->m_Store->LocateNode(current, row))
            {
                frame.bounds = NanToInfinite(current->m_Store->OwnBoundsFromLocal(row));
                frame.leafContent = current->m_Store->OwnContent(row);
            }
            else if (SceneGraphLeaf* leaf = current->GetLeaf())
            {
                const dm::box3 localBoundingBox = leaf->GetLocalBoundingBox();
                if (!localBoundingBox.isempty())
                    frame.bounds = localBoundingBox * current->GetLocalToWorldTransformFloat();
                frame.leafContent = leaf->GetContentFlags();
            }
            frame.content = frame.leafContent;
            return frame;
        };
        auto valid = [&](const SceneGraphNode* current)
        {
            return publish && current->m_CacheEpoch.load(std::memory_order_acquire) == epoch;
        };

        stack.clear();
        stack.push_back(start(node));
        for (;;)
        {
            CacheFrame& top = stack.back();
            if (SceneGraphNode* child = top.nextChild)
            {
                top.nextChild = child->GetNextSibling();
                if (valid(child))
                {
                    top.bounds |= child->m_CachedBounds;
                    top.content |= child->m_CachedSubgraphContent;
                }
                else
                    stack.push_back(start(child));
                continue;
            }

            const CacheFrame done = top;
            stack.pop_back();
            if (publish)
            {
                done.node->m_CachedBounds = done.bounds;
                done.node->m_CachedLeafContent = done.leafContent;
                done.node->m_CachedSubgraphContent = done.content;
                done.node->m_CacheEpoch.store(epoch, std::memory_order_release);
            }
            if (stack.empty())
            {
                outBounds = done.bounds;
                outLeafContent = done.leafContent;
                outContent = done.content;
                return;
            }
            stack.back().bounds |= done.bounds;
            stack.back().content |= done.content;
        }
    }

    void SceneGraphStore::ComputeSubgraphCache(const SceneGraphNode* node)
    {
        std::lock_guard<std::mutex> lock(m_CacheMutex);
        const uint64_t epoch = GetEpoch();
        if (node->m_CacheEpoch.load(std::memory_order_acquire) == epoch)
            return;
        dm::box3 bounds;
        SceneContentFlags leafContent;
        SceneContentFlags content;
        ComputeSubgraph(node, m_CacheStack, epoch, true, bounds, leafContent, content);
    }

    // ------------------------------------------------------------------- subgraph values in Refresh (5.7)
    //
    // Refresh computes the subgraph bounds and content flags of the nodes with children whose subtree changed, into the
    // nodes' caches (derived values, design 4.6), so that the getters (and so the legacy strategies' walks) read them
    // without a lock. A node's own values (its box, content flags, links, leaf) stay in its row only (design R3).
    //   Full mode (the epoch moved since the last pass: a structural change, SetLeaf or InvalidateContent; the world's
    //   structural version moved; or the Bounds system ran outside a Refresh): step 1 runs over every chunk: every
    //   chunk gets its summary and every node with children a mark.
    //   Incremental mode: the Bounds system does step 1 for the chunks it wrote (their boxes changed): the chunks'
    //   summaries, and marks on the paths from the rows with children, and from the other rows' parents, up to the
    //   roots. Nothing marked: nothing more to do.
    //   Step 2, bottom-up: the deep bucket's subtrees (a depth-first walk from each entry, whose parent is shallower),
    //   then the depth buckets from the deepest: each row whose parent is marked adds its subgraph value (its own part,
    //   plus its accumulator when it is marked, or its node's values when it has children and is not marked) to its
    //   parent's accumulator; a marked row stores its value in its node. A chunk whose rows share one parent and have
    //   no children adds its summary (the row-order union of its rows' own parts) instead of its rows.
    // Marks hold the pass's number per entity index; accumulators are merged into by CAS, after a task merged a run of
    // one parent locally. Unions across rows and chunks are taken on keys that order as the floats do (-0 below +0),
    // by min and max, so the result does not depend on the order of the rows or of the threads; it equals the old
    // sequential union except for the sign of a zero that ties with another zero, and NaN (which the old union kept
    // or dropped by position; a chunk with a NaN box has an infinite summary).

    namespace
    {
        inline uint32_t KeyOfFloat(float value) noexcept
        {
            uint32_t bits;
            std::memcpy(&bits, &value, sizeof(bits));
            return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
        }

        inline float FloatOfKey(uint32_t key) noexcept
        {
            const uint32_t bits = (key & 0x80000000u) ? (key & 0x7fffffffu) : ~key;
            float value;
            std::memcpy(&value, &bits, sizeof(value));
            return value;
        }

    }

    struct SceneGraphStore::SubgraphContext
    {
        SceneGraphStore* store;
        uint32_t pass;
        uint64_t epoch;
        uint64_t version;   // the world's version now: summaries computed in this pass
        bool full;
        const Task* tasks;
        std::atomic<bool> failed{ false };
        std::atomic<uint64_t> nodes{ 0 };
        std::atomic<uint64_t> summarised{ 0 };
    };

    bool SceneGraphStore::OwnPart(const RowRef& row, dm::box3& outBounds, SceneContentFlags& outContent) const noexcept
    {
        outBounds = OwnBounds(row);
        outContent = OwnContent(row);
        return HasChildren(row);
    }

    namespace
    {
        void KeysOf(const dm::box3& nanBox, SceneContentFlags content, uint32_t* keys, uint32_t& outContent) noexcept
        {
            const dm::box3 box = NanToInfinite(nanBox);
            keys[0] = KeyOfFloat(box.m_mins.x);
            keys[1] = KeyOfFloat(box.m_mins.y);
            keys[2] = KeyOfFloat(box.m_mins.z);
            keys[3] = KeyOfFloat(box.m_maxs.x);
            keys[4] = KeyOfFloat(box.m_maxs.y);
            keys[5] = KeyOfFloat(box.m_maxs.z);
            outContent = uint32_t(content);
        }

        void EmptyKeys(uint32_t* keys, uint32_t& content) noexcept
        {
            KeysOf(dm::box3::empty(), SceneContentFlags::None, keys, content);
        }

        void UniteKeys(uint32_t* keys, uint32_t& content, const uint32_t* other, uint32_t otherContent) noexcept
        {
            for (int i = 0; i < 3; ++i)
                keys[i] = std::min(keys[i], other[i]);
            for (int i = 3; i < 6; ++i)
                keys[i] = std::max(keys[i], other[i]);
            content |= otherContent;
        }

        dm::box3 BoxOfKeys(const uint32_t* keys) noexcept
        {
            return dm::box3(dm::float3(FloatOfKey(keys[0]), FloatOfKey(keys[1]), FloatOfKey(keys[2])),
                            dm::float3(FloatOfKey(keys[3]), FloatOfKey(keys[4]), FloatOfKey(keys[5])));
        }
    }

    // Marks `entity` and its ancestors up to the first one this pass marked already; each new mark gets an empty
    // accumulator. Threads may mark at once: the first CAS on a mark wins it.
    void SceneGraphStore::MarkPath(Entity entity, uint32_t pass) noexcept
    {
        while (entity != Entity::Null)
        {
            const uint32_t index = GetEntityIndex(entity);
            if (index >= m_SubgraphCapacity)
            {
                m_SubgraphNeedsFull.store(true, std::memory_order_relaxed);   // made during the phases
                return;
            }
            std::atomic<uint32_t>& mark = m_SubgraphMarks[index];
            uint32_t old = mark.load(std::memory_order_relaxed);
            for (;;)
            {
                if (old == pass)
                    return;   // marked already, and so are its ancestors (or will be by the thread that marked it)
                if (mark.compare_exchange_weak(old, pass, std::memory_order_relaxed))
                    break;
            }
            if (!m_SubgraphMarked.load(std::memory_order_relaxed))
                m_SubgraphMarked.store(1, std::memory_order_relaxed);
            SubgraphAccumulator& accumulator = m_SubgraphAccumulators[index];
            uint32_t keys[6];
            uint32_t content;
            EmptyKeys(keys, content);
            for (int i = 0; i < 6; ++i)
                accumulator.keys[i].store(keys[i], std::memory_order_relaxed);
            accumulator.content.store(content, std::memory_order_relaxed);

            RowRef row;
            if (!Locate(entity, row))
                return;
            const Parent* parent = static_cast<const Parent*>(Column(row, SlotParent));
            entity = parent ? parent->entity : Entity::Null;
        }
    }

    // The subgraph values of a node with children into its cache (its own values stay in its row).
    void SceneGraphStore::FinalizeSubgraph(SceneGraphNode* node, const SubgraphKeys& keys, uint64_t epoch) const noexcept
    {
        if (!node)
            return;
        node->m_CachedBounds = BoxOfKeys(keys.keys);
        node->m_CachedSubgraphContent = SceneContentFlags(keys.content);
        node->m_CacheEpoch.store(epoch, std::memory_order_release);
    }

    namespace
    {
        // Merges keys into an accumulator by CAS.
        template <typename Accumulator>
        void MergeInto(Accumulator& accumulator, const uint32_t* keys, uint32_t content) noexcept
        {
            for (int i = 0; i < 6; ++i)
            {
                std::atomic<uint32_t>& target = accumulator.keys[i];
                uint32_t current = target.load(std::memory_order_relaxed);
                if (i < 3)
                {
                    while (keys[i] < current &&
                           !target.compare_exchange_weak(current, keys[i], std::memory_order_relaxed)) { }
                }
                else
                {
                    while (keys[i] > current &&
                           !target.compare_exchange_weak(current, keys[i], std::memory_order_relaxed)) { }
                }
            }
            if (content)
                accumulator.content.fetch_or(content, std::memory_order_relaxed);
        }
    }

    // A run of merges into one parent's accumulator, kept by a task and flushed when the parent changes and at its end,
    // so that the chunks and rows of one parent (all of a flat scene's) cost one CAS merge per task.
    struct SceneGraphStore::SubgraphRun
    {
        Entity parent = Entity::Null;
        uint32_t keys[6];
        uint32_t content = 0;
        uint64_t summarised = 0;   // chunks taken whole, counted per task

        SubgraphRun() noexcept { EmptyKeys(keys, content); }

        void Add(SceneGraphStore& store, Entity target, const uint32_t* otherKeys, uint32_t otherContent) noexcept
        {
            if (target != parent)
            {
                Flush(store);
                parent = target;
            }
            UniteKeys(keys, content, otherKeys, otherContent);
        }

        void Flush(SceneGraphStore& store) noexcept
        {
            if (parent != Entity::Null)
                MergeInto(store.m_SubgraphAccumulators[GetEntityIndex(parent)], keys, content);
            parent = Entity::Null;
            EmptyKeys(keys, content);
        }
    };

    // Step 1 of a full pass, over every chunk: the chunk's summary and the marks (every node with children). The node
    // objects get nothing here: their own values stay in their rows (design R3).
    void SceneGraphStore::RunSubgraphMarks(void* context, uint32_t begin, uint32_t end, uint32_t) noexcept
    {
        SubgraphContext& ctx = *static_cast<SubgraphContext*>(context);
        SceneGraphStore& store = *ctx.store;
        for (uint32_t t = begin; t < end; ++t)
        {
            const Task& task = ctx.tasks[t];
            const Archetype& archetype = store.m_Impl->GetArchetypeStorage(task.archetype);
            const uint32_t count = archetype.chunks[task.chunk].count;
            const uint8_t linksColumn = store.ColumnIndex(archetype, task.archetype, SlotLinks);
            if (linksColumn == c_Absent || count == 0)
                continue;
            auto columnOf = [&](Slot slot) noexcept -> const void*
            {
                const uint8_t column = store.ColumnIndex(archetype, task.archetype, slot);
                return column != c_Absent ? ColumnBase(archetype, task.chunk, column) : nullptr;
            };
            const SceneLinks* links = static_cast<const SceneLinks*>(columnOf(SlotLinks));
            const Parent* parents = static_cast<const Parent*>(columnOf(SlotParent));
            const LeafObject* leaves = static_cast<const LeafObject*>(columnOf(SlotLeaf));
            const WorldBounds* worldBounds = static_cast<const WorldBounds*>(columnOf(SlotWorldBounds));
            const RenderFlags* renderFlags = static_cast<const RenderFlags*>(columnOf(SlotRenderFlags));
            const Entity* entities = archetype.Entities(task.chunk);

            if (task.archetype >= store.m_ChunkSummaries.size() ||
                task.chunk >= store.m_ChunkSummaries[task.archetype].size())
            {
                ctx.failed.store(true, std::memory_order_relaxed);   // EnsureSubgraphStorage covers every chunk
                continue;
            }
            ChunkSummary& summary = store.m_ChunkSummaries[task.archetype][task.chunk];
            summary.single = true;
            summary.parent = parents ? parents[0].entity : Entity::Null;
            dm::box3 summaryBounds = dm::box3::empty();
            uint32_t summaryContent = 0;

            Entity markedParent = Entity::Null;
            for (uint32_t r = 0; r < count; ++r)
            {
                const bool children = links[r].firstChild != Entity::Null;
                const Entity parent = parents ? parents[r].entity : Entity::Null;
                SceneGraphLeaf* leaf = leaves ? leaves[r].object : nullptr;
                const dm::box3 bounds = leaf && worldBounds ? worldBounds[r].bounds : dm::box3::empty();
                const SceneContentFlags own = renderFlags ? SceneContentFlags(renderFlags[r].contentFlags)
                                              : leaf ? SafeContentFlags(leaf) : SceneContentFlags::None;
                if (children || parent != summary.parent)
                    summary.single = false;
                summaryBounds |= NanToInfinite(bounds);
                summaryContent |= uint32_t(own);

                // the paths to mark: a row with children from itself, else from its parent (once per run of rows)
                if (children)
                    store.MarkPath(entities[r], ctx.pass);
                else if (!ctx.full && parent != markedParent)
                {
                    store.MarkPath(parent, ctx.pass);
                    markedParent = parent;
                }
            }
            uint32_t ignored;
            KeysOf(summaryBounds, SceneContentFlags::None, summary.keys.keys, ignored);
            summary.keys.content = summaryContent;
            summary.version = ctx.version;
        }
    }

    // Step 2, one chunk of a depth bucket (not the deep one): its rows' values into their parents' accumulators, its
    // marked rows' values into their nodes.
    void SceneGraphStore::SubgraphChunk(SubgraphContext& ctx, uint32_t archetypeIndex, uint32_t chunk,
                                        SubgraphRun& run) noexcept
    {
        const uint32_t pass = ctx.pass;
        auto marked = [&](Entity entity) noexcept
        {
            const uint32_t index = GetEntityIndex(entity);
            return entity != Entity::Null && index < m_SubgraphCapacity &&
                   m_SubgraphMarks[index].load(std::memory_order_relaxed) == pass;
        };
        // A chunk taken whole from its summary touches nothing but the summary.
        if (archetypeIndex >= m_ChunkSummaries.size() || chunk >= m_ChunkSummaries[archetypeIndex].size())
        {
            ctx.failed.store(true, std::memory_order_relaxed);   // EnsureSubgraphStorage covers every chunk
            return;
        }
        ChunkSummary& summary = m_ChunkSummaries[archetypeIndex][chunk];
        if (summary.version != 0 && summary.single)
        {
            if (marked(summary.parent))
            {
                run.Add(*this, summary.parent, summary.keys.keys, summary.keys.content);
                ++run.summarised;
            }
            return;
        }

        const Archetype& archetype = m_Impl->GetArchetypeStorage(archetypeIndex);
        const uint32_t count = archetype.chunks[chunk].count;
        if (count == 0)
            return;
        const uint8_t parentColumn = ColumnIndex(archetype, archetypeIndex, SlotParent);
        const uint8_t nodeColumn = ColumnIndex(archetype, archetypeIndex, SlotNode);
        const Parent* parents = parentColumn != c_Absent
            ? reinterpret_cast<const Parent*>(ColumnBase(archetype, chunk, parentColumn)) : nullptr;
        const SceneNode* nodes = nodeColumn != c_Absent
            ? reinterpret_cast<const SceneNode*>(ColumnBase(archetype, chunk, nodeColumn)) : nullptr;
        const Entity* entities = archetype.Entities(chunk);

        // The summary: one parent, no row with children. Step 1 of a full pass computes every chunk's; in an incremental
        // pass the Bounds system computed the summaries of the chunks whose boxes changed, and nothing else changes one
        // without a full pass (links, parents, leaves and content flags change with the epoch).
        if (summary.version == 0)
        {
            summary.single = true;
            summary.parent = parents ? parents[0].entity : Entity::Null;
            dm::box3 summaryBounds = dm::box3::empty();
            uint32_t summaryContent = 0;
            for (uint32_t r = 0; r < count; ++r)
            {
                const RowRef row{ &archetype, archetypeIndex, chunk, r };
                dm::box3 bounds;
                SceneContentFlags own;
                const bool children = OwnPart(row, bounds, own);
                if (children || (parents && parents[r].entity != summary.parent))
                    summary.single = false;
                summaryBounds |= NanToInfinite(bounds);
                summaryContent |= uint32_t(own);
            }
            uint32_t ignored;
            KeysOf(summaryBounds, SceneContentFlags::None, summary.keys.keys, ignored);
            summary.keys.content = summaryContent;
            summary.version = ctx.version;
        }
        if (summary.single)
        {
            if (marked(summary.parent))
            {
                run.Add(*this, summary.parent, summary.keys.keys, summary.keys.content);
                ++run.summarised;
            }
            return;
        }

        auto columnOf = [&](Slot slot) noexcept -> const void*
        {
            const uint8_t column = ColumnIndex(archetype, archetypeIndex, slot);
            return column != c_Absent ? ColumnBase(archetype, chunk, column) : nullptr;
        };
        const SceneLinks* links = static_cast<const SceneLinks*>(columnOf(SlotLinks));
        const LeafObject* leaves = static_cast<const LeafObject*>(columnOf(SlotLeaf));
        const WorldBounds* worldBounds = static_cast<const WorldBounds*>(columnOf(SlotWorldBounds));
        const RenderFlags* renderFlags = static_cast<const RenderFlags*>(columnOf(SlotRenderFlags));
        uint64_t computed = 0;
        for (uint32_t r = 0; r < count; ++r)
        {
            const Entity parent = parents ? parents[r].entity : Entity::Null;
            const bool parentMarked = marked(parent);
            const bool rowMarked = marked(entities[r]);
            if (!parentMarked && !rowMarked)
                continue;
            const SceneGraphLeaf* leaf = leaves ? leaves[r].object : nullptr;
            const dm::box3 bounds = leaf && worldBounds ? worldBounds[r].bounds : dm::box3::empty();
            const SceneContentFlags own = renderFlags ? SceneContentFlags(renderFlags[r].contentFlags)
                                          : leaf ? SafeContentFlags(leaf) : SceneContentFlags::None;
            const bool children = links && links[r].firstChild != Entity::Null;
            SubgraphKeys value;
            KeysOf(bounds, own, value.keys, value.content);
            SceneGraphNode* node = nodes ? nodes[r].object : nullptr;
            if (children)
            {
                if (rowMarked)
                {
                    const SubgraphAccumulator& accumulator = m_SubgraphAccumulators[GetEntityIndex(entities[r])];
                    uint32_t keys[6];
                    for (int i = 0; i < 6; ++i)
                        keys[i] = accumulator.keys[i].load(std::memory_order_relaxed);
                    UniteKeys(value.keys, value.content, keys, accumulator.content.load(std::memory_order_relaxed));
                    FinalizeSubgraph(node, value, ctx.epoch);
                    ++computed;
                }
                else if (node)
                {
                    // a subtree that did not change: the values the node keeps from an earlier pass
                    KeysOf(node->m_CachedBounds, node->m_CachedSubgraphContent, value.keys, value.content);
                }
            }
            if (parentMarked)
                run.Add(*this, parent, value.keys, value.content);
        }
        if (computed)
            ctx.nodes.fetch_add(computed, std::memory_order_relaxed);
    }

    void SceneGraphStore::RunSubgraphChunks(void* context, uint32_t begin, uint32_t end, uint32_t) noexcept
    {
        SubgraphContext& ctx = *static_cast<SubgraphContext*>(context);
        SubgraphRun run;
        for (uint32_t t = begin; t < end; ++t)
            ctx.store->SubgraphChunk(ctx, ctx.tasks[t].archetype, ctx.tasks[t].chunk, run);
        run.Flush(*ctx.store);
        if (run.summarised)
            ctx.summarised.fetch_add(run.summarised, std::memory_order_relaxed);
    }

    // The deep bucket: the subtree of each entry (a row whose parent is shallower, or a root) depth first, children in
    // sibling order; every marked node with children in it gets its values, and the entry's value goes to its parent.
    void SceneGraphStore::RunSubgraphDeep(void* context, uint32_t begin, uint32_t end, uint32_t worker) noexcept
    {
        SubgraphContext& ctx = *static_cast<SubgraphContext*>(context);
        SceneGraphStore& store = *ctx.store;
        std::vector<SubgraphFrame>& stack = store.m_SubgraphStacks[worker];
        const uint32_t pass = ctx.pass;
        auto marked = [&](Entity entity) noexcept
        {
            const uint32_t index = GetEntityIndex(entity);
            return entity != Entity::Null && index < store.m_SubgraphCapacity &&
                   store.m_SubgraphMarks[index].load(std::memory_order_relaxed) == pass;
        };
        // The value of `entity` when it needs no walk (no children, or not marked); false when it must be walked.
        auto leafValue = [&](Entity entity, SubgraphKeys& value, SceneContentFlags& own, Entity& firstChild,
                             Entity& nextSibling) noexcept
        {
            RowRef row;
            if (!store.Locate(entity, row))
            {
                EmptyKeys(value.keys, value.content);
                own = SceneContentFlags::None;
                firstChild = nextSibling = Entity::Null;
                return true;
            }
            dm::box3 bounds;
            const bool children = store.OwnPart(row, bounds, own);
            KeysOf(bounds, own, value.keys, value.content);
            const SceneLinks* links = static_cast<const SceneLinks*>(store.Column(row, SlotLinks));
            firstChild = links ? links->firstChild : Entity::Null;
            nextSibling = links ? links->nextSibling : Entity::Null;
            if (!children)
                return true;
            if (!marked(entity))
            {
                const SceneNode* node = static_cast<const SceneNode*>(store.Column(row, SlotNode));
                if (node && node->object)
                    KeysOf(node->object->m_CachedBounds, node->object->m_CachedSubgraphContent, value.keys,
                           value.content);
                return true;
            }
            return false;
        };

        uint64_t computed = 0;
        for (uint32_t e = begin; e < end; ++e)
        {
            const Entity entry = store.m_SubgraphEntries[e];
            SubgraphKeys value;
            SceneContentFlags own;
            Entity firstChild, nextSibling;
            if (!leafValue(entry, value, own, firstChild, nextSibling))
            {
                stack.clear();
                try
                {
                    stack.push_back(SubgraphFrame{ entry, firstChild, value, own });
                    while (!stack.empty())
                    {
                        SubgraphFrame& top = stack.back();
                        if (top.nextChild != Entity::Null)
                        {
                            const Entity child = top.nextChild;
                            SubgraphKeys childValue;
                            SceneContentFlags childOwn;
                            Entity childFirst, childNext;
                            const bool done = leafValue(child, childValue, childOwn, childFirst, childNext);
                            top.nextChild = childNext;
                            if (done)
                                UniteKeys(top.keys.keys, top.keys.content, childValue.keys, childValue.content);
                            else
                                stack.push_back(SubgraphFrame{ child, childFirst, childValue, childOwn });
                            continue;
                        }
                        const SubgraphFrame finished = top;
                        stack.pop_back();
                        store.FinalizeSubgraph(store.NodeOf(finished.entity), finished.keys, ctx.epoch);
                        ++computed;
                        if (stack.empty())
                            value = finished.keys;
                        else
                            UniteKeys(stack.back().keys.keys, stack.back().keys.content, finished.keys.keys,
                                      finished.keys.content);
                    }
                }
                catch (...)
                {
                    ctx.failed.store(true, std::memory_order_relaxed);
                    break;
                }
            }
            // the entry's parent is shallower: its accumulator
            RowRef row;
            if (store.Locate(entry, row))
            {
                const Parent* parent = static_cast<const Parent*>(store.Column(row, SlotParent));
                if (parent && marked(parent->entity))
                    MergeInto(store.m_SubgraphAccumulators[GetEntityIndex(parent->entity)], value.keys, value.content);
            }
        }
        if (computed)
            ctx.nodes.fetch_add(computed, std::memory_order_relaxed);
    }

    bool SceneGraphStore::EnsureSubgraphStorage() noexcept
    {
        if (s_FailSubgraphCountdown.load(std::memory_order_relaxed) >= 0 &&
            s_FailSubgraphCountdown.fetch_sub(1, std::memory_order_relaxed) == 0)
            return false;   // test seam: as out of memory

        // Marks and accumulators by entity index.
        const size_t entities = m_Impl->GetEntityTableSize();
        if (entities > m_SubgraphCapacity)
        {
            const size_t capacity = std::max(entities + entities / 4 + 1024, m_SubgraphCapacity * 2);
            std::unique_ptr<std::atomic<uint32_t>[]> marks(new (std::nothrow) std::atomic<uint32_t>[capacity]);
            std::unique_ptr<SubgraphAccumulator[]> accumulators(new (std::nothrow) SubgraphAccumulator[capacity]);
            if (!marks || !accumulators)
                return false;
            for (size_t i = 0; i < capacity; ++i)
                marks[i].store(0, std::memory_order_relaxed);
            m_SubgraphMarks = std::move(marks);
            m_SubgraphAccumulators = std::move(accumulators);
            m_SubgraphCapacity = capacity;
            m_SubgraphPass = 0;
        }
        // A summary per chunk of every archetype; a stack per thread.
        try
        {
            const uint32_t archetypeCount = m_Impl->GetArchetypeStorageCount();
            if (m_ChunkSummaries.size() < archetypeCount)
                m_ChunkSummaries.resize(archetypeCount);
            for (uint32_t a = 0; a < archetypeCount; ++a)
            {
                const uint32_t chunks = m_Impl->GetArchetypeStorage(a).GetChunkCount();
                if (m_ChunkSummaries[a].size() < chunks)
                    m_ChunkSummaries[a].resize(chunks);
            }
            const size_t threads = m_Jobs ? m_Jobs->GetThreadCount() : 1;
            if (m_SubgraphStacks.size() < threads)
                m_SubgraphStacks.resize(threads);
        }
        catch (...)
        {
            return false;
        }
        return true;
    }

    // A new pass number (marks need no clearing), the mode, and the storage for the world before the phases. A phase
    // that changes the structure (commands played back before it, an exclusive system) is caught by UpdateSubgraphs.
    void SceneGraphStore::PrepareSubgraphs() noexcept
    {
        m_SubgraphPrepared = false;
        // Full when the epoch moved (a structural change through the API, SetLeaf, InvalidateContent) or the world's
        // structure did otherwise (the first transform of a node moves its row to another archetype): rows that joined
        // a chunk make its summary stale.
        m_PreparedStructuralVersion = m_World->GetStructuralVersion();
        m_SubgraphFull = m_SubgraphForceFull || m_SubgraphEpoch.load(std::memory_order_relaxed) != GetEpoch() ||
                         m_PreparedStructuralVersion != m_SubgraphStructuralVersion;
        m_SubgraphMarked.store(0, std::memory_order_relaxed);
        m_SubgraphNeedsFull.store(false, std::memory_order_relaxed);
        if (!EnsureSubgraphStorage())
            return;   // the getters fall back to the on-demand caches
        if (++m_SubgraphPass == 0)
        {
            for (size_t i = 0; i < m_SubgraphCapacity; ++i)
                m_SubgraphMarks[i].store(0, std::memory_order_relaxed);
            m_SubgraphPass = 1;
        }
        m_SubgraphPrepared = true;
    }

    bool SceneGraphStore::HasDynamicBoundsRows() const noexcept
    {
        for (const BoundsArchetype& bounds : m_BoundsArchetypes)
        {
            if (bounds.leaf == c_Absent)
                continue;
            const Archetype& archetype = m_Impl->GetArchetypeStorage(bounds.archetype);
            for (uint32_t c = 0; c < archetype.GetChunkCount(); ++c)
            {
                if (archetype.chunks[c].count != 0)
                    return true;
            }
        }
        return false;
    }

    // An incremental pass's step 1 for a chunk the Bounds system wrote (its boxes changed): the summary (its bounds the
    // chunk's union, ChunkWorldBounds, which the Bounds system just made in row order), and the paths to mark, from a
    // row with children itself, else from its parent (once per run of rows of one parent).
    bool SceneGraphStore::SubgraphFromBounds(const Archetype& archetype, uint32_t archetypeIndex, uint32_t chunk,
                                             const dm::box3& chunkBounds, bool childless) noexcept
    {
        const uint32_t count = archetype.chunks[chunk].count;
        const uint8_t linksColumn = ColumnIndex(archetype, archetypeIndex, SlotLinks);
        if (linksColumn == c_Absent || count == 0)
            return false;
        // E7.4 remedy 4: a chunk whose rows have one parent and no children, as its summary says (links, parents,
        // leaves and content flags change only with a full pass, which computes every summary): the loop below would
        // find the same parent and content and mark the parent once; only the box and the version change.
        if (childless && archetypeIndex < m_ChunkSummaries.size() && chunk < m_ChunkSummaries[archetypeIndex].size())
        {
            ChunkSummary& summary = m_ChunkSummaries[archetypeIndex][chunk];
            if (summary.version != 0 && summary.single)
            {
                if (summary.parent != Entity::Null)
                    MarkPath(summary.parent, m_SubgraphPass);
                uint32_t ignored;
                KeysOf(chunkBounds, SceneContentFlags::None, summary.keys.keys, ignored);
                summary.version = m_World->GetVersion();
                return true;
            }
        }
        const SceneLinks* links = reinterpret_cast<const SceneLinks*>(ColumnBase(archetype, chunk, linksColumn));
        const uint8_t parentColumn = ColumnIndex(archetype, archetypeIndex, SlotParent);
        const Parent* parents = parentColumn != c_Absent
            ? reinterpret_cast<const Parent*>(ColumnBase(archetype, chunk, parentColumn)) : nullptr;
        const Entity* entities = archetype.Entities(chunk);
        auto columnOf = [&](Slot slot) noexcept -> const void*
        {
            const uint8_t column = ColumnIndex(archetype, archetypeIndex, slot);
            return column != c_Absent ? ColumnBase(archetype, chunk, column) : nullptr;
        };
        const LeafObject* leaves = static_cast<const LeafObject*>(columnOf(SlotLeaf));
        const RenderFlags* renderFlags = static_cast<const RenderFlags*>(columnOf(SlotRenderFlags));
        if (archetypeIndex >= m_ChunkSummaries.size() || chunk >= m_ChunkSummaries[archetypeIndex].size())
        {
            m_SubgraphNeedsFull.store(true, std::memory_order_relaxed);   // an archetype or chunk made during the phases
            return false;
        }
        ChunkSummary& summary = m_ChunkSummaries[archetypeIndex][chunk];
        summary.single = true;
        summary.parent = parents ? parents[0].entity : Entity::Null;
        uint32_t summaryContent = 0;
        Entity markedParent = Entity::Null;
        for (uint32_t r = 0; r < count; ++r)
        {
            const bool children = links[r].firstChild != Entity::Null;
            const Entity parent = parents ? parents[r].entity : Entity::Null;
            if (children || parent != summary.parent)
                summary.single = false;
            const SceneGraphLeaf* leaf = leaves ? leaves[r].object : nullptr;
            summaryContent |= renderFlags ? renderFlags[r].contentFlags
                              : leaf ? uint32_t(SafeContentFlags(leaf)) : 0u;
            if (children)
                MarkPath(entities[r], m_SubgraphPass);
            else if (parent != markedParent)
            {
                MarkPath(parent, m_SubgraphPass);
                markedParent = parent;
            }
        }
        uint32_t ignored;
        KeysOf(chunkBounds, SceneContentFlags::None, summary.keys.keys, ignored);
        summary.keys.content = summaryContent;
        summary.version = m_World->GetVersion();
        return false;
    }

    void SceneGraphStore::UpdateSubgraphs() noexcept
    {
        IJobSystem* jobs = m_Jobs;
        const uint32_t threads = WorkingThreads(jobs);
        if (!m_SubgraphPrepared)
            return;   // out of memory before the phases: the getters fall back to the on-demand caches
        m_SubgraphPrepared = false;
        bool full = m_SubgraphFull;
        if (m_World->GetStructuralVersion() != m_PreparedStructuralVersion ||
            m_SubgraphNeedsFull.load(std::memory_order_relaxed))
        {
            // A phase changed the structure (commands played back before it, an exclusive system) or made rows the
            // storage did not cover: the Bounds system's marks and summaries may be incomplete. Again, in full, over
            // the world as it is now.
            SyncArchetypes();
            UpdateSystemLists();
            if (!EnsureSubgraphStorage())
                return;
            if (++m_SubgraphPass == 0)
            {
                for (size_t i = 0; i < m_SubgraphCapacity; ++i)
                    m_SubgraphMarks[i].store(0, std::memory_order_relaxed);
                m_SubgraphPass = 1;
            }
            full = true;
            s_SubgraphRedos.fetch_add(1);
        }
        // 5: a full pass moves the epoch, so that what the lazy path stamped before (on nodes without children too)
        // is not taken for current
        if (full)
            BumpEpoch();
        const uint64_t epoch = GetEpoch();

        SubgraphContext ctx;
        ctx.store = this;
        ctx.pass = m_SubgraphPass;
        ctx.epoch = epoch;
        ctx.version = m_World->GetVersion();
        ctx.full = full;

        try
        {
            // 1. Marks and summaries, in full mode, over every chunk (in incremental mode the Bounds system
            // did it for the chunks it wrote).
            if (full)
            {
                m_SubgraphTasks.clear();
                uint32_t rows = 0;
                for (uint32_t d = 0; d <= c_MaxHierarchyDepth; ++d)
                {
                    for (const LevelArchetype& level : m_Levels[d])
                    {
                        if (level.runtime)
                            continue;   // runtime-only rows (E7.1): in no node's subgraph
                        const Archetype& archetype = m_Impl->GetArchetypeStorage(level.archetype);
                        for (uint32_t c = 0; c < archetype.GetChunkCount(); ++c)
                        {
                            if (archetype.chunks[c].count == 0)
                                continue;
                            m_SubgraphTasks.push_back(Task{ level.archetype, c, true });
                            rows += archetype.chunks[c].count;
                        }
                    }
                }
                ctx.tasks = m_SubgraphTasks.data();
                const uint32_t markTasks = uint32_t(m_SubgraphTasks.size());
                if (jobs && rows >= s_SerialSubgraphRows.load(std::memory_order_relaxed) && markTasks > 1)
                    jobs->ParallelFor(markTasks, std::max(1u, markTasks / (threads * 4)), &RunSubgraphMarks, &ctx);
                else if (markTasks)
                    RunSubgraphMarks(&ctx, 0, markTasks, 0);
            }
            if (m_SubgraphMarked.load(std::memory_order_relaxed) == 0)
            {
                // nothing marked: every node keeps its values
                m_SubgraphForceFull = false;
                m_SubgraphStructuralVersion = m_World->GetStructuralVersion();   // the structure this pass saw
                m_SubgraphEpoch.store(epoch, std::memory_order_release);
                ++m_Counters.subgraphPasses;
                m_Counters.fullSubgraphPasses += full ? 1 : 0;
                return;
            }

            // 2a. The deep bucket's entries: rows whose parent is in a shallower bucket (or roots), when they or their
            // parents are marked.
            m_SubgraphEntries.clear();
            for (const LevelArchetype& level : m_Levels[c_MaxHierarchyDepth])
            {
                const Archetype& archetype = m_Impl->GetArchetypeStorage(level.archetype);
                for (uint32_t c = 0; c < archetype.GetChunkCount(); ++c)
                {
                    const uint32_t count = archetype.chunks[c].count;
                    const Entity* rowEntities = archetype.Entities(c);
                    const Parent* parents = level.parent != c_Absent
                        ? reinterpret_cast<const Parent*>(ColumnBase(archetype, c, level.parent)) : nullptr;
                    for (uint32_t r = 0; r < count; ++r)
                    {
                        const Entity parent = parents ? parents[r].entity : Entity::Null;
                        auto isMarked = [&](Entity entity)
                        {
                            const uint32_t index = GetEntityIndex(entity);
                            return entity != Entity::Null && index < m_SubgraphCapacity &&
                                   m_SubgraphMarks[index].load(std::memory_order_relaxed) == ctx.pass;
                        };
                        if (!isMarked(parent) && !isMarked(rowEntities[r]))
                            continue;
                        bool entry = parent == Entity::Null;
                        if (!entry)
                        {
                            const EntityRecord* record = m_Impl->FindAlive(parent);
                            entry = record && record->archetype < m_ArchetypeLevel.size() &&
                                    m_ArchetypeLevel[record->archetype] != c_MaxHierarchyDepth;
                        }
                        if (entry)
                            m_SubgraphEntries.push_back(rowEntities[r]);
                    }
                }
            }
        }
        catch (...)
        {
            return;   // out of memory: the getters fall back to the on-demand caches
        }
        const uint32_t entryCount = uint32_t(m_SubgraphEntries.size());
        if (jobs && entryCount > 1)
            jobs->ParallelFor(entryCount, 1, &RunSubgraphDeep, &ctx);
        else if (entryCount)
            RunSubgraphDeep(&ctx, 0, entryCount, 0);

        // 2b. The depth buckets from the deepest shallow one up to the roots.
        for (int32_t d = int32_t(c_MaxHierarchyDepth) - 1; d >= 0 && !ctx.failed.load(); --d)
        {
            m_SubgraphTasks.clear();
            uint32_t rows = 0;
            try
            {
                for (const LevelArchetype& level : m_Levels[d])
                {
                    if (level.runtime)
                        continue;   // as in step 1
                    const Archetype& archetype = m_Impl->GetArchetypeStorage(level.archetype);
                    for (uint32_t c = 0; c < archetype.GetChunkCount(); ++c)
                    {
                        if (archetype.chunks[c].count == 0)
                            continue;
                        m_SubgraphTasks.push_back(Task{ level.archetype, c, true });
                        rows += archetype.chunks[c].count;
                    }
                }
            }
            catch (...)
            {
                return;
            }
            const uint32_t taskCount = uint32_t(m_SubgraphTasks.size());
            ctx.tasks = m_SubgraphTasks.data();
            if (jobs && rows >= s_SerialSubgraphRows.load(std::memory_order_relaxed) && taskCount > 1)
                jobs->ParallelFor(taskCount, std::max(1u, taskCount / (threads * 4)), &RunSubgraphChunks, &ctx);
            else if (taskCount)
                RunSubgraphChunks(&ctx, 0, taskCount, 0);
        }
        if (ctx.failed.load())
            return;

        m_SubgraphForceFull = false;
        m_SubgraphStructuralVersion = m_World->GetStructuralVersion();   // the structure this pass saw
        m_SubgraphEpoch.store(epoch, std::memory_order_release);
        ++m_Counters.subgraphPasses;
        m_Counters.fullSubgraphPasses += full ? 1 : 0;
        m_Counters.subgraphNodes += ctx.nodes.load(std::memory_order_relaxed);
        m_Counters.summarisedChunks += ctx.summarised.load(std::memory_order_relaxed);
    }

    // ---------------------------------------------------------------------------- row hints (5.2)

    void SceneGraphStore::RunRowHints(void* context, uint32_t begin, uint32_t end, uint32_t) noexcept
    {
        SceneGraphStore& store = *static_cast<SceneGraphStore*>(context);
        for (uint32_t t = begin; t < end; ++t)
        {
            const Task& task = store.m_SubgraphTasks[t];
            const Archetype& archetype = store.m_Impl->GetArchetypeStorage(task.archetype);
            const uint8_t nodeColumn = store.ColumnIndex(archetype, task.archetype, SlotNode);
            const SceneNode* nodes = reinterpret_cast<const SceneNode*>(ColumnBase(archetype, task.chunk, nodeColumn));
            const uint32_t count = archetype.chunks[task.chunk].count;
            for (uint32_t r = 0; r < count; ++r)
            {
                if (nodes[r].object)
                    RememberRow(nodes[r].object, RowRef{ &archetype, task.archetype, task.chunk, r });
            }
        }
    }

    // The hints of an archetype go stale for all its rows when one row moves (Archetype::rowMoves): after structural
    // changes, Refresh writes them again for the archetypes whose rows moved, so that the getters keep their fast path.
    void SceneGraphStore::RefreshRowHints() noexcept
    {
        const uint32_t archetypeCount = m_Impl->GetArchetypeStorageCount();
        uint32_t rows = 0;
        try
        {
            if (m_HintedMoves.size() < archetypeCount)
                m_HintedMoves.resize(archetypeCount, ~uint64_t(0));
            m_SubgraphTasks.clear();
            for (uint32_t a = 0; a < archetypeCount; ++a)
            {
                const Archetype& archetype = m_Impl->GetArchetypeStorage(a);
                if (m_HintedMoves[a] == archetype.rowMoves || ColumnIndex(archetype, a, SlotNode) == c_Absent)
                    continue;
                m_HintedMoves[a] = archetype.rowMoves;
                for (uint32_t c = 0; c < archetype.GetChunkCount(); ++c)
                {
                    if (archetype.chunks[c].count == 0)
                        continue;
                    m_SubgraphTasks.push_back(Task{ a, c, true });
                    rows += archetype.chunks[c].count;
                }
            }
        }
        catch (...)
        {
            return;   // the hints stay stale: the getters take the entity table
        }
        const uint32_t taskCount = uint32_t(m_SubgraphTasks.size());
        if (m_Jobs && rows >= s_SerialSubgraphRows.load(std::memory_order_relaxed) && taskCount > 1)
            m_Jobs->ParallelFor(taskCount, std::max(1u, taskCount / (WorkingThreads(m_Jobs) * 4)), &RunRowHints, this);
        else if (taskCount)
            RunRowHints(this, 0, taskCount, 0);
    }
}
