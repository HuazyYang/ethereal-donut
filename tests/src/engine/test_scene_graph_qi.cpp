// The scene graph leaves answer QueryInterface for their own class ID and for the class ID of every
// ancestor (nvrhi ADR 0006 / 0007), so that donut::query_cast behaves like the dynamic_cast it replaces:
// accepts the class itself and its bases, refuses siblings and derived classes.

#include <donut/engine/SceneGraph.h>
#include <donut/core/query_cast.h>
#include <donut/tests/utils.h>

#include <type_traits>

using namespace donut;
using namespace donut::engine;

namespace
{
    // Does `leaf` (really an Actual) answer T's class ID exactly when T is Actual or one of its bases?
    template <typename Actual, typename T>
    void check_answers(SceneGraphLeaf* leaf)
    {
        constexpr bool expected = std::is_base_of_v<T, Actual>;

        T* cast = query_cast<T>(leaf);
        CHECK((cast != nullptr) == expected);

        // The same answer through the full QueryInterface: a pointer to the same object, one reference added.
        nvrhi::AutoPtr<T> queried;
        const bool found = NVRHI_SUCCEEDED(leaf->QueryInterface(NVRHI_IID_PPV_ARGS(&queried)));
        CHECK(found == expected);
        if (expected)
        {
            CHECK(queried.Get() == cast);
            CHECK(static_cast<void*>(queried.Get()) == static_cast<void*>(leaf));
        }
    }

    template <typename Actual>
    void check_chain(Actual* actual)
    {
        SceneGraphLeaf* leaf = actual;

        // The leaf is alive and its count is unchanged by the queries.
        leaf->AddRef();
        const nvrhi::FLONG before = leaf->Release();

        check_answers<Actual, SceneGraphLeaf>(leaf);
        check_answers<Actual, MeshInstance>(leaf);
        check_answers<Actual, SkinnedMeshInstance>(leaf);
        check_answers<Actual, SkinnedMeshReference>(leaf);
        check_answers<Actual, SceneCamera>(leaf);
        check_answers<Actual, PerspectiveCamera>(leaf);
        check_answers<Actual, OrthographicCamera>(leaf);
        check_answers<Actual, Light>(leaf);
        check_answers<Actual, DirectionalLight>(leaf);
        check_answers<Actual, SpotLight>(leaf);
        check_answers<Actual, PointLight>(leaf);
        check_answers<Actual, SceneGraphAnimation>(leaf);

        // IWeakReferenceSource is answered by every leaf, and a null ppv only reports support.
        CHECK(leaf->QueryInterface(nvrhi::uuid_of<nvrhi::IWeakReferenceSource>(), nullptr) == nvrhi::FS_OK);
        CHECK(leaf->QueryInterface(nvrhi::uuid_of<nvrhi::IDataBlob>(), nullptr) == nvrhi::FE_NOINTERFACE);

        leaf->AddRef();
        CHECK(leaf->Release() == before);
    }

    template <typename Actual>
    void check_chain(const nvrhi::AutoPtr<Actual>& actual)
    {
        CHECK(actual);
        check_chain(actual.Get());
    }
}

void test_leaf_chains()
{
    auto factory = MAKE_RC_OBJ_PTR(SceneTypeFactory);

    check_chain<DirectionalLight>(MAKE_RC_OBJ_PTR(DirectionalLight));
    check_chain<PointLight>(MAKE_RC_OBJ_PTR(PointLight));
    check_chain<SpotLight>(MAKE_RC_OBJ_PTR(SpotLight));
    check_chain<PerspectiveCamera>(MAKE_RC_OBJ_PTR(PerspectiveCamera));
    check_chain<OrthographicCamera>(MAKE_RC_OBJ_PTR(OrthographicCamera));
    check_chain<SceneGraphAnimation>(MAKE_RC_OBJ_PTR(SceneGraphAnimation));

    auto mesh = factory->CreateMesh();
    check_chain(factory->CreateMeshInstance(mesh));
    check_chain(factory->CreateSkinnedMeshInstance(factory, mesh));
    check_chain<SkinnedMeshReference>(
        MAKE_RC_OBJ_PTR(SkinnedMeshReference, static_cast<SkinnedMeshInstance*>(nullptr)));
}

void test_query_cast_null()
{
    CHECK(query_cast<Light>(static_cast<SceneGraphLeaf*>(nullptr)) == nullptr);
}

int main(int, char**)
{
    try
    {
        test_leaf_chains();
        test_query_cast_null();
    }
    catch (const std::runtime_error& err)
    {
        fprintf(stderr, "%s", err.what());
        return 1;
    }
    return 0;
}
