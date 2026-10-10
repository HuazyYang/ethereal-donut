#pragma once

#include <nvrhi/core/foundation.h>
#include <type_traits>

namespace donut
{
    // Replaces dynamic_cast on nvrhi objects (nvrhi ADR 0006): the object is asked for T's class ID
    // (NVRHI_CLASS_CLSID) through QueryInterface, and the answer is a downcast of the same pointer.
    // Returns null when `object` is null or does not answer T's class ID, like a failed dynamic_cast.
    //
    // The query passes no pointer, so it neither adds a reference nor needs `object` to be alive for
    // longer than the call: the caller's reference keeps the result valid, as it did for dynamic_cast.
    // T must derive from U non-virtually, and every class between them that adds a class ID must route
    // to its parent's table (NVRHI_IMPLEMENTS_ROUTE_PARENT), so that a derived class answers its bases.
    template <typename T, typename U>
    [[nodiscard]] T* query_cast(U* object) noexcept
    {
        static_assert(std::is_base_of_v<nvrhi::IObject, T> && std::is_base_of_v<U, T>,
            "query_cast: T must be an nvrhi::IObject type derived from U");

        if (!object || NVRHI_FAILED(object->QueryInterface(nvrhi::uuid_of<T>(), nullptr)))
            return nullptr;

        return static_cast<T*>(object);
    }
}
