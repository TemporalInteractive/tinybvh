#define TINYBVH_IMPLEMENTATION
#include "tinybvh/tiny_bvh.h"

namespace tinybvh {
    Ray Ray_new(bvhvec3 origin, bvhvec3 direction)
    {
        return Ray(origin, direction);
    }

    BVH* BVH_new()
    {
        return new BVH{};
    }

    BVH_SoA* BVH_SoA_new()
    {
        return new BVH_SoA{};
    }

    BVH4_CPU* BVH4_CPU_new()
    {
        return new BVH4_CPU{};
    }
}