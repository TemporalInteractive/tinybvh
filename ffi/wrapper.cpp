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
}