#include "tinybvh/tiny_bvh.h"

#include <array>
#include <memory>

// Constructors cannot be called from rust, this requires a less elegant workaround
namespace tinybvh {
    Ray Ray_new(bvhvec3 origin, bvhvec3 direction);

    BVH* BVH_new();
}