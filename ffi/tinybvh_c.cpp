/*
tinybvh_c.cpp - implementation of the plain C interface declared in tinybvh_c.h on top of tinybvh.

This is the single translation unit that defines TINYBVH_IMPLEMENTATION. Every opaque handle is a
C++ struct that contains the tinybvh object by value; the C structs mirror the tinybvh types bit for
bit, which the static assertions below verify, so rays, intersections, instances and settings are
converted with reinterpret_cast and never copied.
*/

#define TINYBVH_IMPLEMENTATION
#ifdef _MSC_VER
// BVH<double>::Save stores the vertex count in 32 bits; the narrowing is tinybvh's, not ours.
#pragma warning( push )
#pragma warning( disable: 4244 )
#endif
#include "tinybvh/tiny_bvh.h"
#ifdef _MSC_VER
#pragma warning( pop )
#endif
#include "tinybvh_c.h"

#include <cstddef>
#include <cstring>
#include <new>

namespace tb = tinybvh;

/* ------------------------------------------------------------------------------------------------
   Layout verification
   ------------------------------------------------------------------------------------------------ */

static_assert( sizeof( tbvh_vec3 ) == sizeof( tb::bvhvec3 ), "tbvh_vec3 size" );
static_assert( sizeof( tbvh_vec4 ) == sizeof( tb::bvhvec4 ), "tbvh_vec4 size" );
static_assert( alignof( tbvh_vec4 ) == alignof( tb::bvhvec4 ), "tbvh_vec4 alignment" );
static_assert( sizeof( tbvh_dbl3 ) == sizeof( tb::bvhdbl3 ), "tbvh_dbl3 size" );
static_assert( sizeof( tbvh_mat4 ) == sizeof( tb::bvhmat4 ), "tbvh_mat4 size" );
static_assert( sizeof( tbvh_dblmat4 ) == sizeof( tb::bvhdblmat4 ), "tbvh_dblmat4 size" );
static_assert( sizeof( tbvh_intersection ) == sizeof( tb::Intersection ), "tbvh_intersection size" );
static_assert( sizeof( tbvh_intersection_ex ) == sizeof( tb::IntersectionEx ), "tbvh_intersection_ex size" );
static_assert( sizeof( tbvh_ray ) == sizeof( tb::Ray ), "tbvh_ray size" );
static_assert( alignof( tbvh_ray ) == alignof( tb::Ray ), "tbvh_ray alignment" );
static_assert( sizeof( tbvh_ray_ex ) == sizeof( tb::RayEx ), "tbvh_ray_ex size" );
static_assert( alignof( tbvh_ray_ex ) == alignof( tb::RayEx ), "tbvh_ray_ex alignment" );
static_assert( sizeof( tbvh_blas_instance ) == sizeof( tb::BLASInstance ), "tbvh_blas_instance size" );
static_assert( alignof( tbvh_blas_instance ) == alignof( tb::BLASInstance ), "tbvh_blas_instance alignment" );
static_assert( sizeof( tbvh_blas_instance_ex ) == sizeof( tb::BLASInstanceEx ), "tbvh_blas_instance_ex size" );
static_assert( alignof( tbvh_blas_instance_ex ) == alignof( tb::BLASInstanceEx ), "tbvh_blas_instance_ex alignment" );
static_assert( sizeof( tbvh_build_settings ) == sizeof( tb::BVHBuildSettings ), "tbvh_build_settings size" );
static_assert( sizeof( tbvh_bvh_node ) == sizeof( tb::BVH::BVHNode ), "tbvh_bvh_node size" );
static_assert( sizeof( tbvh_bvh_node_ex ) == sizeof( tb::BVH_Double::BVHNode ), "tbvh_bvh_node_ex size" );
static_assert( sizeof( tbvh_bvh_verbose_node ) == sizeof( tb::BVH_Verbose::BVHNode ), "tbvh_bvh_verbose_node size" );
static_assert( sizeof( tbvh_bvh_gpu_node ) == sizeof( tb::BVH_GPU::BVHNode ), "tbvh_bvh_gpu_node size" );
static_assert( sizeof( tbvh_mbvh4_node ) == sizeof( tb::MBVH<4>::MBVHNode ), "tbvh_mbvh4_node size" );
static_assert( sizeof( tbvh_mbvh8_node ) == sizeof( tb::MBVH<8>::MBVHNode ), "tbvh_mbvh8_node size" );

static_assert( (uint32_t)TBVH_LAYOUT_UNDEFINED == (uint32_t)tb::LAYOUT_UNDEFINED, "layout enum" );
static_assert( (uint32_t)TBVH_LAYOUT_BVH == (uint32_t)tb::LAYOUT_BVH, "layout enum" );
static_assert( (uint32_t)TBVH_LAYOUT_BVH_VERBOSE == (uint32_t)tb::LAYOUT_BVH_VERBOSE, "layout enum" );
static_assert( (uint32_t)TBVH_LAYOUT_BVH_GPU == (uint32_t)tb::LAYOUT_BVH_GPU, "layout enum" );
static_assert( (uint32_t)TBVH_LAYOUT_MBVH == (uint32_t)tb::LAYOUT_MBVH, "layout enum" );
static_assert( (uint32_t)TBVH_LAYOUT_BVH4_CPU == (uint32_t)tb::LAYOUT_BVH4_CPU, "layout enum" );
static_assert( (uint32_t)TBVH_LAYOUT_BVH4_GPU == (uint32_t)tb::LAYOUT_BVH4_GPU, "layout enum" );
static_assert( (uint32_t)TBVH_LAYOUT_CWBVH == (uint32_t)tb::LAYOUT_CWBVH, "layout enum" );
static_assert( (uint32_t)TBVH_LAYOUT_BVH8_SIMD == (uint32_t)tb::LAYOUT_BVH8_SIMD, "layout enum" );
static_assert( (uint32_t)TBVH_LAYOUT_VOXELSET == (uint32_t)tb::LAYOUT_VOXELSET, "layout enum" );
static_assert( (uint32_t)TBVH_LAYOUT_BVH4_AMD_HW == (uint32_t)tb::LAYOUT_BVH4_AMD_HW, "layout enum" );

// The tinybvh types are not standard-layout (ALIGNED members, unions, base classes), so offsetof on
// them is conditionally supported; GCC and Clang warn, MSVC does not. The offsets are what we rely on.
#if defined __GNUC__ || defined __clang__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winvalid-offsetof"
#endif

#define TBVH_SAME_OFFSET( ctype, cfield, cpptype, cppfield ) \
	static_assert( offsetof( ctype, cfield ) == offsetof( cpptype, cppfield ), #ctype "::" #cfield " offset" )

TBVH_SAME_OFFSET( tbvh_intersection, inst, tb::Intersection, inst );
TBVH_SAME_OFFSET( tbvh_intersection, t, tb::Intersection, t );
TBVH_SAME_OFFSET( tbvh_intersection, u, tb::Intersection, u );
TBVH_SAME_OFFSET( tbvh_intersection, v, tb::Intersection, v );
TBVH_SAME_OFFSET( tbvh_intersection, prim, tb::Intersection, prim );
TBVH_SAME_OFFSET( tbvh_intersection, aux_data, tb::Intersection, auxData );
TBVH_SAME_OFFSET( tbvh_intersection, user, tb::Intersection, userInt32 );
TBVH_SAME_OFFSET( tbvh_intersection_ex, inst, tb::IntersectionEx, inst );
TBVH_SAME_OFFSET( tbvh_intersection_ex, t, tb::IntersectionEx, t );
TBVH_SAME_OFFSET( tbvh_intersection_ex, u, tb::IntersectionEx, u );
TBVH_SAME_OFFSET( tbvh_intersection_ex, v, tb::IntersectionEx, v );
TBVH_SAME_OFFSET( tbvh_intersection_ex, prim, tb::IntersectionEx, prim );
TBVH_SAME_OFFSET( tbvh_intersection_ex, aux_data, tb::IntersectionEx, auxData );
TBVH_SAME_OFFSET( tbvh_intersection_ex, user, tb::IntersectionEx, userInt32 );

TBVH_SAME_OFFSET( tbvh_ray, o, tb::Ray, O );
TBVH_SAME_OFFSET( tbvh_ray, mask, tb::Ray, mask );
TBVH_SAME_OFFSET( tbvh_ray, d, tb::Ray, D );
TBVH_SAME_OFFSET( tbvh_ray, inst_idx, tb::Ray, instIdx );
TBVH_SAME_OFFSET( tbvh_ray, rd, tb::Ray, rD );
TBVH_SAME_OFFSET( tbvh_ray, hit, tb::Ray, hit );
static_assert( offsetof( tbvh_ray, hit ) == 44, "tbvh_ray::hit must be at offset 44" );
TBVH_SAME_OFFSET( tbvh_ray_ex, o, tb::RayEx, O );
TBVH_SAME_OFFSET( tbvh_ray_ex, mask, tb::RayEx, mask );
TBVH_SAME_OFFSET( tbvh_ray_ex, d, tb::RayEx, D );
TBVH_SAME_OFFSET( tbvh_ray_ex, inst_idx, tb::RayEx, instIdx );
TBVH_SAME_OFFSET( tbvh_ray_ex, rd, tb::RayEx, rD );
TBVH_SAME_OFFSET( tbvh_ray_ex, hit, tb::RayEx, hit );
static_assert( offsetof( tbvh_ray_ex, hit ) == 88, "tbvh_ray_ex::hit must be at offset 88" );

TBVH_SAME_OFFSET( tbvh_blas_instance, transform, tb::BLASInstance, transform );
TBVH_SAME_OFFSET( tbvh_blas_instance, inv_transform, tb::BLASInstance, invTransform );
TBVH_SAME_OFFSET( tbvh_blas_instance, aabb_min, tb::BLASInstance, aabbMin );
TBVH_SAME_OFFSET( tbvh_blas_instance, blas_idx, tb::BLASInstance, blasIdx );
TBVH_SAME_OFFSET( tbvh_blas_instance, aabb_max, tb::BLASInstance, aabbMax );
TBVH_SAME_OFFSET( tbvh_blas_instance, mask, tb::BLASInstance, mask );
TBVH_SAME_OFFSET( tbvh_blas_instance_ex, transform, tb::BLASInstanceEx, transform );
TBVH_SAME_OFFSET( tbvh_blas_instance_ex, inv_transform, tb::BLASInstanceEx, invTransform );
TBVH_SAME_OFFSET( tbvh_blas_instance_ex, aabb_min, tb::BLASInstanceEx, aabbMin );
TBVH_SAME_OFFSET( tbvh_blas_instance_ex, blas_idx, tb::BLASInstanceEx, blasIdx );
TBVH_SAME_OFFSET( tbvh_blas_instance_ex, aabb_max, tb::BLASInstanceEx, aabbMax );
TBVH_SAME_OFFSET( tbvh_blas_instance_ex, mask, tb::BLASInstanceEx, mask );

TBVH_SAME_OFFSET( tbvh_build_settings, use_presplitting, tb::BVHBuildSettings, usePresplitting );
TBVH_SAME_OFFSET( tbvh_build_settings, use_spatial_splits, tb::BVHBuildSettings, useSpatialSplits );
TBVH_SAME_OFFSET( tbvh_build_settings, min_spatial_overlap, tb::BVHBuildSettings, minSpatialOverlap );
TBVH_SAME_OFFSET( tbvh_build_settings, spatial_skip_smallest_axis, tb::BVHBuildSettings, spatialSkipSmallestAxis );
TBVH_SAME_OFFSET( tbvh_build_settings, presplit_post_pass, tb::BVHBuildSettings, presplitPostPass );
TBVH_SAME_OFFSET( tbvh_build_settings, presplit_factor, tb::BVHBuildSettings, presplitFactor );
TBVH_SAME_OFFSET( tbvh_build_settings, use_full_sweep, tb::BVHBuildSettings, useFullSweep );
TBVH_SAME_OFFSET( tbvh_build_settings, binned_bvh, tb::BVHBuildSettings, binnedBVH );
TBVH_SAME_OFFSET( tbvh_build_settings, bonsai_grid_prims, tb::BVHBuildSettings, bonsaiGridPrims );
TBVH_SAME_OFFSET( tbvh_build_settings, bonsai_prune_factor, tb::BVHBuildSettings, bonsaiPruneFactor );
TBVH_SAME_OFFSET( tbvh_build_settings, use_hploc, tb::BVHBuildSettings, useHPLOC );
TBVH_SAME_OFFSET( tbvh_build_settings, post_optimize, tb::BVHBuildSettings, postOptimize );
TBVH_SAME_OFFSET( tbvh_build_settings, optimize_iterations, tb::BVHBuildSettings, optimizeIterations );
TBVH_SAME_OFFSET( tbvh_build_settings, use_simd_if_available, tb::BVHBuildSettings, useSIMDifavailable );
TBVH_SAME_OFFSET( tbvh_build_settings, enable_threading, tb::BVHBuildSettings, enableThreading );
TBVH_SAME_OFFSET( tbvh_build_settings, discard_intermediates, tb::BVHBuildSettings, discardIntermediates );
TBVH_SAME_OFFSET( tbvh_build_settings, minimize_mem_use, tb::BVHBuildSettings, minimizeMemUse );
static_assert( sizeof( ((tb::BVHBuildSettings*)0)->optimizeIterations ) == sizeof( int32_t ), "optimizeIterations width" );

TBVH_SAME_OFFSET( tbvh_bvh_node, aabb_min, tb::BVH::BVHNode, aabbMin );
TBVH_SAME_OFFSET( tbvh_bvh_node, left_first, tb::BVH::BVHNode, leftFirst );
TBVH_SAME_OFFSET( tbvh_bvh_node, aabb_max, tb::BVH::BVHNode, aabbMax );
TBVH_SAME_OFFSET( tbvh_bvh_node, tri_count, tb::BVH::BVHNode, triCount );
TBVH_SAME_OFFSET( tbvh_bvh_node_ex, aabb_min, tb::BVH_Double::BVHNode, aabbMin );
TBVH_SAME_OFFSET( tbvh_bvh_node_ex, left_first, tb::BVH_Double::BVHNode, leftFirst );
TBVH_SAME_OFFSET( tbvh_bvh_node_ex, aabb_max, tb::BVH_Double::BVHNode, aabbMax );
TBVH_SAME_OFFSET( tbvh_bvh_node_ex, tri_count, tb::BVH_Double::BVHNode, triCount );
TBVH_SAME_OFFSET( tbvh_bvh_verbose_node, aabb_min, tb::BVH_Verbose::BVHNode, aabbMin );
TBVH_SAME_OFFSET( tbvh_bvh_verbose_node, left, tb::BVH_Verbose::BVHNode, left );
TBVH_SAME_OFFSET( tbvh_bvh_verbose_node, aabb_max, tb::BVH_Verbose::BVHNode, aabbMax );
TBVH_SAME_OFFSET( tbvh_bvh_verbose_node, right, tb::BVH_Verbose::BVHNode, right );
TBVH_SAME_OFFSET( tbvh_bvh_verbose_node, tri_count, tb::BVH_Verbose::BVHNode, triCount );
TBVH_SAME_OFFSET( tbvh_bvh_verbose_node, first_tri, tb::BVH_Verbose::BVHNode, firstTri );
TBVH_SAME_OFFSET( tbvh_bvh_verbose_node, parent, tb::BVH_Verbose::BVHNode, parent );
TBVH_SAME_OFFSET( tbvh_bvh_verbose_node, dummy, tb::BVH_Verbose::BVHNode, dummy );
TBVH_SAME_OFFSET( tbvh_bvh_gpu_node, lmin, tb::BVH_GPU::BVHNode, lmin );
TBVH_SAME_OFFSET( tbvh_bvh_gpu_node, left, tb::BVH_GPU::BVHNode, left );
TBVH_SAME_OFFSET( tbvh_bvh_gpu_node, lmax, tb::BVH_GPU::BVHNode, lmax );
TBVH_SAME_OFFSET( tbvh_bvh_gpu_node, right, tb::BVH_GPU::BVHNode, right );
TBVH_SAME_OFFSET( tbvh_bvh_gpu_node, rmin, tb::BVH_GPU::BVHNode, rmin );
TBVH_SAME_OFFSET( tbvh_bvh_gpu_node, tri_count, tb::BVH_GPU::BVHNode, triCount );
TBVH_SAME_OFFSET( tbvh_bvh_gpu_node, rmax, tb::BVH_GPU::BVHNode, rmax );
TBVH_SAME_OFFSET( tbvh_bvh_gpu_node, first_tri, tb::BVH_GPU::BVHNode, firstTri );
TBVH_SAME_OFFSET( tbvh_mbvh4_node, aabb_min, tb::MBVH<4>::MBVHNode, aabbMin );
TBVH_SAME_OFFSET( tbvh_mbvh4_node, first_tri, tb::MBVH<4>::MBVHNode, firstTri );
TBVH_SAME_OFFSET( tbvh_mbvh4_node, aabb_max, tb::MBVH<4>::MBVHNode, aabbMax );
TBVH_SAME_OFFSET( tbvh_mbvh4_node, tri_count, tb::MBVH<4>::MBVHNode, triCount );
TBVH_SAME_OFFSET( tbvh_mbvh4_node, child, tb::MBVH<4>::MBVHNode, child );
TBVH_SAME_OFFSET( tbvh_mbvh4_node, child_count, tb::MBVH<4>::MBVHNode, childCount );
TBVH_SAME_OFFSET( tbvh_mbvh4_node, dummy, tb::MBVH<4>::MBVHNode, dummy );
TBVH_SAME_OFFSET( tbvh_mbvh8_node, aabb_min, tb::MBVH<8>::MBVHNode, aabbMin );
TBVH_SAME_OFFSET( tbvh_mbvh8_node, first_tri, tb::MBVH<8>::MBVHNode, firstTri );
TBVH_SAME_OFFSET( tbvh_mbvh8_node, aabb_max, tb::MBVH<8>::MBVHNode, aabbMax );
TBVH_SAME_OFFSET( tbvh_mbvh8_node, tri_count, tb::MBVH<8>::MBVHNode, triCount );
TBVH_SAME_OFFSET( tbvh_mbvh8_node, child, tb::MBVH<8>::MBVHNode, child );
TBVH_SAME_OFFSET( tbvh_mbvh8_node, child_count, tb::MBVH<8>::MBVHNode, childCount );
TBVH_SAME_OFFSET( tbvh_mbvh8_node, dummy, tb::MBVH<8>::MBVHNode, dummy );

#undef TBVH_SAME_OFFSET

#if defined __GNUC__ || defined __clang__
#pragma GCC diagnostic pop
#endif

/* ------------------------------------------------------------------------------------------------
   Opaque handle definitions
   ------------------------------------------------------------------------------------------------ */

struct tbvh_bvh
{
	tb::BVH bvh;
	tbvh_intersect_fn intersect = nullptr;
	tbvh_occluded_fn occluded = nullptr;
	tbvh_aabb_fn get_aabb = nullptr;
	void* user = nullptr;
};
struct tbvh_bvh_verbose { tb::BVH_Verbose bvh; };
struct tbvh_bvh_gpu { tb::BVH_GPU bvh; };
struct tbvh_mbvh4 { tb::MBVH<4> bvh; };
struct tbvh_mbvh8 { tb::MBVH<8> bvh; };
struct tbvh_bvh4_cpu { tb::BVH4_CPU bvh; };
struct tbvh_bvh4_gpu { tb::BVH4_GPU bvh; };
struct tbvh_bvh4_amd_hw { tb::BVH4_AMD_HW bvh; };
struct tbvh_bvh8_cwbvh { tb::BVH8_CWBVH bvh; };
struct tbvh_bvh8_cpu { tb::BVH8_CPU bvh; };
#ifdef ENABLE_VOXEL_SUPPORT
struct tbvh_voxelset { tb::VoxelSet vs; };
#endif
struct tbvh_bvh_double
{
	tb::BVH_Double bvh;
	tbvh_intersect_ex_fn intersect = nullptr;
	tbvh_occluded_ex_fn occluded = nullptr;
	tbvh_aabb_ex_fn get_aabb = nullptr;
	void* user = nullptr;
};
struct tbvh_bvh4_double { tb::BVH4_Double bvh; };

/* ------------------------------------------------------------------------------------------------
   Conversion helpers
   ------------------------------------------------------------------------------------------------ */

static inline tb::BVHBase* base_of( tbvh_base* b ) { return reinterpret_cast<tb::BVHBase*>( b ); }
static inline const tb::BVHBase* base_of( const tbvh_base* b ) { return reinterpret_cast<const tb::BVHBase*>( b ); }
static inline tb::BVHBaseEx* base_of( tbvh_base_ex* b ) { return reinterpret_cast<tb::BVHBaseEx*>( b ); }
static inline const tb::BVHBaseEx* base_of( const tbvh_base_ex* b ) { return reinterpret_cast<const tb::BVHBaseEx*>( b ); }
// upcast to the shared base first, so the handle pointer is adjusted correctly, then hide it.
static inline tbvh_base* to_base( tb::BVHBase* b ) { return reinterpret_cast<tbvh_base*>( b ); }
static inline const tbvh_base* to_base( const tb::BVHBase* b ) { return reinterpret_cast<const tbvh_base*>( b ); }
static inline tbvh_base_ex* to_base( tb::BVHBaseEx* b ) { return reinterpret_cast<tbvh_base_ex*>( b ); }
static inline const tbvh_base_ex* to_base( const tb::BVHBaseEx* b ) { return reinterpret_cast<const tbvh_base_ex*>( b ); }

static inline tb::Ray& ray_of( tbvh_ray* r ) { return *reinterpret_cast<tb::Ray*>( r ); }
static inline const tb::Ray& ray_of( const tbvh_ray* r ) { return *reinterpret_cast<const tb::Ray*>( r ); }
static inline tb::Ray* rays_of( tbvh_ray* r ) { return reinterpret_cast<tb::Ray*>( r ); }
static inline tb::RayEx& ray_of( tbvh_ray_ex* r ) { return *reinterpret_cast<tb::RayEx*>( r ); }
static inline const tb::RayEx& ray_of( const tbvh_ray_ex* r ) { return *reinterpret_cast<const tb::RayEx*>( r ); }

static inline tb::bvhvec3 vec_of( const tbvh_vec3 v ) { return tb::bvhvec3( v.x, v.y, v.z ); }
static inline tbvh_vec3 vec_to( const tb::bvhvec3& v ) { tbvh_vec3 r = { v.x, v.y, v.z }; return r; }
static inline tb::bvhdbl3 vec_of( const tbvh_dbl3 v ) { return tb::bvhdbl3( v.x, v.y, v.z ); }
static inline tbvh_dbl3 vec_to( const tb::bvhdbl3& v ) { tbvh_dbl3 r = { v.x, v.y, v.z }; return r; }

static inline tb::bvhvec4slice slice_of( const tbvh_vec4* verts, uint32_t count, uint32_t stride )
{
	return tb::bvhvec4slice( reinterpret_cast<const tb::bvhvec4*>( verts ), count, stride );
}
static inline tb::bvhdbl3slice slice_of( const tbvh_dbl3* verts, uint64_t count, uint64_t stride )
{
	return tb::bvhdbl3slice( reinterpret_cast<const tb::bvhdbl3*>( verts ), count, stride );
}

static inline tb::BLASInstance* inst_of( tbvh_blas_instance* i ) { return reinterpret_cast<tb::BLASInstance*>( i ); }
static inline tb::BLASInstanceEx* inst_of( tbvh_blas_instance_ex* i ) { return reinterpret_cast<tb::BLASInstanceEx*>( i ); }

/* ------------------------------------------------------------------------------------------------
   Custom geometry trampolines. tinybvh hands customUserdata to all three callbacks; we store the
   handle there and forward to the C callbacks with the caller's own user pointer.
   ------------------------------------------------------------------------------------------------ */

static void bvh_aabb_trampoline( const uint32_t prim, tb::bvhvec3& bmin, tb::bvhvec3& bmax, void* ud )
{
	tbvh_bvh* h = static_cast<tbvh_bvh*>( ud );
	h->get_aabb( prim, reinterpret_cast<tbvh_vec3*>( &bmin ), reinterpret_cast<tbvh_vec3*>( &bmax ), h->user );
}
static bool bvh_intersect_trampoline( tb::Ray& ray, const uint32_t prim, void* ud )
{
	tbvh_bvh* h = static_cast<tbvh_bvh*>( ud );
	if (!h->intersect) return false;
	return h->intersect( reinterpret_cast<tbvh_ray*>( &ray ), prim, h->user );
}
static bool bvh_occluded_trampoline( const tb::Ray& ray, const uint32_t prim, void* ud )
{
	tbvh_bvh* h = static_cast<tbvh_bvh*>( ud );
	if (h->occluded) return h->occluded( reinterpret_cast<const tbvh_ray*>( &ray ), prim, h->user );
	if (!h->intersect) return false;
	tb::Ray copy = ray; // no any-hit callback: fall back to the closest-hit test.
	h->intersect( reinterpret_cast<tbvh_ray*>( &copy ), prim, h->user );
	return copy.hit.t < ray.hit.t;
}
static void bvh_install_custom( tbvh_bvh* h )
{
	h->bvh.customIntersect = bvh_intersect_trampoline;
	h->bvh.customIsOccluded = bvh_occluded_trampoline;
	h->bvh.customUserdata = h;
}

static void bvh_double_aabb_trampoline( const uint64_t prim, tb::bvhdbl3& bmin, tb::bvhdbl3& bmax, void* ud )
{
	tbvh_bvh_double* h = static_cast<tbvh_bvh_double*>( ud );
	h->get_aabb( prim, reinterpret_cast<tbvh_dbl3*>( &bmin ), reinterpret_cast<tbvh_dbl3*>( &bmax ), h->user );
}
static bool bvh_double_intersect_trampoline( tb::RayEx& ray, const uint64_t prim, void* ud )
{
	tbvh_bvh_double* h = static_cast<tbvh_bvh_double*>( ud );
	if (!h->intersect) return false;
	return h->intersect( reinterpret_cast<tbvh_ray_ex*>( &ray ), prim, h->user );
}
static bool bvh_double_occluded_trampoline( const tb::RayEx& ray, const uint64_t prim, void* ud )
{
	tbvh_bvh_double* h = static_cast<tbvh_bvh_double*>( ud );
	if (h->occluded) return h->occluded( reinterpret_cast<const tbvh_ray_ex*>( &ray ), prim, h->user );
	if (!h->intersect) return false;
	tb::RayEx copy = ray;
	h->intersect( reinterpret_cast<tbvh_ray_ex*>( &copy ), prim, h->user );
	return copy.hit.t < ray.hit.t;
}
static void bvh_double_install_custom( tbvh_bvh_double* h )
{
	h->bvh.customIntersect = bvh_double_intersect_trampoline;
	h->bvh.customIsOccluded = bvh_double_occluded_trampoline;
	h->bvh.customUserdata = h;
}

/* ------------------------------------------------------------------------------------------------
   Library-wide functions
   ------------------------------------------------------------------------------------------------ */

void tbvh_version( int32_t* major, int32_t* minor, int32_t* sub )
{
	if (major) *major = TINY_BVH_VERSION_MAJOR;
	if (minor) *minor = TINY_BVH_VERSION_MINOR;
	if (sub) *sub = TINY_BVH_VERSION_SUB;
}
uint32_t tbvh_bundle_rays( void ) { return TINYBVH_BUNDLE_RAYS; }
uint32_t tbvh_inst_idx_bits( void ) { return INST_IDX_BITS; }
uint32_t tbvh_stack_size( void ) { return TINYBVH_STACK_SIZE; }
bool tbvh_simd_enabled( void )
{
#ifdef TINYBVH_NO_SIMD
	return false;
#else
	return true;
#endif
}
bool tbvh_simd_builder_available( void ) { return tb::impl::BVHSIMDBuilders<float, uint32_t>::available; }
float tbvh_far( void ) { return BVH_FAR; }
double tbvh_dbl_far( void ) { return BVH_DBL_FAR; }
uint32_t tbvh_voxelset_object_dim( void )
{
#ifdef ENABLE_VOXEL_SUPPORT
	return (uint32_t)tb::VoxelSet::objectDim;
#else
	return 0;
#endif
}

size_t tbvh_sizeof( tbvh_struct_id id )
{
	switch (id)
	{
	case TBVH_STRUCT_VEC3: return sizeof( tb::bvhvec3 );
	case TBVH_STRUCT_VEC4: return sizeof( tb::bvhvec4 );
	case TBVH_STRUCT_DBL3: return sizeof( tb::bvhdbl3 );
	case TBVH_STRUCT_MAT4: return sizeof( tb::bvhmat4 );
	case TBVH_STRUCT_DBLMAT4: return sizeof( tb::bvhdblmat4 );
	case TBVH_STRUCT_INTERSECTION: return sizeof( tb::Intersection );
	case TBVH_STRUCT_INTERSECTION_EX: return sizeof( tb::IntersectionEx );
	case TBVH_STRUCT_RAY: return sizeof( tb::Ray );
	case TBVH_STRUCT_RAY_EX: return sizeof( tb::RayEx );
	case TBVH_STRUCT_BLAS_INSTANCE: return sizeof( tb::BLASInstance );
	case TBVH_STRUCT_BLAS_INSTANCE_EX: return sizeof( tb::BLASInstanceEx );
	case TBVH_STRUCT_BUILD_SETTINGS: return sizeof( tb::BVHBuildSettings );
	case TBVH_STRUCT_BVH_NODE: return sizeof( tb::BVH::BVHNode );
	case TBVH_STRUCT_BVH_NODE_EX: return sizeof( tb::BVH_Double::BVHNode );
	case TBVH_STRUCT_BVH_VERBOSE_NODE: return sizeof( tb::BVH_Verbose::BVHNode );
	case TBVH_STRUCT_BVH_GPU_NODE: return sizeof( tb::BVH_GPU::BVHNode );
	case TBVH_STRUCT_MBVH4_NODE: return sizeof( tb::MBVH<4>::MBVHNode );
	case TBVH_STRUCT_MBVH8_NODE: return sizeof( tb::MBVH<8>::MBVHNode );
	default: return 0;
	}
}

// tinybvh's handler has no user pointer; keep the C handler and its pointer in statics.
static tbvh_fatal_error_fn g_fatal_handler = nullptr;
static void* g_fatal_user = nullptr;
static void fatal_error_trampoline( const char* message )
{
	if (g_fatal_handler) g_fatal_handler( message, g_fatal_user );
}
void tbvh_set_fatal_error_handler( tbvh_fatal_error_fn handler, void* user )
{
	g_fatal_handler = handler;
	g_fatal_user = user;
	tb::tinybvh_set_fatal_error_handler( handler ? fatal_error_trampoline : nullptr );
}

void tbvh_shutdown_builtin_pool( void )
{
#if defined ENABLE_THREADED_BUILDS && !defined TINYBVH_NO_BUILTIN_POOL
	tb::tinybvh_shutdown_builtin_pool();
#endif
}

/* Ray helpers. The constructors leave the user payload indeterminate; we zero the whole record first
   so a freshly initialized ray is fully defined. */

void tbvh_ray_init( tbvh_ray* ray, tbvh_vec3 origin, tbvh_vec3 direction, float t, uint32_t mask )
{
	memset( ray, 0, sizeof( tbvh_ray ) );
	new (ray) tb::Ray( vec_of( origin ), vec_of( direction ), t, mask );
}
void tbvh_ray_set_direction( tbvh_ray* ray, tbvh_vec3 direction )
{
	tb::Ray& r = ray_of( ray );
	r.D = vec_of( direction );
	r.rD = tb::tinybvh_rcp( r.D );
}
void tbvh_ray_normalize( tbvh_ray* ray ) { ray_of( ray ).Normalize(); }
void tbvh_ray_ex_init( tbvh_ray_ex* ray, tbvh_dbl3 origin, tbvh_dbl3 direction, double t, uint32_t mask )
{
	memset( ray, 0, sizeof( tbvh_ray_ex ) );
	new (ray) tb::RayEx( vec_of( origin ), vec_of( direction ), t, mask );
}
void tbvh_ray_ex_set_direction( tbvh_ray_ex* ray, tbvh_dbl3 direction )
{
	tb::RayEx& r = ray_of( ray );
	r.D = vec_of( direction );
	r.rD = tb::tinybvh_rcp( r.D );
}
void tbvh_ray_ex_normalize( tbvh_ray_ex* ray ) { ray_of( ray ).Normalize(); }

/* Instance helpers. */

void tbvh_blas_instance_init( tbvh_blas_instance* inst, uint32_t blas_idx )
{
	memset( inst, 0, sizeof( tbvh_blas_instance ) );
	new (inst) tb::BLASInstance( blas_idx );
}
void tbvh_blas_instance_invert_transform( tbvh_blas_instance* inst ) { inst_of( inst )->InvertTransform(); }
void tbvh_blas_instance_update( tbvh_blas_instance* inst, const tbvh_base* blas )
{
	// Update takes a non-const pointer but only reads the bounds of the BLAS.
	inst_of( inst )->Update( const_cast<tb::BVHBase*>( base_of( blas ) ) );
}
void tbvh_blas_instance_ex_init( tbvh_blas_instance_ex* inst, uint64_t blas_idx )
{
	memset( inst, 0, sizeof( tbvh_blas_instance_ex ) );
	new (inst) tb::BLASInstanceEx( blas_idx );
}
void tbvh_blas_instance_ex_invert_transform( tbvh_blas_instance_ex* inst ) { inst_of( inst )->InvertTransform(); }
void tbvh_blas_instance_ex_update( tbvh_blas_instance_ex* inst, const tbvh_base_ex* blas )
{
	inst_of( inst )->Update( const_cast<tb::BVHBaseEx*>( base_of( blas ) ) );
}

/* Transform helpers. */

tbvh_vec3 tbvh_transform_point( tbvh_vec3 p, const tbvh_mat4* m )
{
	return vec_to( tb::tinybvh_transform_point( vec_of( p ), *reinterpret_cast<const tb::bvhmat4*>( m ) ) );
}
tbvh_vec3 tbvh_transform_vector( tbvh_vec3 v, const tbvh_mat4* m )
{
	return vec_to( tb::tinybvh_transform_vector( vec_of( v ), *reinterpret_cast<const tb::bvhmat4*>( m ) ) );
}

/* ------------------------------------------------------------------------------------------------
   tbvh_base / tbvh_base_ex: identical bodies for the two BVHBase instantiations.
   ------------------------------------------------------------------------------------------------ */

#define TBVH_BASE_IMPL( fn, handle, index_t, vec_t ) \
tbvh_layout fn( _layout )( const handle* b ) { return static_cast<tbvh_layout>( base_of( b )->layout ); } \
void fn( _get_settings )( const handle* b, tbvh_build_settings* out ) { memcpy( out, &base_of( b )->settings, sizeof( tbvh_build_settings ) ); } \
void fn( _set_settings )( handle* b, const tbvh_build_settings* settings ) { memcpy( &base_of( b )->settings, settings, sizeof( tbvh_build_settings ) ); } \
float fn( _get_c_trav )( const handle* b ) { return base_of( b )->c_trav; } \
float fn( _get_c_int )( const handle* b ) { return base_of( b )->c_int; } \
void fn( _set_c_trav )( handle* b, float c ) { base_of( b )->c_trav = c; } \
void fn( _set_c_int )( handle* b, float c ) { base_of( b )->c_int = c; } \
uint32_t fn( _get_hq_bins )( const handle* b ) { return base_of( b )->hqbvhbins; } \
void fn( _set_hq_bins )( handle* b, uint32_t bins ) { base_of( b )->hqbvhbins = bins; } \
bool fn( _get_hq_odd_even )( const handle* b ) { return base_of( b )->hqbvhoddeven; } \
void fn( _set_hq_odd_even )( handle* b, bool oddeven ) { base_of( b )->hqbvhoddeven = oddeven; } \
uint32_t fn( _get_hq_leaf_prims )( const handle* b ) { return base_of( b )->hqLeafPrims; } \
void fn( _set_hq_leaf_prims )( handle* b, uint32_t prims ) { base_of( b )->hqLeafPrims = prims; } \
vec_t fn( _aabb_min )( const handle* b ) { return vec_to( base_of( b )->aabbMin ); } \
vec_t fn( _aabb_max )( const handle* b ) { return vec_to( base_of( b )->aabbMax ); } \
index_t fn( _tri_count )( const handle* b ) { return base_of( b )->triCount; } \
index_t fn( _idx_count )( const handle* b ) { return base_of( b )->idxCount; } \
index_t fn( _used_nodes )( const handle* b ) { return base_of( b )->usedNodes; } \
index_t fn( _allocated_nodes )( const handle* b ) { return base_of( b )->allocatedNodes; } \
bool fn( _is_refittable )( const handle* b ) { return base_of( b )->isRefittable(); } \
bool fn( _is_rebuildable )( const handle* b ) { return base_of( b )->isRebuildable(); } \
bool fn( _may_have_holes )( const handle* b ) { return base_of( b )->mayHaveHoles(); } \
bool fn( _has_ordered_nodes )( const handle* b ) { return base_of( b )->hasOrderedNodes(); } \
bool fn( _is_over_aabbs )( const handle* b ) { return base_of( b )->isOverAABBs(); } \
bool fn( _is_over_indices )( const handle* b ) { return base_of( b )->isOverIndices(); } \
void fn( _set_opacity_micro_maps )( handle* b, const uint32_t* map, uint32_t n ) \
{ \
	if (!map || n == 0) base_of( b )->SetOpacityMicroMaps( nullptr, 0 ); \
	else base_of( b )->SetOpacityMicroMaps( const_cast<uint32_t*>( map ), n ); /* tinybvh only reads the map */ \
} \
uint32_t fn( _opacity_micro_map_n )( const handle* b ) { return base_of( b )->opmapN; }

#define TBVH_BASE_FN( suffix ) tbvh_base##suffix
#define TBVH_BASE_EX_FN( suffix ) tbvh_base_ex##suffix
TBVH_BASE_IMPL( TBVH_BASE_FN, tbvh_base, uint32_t, tbvh_vec3 )
TBVH_BASE_IMPL( TBVH_BASE_EX_FN, tbvh_base_ex, uint64_t, tbvh_dbl3 )
#undef TBVH_BASE_FN
#undef TBVH_BASE_EX_FN
#undef TBVH_BASE_IMPL

/* ------------------------------------------------------------------------------------------------
   Shared boilerplate: new / free / base for every single precision layout, and the four triangle
   builders every derived layout has (Build and BuildHQ over a slice, with and without indices).
   ------------------------------------------------------------------------------------------------ */

#define TBVH_HANDLE_BASICS( handle ) \
handle* handle##_new( void ) { return new handle(); } \
void handle##_free( handle* h ) { delete h; } \
tbvh_base* handle##_base( handle* h ) { return to_base( static_cast<tb::BVHBase*>( &h->bvh ) ); } \
const tbvh_base* handle##_base_const( const handle* h ) { return to_base( static_cast<const tb::BVHBase*>( &h->bvh ) ); }

#define TBVH_TRI_BUILDERS( handle ) \
void handle##_build( handle* h, const tbvh_vec4* verts, uint32_t vert_count, uint32_t stride ) \
{ h->bvh.Build( slice_of( verts, vert_count, stride ) ); } \
void handle##_build_indexed( handle* h, const tbvh_vec4* verts, uint32_t vert_count, uint32_t stride, const uint32_t* indices, uint32_t prim_count ) \
{ h->bvh.Build( slice_of( verts, vert_count, stride ), indices, prim_count ); } \
void handle##_build_hq( handle* h, const tbvh_vec4* verts, uint32_t vert_count, uint32_t stride ) \
{ h->bvh.BuildHQ( slice_of( verts, vert_count, stride ) ); } \
void handle##_build_hq_indexed( handle* h, const tbvh_vec4* verts, uint32_t vert_count, uint32_t stride, const uint32_t* indices, uint32_t prim_count ) \
{ h->bvh.BuildHQ( slice_of( verts, vert_count, stride ), indices, prim_count ); }

/* ------------------------------------------------------------------------------------------------
   tbvh_bvh
   ------------------------------------------------------------------------------------------------ */

TBVH_HANDLE_BASICS( tbvh_bvh )
TBVH_TRI_BUILDERS( tbvh_bvh )

void tbvh_bvh_build_quick( tbvh_bvh* h, const tbvh_vec4* verts, uint32_t vert_count, uint32_t stride )
{
	h->bvh.BuildQuick( slice_of( verts, vert_count, stride ) );
}
void tbvh_bvh_build_simd( tbvh_bvh* h, const tbvh_vec4* verts, uint32_t vert_count, uint32_t stride )
{
	h->bvh.BuildSIMD( slice_of( verts, vert_count, stride ) );
}
void tbvh_bvh_build_simd_indexed( tbvh_bvh* h, const tbvh_vec4* verts, uint32_t vert_count, uint32_t stride, const uint32_t* indices, uint32_t prim_count )
{
	h->bvh.BuildSIMD( slice_of( verts, vert_count, stride ), indices, prim_count );
}
void tbvh_bvh_build_aabbs( tbvh_bvh* h, const tbvh_vec4* aabbs, uint32_t aabb_count )
{
	h->bvh.BuildAABB( reinterpret_cast<const tb::bvhvec4*>( aabbs ), aabb_count );
}
void tbvh_bvh_build_custom( tbvh_bvh* h, tbvh_aabb_fn get_aabb, uint32_t prim_count, void* user )
{
	h->get_aabb = get_aabb;
	h->user = user;
	bvh_install_custom( h ); // customUserdata must be set before Build: the AABB callback receives it.
	h->bvh.Build( bvh_aabb_trampoline, prim_count );
}
void tbvh_bvh_set_custom_callbacks( tbvh_bvh* h, tbvh_intersect_fn intersect, tbvh_occluded_fn occluded, void* user )
{
	h->intersect = intersect;
	h->occluded = occluded;
	h->user = user;
	if (!intersect && !occluded)
	{
		// detach: tinybvh's triangle builders do not clear the custom callbacks themselves.
		h->bvh.customIntersect = 0, h->bvh.customIsOccluded = 0, h->bvh.customUserdata = 0;
		return;
	}
	bvh_install_custom( h );
}
void tbvh_bvh_build_tlas( tbvh_bvh* h, tbvh_blas_instance* instances, uint32_t inst_count, tbvh_base** blasses, uint32_t blas_count )
{
	h->bvh.Build( inst_of( instances ), inst_count, reinterpret_cast<tb::BVHBase**>( blasses ), blas_count );
}

int32_t tbvh_bvh_intersect( const tbvh_bvh* h, tbvh_ray* ray ) { return h->bvh.Intersect( ray_of( ray ) ); }
bool tbvh_bvh_is_occluded( const tbvh_bvh* h, const tbvh_ray* ray ) { return h->bvh.IsOccluded( ray_of( ray ) ); }
bool tbvh_bvh_intersect_sphere( const tbvh_bvh* h, tbvh_vec3 pos, float r ) { return h->bvh.IntersectSphere( vec_of( pos ), r ); }
int32_t tbvh_bvh_intersect_bundle( const tbvh_bvh* h, tbvh_ray* rays ) { return h->bvh.IntersectBundle( rays_of( rays ) ); }
int32_t tbvh_bvh_is_occluded_bundle( const tbvh_bvh* h, tbvh_ray* rays, bool* occluded ) { return h->bvh.IsOccludedBundle( rays_of( rays ), occluded ); }

void tbvh_bvh_refit( tbvh_bvh* h, uint32_t node_idx ) { h->bvh.Refit( node_idx ); }
void tbvh_bvh_compact( tbvh_bvh* h )
{
	h->bvh.Compact();
	// BVH::Compact leaves the 'may have holes' flag set (tinybvh 1.9.1), which later makes Refit and
	// Optimize fatal. BVH_Verbose::Compact does clear it, so round-trip through that layout.
	if (h->bvh.mayHaveHoles())
	{
		tb::BVH_Verbose v;
		v.ConvertFrom( h->bvh, true );
		v.Compact();
		h->bvh.ConvertFrom( v, true );
	}
}
void tbvh_bvh_optimize( tbvh_bvh* h, uint32_t iterations, bool extreme, bool stochastic ) { h->bvh.Optimize( iterations, extreme, stochastic ); }
void tbvh_bvh_split_leafs( tbvh_bvh* h, uint32_t max_prims ) { h->bvh.SplitLeafs( max_prims ); }
void tbvh_bvh_merge_leafs( tbvh_bvh* h ) { h->bvh.MergeLeafs(); }
float tbvh_bvh_sah_cost( const tbvh_bvh* h, uint32_t node_idx ) { return h->bvh.SAHCost( node_idx ); }
float tbvh_bvh_epo_cost( const tbvh_bvh* h, uint32_t node_idx ) { return h->bvh.EPOCost( node_idx ); }
uint32_t tbvh_bvh_node_count( const tbvh_bvh* h ) { return h->bvh.NodeCount(); }
uint32_t tbvh_bvh_leaf_count( const tbvh_bvh* h ) { return h->bvh.LeafCount(); }
uint32_t tbvh_bvh_prim_count( const tbvh_bvh* h, uint32_t node_idx ) { return h->bvh.PrimCount( node_idx ); }
uint32_t tbvh_bvh_collect_nodes( const tbvh_bvh* h, uint32_t root, uint32_t* list, uint32_t cap ) { return h->bvh.CollectNodes( root, list, cap ); }
bool tbvh_bvh_is_tlas( const tbvh_bvh* h ) { return h->bvh.isTLAS(); }
bool tbvh_bvh_is_indexed( const tbvh_bvh* h ) { return h->bvh.isIndexed(); }
bool tbvh_bvh_has_custom_geometry( const tbvh_bvh* h ) { return h->bvh.hasCustomGeom(); }

void tbvh_bvh_save( const tbvh_bvh* h, const char* path, bool include_geometry )
{
	// BVH::Save is not declared const but only reads the structure.
	const_cast<tb::BVH&>( h->bvh ).Save( path, include_geometry );
}
bool tbvh_bvh_load( tbvh_bvh* h, const char* path, const tbvh_vec4* verts, uint32_t vert_count, uint32_t stride, const uint32_t* indices, uint32_t prim_count )
{
	const tb::bvhvec4slice s = verts ? slice_of( verts, vert_count, stride ) : tb::bvhvec4slice( nullptr, 0, 0 );
	return h->bvh.Load( path, s, indices, prim_count );
}
bool tbvh_bvh_load_self_contained( tbvh_bvh* h, const char* path )
{
	h->bvh = tb::BVH::Load( path );
	return h->bvh.usedNodes > 0;
}
bool tbvh_bvh_owns_geometry( const tbvh_bvh* h ) { return h->bvh.ownsGeometry; }
bool tbvh_bvh_has_fragments( const tbvh_bvh* h ) { return h->bvh.fragment != 0; }
void tbvh_bvh_free_geometry( tbvh_bvh* h ) { h->bvh.FreeGeometry(); }

const tbvh_bvh_node* tbvh_bvh_nodes( const tbvh_bvh* h, uint32_t* count )
{
	if (count) *count = h->bvh.usedNodes;
	return reinterpret_cast<const tbvh_bvh_node*>( h->bvh.bvhNode );
}
const uint32_t* tbvh_bvh_prim_indices( const tbvh_bvh* h, uint32_t* count )
{
	if (count) *count = h->bvh.idxCount;
	return h->bvh.primIdx;
}
const tbvh_vec4* tbvh_bvh_verts( const tbvh_bvh* h, uint32_t* count, uint32_t* stride )
{
	if (count) *count = h->bvh.verts.count;
	if (stride) *stride = h->bvh.verts.stride;
	return reinterpret_cast<const tbvh_vec4*>( h->bvh.verts.data );
}
const uint32_t* tbvh_bvh_vert_indices( const tbvh_bvh* h ) { return h->bvh.vertIdx; }

void tbvh_bvh_convert_from_verbose( tbvh_bvh* h, const tbvh_bvh_verbose* verbose, bool compact )
{
	h->bvh.ConvertFrom( verbose->bvh, compact );
}

/* ------------------------------------------------------------------------------------------------
   tbvh_bvh_verbose
   ------------------------------------------------------------------------------------------------ */

TBVH_HANDLE_BASICS( tbvh_bvh_verbose )

void tbvh_bvh_verbose_convert_from( tbvh_bvh_verbose* v, const tbvh_bvh* bvh, bool compact ) { v->bvh.ConvertFrom( bvh->bvh, compact ); }
float tbvh_bvh_verbose_sah_cost( const tbvh_bvh_verbose* v, uint32_t node_idx ) { return v->bvh.SAHCost( node_idx ); }
int32_t tbvh_bvh_verbose_node_count( const tbvh_bvh_verbose* v ) { return v->bvh.NodeCount(); }
void tbvh_bvh_verbose_refit( tbvh_bvh_verbose* v, uint32_t node_idx, bool skip_leafs ) { v->bvh.Refit( node_idx, skip_leafs ); }
void tbvh_bvh_verbose_compact( tbvh_bvh_verbose* v ) { v->bvh.Compact(); }
void tbvh_bvh_verbose_sort_indices( tbvh_bvh_verbose* v ) { v->bvh.SortIndices(); }
void tbvh_bvh_verbose_split_leafs( tbvh_bvh_verbose* v, uint32_t max_prims ) { v->bvh.SplitLeafs( max_prims ); }
void tbvh_bvh_verbose_merge_leafs( tbvh_bvh_verbose* v ) { v->bvh.MergeLeafs(); }
void tbvh_bvh_verbose_optimize( tbvh_bvh_verbose* v, uint32_t iterations, bool extreme, bool stochastic ) { v->bvh.Optimize( iterations, extreme, stochastic ); }
const tbvh_bvh_verbose_node* tbvh_bvh_verbose_nodes( const tbvh_bvh_verbose* v, uint32_t* count )
{
	if (count) *count = v->bvh.usedNodes;
	return reinterpret_cast<const tbvh_bvh_verbose_node*>( v->bvh.bvhNode );
}
const uint32_t* tbvh_bvh_verbose_prim_indices( const tbvh_bvh_verbose* v, uint32_t* count )
{
	if (count) *count = v->bvh.idxCount;
	return v->bvh.primIdx;
}

/* ------------------------------------------------------------------------------------------------
   tbvh_bvh_gpu
   ------------------------------------------------------------------------------------------------ */

TBVH_HANDLE_BASICS( tbvh_bvh_gpu )
TBVH_TRI_BUILDERS( tbvh_bvh_gpu )

void tbvh_bvh_gpu_build_aabbs( tbvh_bvh_gpu* g, const tbvh_vec4* aabbs, uint32_t aabb_count )
{
	g->bvh.BuildAABB( reinterpret_cast<const tb::bvhvec4*>( aabbs ), aabb_count );
}
void tbvh_bvh_gpu_build_tlas( tbvh_bvh_gpu* g, tbvh_blas_instance* instances, uint32_t inst_count, tbvh_base** blasses, uint32_t blas_count )
{
	g->bvh.Build( inst_of( instances ), inst_count, reinterpret_cast<tb::BVHBase**>( blasses ), blas_count );
}
void tbvh_bvh_gpu_convert_from( tbvh_bvh_gpu* g, const tbvh_bvh* bvh, bool compact ) { g->bvh.ConvertFrom( bvh->bvh, compact ); }
void tbvh_bvh_gpu_optimize( tbvh_bvh_gpu* g, uint32_t iterations, bool extreme ) { g->bvh.Optimize( iterations, extreme ); }
float tbvh_bvh_gpu_sah_cost( tbvh_bvh_gpu* g, uint32_t node_idx ) { return g->bvh.SAHCost( node_idx ); }
int32_t tbvh_bvh_gpu_intersect( const tbvh_bvh_gpu* g, tbvh_ray* ray ) { return g->bvh.Intersect( ray_of( ray ) ); }
bool tbvh_bvh_gpu_is_occluded( const tbvh_bvh_gpu* g, const tbvh_ray* ray ) { return g->bvh.IsOccluded( ray_of( ray ) ); }
bool tbvh_bvh_gpu_load_self_contained( tbvh_bvh_gpu* g, const char* path )
{
	g->bvh = tb::BVH_GPU::Load( path );
	return g->bvh.usedNodes > 0;
}
const tbvh_bvh_gpu_node* tbvh_bvh_gpu_nodes( const tbvh_bvh_gpu* g, uint32_t* count )
{
	if (count) *count = g->bvh.usedNodes;
	return reinterpret_cast<const tbvh_bvh_gpu_node*>( g->bvh.bvhNode );
}
const uint32_t* tbvh_bvh_gpu_prim_indices( const tbvh_bvh_gpu* g, uint32_t* count )
{
	if (count) *count = g->bvh.bvh.idxCount;
	return g->bvh.bvh.primIdx;
}

/* ------------------------------------------------------------------------------------------------
   tbvh_mbvh4 / tbvh_mbvh8
   ------------------------------------------------------------------------------------------------ */

#define TBVH_MBVH_IMPL( handle, node_t, M ) \
TBVH_HANDLE_BASICS( handle ) \
TBVH_TRI_BUILDERS( handle ) \
void handle##_convert_from( handle* m, tbvh_bvh* bvh, bool compact ) { m->bvh.ConvertFrom( bvh->bvh, compact ); } \
void handle##_optimize( handle* m, uint32_t iterations, bool extreme ) { m->bvh.Optimize( iterations, extreme ); } \
void handle##_refit( handle* m, uint32_t node_idx ) { m->bvh.Refit( node_idx ); } \
uint32_t handle##_leaf_count( const handle* m, uint32_t node_idx ) { return m->bvh.LeafCount( node_idx ); } \
float handle##_sah_cost( const handle* m, uint32_t node_idx ) { return m->bvh.SAHCost( node_idx ); } \
bool handle##_load_self_contained( handle* m, const char* path ) \
{ \
	m->bvh = tb::MBVH<M>::Load( path ); \
	return m->bvh.usedNodes > 0; \
} \
const node_t* handle##_nodes( const handle* m, uint32_t* count ) \
{ \
	if (count) *count = m->bvh.usedNodes; \
	return reinterpret_cast<const node_t*>( m->bvh.mbvhNode ); \
} \
const uint32_t* handle##_prim_indices( const handle* m, uint32_t* count ) \
{ \
	if (count) *count = m->bvh.bvh.idxCount; \
	return m->bvh.bvh.primIdx; \
}

TBVH_MBVH_IMPL( tbvh_mbvh4, tbvh_mbvh4_node, 4 )
TBVH_MBVH_IMPL( tbvh_mbvh8, tbvh_mbvh8_node, 8 )
#undef TBVH_MBVH_IMPL

/* ------------------------------------------------------------------------------------------------
   tbvh_bvh4_cpu / tbvh_bvh8_cpu: the SIMD CPU layouts share everything but the bundle queries.
   ------------------------------------------------------------------------------------------------ */

#define TBVH_CPU_LAYOUT_IMPL( handle, cpptype, mhandle, ownflag, wide ) \
TBVH_HANDLE_BASICS( handle ) \
TBVH_TRI_BUILDERS( handle ) \
void handle##_convert_from( handle* b, mhandle* m, bool leaves_merged ) \
{ \
	/* ConvertFrom works on a copy of the MBVH handle and may reallocate its node pool, which would */ \
	/* leave 'm' with a dangling pointer: transfer ownership of the wide tree to 'b' instead. */ \
	b->bvh.ConvertFrom( m->bvh, leaves_merged ); \
	b->bvh.ownflag = true; \
	m->bvh.ReleaseOwnership(); \
	/* The wide tree's inner binary tree still references the caller's BVH; a later Build would */ \
	/* rebuild in place over those borrowed buffers, so drop the reference now (the layout */ \
	/* carries its own copy of the triangles and is not rebuildable anyway). */ \
	b->bvh.wide.bvh.DropReference( b->bvh.context ); \
	b->bvh.wide.ownBVH = true; \
} \
void handle##_optimize( handle* b, uint32_t iterations, bool extreme ) { b->bvh.Optimize( iterations, extreme ); } \
void handle##_refit( handle* b ) { b->bvh.Refit(); } \
float handle##_sah_cost( const handle* b, uint32_t node_idx ) { return b->bvh.SAHCost( node_idx ); } \
int32_t handle##_intersect( const handle* b, tbvh_ray* ray ) { return b->bvh.Intersect( ray_of( ray ) ); } \
bool handle##_is_occluded( const handle* b, const tbvh_ray* ray ) { return b->bvh.IsOccluded( ray_of( ray ) ); } \
void handle##_save( const handle* b, const char* path ) \
{ \
	const_cast<cpptype&>( b->bvh ).Save( path ); /* Save only reads the structure */ \
} \
bool handle##_load( handle* b, const char* path, uint32_t expected_tris ) { return b->bvh.Load( path, expected_tris ); } \
bool handle##_load_self_contained( handle* b, const char* path ) \
{ \
	b->bvh = cpptype::Load( path ); \
	return b->bvh.usedBlocks > 0; \
}

TBVH_CPU_LAYOUT_IMPL( tbvh_bvh4_cpu, tb::BVH4_CPU, tbvh_mbvh4, ownBVH4, bvh4 )
TBVH_CPU_LAYOUT_IMPL( tbvh_bvh8_cpu, tb::BVH8_CPU, tbvh_mbvh8, ownBVH8, bvh8 )
#undef TBVH_CPU_LAYOUT_IMPL

int32_t tbvh_bvh4_cpu_intersect_bundle( const tbvh_bvh4_cpu* b, tbvh_ray* rays ) { return b->bvh.IntersectBundle( rays_of( rays ) ); }
int32_t tbvh_bvh4_cpu_is_occluded_bundle( const tbvh_bvh4_cpu* b, tbvh_ray* rays, bool* occluded ) { return b->bvh.IsOccludedBundle( rays_of( rays ), occluded ); }
const uint8_t* tbvh_bvh4_cpu_data( const tbvh_bvh4_cpu* b, uint32_t* used_blocks )
{
	if (used_blocks) *used_blocks = b->bvh.usedBlocks;
	return reinterpret_cast<const uint8_t*>( b->bvh.bvh4Data );
}
const uint8_t* tbvh_bvh8_cpu_data( const tbvh_bvh8_cpu* b, uint32_t* used_blocks )
{
	if (used_blocks) *used_blocks = b->bvh.usedBlocks;
	return reinterpret_cast<const uint8_t*>( b->bvh.bvh8Data );
}

/* ------------------------------------------------------------------------------------------------
   tbvh_bvh4_gpu
   ------------------------------------------------------------------------------------------------ */

TBVH_HANDLE_BASICS( tbvh_bvh4_gpu )
TBVH_TRI_BUILDERS( tbvh_bvh4_gpu )

void tbvh_bvh4_gpu_convert_from( tbvh_bvh4_gpu* b, const tbvh_mbvh4* m, bool compact ) { b->bvh.ConvertFrom( m->bvh, compact ); }
void tbvh_bvh4_gpu_optimize( tbvh_bvh4_gpu* b, uint32_t iterations, bool extreme ) { b->bvh.Optimize( iterations, extreme ); }
float tbvh_bvh4_gpu_sah_cost( const tbvh_bvh4_gpu* b, uint32_t node_idx ) { return b->bvh.SAHCost( node_idx ); }
int32_t tbvh_bvh4_gpu_intersect( const tbvh_bvh4_gpu* b, tbvh_ray* ray ) { return b->bvh.Intersect( ray_of( ray ) ); }
bool tbvh_bvh4_gpu_is_occluded( const tbvh_bvh4_gpu* b, const tbvh_ray* ray ) { return b->bvh.IsOccluded( ray_of( ray ) ); }
bool tbvh_bvh4_gpu_load_self_contained( tbvh_bvh4_gpu* b, const char* path )
{
	b->bvh = tb::BVH4_GPU::Load( path );
	return b->bvh.usedBlocks > 0;
}
const tbvh_vec4* tbvh_bvh4_gpu_data( const tbvh_bvh4_gpu* b, uint32_t* used_blocks )
{
	if (used_blocks) *used_blocks = b->bvh.usedBlocks;
	return reinterpret_cast<const tbvh_vec4*>( b->bvh.bvh4Data );
}

/* ------------------------------------------------------------------------------------------------
   tbvh_bvh4_amd_hw
   ------------------------------------------------------------------------------------------------ */

TBVH_HANDLE_BASICS( tbvh_bvh4_amd_hw )
TBVH_TRI_BUILDERS( tbvh_bvh4_amd_hw )

void tbvh_bvh4_amd_hw_convert_from( tbvh_bvh4_amd_hw* b, const tbvh_mbvh4* m ) { b->bvh.ConvertFrom( m->bvh ); }
void tbvh_bvh4_amd_hw_optimize( tbvh_bvh4_amd_hw* b, uint32_t iterations, bool extreme ) { b->bvh.Optimize( iterations, extreme ); }
float tbvh_bvh4_amd_hw_sah_cost( const tbvh_bvh4_amd_hw* b, uint32_t node_idx ) { return b->bvh.SAHCost( node_idx ); }
bool tbvh_bvh4_amd_hw_load_self_contained( tbvh_bvh4_amd_hw* b, const char* path )
{
	b->bvh = tb::BVH4_AMD_HW::Load( path );
	return b->bvh.usedBlocks > 0;
}
const tbvh_vec4* tbvh_bvh4_amd_hw_data( const tbvh_bvh4_amd_hw* b, uint32_t* used_blocks )
{
	if (used_blocks) *used_blocks = b->bvh.usedBlocks;
	return reinterpret_cast<const tbvh_vec4*>( b->bvh.bvh4Data );
}

/* ------------------------------------------------------------------------------------------------
   tbvh_bvh8_cwbvh
   ------------------------------------------------------------------------------------------------ */

TBVH_HANDLE_BASICS( tbvh_bvh8_cwbvh )
TBVH_TRI_BUILDERS( tbvh_bvh8_cwbvh )

void tbvh_bvh8_cwbvh_convert_from( tbvh_bvh8_cwbvh* b, const tbvh_mbvh8* m, bool compact ) { b->bvh.ConvertFrom( m->bvh, compact ); }
void tbvh_bvh8_cwbvh_optimize( tbvh_bvh8_cwbvh* b, uint32_t iterations, bool extreme ) { b->bvh.Optimize( iterations, extreme ); }
float tbvh_bvh8_cwbvh_sah_cost( const tbvh_bvh8_cwbvh* b, uint32_t node_idx ) { return b->bvh.SAHCost( node_idx ); }
int32_t tbvh_bvh8_cwbvh_intersect( const tbvh_bvh8_cwbvh* b, tbvh_ray* ray ) { return b->bvh.Intersect( ray_of( ray ) ); }
bool tbvh_bvh8_cwbvh_is_occluded( const tbvh_bvh8_cwbvh* b, const tbvh_ray* ray ) { return b->bvh.IsOccluded( ray_of( ray ) ); }
void tbvh_bvh8_cwbvh_save( const tbvh_bvh8_cwbvh* b, const char* path )
{
	const_cast<tb::BVH8_CWBVH&>( b->bvh ).Save( path ); // Save only reads the structure
}
bool tbvh_bvh8_cwbvh_load( tbvh_bvh8_cwbvh* b, const char* path, uint32_t expected_tris ) { return b->bvh.Load( path, expected_tris ); }
bool tbvh_bvh8_cwbvh_load_self_contained( tbvh_bvh8_cwbvh* b, const char* path )
{
	b->bvh = tb::BVH8_CWBVH::Load( path );
	return b->bvh.usedBlocks > 0;
}
const tbvh_vec4* tbvh_bvh8_cwbvh_nodes( const tbvh_bvh8_cwbvh* b, uint32_t* used_blocks )
{
	if (used_blocks) *used_blocks = b->bvh.usedBlocks;
	return reinterpret_cast<const tbvh_vec4*>( b->bvh.bvh8Data );
}
const tbvh_vec4* tbvh_bvh8_cwbvh_tris( const tbvh_bvh8_cwbvh* b, uint32_t* used_blocks )
{
	if (used_blocks) *used_blocks = b->bvh.usedTriBlocks;
	return reinterpret_cast<const tbvh_vec4*>( b->bvh.bvh8Tris );
}

/* ------------------------------------------------------------------------------------------------
   tbvh_voxelset. Without ENABLE_VOXEL_SUPPORT the symbols still exist, as stubs.
   ------------------------------------------------------------------------------------------------ */

#ifdef ENABLE_VOXEL_SUPPORT

tbvh_voxelset* tbvh_voxelset_new( void ) { return new tbvh_voxelset(); }
void tbvh_voxelset_free( tbvh_voxelset* v ) { delete v; }
tbvh_base* tbvh_voxelset_base( tbvh_voxelset* v ) { return to_base( static_cast<tb::BVHBase*>( &v->vs ) ); }
const tbvh_base* tbvh_voxelset_base_const( const tbvh_voxelset* v ) { return to_base( static_cast<const tb::BVHBase*>( &v->vs ) ); }
void tbvh_voxelset_set( tbvh_voxelset* v, uint32_t x, uint32_t y, uint32_t z, uint32_t value ) { v->vs.Set( x, y, z, value ); }
void tbvh_voxelset_update_top_grid( tbvh_voxelset* v ) { v->vs.UpdateTopGrid(); }
int32_t tbvh_voxelset_intersect( const tbvh_voxelset* v, tbvh_ray* ray ) { return v->vs.Intersect( ray_of( ray ) ); }
bool tbvh_voxelset_is_occluded( const tbvh_voxelset* v, const tbvh_ray* ray ) { return v->vs.IsOccluded( ray_of( ray ) ); }
tbvh_vec3 tbvh_voxelset_normal( const tbvh_voxelset* v, const tbvh_ray* ray ) { return vec_to( v->vs.GetNormal( ray_of( ray ) ) ); }

#else

tbvh_voxelset* tbvh_voxelset_new( void ) { return nullptr; }
void tbvh_voxelset_free( tbvh_voxelset* ) {}
tbvh_base* tbvh_voxelset_base( tbvh_voxelset* ) { return nullptr; }
const tbvh_base* tbvh_voxelset_base_const( const tbvh_voxelset* ) { return nullptr; }
void tbvh_voxelset_set( tbvh_voxelset*, uint32_t, uint32_t, uint32_t, uint32_t ) {}
void tbvh_voxelset_update_top_grid( tbvh_voxelset* ) {}
int32_t tbvh_voxelset_intersect( const tbvh_voxelset*, tbvh_ray* ) { return 0; }
bool tbvh_voxelset_is_occluded( const tbvh_voxelset*, const tbvh_ray* ) { return false; }
tbvh_vec3 tbvh_voxelset_normal( const tbvh_voxelset*, const tbvh_ray* ) { tbvh_vec3 n = { 0, 0, 0 }; return n; }

#endif

/* ------------------------------------------------------------------------------------------------
   tbvh_bvh_double
   ------------------------------------------------------------------------------------------------ */

tbvh_bvh_double* tbvh_bvh_double_new( void ) { return new tbvh_bvh_double(); }
void tbvh_bvh_double_free( tbvh_bvh_double* b ) { delete b; }
tbvh_base_ex* tbvh_bvh_double_base( tbvh_bvh_double* b ) { return to_base( static_cast<tb::BVHBaseEx*>( &b->bvh ) ); }
const tbvh_base_ex* tbvh_bvh_double_base_const( const tbvh_bvh_double* b ) { return to_base( static_cast<const tb::BVHBaseEx*>( &b->bvh ) ); }

void tbvh_bvh_double_build( tbvh_bvh_double* b, const tbvh_dbl3* verts, uint64_t vert_count, uint64_t stride )
{
	b->bvh.Build( slice_of( verts, vert_count, stride ) );
}
void tbvh_bvh_double_build_indexed( tbvh_bvh_double* b, const tbvh_dbl3* verts, uint64_t vert_count, uint64_t stride, const uint64_t* indices, uint64_t prim_count )
{
	b->bvh.Build( slice_of( verts, vert_count, stride ), indices, prim_count );
}
void tbvh_bvh_double_build_quick( tbvh_bvh_double* b, const tbvh_dbl3* verts, uint64_t vert_count, uint64_t stride )
{
	b->bvh.BuildQuick( slice_of( verts, vert_count, stride ) );
}
void tbvh_bvh_double_build_hq( tbvh_bvh_double* b, const tbvh_dbl3* verts, uint64_t vert_count, uint64_t stride )
{
	b->bvh.BuildHQ( slice_of( verts, vert_count, stride ) );
}
void tbvh_bvh_double_build_hq_indexed( tbvh_bvh_double* b, const tbvh_dbl3* verts, uint64_t vert_count, uint64_t stride, const uint64_t* indices, uint64_t prim_count )
{
	b->bvh.BuildHQ( slice_of( verts, vert_count, stride ), indices, prim_count );
}
void tbvh_bvh_double_build_aabbs( tbvh_bvh_double* b, const tbvh_dbl3* aabbs, uint64_t aabb_count )
{
	b->bvh.BuildAABB( reinterpret_cast<const tb::bvhdbl3*>( aabbs ), aabb_count );
}
void tbvh_bvh_double_build_custom( tbvh_bvh_double* b, tbvh_aabb_ex_fn get_aabb, uint64_t prim_count, void* user )
{
	b->get_aabb = get_aabb;
	b->user = user;
	bvh_double_install_custom( b );
	b->bvh.Build( bvh_double_aabb_trampoline, prim_count );
}
void tbvh_bvh_double_set_custom_callbacks( tbvh_bvh_double* b, tbvh_intersect_ex_fn intersect, tbvh_occluded_ex_fn occluded, void* user )
{
	b->intersect = intersect;
	b->occluded = occluded;
	b->user = user;
	if (!intersect && !occluded)
	{
		b->bvh.customIntersect = 0, b->bvh.customIsOccluded = 0, b->bvh.customUserdata = 0;
		return;
	}
	bvh_double_install_custom( b );
}
void tbvh_bvh_double_build_tlas( tbvh_bvh_double* b, tbvh_blas_instance_ex* instances, uint64_t inst_count, tbvh_base_ex** blasses, uint64_t blas_count )
{
	b->bvh.Build( inst_of( instances ), inst_count, reinterpret_cast<tb::BVHBaseEx**>( blasses ), blas_count );
}

int32_t tbvh_bvh_double_intersect( const tbvh_bvh_double* b, tbvh_ray_ex* ray ) { return b->bvh.Intersect( ray_of( ray ) ); }
bool tbvh_bvh_double_is_occluded( const tbvh_bvh_double* b, const tbvh_ray_ex* ray ) { return b->bvh.IsOccluded( ray_of( ray ) ); }
bool tbvh_bvh_double_intersect_sphere( const tbvh_bvh_double* b, tbvh_dbl3 pos, double r ) { return b->bvh.IntersectSphere( vec_of( pos ), r ); }
void tbvh_bvh_double_refit( tbvh_bvh_double* b, uint64_t node_idx ) { b->bvh.Refit( node_idx ); }
void tbvh_bvh_double_compact( tbvh_bvh_double* b )
{
	b->bvh.Compact();
	if (b->bvh.mayHaveHoles())
	{
		tb::impl::BVH_Verbose<double, uint64_t> v;
		v.ConvertFrom( b->bvh, true );
		v.Compact();
		b->bvh.ConvertFrom( v, true );
	}
}
void tbvh_bvh_double_optimize( tbvh_bvh_double* b, uint32_t iterations, bool extreme, bool stochastic ) { b->bvh.Optimize( iterations, extreme, stochastic ); }
void tbvh_bvh_double_split_leafs( tbvh_bvh_double* b, uint64_t max_prims ) { b->bvh.SplitLeafs( max_prims ); }
void tbvh_bvh_double_merge_leafs( tbvh_bvh_double* b ) { b->bvh.MergeLeafs(); }
double tbvh_bvh_double_sah_cost( const tbvh_bvh_double* b, uint64_t node_idx ) { return b->bvh.SAHCost( node_idx ); }
double tbvh_bvh_double_epo_cost( const tbvh_bvh_double* b, uint64_t node_idx ) { return b->bvh.EPOCost( node_idx ); }
uint64_t tbvh_bvh_double_node_count( const tbvh_bvh_double* b ) { return b->bvh.NodeCount(); }
uint64_t tbvh_bvh_double_leaf_count( const tbvh_bvh_double* b ) { return b->bvh.LeafCount(); }
uint64_t tbvh_bvh_double_prim_count( const tbvh_bvh_double* b, uint64_t node_idx ) { return b->bvh.PrimCount( node_idx ); }
bool tbvh_bvh_double_is_tlas( const tbvh_bvh_double* b ) { return b->bvh.isTLAS(); }
bool tbvh_bvh_double_is_indexed( const tbvh_bvh_double* b ) { return b->bvh.isIndexed(); }
bool tbvh_bvh_double_has_custom_geometry( const tbvh_bvh_double* b ) { return b->bvh.hasCustomGeom(); }
void tbvh_bvh_double_save( const tbvh_bvh_double* b, const char* path, bool include_geometry )
{
	const_cast<tb::BVH_Double&>( b->bvh ).Save( path, include_geometry ); // Save only reads the structure
}
bool tbvh_bvh_double_load( tbvh_bvh_double* b, const char* path, const tbvh_dbl3* verts, uint64_t vert_count, uint64_t stride, const uint64_t* indices, uint64_t prim_count )
{
	const tb::bvhdbl3slice s = verts ? slice_of( verts, vert_count, stride ) : tb::bvhdbl3slice( nullptr, 0, 0 );
	return b->bvh.Load( path, s, indices, prim_count );
}
bool tbvh_bvh_double_load_self_contained( tbvh_bvh_double* b, const char* path )
{
	b->bvh = tb::BVH_Double::Load( path );
	return b->bvh.usedNodes > 0;
}
bool tbvh_bvh_double_owns_geometry( const tbvh_bvh_double* b ) { return b->bvh.ownsGeometry; }
bool tbvh_bvh_double_has_fragments( const tbvh_bvh_double* b ) { return b->bvh.fragment != 0; }
void tbvh_bvh_double_free_geometry( tbvh_bvh_double* b ) { b->bvh.FreeGeometry(); }
const tbvh_bvh_node_ex* tbvh_bvh_double_nodes( const tbvh_bvh_double* b, uint64_t* count )
{
	if (count) *count = b->bvh.usedNodes;
	return reinterpret_cast<const tbvh_bvh_node_ex*>( b->bvh.bvhNode );
}
const uint64_t* tbvh_bvh_double_prim_indices( const tbvh_bvh_double* b, uint64_t* count )
{
	if (count) *count = b->bvh.idxCount;
	return b->bvh.primIdx;
}
const tbvh_dbl3* tbvh_bvh_double_verts( const tbvh_bvh_double* b, uint64_t* count, uint64_t* stride )
{
	if (count) *count = b->bvh.verts.count;
	if (stride) *stride = b->bvh.verts.stride;
	return reinterpret_cast<const tbvh_dbl3*>( b->bvh.verts.data );
}
const uint64_t* tbvh_bvh_double_vert_indices( const tbvh_bvh_double* b ) { return b->bvh.vertIdx; }

/* ------------------------------------------------------------------------------------------------
   tbvh_bvh4_double
   ------------------------------------------------------------------------------------------------ */

tbvh_bvh4_double* tbvh_bvh4_double_new( void ) { return new tbvh_bvh4_double(); }
void tbvh_bvh4_double_free( tbvh_bvh4_double* b ) { delete b; }
tbvh_base_ex* tbvh_bvh4_double_base( tbvh_bvh4_double* b ) { return to_base( static_cast<tb::BVHBaseEx*>( &b->bvh ) ); }
const tbvh_base_ex* tbvh_bvh4_double_base_const( const tbvh_bvh4_double* b ) { return to_base( static_cast<const tb::BVHBaseEx*>( &b->bvh ) ); }

void tbvh_bvh4_double_build( tbvh_bvh4_double* b, const tbvh_dbl3* verts, uint64_t vert_count, uint64_t stride )
{
	b->bvh.Build( slice_of( verts, vert_count, stride ) );
}
void tbvh_bvh4_double_build_indexed( tbvh_bvh4_double* b, const tbvh_dbl3* verts, uint64_t vert_count, uint64_t stride, const uint64_t* indices, uint64_t prim_count )
{
	b->bvh.Build( slice_of( verts, vert_count, stride ), indices, prim_count );
}
void tbvh_bvh4_double_build_hq( tbvh_bvh4_double* b, const tbvh_dbl3* verts, uint64_t vert_count, uint64_t stride )
{
	b->bvh.BuildHQ( slice_of( verts, vert_count, stride ) );
}
void tbvh_bvh4_double_build_hq_indexed( tbvh_bvh4_double* b, const tbvh_dbl3* verts, uint64_t vert_count, uint64_t stride, const uint64_t* indices, uint64_t prim_count )
{
	b->bvh.BuildHQ( slice_of( verts, vert_count, stride ), indices, prim_count );
}
void tbvh_bvh4_double_optimize( tbvh_bvh4_double* b, uint32_t iterations, bool extreme ) { b->bvh.Optimize( iterations, extreme ); }
void tbvh_bvh4_double_refit( tbvh_bvh4_double* b ) { b->bvh.Refit(); }
double tbvh_bvh4_double_sah_cost( const tbvh_bvh4_double* b, uint64_t node_idx ) { return b->bvh.SAHCost( node_idx ); }
int32_t tbvh_bvh4_double_intersect( const tbvh_bvh4_double* b, tbvh_ray_ex* ray ) { return b->bvh.Intersect( ray_of( ray ) ); }
bool tbvh_bvh4_double_is_occluded( const tbvh_bvh4_double* b, const tbvh_ray_ex* ray ) { return b->bvh.IsOccluded( ray_of( ray ) ); }
void tbvh_bvh4_double_save( const tbvh_bvh4_double* b, const char* path )
{
	const_cast<tb::BVH4_Double&>( b->bvh ).Save( path ); // Save only reads the structure
}
bool tbvh_bvh4_double_load( tbvh_bvh4_double* b, const char* path, uint64_t expected_tris ) { return b->bvh.Load( path, expected_tris ); }
bool tbvh_bvh4_double_load_self_contained( tbvh_bvh4_double* b, const char* path )
{
	b->bvh = tb::BVH4_Double::Load( path );
	return b->bvh.usedBlocks > 0;
}
const uint8_t* tbvh_bvh4_double_data( const tbvh_bvh4_double* b, uint32_t* used_blocks )
{
	if (used_blocks) *used_blocks = b->bvh.usedBlocks;
	return reinterpret_cast<const uint8_t*>( b->bvh.bvh4Data );
}

#undef TBVH_HANDLE_BASICS
#undef TBVH_TRI_BUILDERS
