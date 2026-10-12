/*
tinybvh_c.h - a plain C interface over Jacco Bikker's tinybvh (https://github.com/jbikker/tinybvh).

tinybvh 1.9 is a header-only C++ library built from class templates. Templates cannot be called
through a foreign function interface, so this header exposes the single precision (float / uint32_t)
and double precision (double / uint64_t) instantiations as opaque handles with free functions.

Conventions
  * Every layout has tbvh_<layout>_new / tbvh_<layout>_free and tbvh_<layout>_base, which upcasts the
    handle to the shared tbvh_base (tinybvh::BVHBase) for the flags and settings common to all layouts.
  * Geometry is referenced, never copied: vertex and index arrays passed to a build function must stay
    alive and unchanged for as long as the structure is used (this includes derived layouts).
  * Vertex arrays are given as a (pointer, vertex count, stride in bytes) triple, three vertices per
    triangle. The stride must be a multiple of 16; pass sizeof(tbvh_vec4) for a tightly packed array.
  * All struct layouts below mirror the tinybvh types bit for bit; tinybvh_c.cpp verifies this with
    static assertions and tbvh_sizeof() lets a binding verify it at run time.
  * Fatal errors inside tinybvh (invalid input, corrupt trees) call the handler installed with
    tbvh_set_fatal_error_handler, then exit(1) if the handler returns.
*/

#ifndef TINYBVH_C_H_
#define TINYBVH_C_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if defined _MSC_VER
#define TBVH_ALIGN( x ) __declspec( align( x ) )
#else
#define TBVH_ALIGN( x ) __attribute__( ( aligned( x ) ) )
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------------------------------------
   Plain data types
   ------------------------------------------------------------------------------------------------ */

typedef struct tbvh_vec3 { float x, y, z; } tbvh_vec3;                               /* 12 bytes, align 4 */
typedef struct TBVH_ALIGN( 16 ) tbvh_vec4 { float x, y, z, w; } tbvh_vec4;           /* 16 bytes, align 16 */
typedef struct tbvh_dbl3 { double x, y, z; } tbvh_dbl3;                              /* 24 bytes, align 8 */
typedef struct tbvh_mat4 { float cell[16]; } tbvh_mat4;                              /* row-major, 64 bytes */
typedef struct tbvh_dblmat4 { double cell[16]; } tbvh_dblmat4;                       /* row-major, 128 bytes */

/* Hit record of a single precision ray (tinybvh::Intersection). Packed to 4 bytes: 84 bytes. */
#pragma pack( push, 4 )
typedef struct tbvh_intersection
{
	uint32_t inst;        /* instance index, for TLAS traversal */
	float t, u, v;        /* distance along the ray, barycentrics of the hit */
	uint32_t prim;        /* primitive index within the BLAS */
	void* aux_data;       /* user pointer, untouched by tinybvh */
	uint32_t user[14];    /* 56 bytes of user payload, untouched by tinybvh */
} tbvh_intersection;

/* Hit record of a double precision ray (tinybvh::IntersectionEx): 104 bytes. */
typedef struct tbvh_intersection_ex
{
	uint64_t inst;
	double t, u, v;
	uint64_t prim;
	void* aux_data;
	uint32_t user[14];
} tbvh_intersection_ex;
#pragma pack( pop )

/* Single precision ray (tinybvh::Ray): 128 bytes, 64-byte aligned.
   rd must hold the reciprocal direction; use tbvh_ray_init or tbvh_ray_set_direction. */
typedef struct TBVH_ALIGN( 64 ) tbvh_ray
{
	tbvh_vec3 o; uint32_t mask;         /* origin; 16-bit instance mask, 0xFFFF intersects everything */
	tbvh_vec3 d; uint32_t inst_idx;     /* direction; instance index, set during TLAS traversal */
	tbvh_vec3 rd;                       /* reciprocal direction */
	tbvh_intersection hit;              /* offset 44 */
} tbvh_ray;

/* Double precision ray (tinybvh::RayEx): 192 bytes, 64-byte aligned. */
typedef struct TBVH_ALIGN( 64 ) tbvh_ray_ex
{
	tbvh_dbl3 o; uint32_t mask; uint32_t pad0;
	tbvh_dbl3 d; uint64_t inst_idx;
	tbvh_dbl3 rd;
	tbvh_intersection_ex hit;           /* offset 88 */
} tbvh_ray_ex;

/* Instance of a BLAS in a TLAS (tinybvh::BLASInstance): 192 bytes, 64-byte aligned. */
typedef struct TBVH_ALIGN( 64 ) tbvh_blas_instance
{
	tbvh_mat4 transform;      /* object to world, row-major; defaults to identity */
	tbvh_mat4 inv_transform;  /* world to object; computed by the TLAS build */
	tbvh_vec3 aabb_min;       /* world-space bounds; computed by the TLAS build */
	uint32_t blas_idx;        /* index into the BLAS list passed to the TLAS build */
	tbvh_vec3 aabb_max;
	uint32_t mask;            /* 16-bit mask, matched against the ray mask */
} tbvh_blas_instance;

/* Double precision instance (tinybvh::BLASInstanceEx): 320 bytes, 64-byte aligned. */
typedef struct TBVH_ALIGN( 64 ) tbvh_blas_instance_ex
{
	tbvh_dblmat4 transform;
	tbvh_dblmat4 inv_transform;
	tbvh_dbl3 aabb_min;
	uint64_t blas_idx;
	tbvh_dbl3 aabb_max;
	uint32_t mask;
} tbvh_blas_instance_ex;

/* Build settings (tinybvh::BVHBuildSettings): 40 bytes. See the tinybvh README for the builders. */
typedef struct tbvh_build_settings
{
	bool use_presplitting;          /* pre-split triangles before building */
	bool use_spatial_splits;        /* Bonsai HQ, or SBVH with binned_bvh */
	float min_spatial_overlap;      /* SBVH: minimal overlap for a spatial split */
	bool spatial_skip_smallest_axis;
	bool presplit_post_pass;
	float presplit_factor;
	bool use_full_sweep;            /* experimental full-sweep SAH builder */
	bool binned_bvh;                /* binned SAH instead of Bonsai */
	uint32_t bonsai_grid_prims;
	float bonsai_prune_factor;
	bool use_hploc;                 /* H-PLOC builder */
	bool post_optimize;             /* run the optimizer after the build */
	int32_t optimize_iterations;
	bool use_simd_if_available;
	bool enable_threading;
	bool discard_intermediates;     /* free the intermediate BVH after a layout conversion */
	bool minimize_mem_use;
} tbvh_build_settings;

/* Layout identifiers (tinybvh::BVHType). */
typedef enum tbvh_layout
{
	TBVH_LAYOUT_UNDEFINED = 0,
	TBVH_LAYOUT_BVH = 1,
	TBVH_LAYOUT_BVH_VERBOSE = 2,
	TBVH_LAYOUT_BVH_GPU = 3,
	TBVH_LAYOUT_MBVH = 4,
	TBVH_LAYOUT_BVH4_CPU = 5,
	TBVH_LAYOUT_BVH4_GPU = 6,
	TBVH_LAYOUT_CWBVH = 7,
	TBVH_LAYOUT_BVH8_SIMD = 8,
	TBVH_LAYOUT_VOXELSET = 9,
	TBVH_LAYOUT_BVH4_AMD_HW = 10
} tbvh_layout;

/* Node formats that are exposed as arrays. */
typedef struct tbvh_bvh_node          /* tinybvh::BVH::BVHNode, 32 bytes */
{
	tbvh_vec3 aabb_min; uint32_t left_first;   /* leaf: first prim index slot; interior: left child */
	tbvh_vec3 aabb_max; uint32_t tri_count;    /* > 0 for a leaf */
} tbvh_bvh_node;

typedef struct tbvh_bvh_node_ex       /* tinybvh::BVH_Double::BVHNode, 64 bytes */
{
	tbvh_dbl3 aabb_min; uint64_t left_first;
	tbvh_dbl3 aabb_max; uint64_t tri_count;
} tbvh_bvh_node_ex;

typedef struct tbvh_bvh_verbose_node  /* tinybvh::BVH_Verbose::BVHNode, 64 bytes */
{
	tbvh_vec3 aabb_min; uint32_t left;
	tbvh_vec3 aabb_max; uint32_t right;
	uint32_t tri_count, first_tri, parent;
	float dummy[5];
} tbvh_bvh_verbose_node;

typedef struct tbvh_bvh_gpu_node      /* tinybvh::BVH_GPU::BVHNode, 64 bytes (Aila & Laine) */
{
	tbvh_vec3 lmin; uint32_t left;
	tbvh_vec3 lmax; uint32_t right;
	tbvh_vec3 rmin; uint32_t tri_count;
	tbvh_vec3 rmax; uint32_t first_tri;
} tbvh_bvh_gpu_node;

typedef struct tbvh_mbvh4_node        /* tinybvh::MBVH<4>::MBVHNode, 64 bytes */
{
	tbvh_vec3 aabb_min; uint32_t first_tri;
	tbvh_vec3 aabb_max; uint32_t tri_count;
	uint32_t child[4];
	uint32_t child_count;
	uint32_t dummy[3];
} tbvh_mbvh4_node;

typedef struct tbvh_mbvh8_node        /* tinybvh::MBVH<8>::MBVHNode, 80 bytes */
{
	tbvh_vec3 aabb_min; uint32_t first_tri;
	tbvh_vec3 aabb_max; uint32_t tri_count;
	uint32_t child[8];
	uint32_t child_count;
	uint32_t dummy[3];
} tbvh_mbvh8_node;

/* Identifiers for tbvh_sizeof, so a binding can verify its struct mirrors. */
typedef enum tbvh_struct_id
{
	TBVH_STRUCT_VEC3 = 0,
	TBVH_STRUCT_VEC4,
	TBVH_STRUCT_DBL3,
	TBVH_STRUCT_MAT4,
	TBVH_STRUCT_DBLMAT4,
	TBVH_STRUCT_INTERSECTION,
	TBVH_STRUCT_INTERSECTION_EX,
	TBVH_STRUCT_RAY,
	TBVH_STRUCT_RAY_EX,
	TBVH_STRUCT_BLAS_INSTANCE,
	TBVH_STRUCT_BLAS_INSTANCE_EX,
	TBVH_STRUCT_BUILD_SETTINGS,
	TBVH_STRUCT_BVH_NODE,
	TBVH_STRUCT_BVH_NODE_EX,
	TBVH_STRUCT_BVH_VERBOSE_NODE,
	TBVH_STRUCT_BVH_GPU_NODE,
	TBVH_STRUCT_MBVH4_NODE,
	TBVH_STRUCT_MBVH8_NODE,
	TBVH_STRUCT_COUNT
} tbvh_struct_id;

/* ------------------------------------------------------------------------------------------------
   Opaque handles
   ------------------------------------------------------------------------------------------------ */

typedef struct tbvh_base tbvh_base;                 /* tinybvh::BVHBase, single precision */
typedef struct tbvh_base_ex tbvh_base_ex;           /* tinybvh::BVHBaseEx, double precision */
typedef struct tbvh_bvh tbvh_bvh;                   /* tinybvh::BVH */
typedef struct tbvh_bvh_verbose tbvh_bvh_verbose;   /* tinybvh::BVH_Verbose */
typedef struct tbvh_bvh_gpu tbvh_bvh_gpu;           /* tinybvh::BVH_GPU */
typedef struct tbvh_mbvh4 tbvh_mbvh4;               /* tinybvh::MBVH<4> */
typedef struct tbvh_mbvh8 tbvh_mbvh8;               /* tinybvh::MBVH<8> */
typedef struct tbvh_bvh4_cpu tbvh_bvh4_cpu;         /* tinybvh::BVH4_CPU */
typedef struct tbvh_bvh4_gpu tbvh_bvh4_gpu;         /* tinybvh::BVH4_GPU */
typedef struct tbvh_bvh4_amd_hw tbvh_bvh4_amd_hw;   /* tinybvh::BVH4_AMD_HW */
typedef struct tbvh_bvh8_cwbvh tbvh_bvh8_cwbvh;     /* tinybvh::BVH8_CWBVH */
typedef struct tbvh_bvh8_cpu tbvh_bvh8_cpu;         /* tinybvh::BVH8_CPU */
typedef struct tbvh_voxelset tbvh_voxelset;         /* tinybvh::VoxelSet */
typedef struct tbvh_bvh_double tbvh_bvh_double;     /* tinybvh::BVH_Double */
typedef struct tbvh_bvh4_double tbvh_bvh4_double;   /* tinybvh::BVH4_Double */

/* ------------------------------------------------------------------------------------------------
   Callbacks for custom geometry
   ------------------------------------------------------------------------------------------------ */

/* Bounds of primitive 'prim'. May be called from several threads at once during a threaded build. */
typedef void ( *tbvh_aabb_fn )( uint32_t prim, tbvh_vec3* bmin, tbvh_vec3* bmax, void* user );
/* Closest hit test: update ray->hit (t must shrink) and return true on a hit. */
typedef bool ( *tbvh_intersect_fn )( tbvh_ray* ray, uint32_t prim, void* user );
/* Any hit test: return true if the ray hits the primitive before ray->hit.t. */
typedef bool ( *tbvh_occluded_fn )( const tbvh_ray* ray, uint32_t prim, void* user );

typedef void ( *tbvh_aabb_ex_fn )( uint64_t prim, tbvh_dbl3* bmin, tbvh_dbl3* bmax, void* user );
typedef bool ( *tbvh_intersect_ex_fn )( tbvh_ray_ex* ray, uint64_t prim, void* user );
typedef bool ( *tbvh_occluded_ex_fn )( const tbvh_ray_ex* ray, uint64_t prim, void* user );

/* Fatal error reporter. The message is a NUL-terminated string owned by tinybvh for the duration
   of the call. The handler may terminate the process; if it returns, tinybvh calls exit(1). */
typedef void ( *tbvh_fatal_error_fn )( const char* message, void* user );

/* ------------------------------------------------------------------------------------------------
   Library-wide functions
   ------------------------------------------------------------------------------------------------ */

void tbvh_version( int32_t* major, int32_t* minor, int32_t* sub );
uint32_t tbvh_bundle_rays( void );            /* rays per bundle call (TINYBVH_BUNDLE_RAYS) */
uint32_t tbvh_inst_idx_bits( void );          /* INST_IDX_BITS the library was compiled with */
uint32_t tbvh_stack_size( void );             /* TINYBVH_STACK_SIZE */
bool tbvh_simd_enabled( void );               /* true when the full SIMD traversal path (AVX2 / NEON) is compiled in */
bool tbvh_simd_builder_available( void );     /* true when BuildSIMD / Bonsai have a SIMD builder (AVX / NEON) */
float tbvh_far( void );                       /* BVH_FAR, the 'no hit' distance */
double tbvh_dbl_far( void );                  /* BVH_DBL_FAR */
uint32_t tbvh_voxelset_object_dim( void );    /* VoxelSet::objectDim */
size_t tbvh_sizeof( tbvh_struct_id id );      /* sizeof the C++ type behind a mirrored struct; 0 for an unknown id */
void tbvh_set_fatal_error_handler( tbvh_fatal_error_fn handler, void* user );  /* NULL restores the default */
void tbvh_shutdown_builtin_pool( void );      /* release the built-in build thread pool; re-created on demand */

/* Ray helpers. tbvh_ray_init matches the tinybvh::Ray constructor: direction is not normalized. */
void tbvh_ray_init( tbvh_ray* ray, tbvh_vec3 origin, tbvh_vec3 direction, float t, uint32_t mask );
void tbvh_ray_set_direction( tbvh_ray* ray, tbvh_vec3 direction );   /* also updates rd */
void tbvh_ray_normalize( tbvh_ray* ray );                            /* Ray::Normalize */
void tbvh_ray_ex_init( tbvh_ray_ex* ray, tbvh_dbl3 origin, tbvh_dbl3 direction, double t, uint32_t mask );
void tbvh_ray_ex_set_direction( tbvh_ray_ex* ray, tbvh_dbl3 direction );
void tbvh_ray_ex_normalize( tbvh_ray_ex* ray );

/* Instance helpers. */
void tbvh_blas_instance_init( tbvh_blas_instance* inst, uint32_t blas_idx );  /* identity transform, full mask */
void tbvh_blas_instance_invert_transform( tbvh_blas_instance* inst );        /* recompute inv_transform */
void tbvh_blas_instance_update( tbvh_blas_instance* inst, const tbvh_base* blas ); /* inverse + world bounds */
void tbvh_blas_instance_ex_init( tbvh_blas_instance_ex* inst, uint64_t blas_idx );
void tbvh_blas_instance_ex_invert_transform( tbvh_blas_instance_ex* inst );
void tbvh_blas_instance_ex_update( tbvh_blas_instance_ex* inst, const tbvh_base_ex* blas );

/* Transform helpers, row-major matrices as used by the instances. */
tbvh_vec3 tbvh_transform_point( tbvh_vec3 p, const tbvh_mat4* m );
tbvh_vec3 tbvh_transform_vector( tbvh_vec3 v, const tbvh_mat4* m );

/* ------------------------------------------------------------------------------------------------
   tbvh_base: state shared by every single precision layout
   ------------------------------------------------------------------------------------------------ */

tbvh_layout tbvh_base_layout( const tbvh_base* b );
void tbvh_base_get_settings( const tbvh_base* b, tbvh_build_settings* out );
void tbvh_base_set_settings( tbvh_base* b, const tbvh_build_settings* settings );
float tbvh_base_get_c_trav( const tbvh_base* b );
float tbvh_base_get_c_int( const tbvh_base* b );
void tbvh_base_set_c_trav( tbvh_base* b, float c );
void tbvh_base_set_c_int( tbvh_base* b, float c );
uint32_t tbvh_base_get_hq_bins( const tbvh_base* b );        /* hqbvhbins */
void tbvh_base_set_hq_bins( tbvh_base* b, uint32_t bins );   /* clamped to [1, MAXHQBINS] by the caller */
bool tbvh_base_get_hq_odd_even( const tbvh_base* b );        /* hqbvhoddeven */
void tbvh_base_set_hq_odd_even( tbvh_base* b, bool oddeven );
uint32_t tbvh_base_get_hq_leaf_prims( const tbvh_base* b );
void tbvh_base_set_hq_leaf_prims( tbvh_base* b, uint32_t prims );
tbvh_vec3 tbvh_base_aabb_min( const tbvh_base* b );
tbvh_vec3 tbvh_base_aabb_max( const tbvh_base* b );
uint32_t tbvh_base_tri_count( const tbvh_base* b );
uint32_t tbvh_base_idx_count( const tbvh_base* b );
uint32_t tbvh_base_used_nodes( const tbvh_base* b );
uint32_t tbvh_base_allocated_nodes( const tbvh_base* b );
bool tbvh_base_is_refittable( const tbvh_base* b );
bool tbvh_base_is_rebuildable( const tbvh_base* b );
bool tbvh_base_may_have_holes( const tbvh_base* b );
bool tbvh_base_has_ordered_nodes( const tbvh_base* b );
bool tbvh_base_is_over_aabbs( const tbvh_base* b );
bool tbvh_base_is_over_indices( const tbvh_base* b );
/* Opacity micro maps: n x n bits per triangle, (n*n+31)/32 uint32 words per triangle. The data
   must outlive the structure. n == 0 or map == NULL detaches. */
void tbvh_base_set_opacity_micro_maps( tbvh_base* b, const uint32_t* map, uint32_t n );
uint32_t tbvh_base_opacity_micro_map_n( const tbvh_base* b );

/* Double precision counterpart. */
tbvh_layout tbvh_base_ex_layout( const tbvh_base_ex* b );
void tbvh_base_ex_get_settings( const tbvh_base_ex* b, tbvh_build_settings* out );
void tbvh_base_ex_set_settings( tbvh_base_ex* b, const tbvh_build_settings* settings );
float tbvh_base_ex_get_c_trav( const tbvh_base_ex* b );
float tbvh_base_ex_get_c_int( const tbvh_base_ex* b );
void tbvh_base_ex_set_c_trav( tbvh_base_ex* b, float c );
void tbvh_base_ex_set_c_int( tbvh_base_ex* b, float c );
uint32_t tbvh_base_ex_get_hq_bins( const tbvh_base_ex* b );
void tbvh_base_ex_set_hq_bins( tbvh_base_ex* b, uint32_t bins );
bool tbvh_base_ex_get_hq_odd_even( const tbvh_base_ex* b );
void tbvh_base_ex_set_hq_odd_even( tbvh_base_ex* b, bool oddeven );
uint32_t tbvh_base_ex_get_hq_leaf_prims( const tbvh_base_ex* b );
void tbvh_base_ex_set_hq_leaf_prims( tbvh_base_ex* b, uint32_t prims );
tbvh_dbl3 tbvh_base_ex_aabb_min( const tbvh_base_ex* b );
tbvh_dbl3 tbvh_base_ex_aabb_max( const tbvh_base_ex* b );
uint64_t tbvh_base_ex_tri_count( const tbvh_base_ex* b );
uint64_t tbvh_base_ex_idx_count( const tbvh_base_ex* b );
uint64_t tbvh_base_ex_used_nodes( const tbvh_base_ex* b );
uint64_t tbvh_base_ex_allocated_nodes( const tbvh_base_ex* b );
bool tbvh_base_ex_is_refittable( const tbvh_base_ex* b );
bool tbvh_base_ex_is_rebuildable( const tbvh_base_ex* b );
bool tbvh_base_ex_may_have_holes( const tbvh_base_ex* b );
bool tbvh_base_ex_has_ordered_nodes( const tbvh_base_ex* b );
bool tbvh_base_ex_is_over_aabbs( const tbvh_base_ex* b );
bool tbvh_base_ex_is_over_indices( const tbvh_base_ex* b );
void tbvh_base_ex_set_opacity_micro_maps( tbvh_base_ex* b, const uint32_t* map, uint32_t n );
uint32_t tbvh_base_ex_opacity_micro_map_n( const tbvh_base_ex* b );

/* ------------------------------------------------------------------------------------------------
   tbvh_bvh: tinybvh::BVH, the 32-byte node layout every other layout is derived from.
   Also the only layout that can serve as a TLAS.
   ------------------------------------------------------------------------------------------------ */

tbvh_bvh* tbvh_bvh_new( void );
void tbvh_bvh_free( tbvh_bvh* bvh );
tbvh_base* tbvh_bvh_base( tbvh_bvh* bvh );
const tbvh_base* tbvh_bvh_base_const( const tbvh_bvh* bvh );

/* Builders over triangles. 'verts' holds 'vert_count' vertices, three per triangle, 'stride' bytes apart. */
void tbvh_bvh_build( tbvh_bvh* bvh, const tbvh_vec4* verts, uint32_t vert_count, uint32_t stride );
void tbvh_bvh_build_indexed( tbvh_bvh* bvh, const tbvh_vec4* verts, uint32_t vert_count, uint32_t stride, const uint32_t* indices, uint32_t prim_count );
void tbvh_bvh_build_quick( tbvh_bvh* bvh, const tbvh_vec4* verts, uint32_t vert_count, uint32_t stride );
void tbvh_bvh_build_hq( tbvh_bvh* bvh, const tbvh_vec4* verts, uint32_t vert_count, uint32_t stride );
void tbvh_bvh_build_hq_indexed( tbvh_bvh* bvh, const tbvh_vec4* verts, uint32_t vert_count, uint32_t stride, const uint32_t* indices, uint32_t prim_count );
/* BuildSIMD: only valid when tbvh_simd_builder_available() is true. */
void tbvh_bvh_build_simd( tbvh_bvh* bvh, const tbvh_vec4* verts, uint32_t vert_count, uint32_t stride );
void tbvh_bvh_build_simd_indexed( tbvh_bvh* bvh, const tbvh_vec4* verts, uint32_t vert_count, uint32_t stride, const uint32_t* indices, uint32_t prim_count );
/* BVH over axis-aligned boxes: aabbs holds 2 * aabb_count vectors, (min, max) per box. */
void tbvh_bvh_build_aabbs( tbvh_bvh* bvh, const tbvh_vec4* aabbs, uint32_t aabb_count );
/* BVH over custom primitives: bounds come from get_aabb; the intersection callbacks are installed
   with tbvh_bvh_set_custom_callbacks and 'user' is passed to all three. */
void tbvh_bvh_build_custom( tbvh_bvh* bvh, tbvh_aabb_fn get_aabb, uint32_t prim_count, void* user );
void tbvh_bvh_set_custom_callbacks( tbvh_bvh* bvh, tbvh_intersect_fn intersect, tbvh_occluded_fn occluded, void* user );
/* TLAS over instances. Both arrays must outlive the TLAS; the instances are updated in place. */
void tbvh_bvh_build_tlas( tbvh_bvh* bvh, tbvh_blas_instance* instances, uint32_t inst_count, tbvh_base** blasses, uint32_t blas_count );

/* Queries. */
int32_t tbvh_bvh_intersect( const tbvh_bvh* bvh, tbvh_ray* ray );                 /* returns traversal steps */
bool tbvh_bvh_is_occluded( const tbvh_bvh* bvh, const tbvh_ray* ray );
bool tbvh_bvh_intersect_sphere( const tbvh_bvh* bvh, tbvh_vec3 pos, float r );
/* Bundles: 'rays' points at tbvh_bundle_rays() rays; 'occluded' at as many bools. TLAS only. */
int32_t tbvh_bvh_intersect_bundle( const tbvh_bvh* bvh, tbvh_ray* rays );
int32_t tbvh_bvh_is_occluded_bundle( const tbvh_bvh* bvh, tbvh_ray* rays, bool* occluded );

/* Maintenance and metrics. */
void tbvh_bvh_refit( tbvh_bvh* bvh, uint32_t node_idx );
void tbvh_bvh_compact( tbvh_bvh* bvh );
void tbvh_bvh_optimize( tbvh_bvh* bvh, uint32_t iterations, bool extreme, bool stochastic );
void tbvh_bvh_split_leafs( tbvh_bvh* bvh, uint32_t max_prims );
void tbvh_bvh_merge_leafs( tbvh_bvh* bvh );
float tbvh_bvh_sah_cost( const tbvh_bvh* bvh, uint32_t node_idx );
float tbvh_bvh_epo_cost( const tbvh_bvh* bvh, uint32_t node_idx );
uint32_t tbvh_bvh_node_count( const tbvh_bvh* bvh );
uint32_t tbvh_bvh_leaf_count( const tbvh_bvh* bvh );
uint32_t tbvh_bvh_prim_count( const tbvh_bvh* bvh, uint32_t node_idx );
uint32_t tbvh_bvh_collect_nodes( const tbvh_bvh* bvh, uint32_t root, uint32_t* list, uint32_t cap );
bool tbvh_bvh_is_tlas( const tbvh_bvh* bvh );
bool tbvh_bvh_is_indexed( const tbvh_bvh* bvh );
bool tbvh_bvh_has_custom_geometry( const tbvh_bvh* bvh );

/* Serialization. tbvh_bvh_load rebinds the loaded tree to the given geometry (indices may be NULL);
   tbvh_bvh_load_self_contained loads geometry embedded by a save with include_geometry. */
void tbvh_bvh_save( const tbvh_bvh* bvh, const char* path, bool include_geometry );
bool tbvh_bvh_load( tbvh_bvh* bvh, const char* path, const tbvh_vec4* verts, uint32_t vert_count, uint32_t stride, const uint32_t* indices, uint32_t prim_count );
bool tbvh_bvh_load_self_contained( tbvh_bvh* bvh, const char* path );
bool tbvh_bvh_owns_geometry( const tbvh_bvh* bvh );
/* Whether the per-primitive bounds of the build are still available; a tree loaded from a file has
   none, and tbvh_bvh_split_leafs / tbvh_bvh_verbose_split_leafs must not be called on it. */
bool tbvh_bvh_has_fragments( const tbvh_bvh* bvh );
void tbvh_bvh_free_geometry( tbvh_bvh* bvh );

/* Data access. Pointers are valid until the next build / free. */
const tbvh_bvh_node* tbvh_bvh_nodes( const tbvh_bvh* bvh, uint32_t* count );   /* used nodes */
const uint32_t* tbvh_bvh_prim_indices( const tbvh_bvh* bvh, uint32_t* count ); /* idx_count entries */
const tbvh_vec4* tbvh_bvh_verts( const tbvh_bvh* bvh, uint32_t* count, uint32_t* stride );
const uint32_t* tbvh_bvh_vert_indices( const tbvh_bvh* bvh );                  /* 3 * tri_count, or NULL */

/* Conversions. */
void tbvh_bvh_convert_from_verbose( tbvh_bvh* bvh, const tbvh_bvh_verbose* verbose, bool compact );

/* ------------------------------------------------------------------------------------------------
   tbvh_bvh_verbose: tinybvh::BVH_Verbose, 64-byte nodes with parent links, for tree surgery.
   ------------------------------------------------------------------------------------------------ */

tbvh_bvh_verbose* tbvh_bvh_verbose_new( void );
void tbvh_bvh_verbose_free( tbvh_bvh_verbose* v );
tbvh_base* tbvh_bvh_verbose_base( tbvh_bvh_verbose* v );
const tbvh_base* tbvh_bvh_verbose_base_const( const tbvh_bvh_verbose* v );
void tbvh_bvh_verbose_convert_from( tbvh_bvh_verbose* v, const tbvh_bvh* bvh, bool compact );
float tbvh_bvh_verbose_sah_cost( const tbvh_bvh_verbose* v, uint32_t node_idx );
int32_t tbvh_bvh_verbose_node_count( const tbvh_bvh_verbose* v );
void tbvh_bvh_verbose_refit( tbvh_bvh_verbose* v, uint32_t node_idx, bool skip_leafs );
void tbvh_bvh_verbose_compact( tbvh_bvh_verbose* v );
void tbvh_bvh_verbose_sort_indices( tbvh_bvh_verbose* v );
void tbvh_bvh_verbose_split_leafs( tbvh_bvh_verbose* v, uint32_t max_prims );
void tbvh_bvh_verbose_merge_leafs( tbvh_bvh_verbose* v );
void tbvh_bvh_verbose_optimize( tbvh_bvh_verbose* v, uint32_t iterations, bool extreme, bool stochastic );
const tbvh_bvh_verbose_node* tbvh_bvh_verbose_nodes( const tbvh_bvh_verbose* v, uint32_t* count );
const uint32_t* tbvh_bvh_verbose_prim_indices( const tbvh_bvh_verbose* v, uint32_t* count );

/* ------------------------------------------------------------------------------------------------
   tbvh_bvh_gpu: tinybvh::BVH_GPU, 64-byte Aila & Laine nodes for simple GPU kernels.
   ------------------------------------------------------------------------------------------------ */

tbvh_bvh_gpu* tbvh_bvh_gpu_new( void );
void tbvh_bvh_gpu_free( tbvh_bvh_gpu* g );
tbvh_base* tbvh_bvh_gpu_base( tbvh_bvh_gpu* g );
const tbvh_base* tbvh_bvh_gpu_base_const( const tbvh_bvh_gpu* g );
void tbvh_bvh_gpu_build( tbvh_bvh_gpu* g, const tbvh_vec4* verts, uint32_t vert_count, uint32_t stride );
void tbvh_bvh_gpu_build_indexed( tbvh_bvh_gpu* g, const tbvh_vec4* verts, uint32_t vert_count, uint32_t stride, const uint32_t* indices, uint32_t prim_count );
void tbvh_bvh_gpu_build_hq( tbvh_bvh_gpu* g, const tbvh_vec4* verts, uint32_t vert_count, uint32_t stride );
void tbvh_bvh_gpu_build_hq_indexed( tbvh_bvh_gpu* g, const tbvh_vec4* verts, uint32_t vert_count, uint32_t stride, const uint32_t* indices, uint32_t prim_count );
void tbvh_bvh_gpu_build_aabbs( tbvh_bvh_gpu* g, const tbvh_vec4* aabbs, uint32_t aabb_count );
void tbvh_bvh_gpu_build_tlas( tbvh_bvh_gpu* g, tbvh_blas_instance* instances, uint32_t inst_count, tbvh_base** blasses, uint32_t blas_count );
void tbvh_bvh_gpu_convert_from( tbvh_bvh_gpu* g, const tbvh_bvh* bvh, bool compact );
void tbvh_bvh_gpu_optimize( tbvh_bvh_gpu* g, uint32_t iterations, bool extreme );
float tbvh_bvh_gpu_sah_cost( tbvh_bvh_gpu* g, uint32_t node_idx );
int32_t tbvh_bvh_gpu_intersect( const tbvh_bvh_gpu* g, tbvh_ray* ray );
bool tbvh_bvh_gpu_is_occluded( const tbvh_bvh_gpu* g, const tbvh_ray* ray );
bool tbvh_bvh_gpu_load_self_contained( tbvh_bvh_gpu* g, const char* path );
const tbvh_bvh_gpu_node* tbvh_bvh_gpu_nodes( const tbvh_bvh_gpu* g, uint32_t* count );
const uint32_t* tbvh_bvh_gpu_prim_indices( const tbvh_bvh_gpu* g, uint32_t* count );

/* ------------------------------------------------------------------------------------------------
   tbvh_mbvh4 / tbvh_mbvh8: tinybvh::MBVH<4> and MBVH<8>, the wide intermediate layouts.
   These have no traversal of their own.
   ------------------------------------------------------------------------------------------------ */

tbvh_mbvh4* tbvh_mbvh4_new( void );
void tbvh_mbvh4_free( tbvh_mbvh4* m );
tbvh_base* tbvh_mbvh4_base( tbvh_mbvh4* m );
const tbvh_base* tbvh_mbvh4_base_const( const tbvh_mbvh4* m );
void tbvh_mbvh4_build( tbvh_mbvh4* m, const tbvh_vec4* verts, uint32_t vert_count, uint32_t stride );
void tbvh_mbvh4_build_indexed( tbvh_mbvh4* m, const tbvh_vec4* verts, uint32_t vert_count, uint32_t stride, const uint32_t* indices, uint32_t prim_count );
void tbvh_mbvh4_build_hq( tbvh_mbvh4* m, const tbvh_vec4* verts, uint32_t vert_count, uint32_t stride );
void tbvh_mbvh4_build_hq_indexed( tbvh_mbvh4* m, const tbvh_vec4* verts, uint32_t vert_count, uint32_t stride, const uint32_t* indices, uint32_t prim_count );
void tbvh_mbvh4_convert_from( tbvh_mbvh4* m, tbvh_bvh* bvh, bool compact );
void tbvh_mbvh4_optimize( tbvh_mbvh4* m, uint32_t iterations, bool extreme );
void tbvh_mbvh4_refit( tbvh_mbvh4* m, uint32_t node_idx );
uint32_t tbvh_mbvh4_leaf_count( const tbvh_mbvh4* m, uint32_t node_idx );
float tbvh_mbvh4_sah_cost( const tbvh_mbvh4* m, uint32_t node_idx );
bool tbvh_mbvh4_load_self_contained( tbvh_mbvh4* m, const char* path );
const tbvh_mbvh4_node* tbvh_mbvh4_nodes( const tbvh_mbvh4* m, uint32_t* count );
const uint32_t* tbvh_mbvh4_prim_indices( const tbvh_mbvh4* m, uint32_t* count );

tbvh_mbvh8* tbvh_mbvh8_new( void );
void tbvh_mbvh8_free( tbvh_mbvh8* m );
tbvh_base* tbvh_mbvh8_base( tbvh_mbvh8* m );
const tbvh_base* tbvh_mbvh8_base_const( const tbvh_mbvh8* m );
void tbvh_mbvh8_build( tbvh_mbvh8* m, const tbvh_vec4* verts, uint32_t vert_count, uint32_t stride );
void tbvh_mbvh8_build_indexed( tbvh_mbvh8* m, const tbvh_vec4* verts, uint32_t vert_count, uint32_t stride, const uint32_t* indices, uint32_t prim_count );
void tbvh_mbvh8_build_hq( tbvh_mbvh8* m, const tbvh_vec4* verts, uint32_t vert_count, uint32_t stride );
void tbvh_mbvh8_build_hq_indexed( tbvh_mbvh8* m, const tbvh_vec4* verts, uint32_t vert_count, uint32_t stride, const uint32_t* indices, uint32_t prim_count );
void tbvh_mbvh8_convert_from( tbvh_mbvh8* m, tbvh_bvh* bvh, bool compact );
void tbvh_mbvh8_optimize( tbvh_mbvh8* m, uint32_t iterations, bool extreme );
void tbvh_mbvh8_refit( tbvh_mbvh8* m, uint32_t node_idx );
uint32_t tbvh_mbvh8_leaf_count( const tbvh_mbvh8* m, uint32_t node_idx );
float tbvh_mbvh8_sah_cost( const tbvh_mbvh8* m, uint32_t node_idx );
bool tbvh_mbvh8_load_self_contained( tbvh_mbvh8* m, const char* path );
const tbvh_mbvh8_node* tbvh_mbvh8_nodes( const tbvh_mbvh8* m, uint32_t* count );
const uint32_t* tbvh_mbvh8_prim_indices( const tbvh_mbvh8* m, uint32_t* count );

/* ------------------------------------------------------------------------------------------------
   tbvh_bvh4_cpu: tinybvh::BVH4_CPU, the fastest CPU traversal layout (AVX2 / NEON, 4 tris per leaf).
   ------------------------------------------------------------------------------------------------ */

tbvh_bvh4_cpu* tbvh_bvh4_cpu_new( void );
void tbvh_bvh4_cpu_free( tbvh_bvh4_cpu* b );
tbvh_base* tbvh_bvh4_cpu_base( tbvh_bvh4_cpu* b );
const tbvh_base* tbvh_bvh4_cpu_base_const( const tbvh_bvh4_cpu* b );
void tbvh_bvh4_cpu_build( tbvh_bvh4_cpu* b, const tbvh_vec4* verts, uint32_t vert_count, uint32_t stride );
void tbvh_bvh4_cpu_build_indexed( tbvh_bvh4_cpu* b, const tbvh_vec4* verts, uint32_t vert_count, uint32_t stride, const uint32_t* indices, uint32_t prim_count );
void tbvh_bvh4_cpu_build_hq( tbvh_bvh4_cpu* b, const tbvh_vec4* verts, uint32_t vert_count, uint32_t stride );
void tbvh_bvh4_cpu_build_hq_indexed( tbvh_bvh4_cpu* b, const tbvh_vec4* verts, uint32_t vert_count, uint32_t stride, const uint32_t* indices, uint32_t prim_count );
/* Takes over the wide tree: afterwards 'm' is empty (its buffers belong to 'b') and may be freed. */
void tbvh_bvh4_cpu_convert_from( tbvh_bvh4_cpu* b, tbvh_mbvh4* m, bool leaves_merged );
void tbvh_bvh4_cpu_optimize( tbvh_bvh4_cpu* b, uint32_t iterations, bool extreme );
void tbvh_bvh4_cpu_refit( tbvh_bvh4_cpu* b );
float tbvh_bvh4_cpu_sah_cost( const tbvh_bvh4_cpu* b, uint32_t node_idx );
int32_t tbvh_bvh4_cpu_intersect( const tbvh_bvh4_cpu* b, tbvh_ray* ray );
bool tbvh_bvh4_cpu_is_occluded( const tbvh_bvh4_cpu* b, const tbvh_ray* ray );
int32_t tbvh_bvh4_cpu_intersect_bundle( const tbvh_bvh4_cpu* b, tbvh_ray* rays );
int32_t tbvh_bvh4_cpu_is_occluded_bundle( const tbvh_bvh4_cpu* b, tbvh_ray* rays, bool* occluded );
void tbvh_bvh4_cpu_save( const tbvh_bvh4_cpu* b, const char* path );
bool tbvh_bvh4_cpu_load( tbvh_bvh4_cpu* b, const char* path, uint32_t expected_tris );
bool tbvh_bvh4_cpu_load_self_contained( tbvh_bvh4_cpu* b, const char* path );
const uint8_t* tbvh_bvh4_cpu_data( const tbvh_bvh4_cpu* b, uint32_t* used_blocks );  /* 64-byte blocks */

/* ------------------------------------------------------------------------------------------------
   tbvh_bvh8_cpu: tinybvh::BVH8_CPU, 8-wide AVX2 traversal.
   ------------------------------------------------------------------------------------------------ */

tbvh_bvh8_cpu* tbvh_bvh8_cpu_new( void );
void tbvh_bvh8_cpu_free( tbvh_bvh8_cpu* b );
tbvh_base* tbvh_bvh8_cpu_base( tbvh_bvh8_cpu* b );
const tbvh_base* tbvh_bvh8_cpu_base_const( const tbvh_bvh8_cpu* b );
void tbvh_bvh8_cpu_build( tbvh_bvh8_cpu* b, const tbvh_vec4* verts, uint32_t vert_count, uint32_t stride );
void tbvh_bvh8_cpu_build_indexed( tbvh_bvh8_cpu* b, const tbvh_vec4* verts, uint32_t vert_count, uint32_t stride, const uint32_t* indices, uint32_t prim_count );
void tbvh_bvh8_cpu_build_hq( tbvh_bvh8_cpu* b, const tbvh_vec4* verts, uint32_t vert_count, uint32_t stride );
void tbvh_bvh8_cpu_build_hq_indexed( tbvh_bvh8_cpu* b, const tbvh_vec4* verts, uint32_t vert_count, uint32_t stride, const uint32_t* indices, uint32_t prim_count );
/* Takes over the wide tree: afterwards 'm' is empty (its buffers belong to 'b') and may be freed. */
void tbvh_bvh8_cpu_convert_from( tbvh_bvh8_cpu* b, tbvh_mbvh8* m, bool leaves_merged );
void tbvh_bvh8_cpu_optimize( tbvh_bvh8_cpu* b, uint32_t iterations, bool extreme );
void tbvh_bvh8_cpu_refit( tbvh_bvh8_cpu* b );
float tbvh_bvh8_cpu_sah_cost( const tbvh_bvh8_cpu* b, uint32_t node_idx );
int32_t tbvh_bvh8_cpu_intersect( const tbvh_bvh8_cpu* b, tbvh_ray* ray );
bool tbvh_bvh8_cpu_is_occluded( const tbvh_bvh8_cpu* b, const tbvh_ray* ray );
void tbvh_bvh8_cpu_save( const tbvh_bvh8_cpu* b, const char* path );
bool tbvh_bvh8_cpu_load( tbvh_bvh8_cpu* b, const char* path, uint32_t expected_tris );
bool tbvh_bvh8_cpu_load_self_contained( tbvh_bvh8_cpu* b, const char* path );
const uint8_t* tbvh_bvh8_cpu_data( const tbvh_bvh8_cpu* b, uint32_t* used_blocks );  /* 64-byte blocks */

/* ------------------------------------------------------------------------------------------------
   tbvh_bvh4_gpu: tinybvh::BVH4_GPU, compact quantized 4-wide GPU layout.
   ------------------------------------------------------------------------------------------------ */

tbvh_bvh4_gpu* tbvh_bvh4_gpu_new( void );
void tbvh_bvh4_gpu_free( tbvh_bvh4_gpu* b );
tbvh_base* tbvh_bvh4_gpu_base( tbvh_bvh4_gpu* b );
const tbvh_base* tbvh_bvh4_gpu_base_const( const tbvh_bvh4_gpu* b );
void tbvh_bvh4_gpu_build( tbvh_bvh4_gpu* b, const tbvh_vec4* verts, uint32_t vert_count, uint32_t stride );
void tbvh_bvh4_gpu_build_indexed( tbvh_bvh4_gpu* b, const tbvh_vec4* verts, uint32_t vert_count, uint32_t stride, const uint32_t* indices, uint32_t prim_count );
void tbvh_bvh4_gpu_build_hq( tbvh_bvh4_gpu* b, const tbvh_vec4* verts, uint32_t vert_count, uint32_t stride );
void tbvh_bvh4_gpu_build_hq_indexed( tbvh_bvh4_gpu* b, const tbvh_vec4* verts, uint32_t vert_count, uint32_t stride, const uint32_t* indices, uint32_t prim_count );
void tbvh_bvh4_gpu_convert_from( tbvh_bvh4_gpu* b, const tbvh_mbvh4* m, bool compact );
void tbvh_bvh4_gpu_optimize( tbvh_bvh4_gpu* b, uint32_t iterations, bool extreme );
float tbvh_bvh4_gpu_sah_cost( const tbvh_bvh4_gpu* b, uint32_t node_idx );
int32_t tbvh_bvh4_gpu_intersect( const tbvh_bvh4_gpu* b, tbvh_ray* ray );
bool tbvh_bvh4_gpu_is_occluded( const tbvh_bvh4_gpu* b, const tbvh_ray* ray );
bool tbvh_bvh4_gpu_load_self_contained( tbvh_bvh4_gpu* b, const char* path );
const tbvh_vec4* tbvh_bvh4_gpu_data( const tbvh_bvh4_gpu* b, uint32_t* used_blocks );  /* 16-byte blocks */

/* ------------------------------------------------------------------------------------------------
   tbvh_bvh4_amd_hw: tinybvh::BVH4_AMD_HW, AMD hardware BVH format. GPU traversal only.
   ------------------------------------------------------------------------------------------------ */

tbvh_bvh4_amd_hw* tbvh_bvh4_amd_hw_new( void );
void tbvh_bvh4_amd_hw_free( tbvh_bvh4_amd_hw* b );
tbvh_base* tbvh_bvh4_amd_hw_base( tbvh_bvh4_amd_hw* b );
const tbvh_base* tbvh_bvh4_amd_hw_base_const( const tbvh_bvh4_amd_hw* b );
void tbvh_bvh4_amd_hw_build( tbvh_bvh4_amd_hw* b, const tbvh_vec4* verts, uint32_t vert_count, uint32_t stride );
void tbvh_bvh4_amd_hw_build_indexed( tbvh_bvh4_amd_hw* b, const tbvh_vec4* verts, uint32_t vert_count, uint32_t stride, const uint32_t* indices, uint32_t prim_count );
void tbvh_bvh4_amd_hw_build_hq( tbvh_bvh4_amd_hw* b, const tbvh_vec4* verts, uint32_t vert_count, uint32_t stride );
void tbvh_bvh4_amd_hw_build_hq_indexed( tbvh_bvh4_amd_hw* b, const tbvh_vec4* verts, uint32_t vert_count, uint32_t stride, const uint32_t* indices, uint32_t prim_count );
void tbvh_bvh4_amd_hw_convert_from( tbvh_bvh4_amd_hw* b, const tbvh_mbvh4* m );
void tbvh_bvh4_amd_hw_optimize( tbvh_bvh4_amd_hw* b, uint32_t iterations, bool extreme );
float tbvh_bvh4_amd_hw_sah_cost( const tbvh_bvh4_amd_hw* b, uint32_t node_idx );
bool tbvh_bvh4_amd_hw_load_self_contained( tbvh_bvh4_amd_hw* b, const char* path );
const tbvh_vec4* tbvh_bvh4_amd_hw_data( const tbvh_bvh4_amd_hw* b, uint32_t* used_blocks );  /* 16-byte blocks */

/* ------------------------------------------------------------------------------------------------
   tbvh_bvh8_cwbvh: tinybvh::BVH8_CWBVH, compressed wide BVH for GPU rendering.
   ------------------------------------------------------------------------------------------------ */

tbvh_bvh8_cwbvh* tbvh_bvh8_cwbvh_new( void );
void tbvh_bvh8_cwbvh_free( tbvh_bvh8_cwbvh* b );
tbvh_base* tbvh_bvh8_cwbvh_base( tbvh_bvh8_cwbvh* b );
const tbvh_base* tbvh_bvh8_cwbvh_base_const( const tbvh_bvh8_cwbvh* b );
void tbvh_bvh8_cwbvh_build( tbvh_bvh8_cwbvh* b, const tbvh_vec4* verts, uint32_t vert_count, uint32_t stride );
void tbvh_bvh8_cwbvh_build_indexed( tbvh_bvh8_cwbvh* b, const tbvh_vec4* verts, uint32_t vert_count, uint32_t stride, const uint32_t* indices, uint32_t prim_count );
void tbvh_bvh8_cwbvh_build_hq( tbvh_bvh8_cwbvh* b, const tbvh_vec4* verts, uint32_t vert_count, uint32_t stride );
void tbvh_bvh8_cwbvh_build_hq_indexed( tbvh_bvh8_cwbvh* b, const tbvh_vec4* verts, uint32_t vert_count, uint32_t stride, const uint32_t* indices, uint32_t prim_count );
void tbvh_bvh8_cwbvh_convert_from( tbvh_bvh8_cwbvh* b, const tbvh_mbvh8* m, bool compact );
void tbvh_bvh8_cwbvh_optimize( tbvh_bvh8_cwbvh* b, uint32_t iterations, bool extreme );
float tbvh_bvh8_cwbvh_sah_cost( const tbvh_bvh8_cwbvh* b, uint32_t node_idx );
int32_t tbvh_bvh8_cwbvh_intersect( const tbvh_bvh8_cwbvh* b, tbvh_ray* ray );
bool tbvh_bvh8_cwbvh_is_occluded( const tbvh_bvh8_cwbvh* b, const tbvh_ray* ray );
void tbvh_bvh8_cwbvh_save( const tbvh_bvh8_cwbvh* b, const char* path );
bool tbvh_bvh8_cwbvh_load( tbvh_bvh8_cwbvh* b, const char* path, uint32_t expected_tris );
bool tbvh_bvh8_cwbvh_load_self_contained( tbvh_bvh8_cwbvh* b, const char* path );
const tbvh_vec4* tbvh_bvh8_cwbvh_nodes( const tbvh_bvh8_cwbvh* b, uint32_t* used_blocks );  /* 16-byte blocks */
const tbvh_vec4* tbvh_bvh8_cwbvh_tris( const tbvh_bvh8_cwbvh* b, uint32_t* used_blocks );   /* 16-byte blocks */

/* ------------------------------------------------------------------------------------------------
   tbvh_voxelset: tinybvh::VoxelSet, a 256^3 brick-based voxel object in the unit cube, attachable
   as a BLAS in a single precision TLAS.
   ------------------------------------------------------------------------------------------------ */

tbvh_voxelset* tbvh_voxelset_new( void );
void tbvh_voxelset_free( tbvh_voxelset* v );
tbvh_base* tbvh_voxelset_base( tbvh_voxelset* v );
const tbvh_base* tbvh_voxelset_base_const( const tbvh_voxelset* v );
void tbvh_voxelset_set( tbvh_voxelset* v, uint32_t x, uint32_t y, uint32_t z, uint32_t value );  /* 0 clears */
void tbvh_voxelset_update_top_grid( tbvh_voxelset* v );   /* call after a batch of tbvh_voxelset_set */
int32_t tbvh_voxelset_intersect( const tbvh_voxelset* v, tbvh_ray* ray );
bool tbvh_voxelset_is_occluded( const tbvh_voxelset* v, const tbvh_ray* ray );
tbvh_vec3 tbvh_voxelset_normal( const tbvh_voxelset* v, const tbvh_ray* ray );  /* normal at ray->hit */

/* ------------------------------------------------------------------------------------------------
   tbvh_bvh_double: tinybvh::BVH_Double, double precision BVH with 64-bit indices.
   Vertices are tbvh_dbl3, three per triangle, 'stride' bytes apart (sizeof(tbvh_dbl3) when packed).
   ------------------------------------------------------------------------------------------------ */

tbvh_bvh_double* tbvh_bvh_double_new( void );
void tbvh_bvh_double_free( tbvh_bvh_double* b );
tbvh_base_ex* tbvh_bvh_double_base( tbvh_bvh_double* b );
const tbvh_base_ex* tbvh_bvh_double_base_const( const tbvh_bvh_double* b );
void tbvh_bvh_double_build( tbvh_bvh_double* b, const tbvh_dbl3* verts, uint64_t vert_count, uint64_t stride );
void tbvh_bvh_double_build_indexed( tbvh_bvh_double* b, const tbvh_dbl3* verts, uint64_t vert_count, uint64_t stride, const uint64_t* indices, uint64_t prim_count );
void tbvh_bvh_double_build_quick( tbvh_bvh_double* b, const tbvh_dbl3* verts, uint64_t vert_count, uint64_t stride );
void tbvh_bvh_double_build_hq( tbvh_bvh_double* b, const tbvh_dbl3* verts, uint64_t vert_count, uint64_t stride );
void tbvh_bvh_double_build_hq_indexed( tbvh_bvh_double* b, const tbvh_dbl3* verts, uint64_t vert_count, uint64_t stride, const uint64_t* indices, uint64_t prim_count );
void tbvh_bvh_double_build_aabbs( tbvh_bvh_double* b, const tbvh_dbl3* aabbs, uint64_t aabb_count );  /* (min, max) pairs */
void tbvh_bvh_double_build_custom( tbvh_bvh_double* b, tbvh_aabb_ex_fn get_aabb, uint64_t prim_count, void* user );
void tbvh_bvh_double_set_custom_callbacks( tbvh_bvh_double* b, tbvh_intersect_ex_fn intersect, tbvh_occluded_ex_fn occluded, void* user );
void tbvh_bvh_double_build_tlas( tbvh_bvh_double* b, tbvh_blas_instance_ex* instances, uint64_t inst_count, tbvh_base_ex** blasses, uint64_t blas_count );
int32_t tbvh_bvh_double_intersect( const tbvh_bvh_double* b, tbvh_ray_ex* ray );
bool tbvh_bvh_double_is_occluded( const tbvh_bvh_double* b, const tbvh_ray_ex* ray );
bool tbvh_bvh_double_intersect_sphere( const tbvh_bvh_double* b, tbvh_dbl3 pos, double r );
void tbvh_bvh_double_refit( tbvh_bvh_double* b, uint64_t node_idx );
void tbvh_bvh_double_compact( tbvh_bvh_double* b );
void tbvh_bvh_double_optimize( tbvh_bvh_double* b, uint32_t iterations, bool extreme, bool stochastic );
void tbvh_bvh_double_split_leafs( tbvh_bvh_double* b, uint64_t max_prims );
void tbvh_bvh_double_merge_leafs( tbvh_bvh_double* b );
double tbvh_bvh_double_sah_cost( const tbvh_bvh_double* b, uint64_t node_idx );
double tbvh_bvh_double_epo_cost( const tbvh_bvh_double* b, uint64_t node_idx );
uint64_t tbvh_bvh_double_node_count( const tbvh_bvh_double* b );
uint64_t tbvh_bvh_double_leaf_count( const tbvh_bvh_double* b );
uint64_t tbvh_bvh_double_prim_count( const tbvh_bvh_double* b, uint64_t node_idx );
bool tbvh_bvh_double_is_tlas( const tbvh_bvh_double* b );
bool tbvh_bvh_double_is_indexed( const tbvh_bvh_double* b );
bool tbvh_bvh_double_has_custom_geometry( const tbvh_bvh_double* b );
void tbvh_bvh_double_save( const tbvh_bvh_double* b, const char* path, bool include_geometry );
bool tbvh_bvh_double_load( tbvh_bvh_double* b, const char* path, const tbvh_dbl3* verts, uint64_t vert_count, uint64_t stride, const uint64_t* indices, uint64_t prim_count );
bool tbvh_bvh_double_load_self_contained( tbvh_bvh_double* b, const char* path );
bool tbvh_bvh_double_owns_geometry( const tbvh_bvh_double* b );
bool tbvh_bvh_double_has_fragments( const tbvh_bvh_double* b );  /* see tbvh_bvh_has_fragments */
void tbvh_bvh_double_free_geometry( tbvh_bvh_double* b );
const tbvh_bvh_node_ex* tbvh_bvh_double_nodes( const tbvh_bvh_double* b, uint64_t* count );
const uint64_t* tbvh_bvh_double_prim_indices( const tbvh_bvh_double* b, uint64_t* count );
const tbvh_dbl3* tbvh_bvh_double_verts( const tbvh_bvh_double* b, uint64_t* count, uint64_t* stride );
const uint64_t* tbvh_bvh_double_vert_indices( const tbvh_bvh_double* b );

/* ------------------------------------------------------------------------------------------------
   tbvh_bvh4_double: tinybvh::BVH4_Double, double precision 4-wide CPU traversal.
   ------------------------------------------------------------------------------------------------ */

tbvh_bvh4_double* tbvh_bvh4_double_new( void );
void tbvh_bvh4_double_free( tbvh_bvh4_double* b );
tbvh_base_ex* tbvh_bvh4_double_base( tbvh_bvh4_double* b );
const tbvh_base_ex* tbvh_bvh4_double_base_const( const tbvh_bvh4_double* b );
void tbvh_bvh4_double_build( tbvh_bvh4_double* b, const tbvh_dbl3* verts, uint64_t vert_count, uint64_t stride );
void tbvh_bvh4_double_build_indexed( tbvh_bvh4_double* b, const tbvh_dbl3* verts, uint64_t vert_count, uint64_t stride, const uint64_t* indices, uint64_t prim_count );
void tbvh_bvh4_double_build_hq( tbvh_bvh4_double* b, const tbvh_dbl3* verts, uint64_t vert_count, uint64_t stride );
void tbvh_bvh4_double_build_hq_indexed( tbvh_bvh4_double* b, const tbvh_dbl3* verts, uint64_t vert_count, uint64_t stride, const uint64_t* indices, uint64_t prim_count );
void tbvh_bvh4_double_optimize( tbvh_bvh4_double* b, uint32_t iterations, bool extreme );
void tbvh_bvh4_double_refit( tbvh_bvh4_double* b );
double tbvh_bvh4_double_sah_cost( const tbvh_bvh4_double* b, uint64_t node_idx );
int32_t tbvh_bvh4_double_intersect( const tbvh_bvh4_double* b, tbvh_ray_ex* ray );
bool tbvh_bvh4_double_is_occluded( const tbvh_bvh4_double* b, const tbvh_ray_ex* ray );
void tbvh_bvh4_double_save( const tbvh_bvh4_double* b, const char* path );
bool tbvh_bvh4_double_load( tbvh_bvh4_double* b, const char* path, uint64_t expected_tris );
bool tbvh_bvh4_double_load_self_contained( tbvh_bvh4_double* b, const char* path );
const uint8_t* tbvh_bvh4_double_data( const tbvh_bvh4_double* b, uint32_t* used_blocks );  /* 64-byte blocks */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TINYBVH_C_H_ */
