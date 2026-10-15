//! Raw bindings to `ffi/tinybvh_c.h`, the C shim over tinybvh.
//!
//! Hand-written mirror of the header; every struct here matches the C (and therefore tinybvh) layout
//! bit for bit. `tests/layout.rs` checks the sizes against `tbvh_sizeof` at run time and the C shim
//! checks them against tinybvh at compile time.

#![allow(non_camel_case_types, dead_code)]

use core::ffi::{c_char, c_void};

/// Size in bytes of a C pointer; the hit records store a `void*` as plain bytes to keep the
/// 4-byte packing tinybvh uses without needing a packed struct on the Rust side.
pub const PTR_SIZE: usize = core::mem::size_of::<*mut c_void>();

// ---------------------------------------------------------------------------------------------
// Plain data types
// ---------------------------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct tbvh_vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct tbvh_vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct tbvh_dbl3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct tbvh_mat4 {
    pub cell: [f32; 16],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct tbvh_dblmat4 {
    pub cell: [f64; 16],
}

/// Mirror of `tbvh_intersection` (tinybvh::Intersection): 84 bytes, 4-byte aligned.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct tbvh_intersection {
    pub inst: u32,
    pub t: f32,
    pub u: f32,
    pub v: f32,
    pub prim: u32,
    pub aux_data: [u8; PTR_SIZE],
    pub user: [u32; 14],
}

/// Mirror of `tbvh_intersection_ex` (tinybvh::IntersectionEx): 104 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct tbvh_intersection_ex {
    pub inst: u64,
    pub t: f64,
    pub u: f64,
    pub v: f64,
    pub prim: u64,
    pub aux_data: [u8; PTR_SIZE],
    pub user: [u32; 14],
}

/// Mirror of `tbvh_ray` (tinybvh::Ray): 128 bytes, 64-byte aligned, `hit` at offset 44.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct tbvh_ray {
    pub o: tbvh_vec3,
    pub mask: u32,
    pub d: tbvh_vec3,
    pub inst_idx: u32,
    pub rd: tbvh_vec3,
    pub hit: tbvh_intersection,
}

/// Mirror of `tbvh_ray_ex` (tinybvh::RayEx): 192 bytes, 64-byte aligned, `hit` at offset 88.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct tbvh_ray_ex {
    pub o: tbvh_dbl3,
    pub mask: u32,
    pub pad0: u32,
    pub d: tbvh_dbl3,
    pub inst_idx: u64,
    pub rd: tbvh_dbl3,
    pub hit: tbvh_intersection_ex,
}

/// Mirror of `tbvh_blas_instance` (tinybvh::BLASInstance): 192 bytes, 64-byte aligned.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct tbvh_blas_instance {
    pub transform: tbvh_mat4,
    pub inv_transform: tbvh_mat4,
    pub aabb_min: tbvh_vec3,
    pub blas_idx: u32,
    pub aabb_max: tbvh_vec3,
    pub mask: u32,
}

/// Mirror of `tbvh_blas_instance_ex` (tinybvh::BLASInstanceEx): 320 bytes, 64-byte aligned.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct tbvh_blas_instance_ex {
    pub transform: tbvh_dblmat4,
    pub inv_transform: tbvh_dblmat4,
    pub aabb_min: tbvh_dbl3,
    pub blas_idx: u64,
    pub aabb_max: tbvh_dbl3,
    pub mask: u32,
}

/// Mirror of `tbvh_build_settings` (tinybvh::BVHBuildSettings): 40 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct tbvh_build_settings {
    pub use_presplitting: bool,
    pub use_spatial_splits: bool,
    pub min_spatial_overlap: f32,
    pub spatial_skip_smallest_axis: bool,
    pub presplit_post_pass: bool,
    pub presplit_factor: f32,
    pub use_full_sweep: bool,
    pub binned_bvh: bool,
    pub bonsai_grid_prims: u32,
    pub bonsai_prune_factor: f32,
    pub use_hploc: bool,
    pub post_optimize: bool,
    pub optimize_iterations: i32,
    pub use_simd_if_available: bool,
    pub enable_threading: bool,
    pub discard_intermediates: bool,
    pub minimize_mem_use: bool,
}

pub type tbvh_layout = u32;
pub const TBVH_LAYOUT_UNDEFINED: tbvh_layout = 0;
pub const TBVH_LAYOUT_BVH: tbvh_layout = 1;
pub const TBVH_LAYOUT_BVH_VERBOSE: tbvh_layout = 2;
pub const TBVH_LAYOUT_BVH_GPU: tbvh_layout = 3;
pub const TBVH_LAYOUT_MBVH: tbvh_layout = 4;
pub const TBVH_LAYOUT_BVH4_CPU: tbvh_layout = 5;
pub const TBVH_LAYOUT_BVH4_GPU: tbvh_layout = 6;
pub const TBVH_LAYOUT_CWBVH: tbvh_layout = 7;
pub const TBVH_LAYOUT_BVH8_SIMD: tbvh_layout = 8;
pub const TBVH_LAYOUT_VOXELSET: tbvh_layout = 9;
pub const TBVH_LAYOUT_BVH4_AMD_HW: tbvh_layout = 10;

/// tinybvh::BVH::BVHNode, 32 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct tbvh_bvh_node {
    pub aabb_min: tbvh_vec3,
    pub left_first: u32,
    pub aabb_max: tbvh_vec3,
    pub tri_count: u32,
}

/// tinybvh::BVH_Double::BVHNode, 64 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct tbvh_bvh_node_ex {
    pub aabb_min: tbvh_dbl3,
    pub left_first: u64,
    pub aabb_max: tbvh_dbl3,
    pub tri_count: u64,
}

/// tinybvh::BVH_Verbose::BVHNode, 64 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct tbvh_bvh_verbose_node {
    pub aabb_min: tbvh_vec3,
    pub left: u32,
    pub aabb_max: tbvh_vec3,
    pub right: u32,
    pub tri_count: u32,
    pub first_tri: u32,
    pub parent: u32,
    pub dummy: [f32; 5],
}

/// tinybvh::BVH_GPU::BVHNode, 64 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct tbvh_bvh_gpu_node {
    pub lmin: tbvh_vec3,
    pub left: u32,
    pub lmax: tbvh_vec3,
    pub right: u32,
    pub rmin: tbvh_vec3,
    pub tri_count: u32,
    pub rmax: tbvh_vec3,
    pub first_tri: u32,
}

/// tinybvh::MBVH<4>::MBVHNode, 64 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct tbvh_mbvh4_node {
    pub aabb_min: tbvh_vec3,
    pub first_tri: u32,
    pub aabb_max: tbvh_vec3,
    pub tri_count: u32,
    pub child: [u32; 4],
    pub child_count: u32,
    pub dummy: [u32; 3],
}

/// tinybvh::MBVH<8>::MBVHNode, 80 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct tbvh_mbvh8_node {
    pub aabb_min: tbvh_vec3,
    pub first_tri: u32,
    pub aabb_max: tbvh_vec3,
    pub tri_count: u32,
    pub child: [u32; 8],
    pub child_count: u32,
    pub dummy: [u32; 3],
}

pub type tbvh_struct_id = u32;
pub const TBVH_STRUCT_VEC3: tbvh_struct_id = 0;
pub const TBVH_STRUCT_VEC4: tbvh_struct_id = 1;
pub const TBVH_STRUCT_DBL3: tbvh_struct_id = 2;
pub const TBVH_STRUCT_MAT4: tbvh_struct_id = 3;
pub const TBVH_STRUCT_DBLMAT4: tbvh_struct_id = 4;
pub const TBVH_STRUCT_INTERSECTION: tbvh_struct_id = 5;
pub const TBVH_STRUCT_INTERSECTION_EX: tbvh_struct_id = 6;
pub const TBVH_STRUCT_RAY: tbvh_struct_id = 7;
pub const TBVH_STRUCT_RAY_EX: tbvh_struct_id = 8;
pub const TBVH_STRUCT_BLAS_INSTANCE: tbvh_struct_id = 9;
pub const TBVH_STRUCT_BLAS_INSTANCE_EX: tbvh_struct_id = 10;
pub const TBVH_STRUCT_BUILD_SETTINGS: tbvh_struct_id = 11;
pub const TBVH_STRUCT_BVH_NODE: tbvh_struct_id = 12;
pub const TBVH_STRUCT_BVH_NODE_EX: tbvh_struct_id = 13;
pub const TBVH_STRUCT_BVH_VERBOSE_NODE: tbvh_struct_id = 14;
pub const TBVH_STRUCT_BVH_GPU_NODE: tbvh_struct_id = 15;
pub const TBVH_STRUCT_MBVH4_NODE: tbvh_struct_id = 16;
pub const TBVH_STRUCT_MBVH8_NODE: tbvh_struct_id = 17;
pub const TBVH_STRUCT_COUNT: tbvh_struct_id = 18;

// ---------------------------------------------------------------------------------------------
// Opaque handles
// ---------------------------------------------------------------------------------------------

macro_rules! opaque {
    ($($name:ident),* $(,)?) => {
        $(
            #[repr(C)]
            pub struct $name {
                _data: [u8; 0],
                _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
            }
        )*
    };
}

opaque!(
    tbvh_base,
    tbvh_base_ex,
    tbvh_bvh,
    tbvh_bvh_verbose,
    tbvh_bvh_gpu,
    tbvh_mbvh4,
    tbvh_mbvh8,
    tbvh_bvh4_cpu,
    tbvh_bvh4_gpu,
    tbvh_bvh4_amd_hw,
    tbvh_bvh8_cwbvh,
    tbvh_bvh8_cpu,
    tbvh_voxelset,
    tbvh_bvh_double,
    tbvh_bvh4_double,
);

// ---------------------------------------------------------------------------------------------
// Callbacks
// ---------------------------------------------------------------------------------------------

pub type tbvh_aabb_fn = Option<
    unsafe extern "C" fn(prim: u32, bmin: *mut tbvh_vec3, bmax: *mut tbvh_vec3, user: *mut c_void),
>;
pub type tbvh_intersect_fn =
    Option<unsafe extern "C" fn(ray: *mut tbvh_ray, prim: u32, user: *mut c_void) -> bool>;
pub type tbvh_occluded_fn =
    Option<unsafe extern "C" fn(ray: *const tbvh_ray, prim: u32, user: *mut c_void) -> bool>;

pub type tbvh_aabb_ex_fn = Option<
    unsafe extern "C" fn(prim: u64, bmin: *mut tbvh_dbl3, bmax: *mut tbvh_dbl3, user: *mut c_void),
>;
pub type tbvh_intersect_ex_fn =
    Option<unsafe extern "C" fn(ray: *mut tbvh_ray_ex, prim: u64, user: *mut c_void) -> bool>;
pub type tbvh_occluded_ex_fn =
    Option<unsafe extern "C" fn(ray: *const tbvh_ray_ex, prim: u64, user: *mut c_void) -> bool>;

pub type tbvh_fatal_error_fn =
    Option<unsafe extern "C" fn(message: *const c_char, user: *mut c_void)>;

// ---------------------------------------------------------------------------------------------
// Functions
// ---------------------------------------------------------------------------------------------

extern "C" {
    // Library-wide
    pub fn tbvh_version(major: *mut i32, minor: *mut i32, sub: *mut i32);
    pub fn tbvh_bundle_rays() -> u32;
    pub fn tbvh_inst_idx_bits() -> u32;
    pub fn tbvh_stack_size() -> u32;
    pub fn tbvh_simd_enabled() -> bool;
    pub fn tbvh_simd_builder_available() -> bool;
    pub fn tbvh_far() -> f32;
    pub fn tbvh_dbl_far() -> f64;
    pub fn tbvh_voxelset_object_dim() -> u32;
    pub fn tbvh_sizeof(id: tbvh_struct_id) -> usize;
    pub fn tbvh_set_fatal_error_handler(handler: tbvh_fatal_error_fn, user: *mut c_void);
    pub fn tbvh_shutdown_builtin_pool();

    // Ray helpers
    pub fn tbvh_ray_init(
        ray: *mut tbvh_ray,
        origin: tbvh_vec3,
        direction: tbvh_vec3,
        t: f32,
        mask: u32,
    );
    pub fn tbvh_ray_set_direction(ray: *mut tbvh_ray, direction: tbvh_vec3);
    pub fn tbvh_ray_normalize(ray: *mut tbvh_ray);
    pub fn tbvh_ray_ex_init(
        ray: *mut tbvh_ray_ex,
        origin: tbvh_dbl3,
        direction: tbvh_dbl3,
        t: f64,
        mask: u32,
    );
    pub fn tbvh_ray_ex_set_direction(ray: *mut tbvh_ray_ex, direction: tbvh_dbl3);
    pub fn tbvh_ray_ex_normalize(ray: *mut tbvh_ray_ex);

    // Instance helpers
    pub fn tbvh_blas_instance_init(inst: *mut tbvh_blas_instance, blas_idx: u32);
    pub fn tbvh_blas_instance_invert_transform(inst: *mut tbvh_blas_instance);
    pub fn tbvh_blas_instance_update(inst: *mut tbvh_blas_instance, blas: *const tbvh_base);
    pub fn tbvh_blas_instance_ex_init(inst: *mut tbvh_blas_instance_ex, blas_idx: u64);
    pub fn tbvh_blas_instance_ex_invert_transform(inst: *mut tbvh_blas_instance_ex);
    pub fn tbvh_blas_instance_ex_update(
        inst: *mut tbvh_blas_instance_ex,
        blas: *const tbvh_base_ex,
    );
    pub fn tbvh_transform_point(p: tbvh_vec3, m: *const tbvh_mat4) -> tbvh_vec3;
    pub fn tbvh_transform_vector(v: tbvh_vec3, m: *const tbvh_mat4) -> tbvh_vec3;

    // tbvh_base
    pub fn tbvh_base_layout(b: *const tbvh_base) -> tbvh_layout;
    pub fn tbvh_base_get_settings(b: *const tbvh_base, out: *mut tbvh_build_settings);
    pub fn tbvh_base_set_settings(b: *mut tbvh_base, settings: *const tbvh_build_settings);
    pub fn tbvh_base_get_c_trav(b: *const tbvh_base) -> f32;
    pub fn tbvh_base_get_c_int(b: *const tbvh_base) -> f32;
    pub fn tbvh_base_set_c_trav(b: *mut tbvh_base, c: f32);
    pub fn tbvh_base_set_c_int(b: *mut tbvh_base, c: f32);
    pub fn tbvh_base_get_hq_bins(b: *const tbvh_base) -> u32;
    pub fn tbvh_base_set_hq_bins(b: *mut tbvh_base, bins: u32);
    pub fn tbvh_base_get_hq_odd_even(b: *const tbvh_base) -> bool;
    pub fn tbvh_base_set_hq_odd_even(b: *mut tbvh_base, oddeven: bool);
    pub fn tbvh_base_get_hq_leaf_prims(b: *const tbvh_base) -> u32;
    pub fn tbvh_base_set_hq_leaf_prims(b: *mut tbvh_base, prims: u32);
    pub fn tbvh_base_aabb_min(b: *const tbvh_base) -> tbvh_vec3;
    pub fn tbvh_base_aabb_max(b: *const tbvh_base) -> tbvh_vec3;
    pub fn tbvh_base_tri_count(b: *const tbvh_base) -> u32;
    pub fn tbvh_base_idx_count(b: *const tbvh_base) -> u32;
    pub fn tbvh_base_used_nodes(b: *const tbvh_base) -> u32;
    pub fn tbvh_base_allocated_nodes(b: *const tbvh_base) -> u32;
    pub fn tbvh_base_is_refittable(b: *const tbvh_base) -> bool;
    pub fn tbvh_base_is_rebuildable(b: *const tbvh_base) -> bool;
    pub fn tbvh_base_may_have_holes(b: *const tbvh_base) -> bool;
    pub fn tbvh_base_has_ordered_nodes(b: *const tbvh_base) -> bool;
    pub fn tbvh_base_is_over_aabbs(b: *const tbvh_base) -> bool;
    pub fn tbvh_base_is_over_indices(b: *const tbvh_base) -> bool;
    pub fn tbvh_base_set_opacity_micro_maps(b: *mut tbvh_base, map: *const u32, n: u32);
    pub fn tbvh_base_opacity_micro_map_n(b: *const tbvh_base) -> u32;

    // tbvh_base_ex
    pub fn tbvh_base_ex_layout(b: *const tbvh_base_ex) -> tbvh_layout;
    pub fn tbvh_base_ex_get_settings(b: *const tbvh_base_ex, out: *mut tbvh_build_settings);
    pub fn tbvh_base_ex_set_settings(b: *mut tbvh_base_ex, settings: *const tbvh_build_settings);
    pub fn tbvh_base_ex_get_c_trav(b: *const tbvh_base_ex) -> f32;
    pub fn tbvh_base_ex_get_c_int(b: *const tbvh_base_ex) -> f32;
    pub fn tbvh_base_ex_set_c_trav(b: *mut tbvh_base_ex, c: f32);
    pub fn tbvh_base_ex_set_c_int(b: *mut tbvh_base_ex, c: f32);
    pub fn tbvh_base_ex_get_hq_bins(b: *const tbvh_base_ex) -> u32;
    pub fn tbvh_base_ex_set_hq_bins(b: *mut tbvh_base_ex, bins: u32);
    pub fn tbvh_base_ex_get_hq_odd_even(b: *const tbvh_base_ex) -> bool;
    pub fn tbvh_base_ex_set_hq_odd_even(b: *mut tbvh_base_ex, oddeven: bool);
    pub fn tbvh_base_ex_get_hq_leaf_prims(b: *const tbvh_base_ex) -> u32;
    pub fn tbvh_base_ex_set_hq_leaf_prims(b: *mut tbvh_base_ex, prims: u32);
    pub fn tbvh_base_ex_aabb_min(b: *const tbvh_base_ex) -> tbvh_dbl3;
    pub fn tbvh_base_ex_aabb_max(b: *const tbvh_base_ex) -> tbvh_dbl3;
    pub fn tbvh_base_ex_tri_count(b: *const tbvh_base_ex) -> u64;
    pub fn tbvh_base_ex_idx_count(b: *const tbvh_base_ex) -> u64;
    pub fn tbvh_base_ex_used_nodes(b: *const tbvh_base_ex) -> u64;
    pub fn tbvh_base_ex_allocated_nodes(b: *const tbvh_base_ex) -> u64;
    pub fn tbvh_base_ex_is_refittable(b: *const tbvh_base_ex) -> bool;
    pub fn tbvh_base_ex_is_rebuildable(b: *const tbvh_base_ex) -> bool;
    pub fn tbvh_base_ex_may_have_holes(b: *const tbvh_base_ex) -> bool;
    pub fn tbvh_base_ex_has_ordered_nodes(b: *const tbvh_base_ex) -> bool;
    pub fn tbvh_base_ex_is_over_aabbs(b: *const tbvh_base_ex) -> bool;
    pub fn tbvh_base_ex_is_over_indices(b: *const tbvh_base_ex) -> bool;
    pub fn tbvh_base_ex_set_opacity_micro_maps(b: *mut tbvh_base_ex, map: *const u32, n: u32);
    pub fn tbvh_base_ex_opacity_micro_map_n(b: *const tbvh_base_ex) -> u32;

    // tbvh_bvh
    pub fn tbvh_bvh_new() -> *mut tbvh_bvh;
    pub fn tbvh_bvh_free(bvh: *mut tbvh_bvh);
    pub fn tbvh_bvh_base(bvh: *mut tbvh_bvh) -> *mut tbvh_base;
    pub fn tbvh_bvh_base_const(bvh: *const tbvh_bvh) -> *const tbvh_base;
    pub fn tbvh_bvh_build(
        bvh: *mut tbvh_bvh,
        verts: *const tbvh_vec4,
        vert_count: u32,
        stride: u32,
    );
    pub fn tbvh_bvh_build_indexed(
        bvh: *mut tbvh_bvh,
        verts: *const tbvh_vec4,
        vert_count: u32,
        stride: u32,
        indices: *const u32,
        prim_count: u32,
    );
    pub fn tbvh_bvh_build_quick(
        bvh: *mut tbvh_bvh,
        verts: *const tbvh_vec4,
        vert_count: u32,
        stride: u32,
    );
    pub fn tbvh_bvh_build_hq(
        bvh: *mut tbvh_bvh,
        verts: *const tbvh_vec4,
        vert_count: u32,
        stride: u32,
    );
    pub fn tbvh_bvh_build_hq_indexed(
        bvh: *mut tbvh_bvh,
        verts: *const tbvh_vec4,
        vert_count: u32,
        stride: u32,
        indices: *const u32,
        prim_count: u32,
    );
    pub fn tbvh_bvh_build_simd(
        bvh: *mut tbvh_bvh,
        verts: *const tbvh_vec4,
        vert_count: u32,
        stride: u32,
    );
    pub fn tbvh_bvh_build_simd_indexed(
        bvh: *mut tbvh_bvh,
        verts: *const tbvh_vec4,
        vert_count: u32,
        stride: u32,
        indices: *const u32,
        prim_count: u32,
    );
    pub fn tbvh_bvh_build_aabbs(bvh: *mut tbvh_bvh, aabbs: *const tbvh_vec4, aabb_count: u32);
    pub fn tbvh_bvh_build_custom(
        bvh: *mut tbvh_bvh,
        get_aabb: tbvh_aabb_fn,
        prim_count: u32,
        user: *mut c_void,
    );
    pub fn tbvh_bvh_set_custom_callbacks(
        bvh: *mut tbvh_bvh,
        intersect: tbvh_intersect_fn,
        occluded: tbvh_occluded_fn,
        user: *mut c_void,
    );
    pub fn tbvh_bvh_build_tlas(
        bvh: *mut tbvh_bvh,
        instances: *mut tbvh_blas_instance,
        inst_count: u32,
        blasses: *mut *mut tbvh_base,
        blas_count: u32,
    );
    pub fn tbvh_bvh_intersect(bvh: *const tbvh_bvh, ray: *mut tbvh_ray) -> i32;
    pub fn tbvh_bvh_is_occluded(bvh: *const tbvh_bvh, ray: *const tbvh_ray) -> bool;
    pub fn tbvh_bvh_intersect_sphere(bvh: *const tbvh_bvh, pos: tbvh_vec3, r: f32) -> bool;
    pub fn tbvh_bvh_intersect_bundle(bvh: *const tbvh_bvh, rays: *mut tbvh_ray) -> i32;
    pub fn tbvh_bvh_is_occluded_bundle(
        bvh: *const tbvh_bvh,
        rays: *mut tbvh_ray,
        occluded: *mut bool,
    ) -> i32;
    pub fn tbvh_bvh_refit(bvh: *mut tbvh_bvh, node_idx: u32);
    pub fn tbvh_bvh_compact(bvh: *mut tbvh_bvh);
    pub fn tbvh_bvh_optimize(bvh: *mut tbvh_bvh, iterations: u32, extreme: bool, stochastic: bool);
    pub fn tbvh_bvh_split_leafs(bvh: *mut tbvh_bvh, max_prims: u32);
    pub fn tbvh_bvh_merge_leafs(bvh: *mut tbvh_bvh);
    pub fn tbvh_bvh_sah_cost(bvh: *const tbvh_bvh, node_idx: u32) -> f32;
    pub fn tbvh_bvh_epo_cost(bvh: *const tbvh_bvh, node_idx: u32) -> f32;
    pub fn tbvh_bvh_node_count(bvh: *const tbvh_bvh) -> u32;
    pub fn tbvh_bvh_leaf_count(bvh: *const tbvh_bvh) -> u32;
    pub fn tbvh_bvh_prim_count(bvh: *const tbvh_bvh, node_idx: u32) -> u32;
    pub fn tbvh_bvh_collect_nodes(bvh: *const tbvh_bvh, root: u32, list: *mut u32, cap: u32)
        -> u32;
    pub fn tbvh_bvh_is_tlas(bvh: *const tbvh_bvh) -> bool;
    pub fn tbvh_bvh_is_indexed(bvh: *const tbvh_bvh) -> bool;
    pub fn tbvh_bvh_has_custom_geometry(bvh: *const tbvh_bvh) -> bool;
    pub fn tbvh_bvh_save(bvh: *const tbvh_bvh, path: *const c_char, include_geometry: bool);
    pub fn tbvh_bvh_load(
        bvh: *mut tbvh_bvh,
        path: *const c_char,
        verts: *const tbvh_vec4,
        vert_count: u32,
        stride: u32,
        indices: *const u32,
        prim_count: u32,
    ) -> bool;
    pub fn tbvh_bvh_load_self_contained(bvh: *mut tbvh_bvh, path: *const c_char) -> bool;
    pub fn tbvh_bvh_owns_geometry(bvh: *const tbvh_bvh) -> bool;
    pub fn tbvh_bvh_has_fragments(bvh: *const tbvh_bvh) -> bool;
    pub fn tbvh_bvh_free_geometry(bvh: *mut tbvh_bvh);
    pub fn tbvh_bvh_nodes(bvh: *const tbvh_bvh, count: *mut u32) -> *const tbvh_bvh_node;
    pub fn tbvh_bvh_prim_indices(bvh: *const tbvh_bvh, count: *mut u32) -> *const u32;
    pub fn tbvh_bvh_verts(
        bvh: *const tbvh_bvh,
        count: *mut u32,
        stride: *mut u32,
    ) -> *const tbvh_vec4;
    pub fn tbvh_bvh_vert_indices(bvh: *const tbvh_bvh) -> *const u32;
    pub fn tbvh_bvh_convert_from_verbose(
        bvh: *mut tbvh_bvh,
        verbose: *const tbvh_bvh_verbose,
        compact: bool,
    );

    // tbvh_bvh_verbose
    pub fn tbvh_bvh_verbose_new() -> *mut tbvh_bvh_verbose;
    pub fn tbvh_bvh_verbose_free(v: *mut tbvh_bvh_verbose);
    pub fn tbvh_bvh_verbose_base(v: *mut tbvh_bvh_verbose) -> *mut tbvh_base;
    pub fn tbvh_bvh_verbose_base_const(v: *const tbvh_bvh_verbose) -> *const tbvh_base;
    pub fn tbvh_bvh_verbose_convert_from(
        v: *mut tbvh_bvh_verbose,
        bvh: *const tbvh_bvh,
        compact: bool,
    );
    pub fn tbvh_bvh_verbose_sah_cost(v: *const tbvh_bvh_verbose, node_idx: u32) -> f32;
    pub fn tbvh_bvh_verbose_node_count(v: *const tbvh_bvh_verbose) -> i32;
    pub fn tbvh_bvh_verbose_refit(v: *mut tbvh_bvh_verbose, node_idx: u32, skip_leafs: bool);
    pub fn tbvh_bvh_verbose_compact(v: *mut tbvh_bvh_verbose);
    pub fn tbvh_bvh_verbose_sort_indices(v: *mut tbvh_bvh_verbose);
    pub fn tbvh_bvh_verbose_split_leafs(v: *mut tbvh_bvh_verbose, max_prims: u32);
    pub fn tbvh_bvh_verbose_merge_leafs(v: *mut tbvh_bvh_verbose);
    pub fn tbvh_bvh_verbose_optimize(
        v: *mut tbvh_bvh_verbose,
        iterations: u32,
        extreme: bool,
        stochastic: bool,
    );
    pub fn tbvh_bvh_verbose_nodes(
        v: *const tbvh_bvh_verbose,
        count: *mut u32,
    ) -> *const tbvh_bvh_verbose_node;
    pub fn tbvh_bvh_verbose_prim_indices(v: *const tbvh_bvh_verbose, count: *mut u32)
        -> *const u32;

    // tbvh_bvh_gpu
    pub fn tbvh_bvh_gpu_new() -> *mut tbvh_bvh_gpu;
    pub fn tbvh_bvh_gpu_free(g: *mut tbvh_bvh_gpu);
    pub fn tbvh_bvh_gpu_base(g: *mut tbvh_bvh_gpu) -> *mut tbvh_base;
    pub fn tbvh_bvh_gpu_base_const(g: *const tbvh_bvh_gpu) -> *const tbvh_base;
    pub fn tbvh_bvh_gpu_build(
        g: *mut tbvh_bvh_gpu,
        verts: *const tbvh_vec4,
        vert_count: u32,
        stride: u32,
    );
    pub fn tbvh_bvh_gpu_build_indexed(
        g: *mut tbvh_bvh_gpu,
        verts: *const tbvh_vec4,
        vert_count: u32,
        stride: u32,
        indices: *const u32,
        prim_count: u32,
    );
    pub fn tbvh_bvh_gpu_build_hq(
        g: *mut tbvh_bvh_gpu,
        verts: *const tbvh_vec4,
        vert_count: u32,
        stride: u32,
    );
    pub fn tbvh_bvh_gpu_build_hq_indexed(
        g: *mut tbvh_bvh_gpu,
        verts: *const tbvh_vec4,
        vert_count: u32,
        stride: u32,
        indices: *const u32,
        prim_count: u32,
    );
    pub fn tbvh_bvh_gpu_build_aabbs(g: *mut tbvh_bvh_gpu, aabbs: *const tbvh_vec4, aabb_count: u32);
    pub fn tbvh_bvh_gpu_build_tlas(
        g: *mut tbvh_bvh_gpu,
        instances: *mut tbvh_blas_instance,
        inst_count: u32,
        blasses: *mut *mut tbvh_base,
        blas_count: u32,
    );
    pub fn tbvh_bvh_gpu_convert_from(g: *mut tbvh_bvh_gpu, bvh: *const tbvh_bvh, compact: bool);
    pub fn tbvh_bvh_gpu_optimize(g: *mut tbvh_bvh_gpu, iterations: u32, extreme: bool);
    pub fn tbvh_bvh_gpu_sah_cost(g: *mut tbvh_bvh_gpu, node_idx: u32) -> f32;
    pub fn tbvh_bvh_gpu_intersect(g: *const tbvh_bvh_gpu, ray: *mut tbvh_ray) -> i32;
    pub fn tbvh_bvh_gpu_is_occluded(g: *const tbvh_bvh_gpu, ray: *const tbvh_ray) -> bool;
    pub fn tbvh_bvh_gpu_load_self_contained(g: *mut tbvh_bvh_gpu, path: *const c_char) -> bool;
    pub fn tbvh_bvh_gpu_nodes(g: *const tbvh_bvh_gpu, count: *mut u32) -> *const tbvh_bvh_gpu_node;
    pub fn tbvh_bvh_gpu_prim_indices(g: *const tbvh_bvh_gpu, count: *mut u32) -> *const u32;

    // tbvh_mbvh4
    pub fn tbvh_mbvh4_new() -> *mut tbvh_mbvh4;
    pub fn tbvh_mbvh4_free(m: *mut tbvh_mbvh4);
    pub fn tbvh_mbvh4_base(m: *mut tbvh_mbvh4) -> *mut tbvh_base;
    pub fn tbvh_mbvh4_base_const(m: *const tbvh_mbvh4) -> *const tbvh_base;
    pub fn tbvh_mbvh4_build(
        m: *mut tbvh_mbvh4,
        verts: *const tbvh_vec4,
        vert_count: u32,
        stride: u32,
    );
    pub fn tbvh_mbvh4_build_indexed(
        m: *mut tbvh_mbvh4,
        verts: *const tbvh_vec4,
        vert_count: u32,
        stride: u32,
        indices: *const u32,
        prim_count: u32,
    );
    pub fn tbvh_mbvh4_build_hq(
        m: *mut tbvh_mbvh4,
        verts: *const tbvh_vec4,
        vert_count: u32,
        stride: u32,
    );
    pub fn tbvh_mbvh4_build_hq_indexed(
        m: *mut tbvh_mbvh4,
        verts: *const tbvh_vec4,
        vert_count: u32,
        stride: u32,
        indices: *const u32,
        prim_count: u32,
    );
    pub fn tbvh_mbvh4_convert_from(m: *mut tbvh_mbvh4, bvh: *mut tbvh_bvh, compact: bool);
    pub fn tbvh_mbvh4_optimize(m: *mut tbvh_mbvh4, iterations: u32, extreme: bool);
    pub fn tbvh_mbvh4_refit(m: *mut tbvh_mbvh4, node_idx: u32);
    pub fn tbvh_mbvh4_leaf_count(m: *const tbvh_mbvh4, node_idx: u32) -> u32;
    pub fn tbvh_mbvh4_sah_cost(m: *const tbvh_mbvh4, node_idx: u32) -> f32;
    pub fn tbvh_mbvh4_load_self_contained(m: *mut tbvh_mbvh4, path: *const c_char) -> bool;
    pub fn tbvh_mbvh4_nodes(m: *const tbvh_mbvh4, count: *mut u32) -> *const tbvh_mbvh4_node;
    pub fn tbvh_mbvh4_prim_indices(m: *const tbvh_mbvh4, count: *mut u32) -> *const u32;

    // tbvh_mbvh8
    pub fn tbvh_mbvh8_new() -> *mut tbvh_mbvh8;
    pub fn tbvh_mbvh8_free(m: *mut tbvh_mbvh8);
    pub fn tbvh_mbvh8_base(m: *mut tbvh_mbvh8) -> *mut tbvh_base;
    pub fn tbvh_mbvh8_base_const(m: *const tbvh_mbvh8) -> *const tbvh_base;
    pub fn tbvh_mbvh8_build(
        m: *mut tbvh_mbvh8,
        verts: *const tbvh_vec4,
        vert_count: u32,
        stride: u32,
    );
    pub fn tbvh_mbvh8_build_indexed(
        m: *mut tbvh_mbvh8,
        verts: *const tbvh_vec4,
        vert_count: u32,
        stride: u32,
        indices: *const u32,
        prim_count: u32,
    );
    pub fn tbvh_mbvh8_build_hq(
        m: *mut tbvh_mbvh8,
        verts: *const tbvh_vec4,
        vert_count: u32,
        stride: u32,
    );
    pub fn tbvh_mbvh8_build_hq_indexed(
        m: *mut tbvh_mbvh8,
        verts: *const tbvh_vec4,
        vert_count: u32,
        stride: u32,
        indices: *const u32,
        prim_count: u32,
    );
    pub fn tbvh_mbvh8_convert_from(m: *mut tbvh_mbvh8, bvh: *mut tbvh_bvh, compact: bool);
    pub fn tbvh_mbvh8_optimize(m: *mut tbvh_mbvh8, iterations: u32, extreme: bool);
    pub fn tbvh_mbvh8_refit(m: *mut tbvh_mbvh8, node_idx: u32);
    pub fn tbvh_mbvh8_leaf_count(m: *const tbvh_mbvh8, node_idx: u32) -> u32;
    pub fn tbvh_mbvh8_sah_cost(m: *const tbvh_mbvh8, node_idx: u32) -> f32;
    pub fn tbvh_mbvh8_load_self_contained(m: *mut tbvh_mbvh8, path: *const c_char) -> bool;
    pub fn tbvh_mbvh8_nodes(m: *const tbvh_mbvh8, count: *mut u32) -> *const tbvh_mbvh8_node;
    pub fn tbvh_mbvh8_prim_indices(m: *const tbvh_mbvh8, count: *mut u32) -> *const u32;

    // tbvh_bvh4_cpu
    pub fn tbvh_bvh4_cpu_new() -> *mut tbvh_bvh4_cpu;
    pub fn tbvh_bvh4_cpu_free(b: *mut tbvh_bvh4_cpu);
    pub fn tbvh_bvh4_cpu_base(b: *mut tbvh_bvh4_cpu) -> *mut tbvh_base;
    pub fn tbvh_bvh4_cpu_base_const(b: *const tbvh_bvh4_cpu) -> *const tbvh_base;
    pub fn tbvh_bvh4_cpu_build(
        b: *mut tbvh_bvh4_cpu,
        verts: *const tbvh_vec4,
        vert_count: u32,
        stride: u32,
    );
    pub fn tbvh_bvh4_cpu_build_indexed(
        b: *mut tbvh_bvh4_cpu,
        verts: *const tbvh_vec4,
        vert_count: u32,
        stride: u32,
        indices: *const u32,
        prim_count: u32,
    );
    pub fn tbvh_bvh4_cpu_build_hq(
        b: *mut tbvh_bvh4_cpu,
        verts: *const tbvh_vec4,
        vert_count: u32,
        stride: u32,
    );
    pub fn tbvh_bvh4_cpu_build_hq_indexed(
        b: *mut tbvh_bvh4_cpu,
        verts: *const tbvh_vec4,
        vert_count: u32,
        stride: u32,
        indices: *const u32,
        prim_count: u32,
    );
    pub fn tbvh_bvh4_cpu_convert_from(
        b: *mut tbvh_bvh4_cpu,
        m: *mut tbvh_mbvh4,
        leaves_merged: bool,
    );
    pub fn tbvh_bvh4_cpu_optimize(b: *mut tbvh_bvh4_cpu, iterations: u32, extreme: bool);
    pub fn tbvh_bvh4_cpu_refit(b: *mut tbvh_bvh4_cpu);
    pub fn tbvh_bvh4_cpu_sah_cost(b: *const tbvh_bvh4_cpu, node_idx: u32) -> f32;
    pub fn tbvh_bvh4_cpu_intersect(b: *const tbvh_bvh4_cpu, ray: *mut tbvh_ray) -> i32;
    pub fn tbvh_bvh4_cpu_is_occluded(b: *const tbvh_bvh4_cpu, ray: *const tbvh_ray) -> bool;
    pub fn tbvh_bvh4_cpu_intersect_bundle(b: *const tbvh_bvh4_cpu, rays: *mut tbvh_ray) -> i32;
    pub fn tbvh_bvh4_cpu_is_occluded_bundle(
        b: *const tbvh_bvh4_cpu,
        rays: *mut tbvh_ray,
        occluded: *mut bool,
    ) -> i32;
    pub fn tbvh_bvh4_cpu_save(b: *const tbvh_bvh4_cpu, path: *const c_char);
    pub fn tbvh_bvh4_cpu_load(
        b: *mut tbvh_bvh4_cpu,
        path: *const c_char,
        expected_tris: u32,
    ) -> bool;
    pub fn tbvh_bvh4_cpu_load_self_contained(b: *mut tbvh_bvh4_cpu, path: *const c_char) -> bool;
    pub fn tbvh_bvh4_cpu_data(b: *const tbvh_bvh4_cpu, used_blocks: *mut u32) -> *const u8;

    // tbvh_bvh8_cpu
    pub fn tbvh_bvh8_cpu_new() -> *mut tbvh_bvh8_cpu;
    pub fn tbvh_bvh8_cpu_free(b: *mut tbvh_bvh8_cpu);
    pub fn tbvh_bvh8_cpu_base(b: *mut tbvh_bvh8_cpu) -> *mut tbvh_base;
    pub fn tbvh_bvh8_cpu_base_const(b: *const tbvh_bvh8_cpu) -> *const tbvh_base;
    pub fn tbvh_bvh8_cpu_build(
        b: *mut tbvh_bvh8_cpu,
        verts: *const tbvh_vec4,
        vert_count: u32,
        stride: u32,
    );
    pub fn tbvh_bvh8_cpu_build_indexed(
        b: *mut tbvh_bvh8_cpu,
        verts: *const tbvh_vec4,
        vert_count: u32,
        stride: u32,
        indices: *const u32,
        prim_count: u32,
    );
    pub fn tbvh_bvh8_cpu_build_hq(
        b: *mut tbvh_bvh8_cpu,
        verts: *const tbvh_vec4,
        vert_count: u32,
        stride: u32,
    );
    pub fn tbvh_bvh8_cpu_build_hq_indexed(
        b: *mut tbvh_bvh8_cpu,
        verts: *const tbvh_vec4,
        vert_count: u32,
        stride: u32,
        indices: *const u32,
        prim_count: u32,
    );
    pub fn tbvh_bvh8_cpu_convert_from(
        b: *mut tbvh_bvh8_cpu,
        m: *mut tbvh_mbvh8,
        leaves_merged: bool,
    );
    pub fn tbvh_bvh8_cpu_optimize(b: *mut tbvh_bvh8_cpu, iterations: u32, extreme: bool);
    pub fn tbvh_bvh8_cpu_refit(b: *mut tbvh_bvh8_cpu);
    pub fn tbvh_bvh8_cpu_sah_cost(b: *const tbvh_bvh8_cpu, node_idx: u32) -> f32;
    pub fn tbvh_bvh8_cpu_intersect(b: *const tbvh_bvh8_cpu, ray: *mut tbvh_ray) -> i32;
    pub fn tbvh_bvh8_cpu_is_occluded(b: *const tbvh_bvh8_cpu, ray: *const tbvh_ray) -> bool;
    pub fn tbvh_bvh8_cpu_save(b: *const tbvh_bvh8_cpu, path: *const c_char);
    pub fn tbvh_bvh8_cpu_load(
        b: *mut tbvh_bvh8_cpu,
        path: *const c_char,
        expected_tris: u32,
    ) -> bool;
    pub fn tbvh_bvh8_cpu_load_self_contained(b: *mut tbvh_bvh8_cpu, path: *const c_char) -> bool;
    pub fn tbvh_bvh8_cpu_data(b: *const tbvh_bvh8_cpu, used_blocks: *mut u32) -> *const u8;

    // tbvh_bvh4_gpu
    pub fn tbvh_bvh4_gpu_new() -> *mut tbvh_bvh4_gpu;
    pub fn tbvh_bvh4_gpu_free(b: *mut tbvh_bvh4_gpu);
    pub fn tbvh_bvh4_gpu_base(b: *mut tbvh_bvh4_gpu) -> *mut tbvh_base;
    pub fn tbvh_bvh4_gpu_base_const(b: *const tbvh_bvh4_gpu) -> *const tbvh_base;
    pub fn tbvh_bvh4_gpu_build(
        b: *mut tbvh_bvh4_gpu,
        verts: *const tbvh_vec4,
        vert_count: u32,
        stride: u32,
    );
    pub fn tbvh_bvh4_gpu_build_indexed(
        b: *mut tbvh_bvh4_gpu,
        verts: *const tbvh_vec4,
        vert_count: u32,
        stride: u32,
        indices: *const u32,
        prim_count: u32,
    );
    pub fn tbvh_bvh4_gpu_build_hq(
        b: *mut tbvh_bvh4_gpu,
        verts: *const tbvh_vec4,
        vert_count: u32,
        stride: u32,
    );
    pub fn tbvh_bvh4_gpu_build_hq_indexed(
        b: *mut tbvh_bvh4_gpu,
        verts: *const tbvh_vec4,
        vert_count: u32,
        stride: u32,
        indices: *const u32,
        prim_count: u32,
    );
    pub fn tbvh_bvh4_gpu_convert_from(b: *mut tbvh_bvh4_gpu, m: *const tbvh_mbvh4, compact: bool);
    pub fn tbvh_bvh4_gpu_optimize(b: *mut tbvh_bvh4_gpu, iterations: u32, extreme: bool);
    pub fn tbvh_bvh4_gpu_sah_cost(b: *const tbvh_bvh4_gpu, node_idx: u32) -> f32;
    pub fn tbvh_bvh4_gpu_intersect(b: *const tbvh_bvh4_gpu, ray: *mut tbvh_ray) -> i32;
    pub fn tbvh_bvh4_gpu_is_occluded(b: *const tbvh_bvh4_gpu, ray: *const tbvh_ray) -> bool;
    pub fn tbvh_bvh4_gpu_load_self_contained(b: *mut tbvh_bvh4_gpu, path: *const c_char) -> bool;
    pub fn tbvh_bvh4_gpu_data(b: *const tbvh_bvh4_gpu, used_blocks: *mut u32) -> *const tbvh_vec4;

    // tbvh_bvh4_amd_hw
    pub fn tbvh_bvh4_amd_hw_new() -> *mut tbvh_bvh4_amd_hw;
    pub fn tbvh_bvh4_amd_hw_free(b: *mut tbvh_bvh4_amd_hw);
    pub fn tbvh_bvh4_amd_hw_base(b: *mut tbvh_bvh4_amd_hw) -> *mut tbvh_base;
    pub fn tbvh_bvh4_amd_hw_base_const(b: *const tbvh_bvh4_amd_hw) -> *const tbvh_base;
    pub fn tbvh_bvh4_amd_hw_build(
        b: *mut tbvh_bvh4_amd_hw,
        verts: *const tbvh_vec4,
        vert_count: u32,
        stride: u32,
    );
    pub fn tbvh_bvh4_amd_hw_build_indexed(
        b: *mut tbvh_bvh4_amd_hw,
        verts: *const tbvh_vec4,
        vert_count: u32,
        stride: u32,
        indices: *const u32,
        prim_count: u32,
    );
    pub fn tbvh_bvh4_amd_hw_build_hq(
        b: *mut tbvh_bvh4_amd_hw,
        verts: *const tbvh_vec4,
        vert_count: u32,
        stride: u32,
    );
    pub fn tbvh_bvh4_amd_hw_build_hq_indexed(
        b: *mut tbvh_bvh4_amd_hw,
        verts: *const tbvh_vec4,
        vert_count: u32,
        stride: u32,
        indices: *const u32,
        prim_count: u32,
    );
    pub fn tbvh_bvh4_amd_hw_convert_from(b: *mut tbvh_bvh4_amd_hw, m: *const tbvh_mbvh4);
    pub fn tbvh_bvh4_amd_hw_optimize(b: *mut tbvh_bvh4_amd_hw, iterations: u32, extreme: bool);
    pub fn tbvh_bvh4_amd_hw_sah_cost(b: *const tbvh_bvh4_amd_hw, node_idx: u32) -> f32;
    pub fn tbvh_bvh4_amd_hw_load_self_contained(
        b: *mut tbvh_bvh4_amd_hw,
        path: *const c_char,
    ) -> bool;
    pub fn tbvh_bvh4_amd_hw_data(
        b: *const tbvh_bvh4_amd_hw,
        used_blocks: *mut u32,
    ) -> *const tbvh_vec4;

    // tbvh_bvh8_cwbvh
    pub fn tbvh_bvh8_cwbvh_new() -> *mut tbvh_bvh8_cwbvh;
    pub fn tbvh_bvh8_cwbvh_free(b: *mut tbvh_bvh8_cwbvh);
    pub fn tbvh_bvh8_cwbvh_base(b: *mut tbvh_bvh8_cwbvh) -> *mut tbvh_base;
    pub fn tbvh_bvh8_cwbvh_base_const(b: *const tbvh_bvh8_cwbvh) -> *const tbvh_base;
    pub fn tbvh_bvh8_cwbvh_build(
        b: *mut tbvh_bvh8_cwbvh,
        verts: *const tbvh_vec4,
        vert_count: u32,
        stride: u32,
    );
    pub fn tbvh_bvh8_cwbvh_build_indexed(
        b: *mut tbvh_bvh8_cwbvh,
        verts: *const tbvh_vec4,
        vert_count: u32,
        stride: u32,
        indices: *const u32,
        prim_count: u32,
    );
    pub fn tbvh_bvh8_cwbvh_build_hq(
        b: *mut tbvh_bvh8_cwbvh,
        verts: *const tbvh_vec4,
        vert_count: u32,
        stride: u32,
    );
    pub fn tbvh_bvh8_cwbvh_build_hq_indexed(
        b: *mut tbvh_bvh8_cwbvh,
        verts: *const tbvh_vec4,
        vert_count: u32,
        stride: u32,
        indices: *const u32,
        prim_count: u32,
    );
    pub fn tbvh_bvh8_cwbvh_convert_from(
        b: *mut tbvh_bvh8_cwbvh,
        m: *const tbvh_mbvh8,
        compact: bool,
    );
    pub fn tbvh_bvh8_cwbvh_optimize(b: *mut tbvh_bvh8_cwbvh, iterations: u32, extreme: bool);
    pub fn tbvh_bvh8_cwbvh_sah_cost(b: *const tbvh_bvh8_cwbvh, node_idx: u32) -> f32;
    pub fn tbvh_bvh8_cwbvh_intersect(b: *const tbvh_bvh8_cwbvh, ray: *mut tbvh_ray) -> i32;
    pub fn tbvh_bvh8_cwbvh_is_occluded(b: *const tbvh_bvh8_cwbvh, ray: *const tbvh_ray) -> bool;
    pub fn tbvh_bvh8_cwbvh_save(b: *const tbvh_bvh8_cwbvh, path: *const c_char);
    pub fn tbvh_bvh8_cwbvh_load(
        b: *mut tbvh_bvh8_cwbvh,
        path: *const c_char,
        expected_tris: u32,
    ) -> bool;
    pub fn tbvh_bvh8_cwbvh_load_self_contained(
        b: *mut tbvh_bvh8_cwbvh,
        path: *const c_char,
    ) -> bool;
    pub fn tbvh_bvh8_cwbvh_nodes(
        b: *const tbvh_bvh8_cwbvh,
        used_blocks: *mut u32,
    ) -> *const tbvh_vec4;
    pub fn tbvh_bvh8_cwbvh_tris(
        b: *const tbvh_bvh8_cwbvh,
        used_blocks: *mut u32,
    ) -> *const tbvh_vec4;

    // tbvh_voxelset
    pub fn tbvh_voxelset_new() -> *mut tbvh_voxelset;
    pub fn tbvh_voxelset_free(v: *mut tbvh_voxelset);
    pub fn tbvh_voxelset_base(v: *mut tbvh_voxelset) -> *mut tbvh_base;
    pub fn tbvh_voxelset_base_const(v: *const tbvh_voxelset) -> *const tbvh_base;
    pub fn tbvh_voxelset_set(v: *mut tbvh_voxelset, x: u32, y: u32, z: u32, value: u32);
    pub fn tbvh_voxelset_update_top_grid(v: *mut tbvh_voxelset);
    pub fn tbvh_voxelset_intersect(v: *const tbvh_voxelset, ray: *mut tbvh_ray) -> i32;
    pub fn tbvh_voxelset_is_occluded(v: *const tbvh_voxelset, ray: *const tbvh_ray) -> bool;
    pub fn tbvh_voxelset_normal(v: *const tbvh_voxelset, ray: *const tbvh_ray) -> tbvh_vec3;

    // tbvh_bvh_double
    pub fn tbvh_bvh_double_new() -> *mut tbvh_bvh_double;
    pub fn tbvh_bvh_double_free(b: *mut tbvh_bvh_double);
    pub fn tbvh_bvh_double_base(b: *mut tbvh_bvh_double) -> *mut tbvh_base_ex;
    pub fn tbvh_bvh_double_base_const(b: *const tbvh_bvh_double) -> *const tbvh_base_ex;
    pub fn tbvh_bvh_double_build(
        b: *mut tbvh_bvh_double,
        verts: *const tbvh_dbl3,
        vert_count: u64,
        stride: u64,
    );
    pub fn tbvh_bvh_double_build_indexed(
        b: *mut tbvh_bvh_double,
        verts: *const tbvh_dbl3,
        vert_count: u64,
        stride: u64,
        indices: *const u64,
        prim_count: u64,
    );
    pub fn tbvh_bvh_double_build_quick(
        b: *mut tbvh_bvh_double,
        verts: *const tbvh_dbl3,
        vert_count: u64,
        stride: u64,
    );
    pub fn tbvh_bvh_double_build_hq(
        b: *mut tbvh_bvh_double,
        verts: *const tbvh_dbl3,
        vert_count: u64,
        stride: u64,
    );
    pub fn tbvh_bvh_double_build_hq_indexed(
        b: *mut tbvh_bvh_double,
        verts: *const tbvh_dbl3,
        vert_count: u64,
        stride: u64,
        indices: *const u64,
        prim_count: u64,
    );
    pub fn tbvh_bvh_double_build_aabbs(
        b: *mut tbvh_bvh_double,
        aabbs: *const tbvh_dbl3,
        aabb_count: u64,
    );
    pub fn tbvh_bvh_double_build_custom(
        b: *mut tbvh_bvh_double,
        get_aabb: tbvh_aabb_ex_fn,
        prim_count: u64,
        user: *mut c_void,
    );
    pub fn tbvh_bvh_double_set_custom_callbacks(
        b: *mut tbvh_bvh_double,
        intersect: tbvh_intersect_ex_fn,
        occluded: tbvh_occluded_ex_fn,
        user: *mut c_void,
    );
    pub fn tbvh_bvh_double_build_tlas(
        b: *mut tbvh_bvh_double,
        instances: *mut tbvh_blas_instance_ex,
        inst_count: u64,
        blasses: *mut *mut tbvh_base_ex,
        blas_count: u64,
    );
    pub fn tbvh_bvh_double_intersect(b: *const tbvh_bvh_double, ray: *mut tbvh_ray_ex) -> i32;
    pub fn tbvh_bvh_double_is_occluded(b: *const tbvh_bvh_double, ray: *const tbvh_ray_ex) -> bool;
    pub fn tbvh_bvh_double_intersect_sphere(
        b: *const tbvh_bvh_double,
        pos: tbvh_dbl3,
        r: f64,
    ) -> bool;
    pub fn tbvh_bvh_double_refit(b: *mut tbvh_bvh_double, node_idx: u64);
    pub fn tbvh_bvh_double_compact(b: *mut tbvh_bvh_double);
    pub fn tbvh_bvh_double_optimize(
        b: *mut tbvh_bvh_double,
        iterations: u32,
        extreme: bool,
        stochastic: bool,
    );
    pub fn tbvh_bvh_double_split_leafs(b: *mut tbvh_bvh_double, max_prims: u64);
    pub fn tbvh_bvh_double_merge_leafs(b: *mut tbvh_bvh_double);
    pub fn tbvh_bvh_double_sah_cost(b: *const tbvh_bvh_double, node_idx: u64) -> f64;
    pub fn tbvh_bvh_double_epo_cost(b: *const tbvh_bvh_double, node_idx: u64) -> f64;
    pub fn tbvh_bvh_double_node_count(b: *const tbvh_bvh_double) -> u64;
    pub fn tbvh_bvh_double_leaf_count(b: *const tbvh_bvh_double) -> u64;
    pub fn tbvh_bvh_double_prim_count(b: *const tbvh_bvh_double, node_idx: u64) -> u64;
    pub fn tbvh_bvh_double_is_tlas(b: *const tbvh_bvh_double) -> bool;
    pub fn tbvh_bvh_double_is_indexed(b: *const tbvh_bvh_double) -> bool;
    pub fn tbvh_bvh_double_has_custom_geometry(b: *const tbvh_bvh_double) -> bool;
    pub fn tbvh_bvh_double_save(
        b: *const tbvh_bvh_double,
        path: *const c_char,
        include_geometry: bool,
    );
    pub fn tbvh_bvh_double_load(
        b: *mut tbvh_bvh_double,
        path: *const c_char,
        verts: *const tbvh_dbl3,
        vert_count: u64,
        stride: u64,
        indices: *const u64,
        prim_count: u64,
    ) -> bool;
    pub fn tbvh_bvh_double_load_self_contained(
        b: *mut tbvh_bvh_double,
        path: *const c_char,
    ) -> bool;
    pub fn tbvh_bvh_double_owns_geometry(b: *const tbvh_bvh_double) -> bool;
    pub fn tbvh_bvh_double_has_fragments(b: *const tbvh_bvh_double) -> bool;
    pub fn tbvh_bvh_double_free_geometry(b: *mut tbvh_bvh_double);
    pub fn tbvh_bvh_double_nodes(
        b: *const tbvh_bvh_double,
        count: *mut u64,
    ) -> *const tbvh_bvh_node_ex;
    pub fn tbvh_bvh_double_prim_indices(b: *const tbvh_bvh_double, count: *mut u64) -> *const u64;
    pub fn tbvh_bvh_double_verts(
        b: *const tbvh_bvh_double,
        count: *mut u64,
        stride: *mut u64,
    ) -> *const tbvh_dbl3;
    pub fn tbvh_bvh_double_vert_indices(b: *const tbvh_bvh_double) -> *const u64;

    // tbvh_bvh4_double
    pub fn tbvh_bvh4_double_new() -> *mut tbvh_bvh4_double;
    pub fn tbvh_bvh4_double_free(b: *mut tbvh_bvh4_double);
    pub fn tbvh_bvh4_double_base(b: *mut tbvh_bvh4_double) -> *mut tbvh_base_ex;
    pub fn tbvh_bvh4_double_base_const(b: *const tbvh_bvh4_double) -> *const tbvh_base_ex;
    pub fn tbvh_bvh4_double_build(
        b: *mut tbvh_bvh4_double,
        verts: *const tbvh_dbl3,
        vert_count: u64,
        stride: u64,
    );
    pub fn tbvh_bvh4_double_build_indexed(
        b: *mut tbvh_bvh4_double,
        verts: *const tbvh_dbl3,
        vert_count: u64,
        stride: u64,
        indices: *const u64,
        prim_count: u64,
    );
    pub fn tbvh_bvh4_double_build_hq(
        b: *mut tbvh_bvh4_double,
        verts: *const tbvh_dbl3,
        vert_count: u64,
        stride: u64,
    );
    pub fn tbvh_bvh4_double_build_hq_indexed(
        b: *mut tbvh_bvh4_double,
        verts: *const tbvh_dbl3,
        vert_count: u64,
        stride: u64,
        indices: *const u64,
        prim_count: u64,
    );
    pub fn tbvh_bvh4_double_optimize(b: *mut tbvh_bvh4_double, iterations: u32, extreme: bool);
    pub fn tbvh_bvh4_double_refit(b: *mut tbvh_bvh4_double);
    pub fn tbvh_bvh4_double_sah_cost(b: *const tbvh_bvh4_double, node_idx: u64) -> f64;
    pub fn tbvh_bvh4_double_intersect(b: *const tbvh_bvh4_double, ray: *mut tbvh_ray_ex) -> i32;
    pub fn tbvh_bvh4_double_is_occluded(
        b: *const tbvh_bvh4_double,
        ray: *const tbvh_ray_ex,
    ) -> bool;
    pub fn tbvh_bvh4_double_save(b: *const tbvh_bvh4_double, path: *const c_char);
    pub fn tbvh_bvh4_double_load(
        b: *mut tbvh_bvh4_double,
        path: *const c_char,
        expected_tris: u64,
    ) -> bool;
    pub fn tbvh_bvh4_double_load_self_contained(
        b: *mut tbvh_bvh4_double,
        path: *const c_char,
    ) -> bool;
    pub fn tbvh_bvh4_double_data(b: *const tbvh_bvh4_double, used_blocks: *mut u32) -> *const u8;
}

// Compile-time layout checks for the mirrored structs (64-bit targets).
const _: () = {
    use core::mem::{align_of, size_of};
    assert!(size_of::<tbvh_vec3>() == 12);
    assert!(size_of::<tbvh_vec4>() == 16 && align_of::<tbvh_vec4>() == 16);
    assert!(size_of::<tbvh_dbl3>() == 24);
    assert!(size_of::<tbvh_mat4>() == 64);
    assert!(size_of::<tbvh_dblmat4>() == 128);
    assert!(size_of::<tbvh_build_settings>() == 40);
    assert!(size_of::<tbvh_bvh_node>() == 32);
    assert!(size_of::<tbvh_bvh_node_ex>() == 64);
    assert!(size_of::<tbvh_bvh_verbose_node>() == 64);
    assert!(size_of::<tbvh_bvh_gpu_node>() == 64);
    assert!(size_of::<tbvh_mbvh4_node>() == 64);
    assert!(size_of::<tbvh_mbvh8_node>() == 80);
    if PTR_SIZE == 8 {
        assert!(size_of::<tbvh_intersection>() == 84);
        assert!(size_of::<tbvh_intersection_ex>() == 104);
        assert!(size_of::<tbvh_ray>() == 128 && align_of::<tbvh_ray>() == 64);
        assert!(size_of::<tbvh_ray_ex>() == 192 && align_of::<tbvh_ray_ex>() == 64);
        assert!(size_of::<tbvh_blas_instance>() == 192);
        assert!(size_of::<tbvh_blas_instance_ex>() == 320);
    }
};
