//! Build settings and layout identifiers shared by every acceleration structure.

use crate::ffi;

/// Build configuration (tinybvh's `BVHBuildSettings`).
///
/// Settings are per structure and are read when a build starts. Derived layouts pass them on to
/// the intermediate structures they build internally, so a [`Bvh4Cpu`](crate::Bvh4Cpu) with
/// `use_spatial_splits` set builds its binary tree with Bonsai HQ.
///
/// The builder is selected as follows (see the tinybvh README):
///
/// | setting | builder |
/// | --- | --- |
/// | *(default)* | Bonsai, a fast parallel SIMD builder |
/// | `use_spatial_splits` | Bonsai HQ, with SAH-guided triangle splitting |
/// | `binned_bvh` | binned SAH, using the SIMD builder |
/// | `binned_bvh` + `use_spatial_splits` | SBVH |
/// | `use_simd_if_available = false` | scalar reference binned SAH builder, or SBVH |
/// | `use_full_sweep` | full-sweep SAH, for experiments |
/// | `use_hploc` | H-PLOC |
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BuildSettings {
    /// Pre-split large triangles before building.
    pub use_presplitting: bool,
    /// Split triangles during construction: Bonsai HQ, or SBVH together with `binned_bvh`.
    /// A tree built with spatial splits cannot be refitted.
    pub use_spatial_splits: bool,
    /// SBVH: minimal overlap of the child boxes before a spatial split is considered.
    pub min_spatial_overlap: f32,
    /// SBVH: skip the smallest axis when looking for spatial splits.
    pub spatial_skip_smallest_axis: bool,
    /// Try to un-split primitives in leaves after a presplit build.
    pub presplit_post_pass: bool,
    /// Presplit budget relative to the input size.
    pub presplit_factor: f32,
    /// Full-sweep SAH builder, for experiments only.
    pub use_full_sweep: bool,
    /// Binned SAH builders instead of Bonsai.
    pub binned_bvh: bool,
    /// Bonsai: grid resolution target, in primitives per grid cell.
    pub bonsai_grid_prims: u32,
    /// Bonsai: prune mini trees larger than this fraction of the average; 0 disables pruning.
    pub bonsai_prune_factor: f32,
    /// H-PLOC builder.
    pub use_hploc: bool,
    /// Run the tree-rotation optimizer after the build.
    pub post_optimize: bool,
    /// Iterations for `post_optimize`.
    pub optimize_iterations: i32,
    /// Use the SIMD builder when the platform has one.
    pub use_simd_if_available: bool,
    /// Build on the process-wide thread pool. Disable when the application itself builds many
    /// structures in parallel.
    pub enable_threading: bool,
    /// Free the intermediate binary tree after a layout conversion. Saves memory, but the derived
    /// layout can then no longer be refitted or optimized.
    pub discard_intermediates: bool,
    /// Size intermediate buffers exactly, at a small cost in build time.
    pub minimize_mem_use: bool,
}

impl Default for BuildSettings {
    /// The tinybvh defaults: Bonsai, SIMD and threading on, no spatial splits.
    fn default() -> Self {
        BuildSettings {
            use_presplitting: false,
            use_spatial_splits: false,
            min_spatial_overlap: 1e-3,
            spatial_skip_smallest_axis: true,
            presplit_post_pass: true,
            presplit_factor: 0.3,
            use_full_sweep: false,
            binned_bvh: false,
            bonsai_grid_prims: 256,
            bonsai_prune_factor: 0.1,
            use_hploc: false,
            post_optimize: false,
            optimize_iterations: 25,
            use_simd_if_available: true,
            enable_threading: true,
            discard_intermediates: false,
            minimize_mem_use: true,
        }
    }
}

impl BuildSettings {
    /// Settings for the highest quality builder: spatial splits (Bonsai HQ).
    pub fn high_quality() -> Self {
        BuildSettings {
            use_spatial_splits: true,
            ..Default::default()
        }
    }

    /// Settings for a plain binned SAH build without spatial splits, which keeps the tree refittable.
    pub fn refittable() -> Self {
        BuildSettings {
            use_spatial_splits: false,
            use_presplitting: false,
            ..Default::default()
        }
    }

    pub(crate) fn to_ffi(self) -> ffi::tbvh_build_settings {
        ffi::tbvh_build_settings {
            use_presplitting: self.use_presplitting,
            use_spatial_splits: self.use_spatial_splits,
            min_spatial_overlap: self.min_spatial_overlap,
            spatial_skip_smallest_axis: self.spatial_skip_smallest_axis,
            presplit_post_pass: self.presplit_post_pass,
            presplit_factor: self.presplit_factor,
            use_full_sweep: self.use_full_sweep,
            binned_bvh: self.binned_bvh,
            bonsai_grid_prims: self.bonsai_grid_prims,
            bonsai_prune_factor: self.bonsai_prune_factor,
            use_hploc: self.use_hploc,
            post_optimize: self.post_optimize,
            optimize_iterations: self.optimize_iterations,
            use_simd_if_available: self.use_simd_if_available,
            enable_threading: self.enable_threading,
            discard_intermediates: self.discard_intermediates,
            minimize_mem_use: self.minimize_mem_use,
        }
    }

    pub(crate) fn from_ffi(s: &ffi::tbvh_build_settings) -> Self {
        BuildSettings {
            use_presplitting: s.use_presplitting,
            use_spatial_splits: s.use_spatial_splits,
            min_spatial_overlap: s.min_spatial_overlap,
            spatial_skip_smallest_axis: s.spatial_skip_smallest_axis,
            presplit_post_pass: s.presplit_post_pass,
            presplit_factor: s.presplit_factor,
            use_full_sweep: s.use_full_sweep,
            binned_bvh: s.binned_bvh,
            bonsai_grid_prims: s.bonsai_grid_prims,
            bonsai_prune_factor: s.bonsai_prune_factor,
            use_hploc: s.use_hploc,
            post_optimize: s.post_optimize,
            optimize_iterations: s.optimize_iterations,
            use_simd_if_available: s.use_simd_if_available,
            enable_threading: s.enable_threading,
            discard_intermediates: s.discard_intermediates,
            minimize_mem_use: s.minimize_mem_use,
        }
    }
}

/// Identifies the memory layout of an acceleration structure (tinybvh's `BVHType`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Layout {
    /// Not built, or unknown.
    Undefined,
    /// [`Bvh`](crate::Bvh): 32-byte nodes.
    Bvh,
    /// [`BvhVerbose`](crate::BvhVerbose): 64-byte nodes with parent links.
    BvhVerbose,
    /// [`BvhGpu`](crate::BvhGpu): Aila & Laine nodes.
    BvhGpu,
    /// [`Mbvh4`](crate::Mbvh4) or [`Mbvh8`](crate::Mbvh8): wide intermediate layouts.
    Mbvh,
    /// [`Bvh4Cpu`](crate::Bvh4Cpu): 4-wide SIMD CPU traversal.
    Bvh4Cpu,
    /// [`Bvh4Gpu`](crate::Bvh4Gpu): quantized 4-wide GPU layout.
    Bvh4Gpu,
    /// [`Bvh8Cwbvh`](crate::Bvh8Cwbvh): compressed wide BVH.
    Cwbvh,
    /// [`Bvh8Cpu`](crate::Bvh8Cpu): 8-wide SIMD CPU traversal.
    Bvh8Cpu,
    /// [`VoxelSet`](crate::VoxelSet).
    VoxelSet,
    /// [`Bvh4AmdHw`](crate::Bvh4AmdHw): AMD hardware format.
    Bvh4AmdHw,
}

impl Layout {
    pub(crate) fn from_raw(raw: ffi::tbvh_layout) -> Layout {
        match raw {
            ffi::TBVH_LAYOUT_BVH => Layout::Bvh,
            ffi::TBVH_LAYOUT_BVH_VERBOSE => Layout::BvhVerbose,
            ffi::TBVH_LAYOUT_BVH_GPU => Layout::BvhGpu,
            ffi::TBVH_LAYOUT_MBVH => Layout::Mbvh,
            ffi::TBVH_LAYOUT_BVH4_CPU => Layout::Bvh4Cpu,
            ffi::TBVH_LAYOUT_BVH4_GPU => Layout::Bvh4Gpu,
            ffi::TBVH_LAYOUT_CWBVH => Layout::Cwbvh,
            ffi::TBVH_LAYOUT_BVH8_SIMD => Layout::Bvh8Cpu,
            ffi::TBVH_LAYOUT_VOXELSET => Layout::VoxelSet,
            ffi::TBVH_LAYOUT_BVH4_AMD_HW => Layout::Bvh4AmdHw,
            _ => Layout::Undefined,
        }
    }
}
