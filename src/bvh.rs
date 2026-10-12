//! [`Bvh`]: tinybvh's default layout with 32-byte nodes.

use core::ptr::NonNull;
use std::path::Path;
use std::sync::Arc;

use glam::{Vec3, Vec4};

use crate::error::{Error, Result};
use crate::ffi;
use crate::geometry::{self, CustomGeometry, CustomHolder, Geometry};
use crate::instance::{sealed, Blas, BlasInstance};
use crate::ray::Ray;

/// A node of a [`Bvh`] (tinybvh's `BVH::BVHNode`): 32 bytes, two per cache line.
///
/// Interior nodes store the index of their left child; the right child is `left + 1`. Leaves store
/// the first slot in the [primitive index array](Bvh::prim_indices) and the number of primitives.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BvhNode {
    /// Minimum corner of the node bounds.
    pub aabb_min: Vec3,
    /// Left child index for an interior node; first primitive index slot for a leaf.
    pub left_first: u32,
    /// Maximum corner of the node bounds.
    pub aabb_max: Vec3,
    /// Number of primitives for a leaf; 0 for an interior node.
    pub tri_count: u32,
}

impl BvhNode {
    /// Whether the node is a leaf. Empty leaves do not exist, so this is `tri_count > 0`.
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.tri_count > 0
    }

    /// The children of an interior node as `(left, right)`; `None` for a leaf.
    #[inline]
    pub fn children(&self) -> Option<(u32, u32)> {
        if self.is_leaf() {
            None
        } else {
            Some((self.left_first, self.left_first + 1))
        }
    }

    /// The range into [`Bvh::prim_indices`] covered by a leaf; empty for an interior node.
    #[inline]
    pub fn prim_range(&self) -> core::ops::Range<usize> {
        if self.is_leaf() {
            self.left_first as usize..(self.left_first + self.tri_count) as usize
        } else {
            0..0
        }
    }

    /// Half the surface area of the node bounds, the quantity the SAH works with.
    #[inline]
    pub fn half_area(&self) -> f32 {
        let e = self.aabb_max - self.aabb_min;
        e.x * e.y + e.y * e.z + e.z * e.x
    }
}

const _: () = assert!(core::mem::size_of::<BvhNode>() == core::mem::size_of::<ffi::tbvh_bvh_node>());

/// What the structure was built over; keeps the referenced data alive.
enum Source {
    /// Nothing built yet.
    Empty,
    /// Triangles owned by the structure.
    Triangles(Geometry),
    /// Custom primitives.
    Custom(CustomHolder),
    /// A TLAS over instances of other structures.
    Tlas { instances: Vec<BlasInstance>, blasses: Vec<Arc<dyn Blas>>, blas_ptrs: Vec<*mut ffi::tbvh_base> },
    /// Loaded from a self-contained file; tinybvh owns the geometry.
    Loaded,
}

/// The standard BVH layout (tinybvh's `BVH`): 32-byte nodes, cross-platform, and the basis of
/// every other layout.
///
/// A `Bvh` can be built over a triangle soup, indexed triangles, [custom primitives](CustomGeometry)
/// or, as a top-level acceleration structure, over [`BlasInstance`]s of other structures. It is the
/// only layout that can be a TLAS, and the only one with the full set of maintenance operations:
/// [refitting](Bvh::refit), [optimization](Bvh::optimize), [compaction](Bvh::compact), leaf
/// surgery and [serialization](Bvh::save).
///
/// For raw traversal speed on the CPU use [`Bvh4Cpu`](crate::Bvh4Cpu); use `Bvh` when you need
/// a TLAS, custom primitives, or access to the tree itself.
///
/// # Example
///
/// ```
/// use glam::{Vec3, Vec4};
/// use tinybvh::{Bvh, BuildSettings, Ray};
///
/// let tri = |z: f32| [Vec4::new(-1.0, -1.0, z, 0.0), Vec4::new(1.0, -1.0, z, 0.0), Vec4::new(0.0, 1.0, z, 0.0)];
/// let mut vertices = Vec::new();
/// vertices.extend(tri(5.0));
/// vertices.extend(tri(3.0));
///
/// let mut bvh = Bvh::new().with_settings(&BuildSettings::high_quality());
/// bvh.build(vertices);
///
/// let mut ray = Ray::new(Vec3::ZERO, Vec3::Z);
/// bvh.intersect(&mut ray);
/// assert_eq!(ray.hit.prim, 1); // the closer triangle
/// assert!(bvh.is_occluded(&Ray::with_t_max(Vec3::ZERO, Vec3::Z, 4.0)));
/// assert!(!bvh.is_occluded(&Ray::with_t_max(Vec3::ZERO, Vec3::Z, 2.0)));
/// ```
pub struct Bvh {
    raw: NonNull<ffi::tbvh_bvh>,
    source: Source,
    opacity_map: Option<Vec<u32>>,
}

impl_base_accessors!(Bvh, tbvh_bvh_base, tbvh_bvh_base_const);
impl_opacity_maps!(Bvh);

impl Bvh {
    /// An empty structure with default [`BuildSettings`](crate::BuildSettings).
    pub fn new() -> Self {
        crate::install_fatal_trampoline();
        let raw = unsafe { ffi::tbvh_bvh_new() };
        Bvh { raw: NonNull::new(raw).expect("tinybvh: allocation failed"), source: Source::Empty, opacity_map: None }
    }

    // --- builders ------------------------------------------------------------------------------

    /// Builds over triangles with the builder selected by the [settings](Self::settings)
    /// (Bonsai by default). Any previous content is replaced.
    ///
    /// Accepts a `Vec<Vec4>` with three vertices per triangle, a `(Vec<Vec4>, Vec<u32>)` pair for
    /// indexed triangles, or a [`Geometry`]. Panics if the geometry is empty, not a multiple of
    /// three, or has an index out of range.
    pub fn build(&mut self, geometry: impl Into<Geometry>) {
        let geometry = geometry.into();
        geometry.validate();
        self.reset_source();
        unsafe {
            match geometry.index_args() {
                None => ffi::tbvh_bvh_build(self.raw.as_ptr(), geometry.vert_ptr(), geometry.vert_count(), geometry::VEC4_STRIDE),
                Some((idx, prims)) => ffi::tbvh_bvh_build_indexed(
                    self.raw.as_ptr(),
                    geometry.vert_ptr(),
                    geometry.vert_count(),
                    geometry::VEC4_STRIDE,
                    idx,
                    prims,
                ),
            }
        }
        self.source = Source::Triangles(geometry);
    }

    /// Builds with spatial splits for a higher quality tree (tinybvh's `BuildHQ`). The result
    /// cannot be [refitted](Self::refit).
    pub fn build_hq(&mut self, geometry: impl Into<Geometry>) {
        let geometry = geometry.into();
        geometry.validate();
        self.reset_source();
        unsafe {
            match geometry.index_args() {
                None => ffi::tbvh_bvh_build_hq(self.raw.as_ptr(), geometry.vert_ptr(), geometry.vert_count(), geometry::VEC4_STRIDE),
                Some((idx, prims)) => ffi::tbvh_bvh_build_hq_indexed(
                    self.raw.as_ptr(),
                    geometry.vert_ptr(),
                    geometry.vert_count(),
                    geometry::VEC4_STRIDE,
                    idx,
                    prims,
                ),
            }
        }
        self.source = Source::Triangles(geometry);
    }

    /// Builds with the single-function midpoint-split builder (tinybvh's `BuildQuick`): fast but
    /// low quality, for reference and debugging. Only supports triangle soups; indexed geometry
    /// panics.
    pub fn build_quick(&mut self, geometry: impl Into<Geometry>) {
        let geometry = geometry.into();
        geometry.validate();
        assert!(geometry.indices.is_none(), "tinybvh: build_quick does not support indexed geometry");
        self.reset_source();
        unsafe { ffi::tbvh_bvh_build_quick(self.raw.as_ptr(), geometry.vert_ptr(), geometry.vert_count(), geometry::VEC4_STRIDE) };
        self.source = Source::Triangles(geometry);
    }

    /// Builds with the binned SAH SIMD builder directly (tinybvh's `BuildSIMD`), bypassing the
    /// settings-based selection. Fails when tinybvh was compiled without a SIMD builder; see
    /// [`simd_builder_available`](crate::simd_builder_available).
    pub fn build_simd(&mut self, geometry: impl Into<Geometry>) -> Result<()> {
        if !crate::simd_builder_available() {
            return Err(Error::SimdBuilderUnavailable);
        }
        let geometry = geometry.into();
        geometry.validate();
        self.reset_source();
        unsafe {
            match geometry.index_args() {
                None => ffi::tbvh_bvh_build_simd(self.raw.as_ptr(), geometry.vert_ptr(), geometry.vert_count(), geometry::VEC4_STRIDE),
                Some((idx, prims)) => ffi::tbvh_bvh_build_simd_indexed(
                    self.raw.as_ptr(),
                    geometry.vert_ptr(),
                    geometry.vert_count(),
                    geometry::VEC4_STRIDE,
                    idx,
                    prims,
                ),
            }
        }
        self.source = Source::Triangles(geometry);
        Ok(())
    }

    /// Builds over user-defined primitives. The tree is built over the bounds reported by
    /// [`CustomGeometry::aabb`]; queries call the geometry's intersection methods. Panics if the
    /// geometry reports zero primitives.
    pub fn build_custom<G: CustomGeometry + 'static>(&mut self, geometry: G) {
        self.build_custom_boxed(Box::new(geometry));
    }

    /// [`build_custom`](Self::build_custom) for an already boxed geometry.
    pub fn build_custom_boxed(&mut self, geometry: Box<dyn CustomGeometry>) {
        let prim_count = geometry.prim_count();
        assert!(prim_count > 0, "tinybvh: custom geometry has no primitives");
        self.reset_source();
        let holder = CustomHolder::new(geometry);
        unsafe {
            ffi::tbvh_bvh_set_custom_callbacks(
                self.raw.as_ptr(),
                CustomHolder::intersect_fn(),
                CustomHolder::occluded_fn(),
                holder.user_ptr(),
            );
            ffi::tbvh_bvh_build_custom(self.raw.as_ptr(), CustomHolder::aabb_fn(), prim_count, holder.user_ptr());
        }
        self.source = Source::Custom(holder);
        geometry::rethrow_pending_panic();
    }

    /// Builds a top-level acceleration structure over `instances`, each referring by
    /// [`blas_idx`](BlasInstance::blas_idx) to one of `blasses`. The instances' inverse
    /// transforms and world bounds are computed here. The TLAS keeps the bottom-level structures
    /// alive through the `Arc`s.
    ///
    /// Panics if there are no instances, an instance refers to a BLAS index out of range, or a
    /// BLAS has not been built.
    pub fn build_tlas(&mut self, instances: Vec<BlasInstance>, blasses: Vec<Arc<dyn Blas>>) {
        assert!(!instances.is_empty(), "tinybvh: a TLAS needs at least one instance");
        assert!(!blasses.is_empty(), "tinybvh: a TLAS needs at least one BLAS");
        for (i, inst) in instances.iter().enumerate() {
            assert!(
                (inst.blas_idx as usize) < blasses.len(),
                "tinybvh: instance {i} refers to BLAS {} but only {} were given",
                inst.blas_idx,
                blasses.len()
            );
        }
        for (i, blas) in blasses.iter().enumerate() {
            assert!(blas.blas_is_ready(), "tinybvh: BLAS {i} has not been built");
        }
        let blas_ptrs: Vec<*mut ffi::tbvh_base> = blasses.iter().map(|b| b.blas_base_ptr() as *mut _).collect();
        self.reset_source();
        self.source = Source::Tlas { instances, blasses, blas_ptrs };
        self.run_tlas_build();
    }

    /// Rebuilds the TLAS over its current instances, after transforms were changed through
    /// [`instances_mut`](Self::instances_mut) or a BLAS was refitted. Much cheaper than
    /// [`build_tlas`](Self::build_tlas) for animation, since nothing is re-validated or moved.
    pub fn rebuild_tlas(&mut self) -> Result<()> {
        match self.source {
            Source::Tlas { .. } => {
                self.run_tlas_build();
                Ok(())
            }
            _ => Err(Error::NotTlas),
        }
    }

    fn run_tlas_build(&mut self) {
        let raw = self.raw.as_ptr();
        if let Source::Tlas { instances, blas_ptrs, .. } = &mut self.source {
            unsafe {
                ffi::tbvh_bvh_build_tlas(
                    raw,
                    instances.as_mut_ptr() as *mut ffi::tbvh_blas_instance,
                    instances.len() as u32,
                    blas_ptrs.as_mut_ptr(),
                    blas_ptrs.len() as u32,
                );
            }
        }
    }

    fn reset_source(&mut self) {
        // Drop the old data only after tinybvh stops referencing it, i.e. after the new build;
        // the callbacks are cleared here so a custom build cannot run against freed state.
        if matches!(self.source, Source::Custom(_)) {
            unsafe { ffi::tbvh_bvh_set_custom_callbacks(self.raw.as_ptr(), None, None, core::ptr::null_mut()) };
        }
        self.source = Source::Empty;
    }

    // --- queries ---------------------------------------------------------------------------------

    /// Whether a tree is available for queries.
    pub fn is_built(&self) -> bool {
        self.used_nodes() > 0 && !matches!(self.source, Source::Empty)
    }

    #[inline]
    fn assert_built(&self) {
        assert!(self.is_built(), "tinybvh: the Bvh has not been built");
    }

    /// Finds the closest hit along `ray` and records it in `ray.hit`. Returns the number of
    /// traversal steps, a measure of the cost of the query.
    ///
    /// Panics if the structure has not been built.
    pub fn intersect(&self, ray: &mut Ray) -> u32 {
        self.assert_built();
        ray.assert_finite();
        let steps = unsafe { ffi::tbvh_bvh_intersect(self.raw.as_ptr(), ray.as_ffi_mut()) };
        geometry::rethrow_pending_panic();
        steps.max(0) as u32
    }

    /// Whether anything blocks `ray` before `ray.hit.t`. Cheaper than [`intersect`](Self::intersect)
    /// for shadow rays.
    pub fn is_occluded(&self, ray: &Ray) -> bool {
        self.assert_built();
        ray.assert_finite();
        let occluded = unsafe { ffi::tbvh_bvh_is_occluded(self.raw.as_ptr(), ray.as_ffi()) };
        geometry::rethrow_pending_panic();
        occluded
    }

    /// Whether any triangle of the structure intersects the sphere. Panics for structures without
    /// triangle geometry (custom primitives, TLAS).
    pub fn intersect_sphere(&self, center: Vec3, radius: f32) -> bool {
        self.assert_built();
        assert!(self.has_vertex_data(), "tinybvh: intersect_sphere needs triangle geometry");
        unsafe { ffi::tbvh_bvh_intersect_sphere(self.raw.as_ptr(), geometry::vec3_to_ffi(center), radius) }
    }

    /// Traces a bundle of coherent rays at once (WiVeC traversal). Only available on a TLAS;
    /// `rays.len()` must equal [`bundle_rays`](crate::bundle_rays). Returns the traversal cost.
    pub fn intersect_bundle(&self, rays: &mut [Ray]) -> Result<u32> {
        self.assert_built();
        if !self.is_tlas() {
            return Err(Error::NotTlas);
        }
        let n = crate::bundle_rays();
        assert!(rays.len() == n, "tinybvh: a bundle holds exactly {n} rays, got {}", rays.len());
        rays.iter().for_each(Ray::assert_finite);
        let steps = unsafe { ffi::tbvh_bvh_intersect_bundle(self.raw.as_ptr(), rays.as_mut_ptr() as *mut ffi::tbvh_ray) };
        geometry::rethrow_pending_panic();
        Ok(steps.max(0) as u32)
    }

    /// Occlusion test for a bundle of rays; `occluded[i]` receives the result for `rays[i]`.
    /// Only available on a TLAS; both slices must have [`bundle_rays`](crate::bundle_rays)
    /// entries.
    pub fn is_occluded_bundle(&self, rays: &mut [Ray], occluded: &mut [bool]) -> Result<u32> {
        self.assert_built();
        if !self.is_tlas() {
            return Err(Error::NotTlas);
        }
        let n = crate::bundle_rays();
        assert!(rays.len() == n && occluded.len() == n, "tinybvh: a bundle holds exactly {n} rays");
        rays.iter().for_each(Ray::assert_finite);
        let steps = unsafe {
            ffi::tbvh_bvh_is_occluded_bundle(self.raw.as_ptr(), rays.as_mut_ptr() as *mut ffi::tbvh_ray, occluded.as_mut_ptr())
        };
        geometry::rethrow_pending_panic();
        Ok(steps.max(0) as u32)
    }

    // --- maintenance -----------------------------------------------------------------------------

    /// Updates the node bounds after the vertices changed through [`vertices_mut`](Self::vertices_mut),
    /// keeping the topology. Much faster than a rebuild; suitable for subtle animation.
    ///
    /// Fails for trees built with spatial splits or presplitting, for a TLAS (rebuild it), and
    /// for structures without triangle geometry. A tree with unused nodes is compacted first.
    pub fn refit(&mut self) -> Result<()> {
        if !self.is_built() {
            return Err(Error::NotBuilt);
        }
        if !matches!(self.source, Source::Triangles(_) | Source::Loaded) || !self.has_vertex_data() {
            return Err(Error::NoGeometry);
        }
        if self.is_tlas() || !self.is_refittable() {
            return Err(Error::NotRefittable);
        }
        if self.may_have_holes() {
            self.compact();
            if self.may_have_holes() {
                return Err(Error::HasHoles);
            }
        }
        unsafe { ffi::tbvh_bvh_refit(self.raw.as_ptr(), 0) };
        Ok(())
    }

    /// Removes unused nodes and orders the nodes depth-first. Required before refitting a tree
    /// that [may have holes](Self::may_have_holes); done automatically by [`refit`](Self::refit).
    pub fn compact(&mut self) {
        self.assert_built();
        unsafe { ffi::tbvh_bvh_compact(self.raw.as_ptr()) };
    }

    /// Improves the tree with insertion-based tree rotations (Bittner et al.), lowering the SAH
    /// cost and raising traversal performance. `iterations` of 25 is the tinybvh default.
    pub fn optimize(&mut self, iterations: u32) {
        self.optimize_with(iterations, false, false);
    }

    /// [`optimize`](Self::optimize) with tinybvh's extra knobs: `extreme` considers every node as
    /// a candidate for reinsertion; `stochastic` randomizes candidate selection.
    pub fn optimize_with(&mut self, iterations: u32, extreme: bool, stochastic: bool) {
        self.assert_built();
        if self.may_have_holes() {
            self.compact();
            assert!(!self.may_have_holes(), "tinybvh: the tree still has holes after compaction");
        }
        unsafe { ffi::tbvh_bvh_optimize(self.raw.as_ptr(), iterations, extreme, stochastic) };
    }

    /// Splits leaves until none holds more than `max_prims` primitives.
    ///
    /// Needs the per-primitive bounds kept from the build; a tree loaded from a file has none and
    /// fails with [`NotRebuildable`](Error::NotRebuildable).
    pub fn split_leafs(&mut self, max_prims: u32) -> Result<()> {
        if !self.is_built() {
            return Err(Error::NotBuilt);
        }
        assert!(max_prims > 0, "tinybvh: max_prims must be at least 1");
        if !self.has_fragments() {
            return Err(Error::NotRebuildable);
        }
        unsafe { ffi::tbvh_bvh_split_leafs(self.raw.as_ptr(), max_prims) };
        Ok(())
    }

    /// Whether the per-primitive bounds of the build are still available (false after a load).
    #[doc(hidden)]
    pub fn has_fragments(&self) -> bool {
        unsafe { ffi::tbvh_bvh_has_fragments(self.raw.as_ptr()) }
    }

    /// Merges leaves where the SAH says it pays off.
    pub fn merge_leafs(&mut self) {
        self.assert_built();
        if self.may_have_holes() {
            self.compact();
        }
        unsafe { ffi::tbvh_bvh_merge_leafs(self.raw.as_ptr()) };
    }

    // --- metrics ---------------------------------------------------------------------------------

    /// SAH cost of the subtree rooted at `node` (0 for the whole tree): lower is better.
    pub fn sah_cost(&self, node: u32) -> f32 {
        self.assert_built();
        assert!(node < self.used_nodes(), "tinybvh: node index out of range");
        unsafe { ffi::tbvh_bvh_sah_cost(self.raw.as_ptr(), node) }
    }

    /// End-point overlap cost (Aila et al. 2013) of the subtree rooted at `node`, a quality metric
    /// that accounts for primitives sticking out of their nodes. Needs triangle geometry.
    pub fn epo_cost(&self, node: u32) -> Result<f32> {
        if !self.is_built() {
            return Err(Error::NotBuilt);
        }
        if !self.has_vertex_data() {
            return Err(Error::NoGeometry);
        }
        assert!(node < self.used_nodes(), "tinybvh: node index out of range");
        Ok(unsafe { ffi::tbvh_bvh_epo_cost(self.raw.as_ptr(), node) })
    }

    /// Number of reachable nodes in the tree.
    pub fn node_count(&self) -> u32 {
        if !self.is_built() {
            return 0;
        }
        unsafe { ffi::tbvh_bvh_node_count(self.raw.as_ptr()) }
    }

    /// Number of leaves in the tree.
    pub fn leaf_count(&self) -> u32 {
        if !self.is_built() {
            return 0;
        }
        unsafe { ffi::tbvh_bvh_leaf_count(self.raw.as_ptr()) }
    }

    /// Number of primitive references in the subtree rooted at `node`.
    pub fn prim_count_in(&self, node: u32) -> u32 {
        self.assert_built();
        assert!(node < self.used_nodes(), "tinybvh: node index out of range");
        unsafe { ffi::tbvh_bvh_prim_count(self.raw.as_ptr(), node) }
    }

    /// Indices of all nodes in the subtree rooted at `root`, in traversal order.
    pub fn collect_nodes(&self, root: u32) -> Vec<u32> {
        self.assert_built();
        assert!(root < self.used_nodes(), "tinybvh: node index out of range");
        let cap = self.used_nodes();
        let mut list = vec![0u32; cap as usize];
        let n = unsafe { ffi::tbvh_bvh_collect_nodes(self.raw.as_ptr(), root, list.as_mut_ptr(), cap) };
        list.truncate(n as usize);
        list
    }

    /// Whether this structure is a top-level acceleration structure.
    pub fn is_tlas(&self) -> bool {
        unsafe { ffi::tbvh_bvh_is_tlas(self.raw.as_ptr()) }
    }

    /// Whether the structure was built over indexed triangles.
    pub fn is_indexed(&self) -> bool {
        unsafe { ffi::tbvh_bvh_is_indexed(self.raw.as_ptr()) }
    }

    /// Whether the structure was built over [custom primitives](CustomGeometry).
    pub fn has_custom_geometry(&self) -> bool {
        unsafe { ffi::tbvh_bvh_has_custom_geometry(self.raw.as_ptr()) }
    }

    fn has_vertex_data(&self) -> bool {
        let (mut count, mut stride) = (0u32, 0u32);
        let ptr = unsafe { ffi::tbvh_bvh_verts(self.raw.as_ptr(), &mut count, &mut stride) };
        !ptr.is_null() && count > 0
    }

    // --- serialization ---------------------------------------------------------------------------

    /// Writes the tree to a file, including the triangle geometry when `include_geometry` is set
    /// so that [`load_self_contained`](Self::load_self_contained) can restore it on its own.
    /// A TLAS cannot be saved.
    pub fn save(&self, path: impl AsRef<Path>, include_geometry: bool) -> Result<()> {
        if !self.is_built() {
            return Err(Error::NotBuilt);
        }
        if self.is_tlas() {
            return Err(Error::NotTlas);
        }
        let cpath = geometry::path_to_cstring(path.as_ref())?;
        unsafe { ffi::tbvh_bvh_save(self.raw.as_ptr(), cpath.as_ptr(), include_geometry) };
        Ok(())
    }

    /// Loads a tree saved with [`save`](Self::save) and binds it to `geometry`, which must be the
    /// data the tree was built over (same triangle count and indexing). Any previous content is
    /// replaced; on failure the structure is left empty.
    pub fn load(&mut self, path: impl AsRef<Path>, geometry: impl Into<Geometry>) -> Result<()> {
        let geometry = geometry.into();
        geometry.validate();
        let cpath = geometry::path_to_cstring(path.as_ref())?;
        self.reset_source();
        let (idx, prims) = geometry.index_args().unwrap_or((core::ptr::null(), 0));
        let ok = unsafe {
            ffi::tbvh_bvh_load(
                self.raw.as_ptr(),
                cpath.as_ptr(),
                geometry.vert_ptr(),
                geometry.vert_count(),
                geometry::VEC4_STRIDE,
                idx,
                prims,
            )
        };
        if ok {
            self.source = Source::Triangles(geometry);
            Ok(())
        } else {
            Err(Error::LoadFailed(path.as_ref().display().to_string()))
        }
    }

    /// Loads a tree together with the geometry embedded in the file by
    /// [`save`](Self::save) with `include_geometry`. Fails if the file carries no geometry.
    pub fn load_self_contained(path: impl AsRef<Path>) -> Result<Bvh> {
        let cpath = geometry::path_to_cstring(path.as_ref())?;
        let mut bvh = Bvh::new();
        let ok = unsafe { ffi::tbvh_bvh_load_self_contained(bvh.raw.as_ptr(), cpath.as_ptr()) };
        if !ok {
            return Err(Error::LoadFailed(path.as_ref().display().to_string()));
        }
        if !bvh.has_vertex_data() {
            return Err(Error::NoGeometry);
        }
        bvh.source = Source::Loaded;
        Ok(bvh)
    }

    // --- data access -----------------------------------------------------------------------------

    /// The nodes in use. Node 0 is the root.
    pub fn nodes(&self) -> &[BvhNode] {
        let mut count = 0u32;
        let ptr = unsafe { ffi::tbvh_bvh_nodes(self.raw.as_ptr(), &mut count) };
        if ptr.is_null() || count == 0 {
            return &[];
        }
        unsafe { core::slice::from_raw_parts(ptr as *const BvhNode, count as usize) }
    }

    /// The primitive index array that leaves refer to.
    pub fn prim_indices(&self) -> &[u32] {
        let mut count = 0u32;
        let ptr = unsafe { ffi::tbvh_bvh_prim_indices(self.raw.as_ptr(), &mut count) };
        if ptr.is_null() || count == 0 {
            return &[];
        }
        unsafe { core::slice::from_raw_parts(ptr, count as usize) }
    }

    /// The triangle geometry the tree was built over, if any.
    pub fn geometry(&self) -> Option<&Geometry> {
        match &self.source {
            Source::Triangles(g) => Some(g),
            _ => None,
        }
    }

    /// The vertices the tree references: the owned geometry, or the geometry loaded from a
    /// self-contained file.
    pub fn vertices(&self) -> Option<&[Vec4]> {
        match &self.source {
            Source::Triangles(g) => Some(&g.vertices),
            Source::Loaded => {
                let (mut count, mut stride) = (0u32, 0u32);
                let ptr = unsafe { ffi::tbvh_bvh_verts(self.raw.as_ptr(), &mut count, &mut stride) };
                if ptr.is_null() || count == 0 || stride != geometry::VEC4_STRIDE {
                    None
                } else {
                    Some(unsafe { core::slice::from_raw_parts(ptr as *const Vec4, count as usize) })
                }
            }
            _ => None,
        }
    }

    /// Mutable access to the vertices, for animation followed by [`refit`](Self::refit). The
    /// number of vertices cannot change. Only available for owned geometry.
    pub fn vertices_mut(&mut self) -> Option<&mut [Vec4]> {
        match &mut self.source {
            Source::Triangles(g) => Some(&mut g.vertices),
            _ => None,
        }
    }

    /// The vertex indices for indexed geometry.
    pub fn indices(&self) -> Option<&[u32]> {
        match &self.source {
            Source::Triangles(g) => g.indices.as_deref(),
            Source::Loaded => {
                let ptr = unsafe { ffi::tbvh_bvh_vert_indices(self.raw.as_ptr()) };
                if ptr.is_null() {
                    None
                } else {
                    Some(unsafe { core::slice::from_raw_parts(ptr, self.prim_count() as usize * 3) })
                }
            }
            _ => None,
        }
    }

    /// The custom geometry the tree was built over, if any.
    pub fn custom_geometry(&self) -> Option<&dyn CustomGeometry> {
        match &self.source {
            Source::Custom(holder) => Some(holder.geometry()),
            _ => None,
        }
    }

    /// The instances of a TLAS.
    pub fn instances(&self) -> &[BlasInstance] {
        match &self.source {
            Source::Tlas { instances, .. } => instances,
            _ => &[],
        }
    }

    /// Mutable access to the instances of a TLAS, to change transforms or masks. Call
    /// [`rebuild_tlas`](Self::rebuild_tlas) afterwards; until then queries use stale bounds.
    pub fn instances_mut(&mut self) -> &mut [BlasInstance] {
        match &mut self.source {
            Source::Tlas { instances, .. } => instances,
            _ => &mut [],
        }
    }

    /// The bottom-level structures of a TLAS.
    pub fn blasses(&self) -> &[Arc<dyn Blas>] {
        match &self.source {
            Source::Tlas { blasses, .. } => blasses,
            _ => &[],
        }
    }

    /// Releases everything the structure holds, including the geometry, and returns it to the
    /// empty state.
    pub fn clear(&mut self) {
        self.reset_source();
        unsafe {
            ffi::tbvh_bvh_free(self.raw.as_ptr());
            self.raw = NonNull::new(ffi::tbvh_bvh_new()).expect("tinybvh: allocation failed");
        }
        self.opacity_map = None;
    }

    /// Takes the owned triangle geometry out of the structure, leaving it empty.
    pub fn into_geometry(mut self) -> Option<Geometry> {
        let source = core::mem::replace(&mut self.source, Source::Empty);
        unsafe {
            ffi::tbvh_bvh_free(self.raw.as_ptr());
            self.raw = NonNull::new(ffi::tbvh_bvh_new()).expect("tinybvh: allocation failed");
        }
        match source {
            Source::Triangles(g) => Some(g),
            _ => None,
        }
    }

    #[doc(hidden)]
    pub fn raw_ptr(&self) -> *const ffi::tbvh_bvh {
        self.raw.as_ptr()
    }

    #[doc(hidden)]
    pub fn raw_ptr_mut(&mut self) -> *mut ffi::tbvh_bvh {
        self.raw.as_ptr()
    }
}

impl Default for Bvh {
    fn default() -> Self {
        Bvh::new()
    }
}

impl Drop for Bvh {
    fn drop(&mut self) {
        // The C++ object goes first: it references the geometry and callbacks dropped afterwards.
        unsafe { ffi::tbvh_bvh_free(self.raw.as_ptr()) };
    }
}

impl core::fmt::Debug for Bvh {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Bvh")
            .field("built", &self.is_built())
            .field("prim_count", &self.prim_count())
            .field("used_nodes", &self.used_nodes())
            .field("is_tlas", &self.is_tlas())
            .field("custom", &self.has_custom_geometry())
            .finish()
    }
}

// Queries take &self and only read the tree; builds take &mut self. The referenced geometry is
// owned here, custom geometry is Send + Sync by trait bound, and BLASes are shared via Arc.
unsafe impl Send for Bvh {}
unsafe impl Sync for Bvh {}

impl sealed::Sealed for Bvh {}
impl Blas for Bvh {
    fn blas_base_ptr(&self) -> *const ffi::tbvh_base {
        self.base_ptr()
    }

    fn blas_is_ready(&self) -> bool {
        self.is_built()
    }
}

impl From<Vec<Vec4>> for Geometry {
    fn from(vertices: Vec<Vec4>) -> Self {
        Geometry::new(vertices)
    }
}

impl From<(Vec<Vec4>, Vec<u32>)> for Geometry {
    fn from((vertices, indices): (Vec<Vec4>, Vec<u32>)) -> Self {
        Geometry::indexed(vertices, indices)
    }
}
