//! Double precision layouts: [`BvhDouble`] (tinybvh's `BVH_Double`) and [`Bvh4Double`]
//! (`BVH4_Double`), with `f64` coordinates and 64-bit indices.

use core::ptr::NonNull;
use std::path::Path;
use std::sync::Arc;

use glam::DVec3;

use crate::error::{Error, Result};
use crate::ffi;
use crate::geometry::{self, CustomGeometryEx, CustomHolderEx, GeometryEx};
use crate::instance::{sealed, BlasEx, BlasInstanceEx};
use crate::ray::RayEx;

/// A node of a [`BvhDouble`] (tinybvh's `BVH_Double::BVHNode`): 64 bytes, one per cache line.
///
/// Interior nodes store the index of their left child; the right child is `left + 1`. Leaves store
/// the first slot in the [primitive index array](BvhDouble::prim_indices) and the number of
/// primitives.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BvhNodeEx {
    /// Minimum corner of the node bounds.
    pub aabb_min: DVec3,
    /// Left child index for an interior node; first primitive index slot for a leaf.
    pub left_first: u64,
    /// Maximum corner of the node bounds.
    pub aabb_max: DVec3,
    /// Number of primitives for a leaf; 0 for an interior node.
    pub tri_count: u64,
}

impl BvhNodeEx {
    /// Whether the node is a leaf. Empty leaves do not exist, so this is `tri_count > 0`.
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.tri_count > 0
    }

    /// The children of an interior node as `(left, right)`; `None` for a leaf.
    #[inline]
    pub fn children(&self) -> Option<(u64, u64)> {
        if self.is_leaf() {
            None
        } else {
            Some((self.left_first, self.left_first + 1))
        }
    }

    /// The range into [`BvhDouble::prim_indices`] covered by a leaf; empty for an interior node.
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
    pub fn half_area(&self) -> f64 {
        let e = self.aabb_max - self.aabb_min;
        e.x * e.y + e.y * e.z + e.z * e.x
    }
}

const _: () = {
    use core::mem::{align_of, size_of};
    assert!(size_of::<BvhNodeEx>() == size_of::<ffi::tbvh_bvh_node_ex>());
    assert!(align_of::<BvhNodeEx>() == align_of::<ffi::tbvh_bvh_node_ex>());
    assert!(core::mem::offset_of!(BvhNodeEx, left_first) == 24);
    assert!(core::mem::offset_of!(BvhNodeEx, aabb_max) == 32);
    assert!(core::mem::offset_of!(BvhNodeEx, tri_count) == 56);
};

/// What a [`BvhDouble`] was built over; keeps the referenced data alive.
enum Source {
    /// Nothing built yet.
    Empty,
    /// Triangles owned by the structure.
    Triangles(GeometryEx),
    /// Custom primitives.
    Custom(CustomHolderEx),
    /// A TLAS over instances of other structures.
    Tlas { instances: Vec<BlasInstanceEx>, blasses: Vec<Arc<dyn BlasEx>>, blas_ptrs: Vec<*mut ffi::tbvh_base_ex> },
    /// Loaded from a self-contained file; tinybvh owns the geometry.
    Loaded,
}

/// The double precision counterpart of [`Bvh`](crate::Bvh) (tinybvh's `BVH_Double`): 64-byte
/// nodes with `f64` bounds and 64-bit indices, for scenes that exceed the range or precision of
/// single precision floats.
///
/// A `BvhDouble` can be built over a triangle soup, indexed triangles, [custom
/// primitives](CustomGeometryEx) or, as a top-level acceleration structure, over
/// [`BlasInstanceEx`]s of other double precision structures ([`BvhDouble`] and [`Bvh4Double`]).
/// It supports the same maintenance operations as [`Bvh`](crate::Bvh): [refitting](Self::refit),
/// [optimization](Self::optimize), [compaction](Self::compact), leaf surgery and
/// [serialization](Self::save). There is no double precision SIMD builder, so the Bonsai
/// settings fall back to the scalar reference builders; ray bundles and opacity micro maps are not
/// available in double precision.
///
/// # Example
///
/// ```
/// use glam::DVec3;
/// use tinybvh::{BvhDouble, RayEx};
///
/// let tri = |z: f64| [DVec3::new(-1.0, -1.0, z), DVec3::new(1.0, -1.0, z), DVec3::new(0.0, 1.0, z)];
/// let mut vertices = Vec::new();
/// vertices.extend(tri(5.0));
/// vertices.extend(tri(3.0));
///
/// let mut bvh = BvhDouble::new();
/// bvh.build(vertices);
///
/// let mut ray = RayEx::new(DVec3::ZERO, DVec3::Z);
/// bvh.intersect(&mut ray);
/// assert_eq!(ray.hit.prim, 1); // the closer triangle
/// assert!((ray.hit.t - 3.0).abs() < 1e-12);
/// assert!(bvh.is_occluded(&RayEx::with_t_max(DVec3::ZERO, DVec3::Z, 4.0)));
/// assert!(!bvh.is_occluded(&RayEx::with_t_max(DVec3::ZERO, DVec3::Z, 2.0)));
/// ```
pub struct BvhDouble {
    raw: NonNull<ffi::tbvh_bvh_double>,
    source: Source,
}

impl_base_accessors_ex!(BvhDouble, tbvh_bvh_double_base, tbvh_bvh_double_base_const);

impl BvhDouble {
    /// An empty structure with default [`BuildSettings`](crate::BuildSettings).
    pub fn new() -> Self {
        crate::install_fatal_trampoline();
        let raw = unsafe { ffi::tbvh_bvh_double_new() };
        BvhDouble { raw: NonNull::new(raw).expect("tinybvh: allocation failed"), source: Source::Empty }
    }

    // --- builders ------------------------------------------------------------------------------

    /// Builds over triangles with the builder selected by the [settings](Self::settings). Any
    /// previous content is replaced.
    ///
    /// Accepts a `Vec<DVec3>` with three vertices per triangle, a `(Vec<DVec3>, Vec<u64>)` pair
    /// for indexed triangles, or a [`GeometryEx`]. Panics if the geometry is empty, not a multiple
    /// of three, or has an index out of range.
    pub fn build(&mut self, geometry: impl Into<GeometryEx>) {
        let geometry = geometry.into();
        geometry.validate();
        self.reset_source();
        unsafe {
            match geometry.index_args() {
                None => ffi::tbvh_bvh_double_build(
                    self.raw.as_ptr(),
                    geometry.vert_ptr(),
                    geometry.vert_count(),
                    geometry::DVEC3_STRIDE,
                ),
                Some((idx, prims)) => ffi::tbvh_bvh_double_build_indexed(
                    self.raw.as_ptr(),
                    geometry.vert_ptr(),
                    geometry.vert_count(),
                    geometry::DVEC3_STRIDE,
                    idx,
                    prims,
                ),
            }
        }
        self.source = Source::Triangles(geometry);
    }

    /// Builds with spatial splits for a higher quality tree (tinybvh's `BuildHQ`). The result
    /// cannot be [refitted](Self::refit).
    pub fn build_hq(&mut self, geometry: impl Into<GeometryEx>) {
        let geometry = geometry.into();
        geometry.validate();
        self.reset_source();
        unsafe {
            match geometry.index_args() {
                None => ffi::tbvh_bvh_double_build_hq(
                    self.raw.as_ptr(),
                    geometry.vert_ptr(),
                    geometry.vert_count(),
                    geometry::DVEC3_STRIDE,
                ),
                Some((idx, prims)) => ffi::tbvh_bvh_double_build_hq_indexed(
                    self.raw.as_ptr(),
                    geometry.vert_ptr(),
                    geometry.vert_count(),
                    geometry::DVEC3_STRIDE,
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
    pub fn build_quick(&mut self, geometry: impl Into<GeometryEx>) {
        let geometry = geometry.into();
        geometry.validate();
        assert!(geometry.indices.is_none(), "tinybvh: build_quick does not support indexed geometry");
        self.reset_source();
        unsafe {
            ffi::tbvh_bvh_double_build_quick(
                self.raw.as_ptr(),
                geometry.vert_ptr(),
                geometry.vert_count(),
                geometry::DVEC3_STRIDE,
            )
        };
        self.source = Source::Triangles(geometry);
    }

    /// Builds over user-defined primitives. The tree is built over the bounds reported by
    /// [`CustomGeometryEx::aabb`]; queries call the geometry's intersection methods. Panics if
    /// the geometry reports zero primitives.
    pub fn build_custom<G: CustomGeometryEx + 'static>(&mut self, geometry: G) {
        self.build_custom_boxed(Box::new(geometry));
    }

    /// [`build_custom`](Self::build_custom) for an already boxed geometry.
    pub fn build_custom_boxed(&mut self, geometry: Box<dyn CustomGeometryEx>) {
        let prim_count = geometry.prim_count();
        assert!(prim_count > 0, "tinybvh: custom geometry has no primitives");
        self.reset_source();
        let holder = CustomHolderEx::new(geometry);
        unsafe {
            ffi::tbvh_bvh_double_set_custom_callbacks(
                self.raw.as_ptr(),
                CustomHolderEx::intersect_fn(),
                CustomHolderEx::occluded_fn(),
                holder.user_ptr(),
            );
            ffi::tbvh_bvh_double_build_custom(self.raw.as_ptr(), CustomHolderEx::aabb_fn(), prim_count, holder.user_ptr());
        }
        self.source = Source::Custom(holder);
        geometry::rethrow_pending_panic();
    }

    /// Builds a top-level acceleration structure over `instances`, each referring by
    /// [`blas_idx`](BlasInstanceEx::blas_idx) to one of `blasses`. The instances' inverse
    /// transforms and world bounds are computed here. The TLAS keeps the bottom-level structures
    /// alive through the `Arc`s.
    ///
    /// Panics if there are no instances, an instance refers to a BLAS index out of range, or a
    /// BLAS has not been built.
    pub fn build_tlas(&mut self, instances: Vec<BlasInstanceEx>, blasses: Vec<Arc<dyn BlasEx>>) {
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
        let blas_ptrs: Vec<*mut ffi::tbvh_base_ex> = blasses.iter().map(|b| b.blas_base_ptr() as *mut _).collect();
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
                ffi::tbvh_bvh_double_build_tlas(
                    raw,
                    instances.as_mut_ptr() as *mut ffi::tbvh_blas_instance_ex,
                    instances.len() as u64,
                    blas_ptrs.as_mut_ptr(),
                    blas_ptrs.len() as u64,
                );
            }
        }
    }

    fn reset_source(&mut self) {
        // Drop the old data only after tinybvh stops referencing it, i.e. after the new build;
        // the callbacks are cleared here so a custom build cannot run against freed state.
        // tinybvh's own builders leave the callbacks in place, the shim clears them on request.
        if matches!(self.source, Source::Custom(_)) {
            unsafe { ffi::tbvh_bvh_double_set_custom_callbacks(self.raw.as_ptr(), None, None, core::ptr::null_mut()) };
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
        assert!(self.is_built(), "tinybvh: the BvhDouble has not been built");
    }

    /// Finds the closest hit along `ray` and records it in `ray.hit`. Returns the number of
    /// traversal steps, a measure of the cost of the query.
    ///
    /// Panics if the structure has not been built or the ray contains NaN.
    pub fn intersect(&self, ray: &mut RayEx) -> u32 {
        self.assert_built();
        ray.assert_finite();
        let steps = unsafe { ffi::tbvh_bvh_double_intersect(self.raw.as_ptr(), ray.as_ffi_mut()) };
        geometry::rethrow_pending_panic();
        steps.max(0) as u32
    }

    /// Whether anything blocks `ray` strictly before `ray.hit.t`. Cheaper than
    /// [`intersect`](Self::intersect) for shadow rays. A ray that already records a hit at `t`
    /// is therefore not occluded by the primitive it hit.
    pub fn is_occluded(&self, ray: &RayEx) -> bool {
        self.assert_built();
        ray.assert_finite();
        let occluded = unsafe { ffi::tbvh_bvh_double_is_occluded(self.raw.as_ptr(), ray.as_ffi()) };
        geometry::rethrow_pending_panic();
        occluded
    }

    /// Whether any triangle of the structure intersects the sphere. Panics unless the structure
    /// was built over (or loaded with) triangle geometry.
    pub fn intersect_sphere(&self, center: DVec3, radius: f64) -> bool {
        self.assert_built();
        assert!(self.has_vertex_data(), "tinybvh: intersect_sphere needs triangle geometry");
        unsafe { ffi::tbvh_bvh_double_intersect_sphere(self.raw.as_ptr(), geometry::dvec3_to_ffi(center), radius) }
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
        unsafe { ffi::tbvh_bvh_double_refit(self.raw.as_ptr(), 0) };
        Ok(())
    }

    /// Removes unused nodes and orders the nodes depth-first. Required before refitting a tree
    /// that [may have holes](Self::may_have_holes); done automatically by [`refit`](Self::refit).
    pub fn compact(&mut self) {
        self.assert_built();
        unsafe { ffi::tbvh_bvh_double_compact(self.raw.as_ptr()) };
    }

    /// Improves the tree with insertion-based tree rotations (Bittner et al.), lowering the SAH
    /// cost and raising traversal performance. `iterations` of 25 is the tinybvh default.
    pub fn optimize(&mut self, iterations: u32) {
        self.optimize_with(iterations, false, false);
    }

    /// [`optimize`](Self::optimize) with tinybvh's extra knobs: `extreme` considers every node as
    /// a candidate for reinsertion; `stochastic` randomizes candidate selection.
    ///
    /// A tree with unused nodes is compacted first.
    pub fn optimize_with(&mut self, iterations: u32, extreme: bool, stochastic: bool) {
        self.assert_built();
        if self.may_have_holes() {
            self.compact();
            assert!(!self.may_have_holes(), "tinybvh: the tree still has holes after compaction");
        }
        unsafe { ffi::tbvh_bvh_double_optimize(self.raw.as_ptr(), iterations, extreme, stochastic) };
    }

    /// Splits leaves until none holds more than `max_prims` primitives.
    ///
    /// Needs the per-primitive bounds kept from the build; a tree loaded from a file has none and
    /// fails with [`NotRebuildable`](Error::NotRebuildable).
    pub fn split_leafs(&mut self, max_prims: u64) -> Result<()> {
        if !self.is_built() {
            return Err(Error::NotBuilt);
        }
        assert!(max_prims > 0, "tinybvh: max_prims must be at least 1");
        if !unsafe { ffi::tbvh_bvh_double_has_fragments(self.raw.as_ptr()) } {
            return Err(Error::NotRebuildable);
        }
        unsafe { ffi::tbvh_bvh_double_split_leafs(self.raw.as_ptr(), max_prims) };
        Ok(())
    }

    /// Merges leaves where the SAH says it pays off. Afterwards the tree
    /// [may have holes](Self::may_have_holes) until it is [compacted](Self::compact).
    pub fn merge_leafs(&mut self) {
        self.assert_built();
        if self.may_have_holes() {
            self.compact();
        }
        unsafe { ffi::tbvh_bvh_double_merge_leafs(self.raw.as_ptr()) };
    }

    // --- metrics ---------------------------------------------------------------------------------

    /// SAH cost of the subtree rooted at `node` (0 for the whole tree): lower is better.
    pub fn sah_cost(&self, node: u64) -> f64 {
        self.assert_built();
        assert!(node < self.used_nodes(), "tinybvh: node index out of range");
        unsafe { ffi::tbvh_bvh_double_sah_cost(self.raw.as_ptr(), node) }
    }

    /// End-point overlap cost (Aila et al. 2013) of the subtree rooted at `node`, a quality metric
    /// that accounts for primitives sticking out of their nodes. Needs triangle geometry.
    pub fn epo_cost(&self, node: u64) -> Result<f64> {
        if !self.is_built() {
            return Err(Error::NotBuilt);
        }
        if !self.has_vertex_data() {
            return Err(Error::NoGeometry);
        }
        assert!(node < self.used_nodes(), "tinybvh: node index out of range");
        Ok(unsafe { ffi::tbvh_bvh_double_epo_cost(self.raw.as_ptr(), node) })
    }

    /// Number of reachable nodes in the tree.
    pub fn node_count(&self) -> u64 {
        if !self.is_built() {
            return 0;
        }
        unsafe { ffi::tbvh_bvh_double_node_count(self.raw.as_ptr()) }
    }

    /// Number of leaves in the tree.
    pub fn leaf_count(&self) -> u64 {
        if !self.is_built() {
            return 0;
        }
        unsafe { ffi::tbvh_bvh_double_leaf_count(self.raw.as_ptr()) }
    }

    /// Number of primitive references in the subtree rooted at `node`.
    pub fn prim_count_in(&self, node: u64) -> u64 {
        self.assert_built();
        assert!(node < self.used_nodes(), "tinybvh: node index out of range");
        unsafe { ffi::tbvh_bvh_double_prim_count(self.raw.as_ptr(), node) }
    }

    /// Whether this structure is a top-level acceleration structure.
    pub fn is_tlas(&self) -> bool {
        unsafe { ffi::tbvh_bvh_double_is_tlas(self.raw.as_ptr()) }
    }

    /// Whether the structure was built over indexed triangles.
    pub fn is_indexed(&self) -> bool {
        unsafe { ffi::tbvh_bvh_double_is_indexed(self.raw.as_ptr()) }
    }

    /// Whether the structure was built over [custom primitives](CustomGeometryEx).
    pub fn has_custom_geometry(&self) -> bool {
        unsafe { ffi::tbvh_bvh_double_has_custom_geometry(self.raw.as_ptr()) }
    }

    fn has_vertex_data(&self) -> bool {
        let (mut count, mut stride) = (0u64, 0u64);
        let ptr = unsafe { ffi::tbvh_bvh_double_verts(self.raw.as_ptr(), &mut count, &mut stride) };
        !ptr.is_null() && count > 0
    }

    // --- serialization ---------------------------------------------------------------------------

    /// Writes the tree to a file, including the triangle geometry when `include_geometry` is set
    /// so that [`load_self_contained`](Self::load_self_contained) can restore it on its own.
    /// A TLAS cannot be saved.
    ///
    /// tinybvh stores the vertex count of the embedded geometry as a 32-bit number, so a file
    /// with more than 2^32 - 1 vertices cannot be restored self-contained; bind such a tree to
    /// its geometry with [`load`](Self::load) instead.
    pub fn save(&self, path: impl AsRef<Path>, include_geometry: bool) -> Result<()> {
        if !self.is_built() {
            return Err(Error::NotBuilt);
        }
        if self.is_tlas() {
            return Err(Error::NotTlas);
        }
        let cpath = geometry::path_to_cstring(path.as_ref())?;
        unsafe { ffi::tbvh_bvh_double_save(self.raw.as_ptr(), cpath.as_ptr(), include_geometry) };
        Ok(())
    }

    /// Loads a tree saved with [`save`](Self::save) and binds it to `geometry`, which must be the
    /// data the tree was built over (same triangle count and indexing). Any previous content is
    /// replaced; on failure the structure is left empty.
    pub fn load(&mut self, path: impl AsRef<Path>, geometry: impl Into<GeometryEx>) -> Result<()> {
        let geometry = geometry.into();
        geometry.validate();
        let cpath = geometry::path_to_cstring(path.as_ref())?;
        self.reset_source();
        let (idx, prims) = geometry.index_args().unwrap_or((core::ptr::null(), 0));
        let ok = unsafe {
            ffi::tbvh_bvh_double_load(
                self.raw.as_ptr(),
                cpath.as_ptr(),
                geometry.vert_ptr(),
                geometry.vert_count(),
                geometry::DVEC3_STRIDE,
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
    pub fn load_self_contained(path: impl AsRef<Path>) -> Result<BvhDouble> {
        let cpath = geometry::path_to_cstring(path.as_ref())?;
        let mut bvh = BvhDouble::new();
        let ok = unsafe { ffi::tbvh_bvh_double_load_self_contained(bvh.raw.as_ptr(), cpath.as_ptr()) };
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

    /// The nodes in use. Node 0 is the root; node 1 is unused padding.
    pub fn nodes(&self) -> &[BvhNodeEx] {
        let mut count = 0u64;
        let ptr = unsafe { ffi::tbvh_bvh_double_nodes(self.raw.as_ptr(), &mut count) };
        if ptr.is_null() || count == 0 {
            return &[];
        }
        unsafe { core::slice::from_raw_parts(ptr as *const BvhNodeEx, count as usize) }
    }

    /// The primitive index array that leaves refer to.
    pub fn prim_indices(&self) -> &[u64] {
        let mut count = 0u64;
        let ptr = unsafe { ffi::tbvh_bvh_double_prim_indices(self.raw.as_ptr(), &mut count) };
        if ptr.is_null() || count == 0 {
            return &[];
        }
        unsafe { core::slice::from_raw_parts(ptr, count as usize) }
    }

    /// The triangle geometry the tree was built over, if any.
    pub fn geometry(&self) -> Option<&GeometryEx> {
        match &self.source {
            Source::Triangles(g) => Some(g),
            _ => None,
        }
    }

    /// The vertices the tree references: the owned geometry, or the geometry loaded from a
    /// self-contained file.
    pub fn vertices(&self) -> Option<&[DVec3]> {
        match &self.source {
            Source::Triangles(g) => Some(&g.vertices),
            Source::Loaded => {
                let (mut count, mut stride) = (0u64, 0u64);
                let ptr = unsafe { ffi::tbvh_bvh_double_verts(self.raw.as_ptr(), &mut count, &mut stride) };
                if ptr.is_null() || count == 0 || stride != geometry::DVEC3_STRIDE {
                    None
                } else {
                    Some(unsafe { core::slice::from_raw_parts(ptr as *const DVec3, count as usize) })
                }
            }
            _ => None,
        }
    }

    /// Mutable access to the vertices, for animation followed by [`refit`](Self::refit). The
    /// number of vertices cannot change. Only available for owned geometry.
    pub fn vertices_mut(&mut self) -> Option<&mut [DVec3]> {
        match &mut self.source {
            Source::Triangles(g) => Some(&mut g.vertices),
            _ => None,
        }
    }

    /// The vertex indices for indexed geometry.
    pub fn indices(&self) -> Option<&[u64]> {
        match &self.source {
            Source::Triangles(g) => g.indices.as_deref(),
            Source::Loaded => {
                let ptr = unsafe { ffi::tbvh_bvh_double_vert_indices(self.raw.as_ptr()) };
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
    pub fn custom_geometry(&self) -> Option<&dyn CustomGeometryEx> {
        match &self.source {
            Source::Custom(holder) => Some(holder.geometry()),
            _ => None,
        }
    }

    /// The instances of a TLAS.
    pub fn instances(&self) -> &[BlasInstanceEx] {
        match &self.source {
            Source::Tlas { instances, .. } => instances,
            _ => &[],
        }
    }

    /// Mutable access to the instances of a TLAS, to change transforms or masks. Call
    /// [`rebuild_tlas`](Self::rebuild_tlas) afterwards; until then queries use stale bounds.
    pub fn instances_mut(&mut self) -> &mut [BlasInstanceEx] {
        match &mut self.source {
            Source::Tlas { instances, .. } => instances,
            _ => &mut [],
        }
    }

    /// The bottom-level structures of a TLAS.
    pub fn blasses(&self) -> &[Arc<dyn BlasEx>] {
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
            ffi::tbvh_bvh_double_free(self.raw.as_ptr());
            self.raw = NonNull::new(ffi::tbvh_bvh_double_new()).expect("tinybvh: allocation failed");
        }
    }

    /// Takes the owned triangle geometry out of the structure, leaving it empty.
    pub fn into_geometry(mut self) -> Option<GeometryEx> {
        let source = core::mem::replace(&mut self.source, Source::Empty);
        unsafe {
            ffi::tbvh_bvh_double_free(self.raw.as_ptr());
            self.raw = NonNull::new(ffi::tbvh_bvh_double_new()).expect("tinybvh: allocation failed");
        }
        match source {
            Source::Triangles(g) => Some(g),
            _ => None,
        }
    }

    #[doc(hidden)]
    pub fn raw_ptr(&self) -> *const ffi::tbvh_bvh_double {
        self.raw.as_ptr()
    }

    #[doc(hidden)]
    pub fn raw_ptr_mut(&mut self) -> *mut ffi::tbvh_bvh_double {
        self.raw.as_ptr()
    }
}

impl Default for BvhDouble {
    fn default() -> Self {
        BvhDouble::new()
    }
}

impl Drop for BvhDouble {
    fn drop(&mut self) {
        // The C++ object goes first: it references the geometry and callbacks dropped afterwards.
        unsafe { ffi::tbvh_bvh_double_free(self.raw.as_ptr()) };
    }
}

impl core::fmt::Debug for BvhDouble {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BvhDouble")
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
unsafe impl Send for BvhDouble {}
unsafe impl Sync for BvhDouble {}

impl sealed::Sealed for BvhDouble {}
impl BlasEx for BvhDouble {
    fn blas_base_ptr(&self) -> *const ffi::tbvh_base_ex {
        self.base_ptr()
    }

    fn blas_is_ready(&self) -> bool {
        self.is_built()
    }
}

impl From<Vec<DVec3>> for GeometryEx {
    fn from(vertices: Vec<DVec3>) -> Self {
        GeometryEx::new(vertices)
    }
}

impl From<(Vec<DVec3>, Vec<u64>)> for GeometryEx {
    fn from((vertices, indices): (Vec<DVec3>, Vec<u64>)) -> Self {
        GeometryEx::indexed(vertices, indices)
    }
}

// ---------------------------------------------------------------------------------------------
// Bvh4Double
// ---------------------------------------------------------------------------------------------

/// What a [`Bvh4Double`] was built over.
enum Source4 {
    /// Nothing built yet.
    Empty,
    /// Triangles owned by the structure, built here; the intermediate binary tree references them.
    Triangles(GeometryEx),
    /// Loaded from a file. The layout embeds its triangle data, so tinybvh references nothing;
    /// the geometry handed to [`Bvh4Double::load`] is kept for the caller's convenience.
    Loaded(Option<GeometryEx>),
}

/// The double precision 4-wide CPU traversal layout (tinybvh's `BVH4_Double`): the counterpart of
/// [`Bvh4Cpu`](crate::Bvh4Cpu), with `f64` bounds in 256-byte nodes and triangle data stored
/// inline in 64-byte blocks.
///
/// A `Bvh4Double` is built from a [`BvhDouble`] internally and keeps that binary tree (unless
/// `discard_intermediates` is set) so that it can be [refitted](Self::refit) and
/// [optimized](Self::optimize). It can be instanced in a [`BvhDouble`] TLAS. Nodes are not exposed
/// individually; [`data`](Self::data) gives the raw blocks, and the node counters inherited from
/// the base accessors ([`used_nodes`](Self::used_nodes), [`allocated_nodes`](Self::allocated_nodes))
/// are not maintained by tinybvh for this layout and read 0.
///
/// # Example
///
/// ```
/// use glam::DVec3;
/// use tinybvh::{Bvh4Double, RayEx};
///
/// let tri = |z: f64| [DVec3::new(-1.0, -1.0, z), DVec3::new(1.0, -1.0, z), DVec3::new(0.0, 1.0, z)];
/// let mut vertices = Vec::new();
/// vertices.extend(tri(5.0));
/// vertices.extend(tri(3.0));
///
/// let mut bvh = Bvh4Double::new();
/// bvh.build(vertices);
///
/// let mut ray = RayEx::new(DVec3::ZERO, DVec3::Z);
/// bvh.intersect(&mut ray);
/// assert_eq!(ray.hit.prim, 1);
/// assert!((ray.hit.t - 3.0).abs() < 1e-12);
/// assert!(!bvh.is_occluded(&ray)); // nothing strictly before the recorded hit
/// assert!(bvh.is_occluded(&RayEx::new(DVec3::ZERO, DVec3::Z)));
/// ```
pub struct Bvh4Double {
    raw: NonNull<ffi::tbvh_bvh4_double>,
    source: Source4,
    /// Set when the build ran with `discard_intermediates`: no binary tree to refit or optimize.
    intermediates_discarded: bool,
}

impl_base_accessors_ex!(Bvh4Double, tbvh_bvh4_double_base, tbvh_bvh4_double_base_const);

impl Bvh4Double {
    /// An empty structure with default [`BuildSettings`](crate::BuildSettings).
    pub fn new() -> Self {
        crate::install_fatal_trampoline();
        let raw = unsafe { ffi::tbvh_bvh4_double_new() };
        Bvh4Double {
            raw: NonNull::new(raw).expect("tinybvh: allocation failed"),
            source: Source4::Empty,
            intermediates_discarded: false,
        }
    }

    // --- builders ------------------------------------------------------------------------------

    /// Builds over triangles with the builder selected by the [settings](Self::settings). Any
    /// previous content is replaced.
    ///
    /// Accepts a `Vec<DVec3>` with three vertices per triangle, a `(Vec<DVec3>, Vec<u64>)` pair
    /// for indexed triangles, or a [`GeometryEx`]. Panics if the geometry is empty, not a multiple
    /// of three, or has an index out of range.
    pub fn build(&mut self, geometry: impl Into<GeometryEx>) {
        let geometry = geometry.into();
        geometry.validate();
        self.source = Source4::Empty;
        self.intermediates_discarded = self.settings().discard_intermediates;
        unsafe {
            match geometry.index_args() {
                None => ffi::tbvh_bvh4_double_build(
                    self.raw.as_ptr(),
                    geometry.vert_ptr(),
                    geometry.vert_count(),
                    geometry::DVEC3_STRIDE,
                ),
                Some((idx, prims)) => ffi::tbvh_bvh4_double_build_indexed(
                    self.raw.as_ptr(),
                    geometry.vert_ptr(),
                    geometry.vert_count(),
                    geometry::DVEC3_STRIDE,
                    idx,
                    prims,
                ),
            }
        }
        self.source = Source4::Triangles(geometry);
    }

    /// Builds with spatial splits for a higher quality tree (tinybvh's `BuildHQ`). The result
    /// cannot be [refitted](Self::refit).
    pub fn build_hq(&mut self, geometry: impl Into<GeometryEx>) {
        let geometry = geometry.into();
        geometry.validate();
        self.source = Source4::Empty;
        self.intermediates_discarded = self.settings().discard_intermediates;
        unsafe {
            match geometry.index_args() {
                None => ffi::tbvh_bvh4_double_build_hq(
                    self.raw.as_ptr(),
                    geometry.vert_ptr(),
                    geometry.vert_count(),
                    geometry::DVEC3_STRIDE,
                ),
                Some((idx, prims)) => ffi::tbvh_bvh4_double_build_hq_indexed(
                    self.raw.as_ptr(),
                    geometry.vert_ptr(),
                    geometry.vert_count(),
                    geometry::DVEC3_STRIDE,
                    idx,
                    prims,
                ),
            }
        }
        self.source = Source4::Triangles(geometry);
    }

    // --- queries ---------------------------------------------------------------------------------

    /// Whether a tree is available for queries.
    pub fn is_built(&self) -> bool {
        self.used_blocks() > 0 && !matches!(self.source, Source4::Empty)
    }

    #[inline]
    fn assert_built(&self) {
        assert!(self.is_built(), "tinybvh: the Bvh4Double has not been built");
    }

    /// Whether the intermediate binary tree is available, which refitting, optimization and the
    /// SAH cost need: false after a build with `discard_intermediates` or a load from file.
    fn has_intermediates(&self) -> bool {
        matches!(self.source, Source4::Triangles(_)) && !self.intermediates_discarded
    }

    /// Finds the closest hit along `ray` and records it in `ray.hit`. Returns the number of
    /// traversal steps, which the double precision 4-wide kernel only counts in debug builds of
    /// the C++ library; it is 0 otherwise.
    ///
    /// Panics if the structure has not been built or the ray contains NaN.
    pub fn intersect(&self, ray: &mut RayEx) -> u32 {
        self.assert_built();
        ray.assert_finite();
        let steps = unsafe { ffi::tbvh_bvh4_double_intersect(self.raw.as_ptr(), ray.as_ffi_mut()) };
        steps.max(0) as u32
    }

    /// Whether anything blocks `ray` strictly before `ray.hit.t`.
    pub fn is_occluded(&self, ray: &RayEx) -> bool {
        self.assert_built();
        ray.assert_finite();
        unsafe { ffi::tbvh_bvh4_double_is_occluded(self.raw.as_ptr(), ray.as_ffi()) }
    }

    // --- maintenance -----------------------------------------------------------------------------

    /// Updates the bounds after the vertices changed through [`vertices_mut`](Self::vertices_mut):
    /// refits the intermediate binary tree and rebuilds the wide layout from it.
    ///
    /// Fails with [`NotRebuildable`](Error::NotRebuildable) when the intermediate tree is gone
    /// (loaded from a file or built with `discard_intermediates`), with
    /// [`NotRefittable`](Error::NotRefittable) after a build with spatial splits or presplitting,
    /// and with [`HasHoles`](Error::HasHoles) when the intermediate tree has unused nodes.
    pub fn refit(&mut self) -> Result<()> {
        if !self.is_built() {
            return Err(Error::NotBuilt);
        }
        if !self.is_rebuildable() || !self.has_intermediates() {
            return Err(Error::NotRebuildable);
        }
        if !self.is_refittable() {
            return Err(Error::NotRefittable);
        }
        if self.may_have_holes() {
            return Err(Error::HasHoles);
        }
        unsafe { ffi::tbvh_bvh4_double_refit(self.raw.as_ptr()) };
        Ok(())
    }

    /// Improves the intermediate tree with tree rotations and rebuilds the wide layout from it.
    /// `iterations` of 25 is the tinybvh default.
    ///
    /// Fails with [`NotRebuildable`](Error::NotRebuildable) when the intermediate tree is gone
    /// (loaded from a file or built with `discard_intermediates`) and with
    /// [`HasHoles`](Error::HasHoles) when it has unused nodes.
    pub fn optimize(&mut self, iterations: u32) -> Result<()> {
        self.optimize_with(iterations, false)
    }

    /// [`optimize`](Self::optimize) with tinybvh's `extreme` knob, which considers every node as
    /// a candidate for reinsertion.
    pub fn optimize_with(&mut self, iterations: u32, extreme: bool) -> Result<()> {
        if !self.is_built() {
            return Err(Error::NotBuilt);
        }
        if !self.is_rebuildable() || !self.has_intermediates() {
            return Err(Error::NotRebuildable);
        }
        if self.may_have_holes() {
            return Err(Error::HasHoles);
        }
        unsafe { ffi::tbvh_bvh4_double_optimize(self.raw.as_ptr(), iterations, extreme) };
        Ok(())
    }

    /// SAH cost of the whole intermediate 4-wide tree: lower is better. Panics when the
    /// intermediate tree is gone (loaded from a file or built with `discard_intermediates`).
    ///
    /// Only the whole tree can be scored: tinybvh does not report the node count of the
    /// intermediate tree for this layout ([`used_nodes`](Self::used_nodes) stays 0), so a
    /// subtree index could not be validated.
    pub fn sah_cost(&self) -> f64 {
        self.assert_built();
        assert!(
            self.has_intermediates(),
            "tinybvh: the Bvh4Double has no intermediate tree (loaded from a file or built with discard_intermediates)"
        );
        unsafe { ffi::tbvh_bvh4_double_sah_cost(self.raw.as_ptr(), 0) }
    }

    // --- serialization ---------------------------------------------------------------------------

    /// Writes the layout to a file. The triangle data is part of the layout, so the file is
    /// always self-contained.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        if !self.is_built() {
            return Err(Error::NotBuilt);
        }
        let cpath = geometry::path_to_cstring(path.as_ref())?;
        unsafe { ffi::tbvh_bvh4_double_save(self.raw.as_ptr(), cpath.as_ptr()) };
        Ok(())
    }

    /// Loads a layout saved with [`save`](Self::save). `geometry` must have the triangle count
    /// the layout was built over; it is kept for [`geometry`](Self::geometry) but not referenced
    /// by the layout, which carries its own copy of the triangles. Any previous content is
    /// replaced; on failure the structure is left empty. A loaded layout cannot be refitted or
    /// optimized.
    pub fn load(&mut self, path: impl AsRef<Path>, geometry: impl Into<GeometryEx>) -> Result<()> {
        let geometry = geometry.into();
        geometry.validate();
        let cpath = geometry::path_to_cstring(path.as_ref())?;
        self.source = Source4::Empty;
        let ok = unsafe { ffi::tbvh_bvh4_double_load(self.raw.as_ptr(), cpath.as_ptr(), geometry.triangle_count() as u64) };
        if ok {
            self.source = Source4::Loaded(Some(geometry));
            Ok(())
        } else {
            // tinybvh leaves the old layout in place when the header or triangle count does not
            // match; release it so the structure is empty as documented.
            let settings = self.settings();
            self.clear();
            self.set_settings(&settings);
            Err(Error::LoadFailed(path.as_ref().display().to_string()))
        }
    }

    /// Loads a layout saved with [`save`](Self::save), or converts a [`BvhDouble`] file that was
    /// saved with its geometry. The result cannot be refitted or optimized.
    pub fn load_self_contained(path: impl AsRef<Path>) -> Result<Bvh4Double> {
        let cpath = geometry::path_to_cstring(path.as_ref())?;
        let mut bvh = Bvh4Double::new();
        let ok = unsafe { ffi::tbvh_bvh4_double_load_self_contained(bvh.raw.as_ptr(), cpath.as_ptr()) };
        if !ok {
            return Err(Error::LoadFailed(path.as_ref().display().to_string()));
        }
        bvh.source = Source4::Loaded(None);
        Ok(bvh)
    }

    // --- data access -----------------------------------------------------------------------------

    /// Number of 64-byte blocks in use by the layout; 0 before a build.
    pub fn used_blocks(&self) -> u32 {
        let mut used = 0u32;
        unsafe { ffi::tbvh_bvh4_double_data(self.raw.as_ptr(), &mut used) };
        used
    }

    /// The raw layout: [`used_blocks`](Self::used_blocks) blocks of 64 bytes, interleaving
    /// 256-byte interior nodes and leaf blocks with the triangle data. Empty before a build.
    pub fn data(&self) -> &[u8] {
        if !self.is_built() {
            return &[];
        }
        let mut used = 0u32;
        let ptr = unsafe { ffi::tbvh_bvh4_double_data(self.raw.as_ptr(), &mut used) };
        if ptr.is_null() || used == 0 {
            return &[];
        }
        unsafe { core::slice::from_raw_parts(ptr, used as usize * 64) }
    }

    /// The triangle geometry the layout was built over or loaded with, if any.
    pub fn geometry(&self) -> Option<&GeometryEx> {
        match &self.source {
            Source4::Triangles(g) | Source4::Loaded(Some(g)) => Some(g),
            _ => None,
        }
    }

    /// The owned vertices, if any.
    pub fn vertices(&self) -> Option<&[DVec3]> {
        self.geometry().map(|g| g.vertices.as_slice())
    }

    /// Mutable access to the vertices, for animation followed by [`refit`](Self::refit). The
    /// number of vertices cannot change. Only available for geometry the layout was built over
    /// here; a loaded layout does not reference its geometry.
    pub fn vertices_mut(&mut self) -> Option<&mut [DVec3]> {
        match &mut self.source {
            Source4::Triangles(g) => Some(&mut g.vertices),
            _ => None,
        }
    }

    /// The vertex indices for indexed geometry.
    pub fn indices(&self) -> Option<&[u64]> {
        self.geometry().and_then(|g| g.indices.as_deref())
    }

    /// Releases everything the structure holds, including the geometry, and returns it to the
    /// empty state.
    pub fn clear(&mut self) {
        self.source = Source4::Empty;
        self.intermediates_discarded = false;
        unsafe {
            ffi::tbvh_bvh4_double_free(self.raw.as_ptr());
            self.raw = NonNull::new(ffi::tbvh_bvh4_double_new()).expect("tinybvh: allocation failed");
        }
    }

    /// Takes the owned triangle geometry out of the structure, leaving it empty.
    pub fn into_geometry(mut self) -> Option<GeometryEx> {
        let source = core::mem::replace(&mut self.source, Source4::Empty);
        unsafe {
            ffi::tbvh_bvh4_double_free(self.raw.as_ptr());
            self.raw = NonNull::new(ffi::tbvh_bvh4_double_new()).expect("tinybvh: allocation failed");
        }
        match source {
            Source4::Triangles(g) | Source4::Loaded(Some(g)) => Some(g),
            _ => None,
        }
    }

    #[doc(hidden)]
    pub fn raw_ptr(&self) -> *const ffi::tbvh_bvh4_double {
        self.raw.as_ptr()
    }

    #[doc(hidden)]
    pub fn raw_ptr_mut(&mut self) -> *mut ffi::tbvh_bvh4_double {
        self.raw.as_ptr()
    }
}

impl Default for Bvh4Double {
    fn default() -> Self {
        Bvh4Double::new()
    }
}

impl Drop for Bvh4Double {
    fn drop(&mut self) {
        // The C++ object goes first: its intermediate tree references the geometry dropped afterwards.
        unsafe { ffi::tbvh_bvh4_double_free(self.raw.as_ptr()) };
    }
}

impl core::fmt::Debug for Bvh4Double {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Bvh4Double")
            .field("built", &self.is_built())
            .field("prim_count", &self.prim_count())
            .field("used_blocks", &self.used_blocks())
            .field("has_intermediates", &self.has_intermediates())
            .finish()
    }
}

// Queries take &self and only read the layout; builds take &mut self. The geometry is owned here.
unsafe impl Send for Bvh4Double {}
unsafe impl Sync for Bvh4Double {}

impl sealed::Sealed for Bvh4Double {}
impl BlasEx for Bvh4Double {
    fn blas_base_ptr(&self) -> *const ffi::tbvh_base_ex {
        self.base_ptr()
    }

    fn blas_is_ready(&self) -> bool {
        self.is_built()
    }
}
