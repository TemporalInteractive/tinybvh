//! GPU node layouts: [`BvhGpu`] (Aila & Laine), [`Bvh4Gpu`] (quantized 4-wide), [`Bvh8Cwbvh`]
//! (compressed wide BVH) and [`Bvh4AmdHw`] (AMD hardware format).
//!
//! These layouts exist to be uploaded to a GPU: their node data is exposed as plain slices.
//! tinybvh also provides reference CPU traversal for the first three, which this crate exposes as
//! `intersect` / `is_occluded` so that the data can be validated without a GPU; it is not fast.
//! `Bvh4AmdHw` has no CPU traversal at all.

use core::ptr::NonNull;
use std::path::Path;
use std::sync::Arc;

use glam::{Vec3, Vec4};

use crate::bvh::Bvh;
use crate::error::{Error, Result};
use crate::ffi;
use crate::geometry::{self, Geometry};
use crate::instance::{Blas, BlasInstance};
use crate::mbvh::{Mbvh4, Mbvh8};
use crate::ray::Ray;

// ---------------------------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------------------------

/// Reference Möller–Trumbore test over `count` triangles, mirroring tinybvh's own test; used when
/// tinybvh's conversion left a layout without CPU-traversable triangle data (see
/// [`BvhGpu::intersect`]). Returns whether anything was hit; with `any_hit` the first hit ends
/// the search and `ray` is not modified.
fn intersect_triangles(
    verts: &[Vec4],
    indices: Option<&[u32]>,
    count: usize,
    ray: &mut Ray,
    any_hit: bool,
) -> bool {
    let origin = ray.origin;
    let dir = ray.direction();
    let mut hit = false;
    for prim in 0..count {
        let [i0, i1, i2] = match indices {
            Some(idx) => [
                idx[prim * 3] as usize,
                idx[prim * 3 + 1] as usize,
                idx[prim * 3 + 2] as usize,
            ],
            None => [prim * 3, prim * 3 + 1, prim * 3 + 2],
        };
        let v0 = verts[i0].truncate();
        let e1 = verts[i1].truncate() - v0;
        let e2 = verts[i2].truncate() - v0;
        let h = dir.cross(e2);
        let a = e1.dot(h);
        if a.abs() < 1e-10 {
            continue;
        }
        let f = 1.0 / a;
        let s = origin - v0;
        let u = f * s.dot(h);
        if !(0.0..=1.0).contains(&u) {
            continue;
        }
        let q = s.cross(e1);
        let v = f * dir.dot(q);
        if v < 0.0 || u + v > 1.0 {
            continue;
        }
        let t = f * e2.dot(q);
        if t <= 0.0 || t >= ray.hit.t {
            continue;
        }
        if any_hit {
            return true;
        }
        ray.hit.t = t;
        ray.hit.u = u;
        ray.hit.v = v;
        ray.hit.prim = prim as u32;
        hit = true;
    }
    hit
}

/// Slice of 16-byte blocks returned by the `*_data` / `*_nodes` / `*_tris` C functions.
unsafe fn block_slice<'a>(ptr: *const ffi::tbvh_vec4, used_blocks: u32) -> &'a [Vec4] {
    if ptr.is_null() || used_blocks == 0 {
        return &[];
    }
    core::slice::from_raw_parts(ptr as *const Vec4, used_blocks as usize)
}

// ---------------------------------------------------------------------------------------------
// BvhGpu
// ---------------------------------------------------------------------------------------------

/// A node of a [`BvhGpu`] (tinybvh's `BVH_GPU::BVHNode`): 64 bytes, the layout of Aila & Laine.
///
/// A node stores the bounds of its two children rather than its own. Each child reference is
/// either the index of another node, or a leaf encoded inline: the high bit is set, bits 24..30
/// hold the primitive count (up to 127) and bits 0..23 the first slot in the
/// [primitive index array](BvhGpu::prim_indices). Use [`child_is_leaf`](Self::child_is_leaf),
/// [`child_node`](Self::child_node) and [`child_prim_range`](Self::child_prim_range) to decode a
/// reference. Nodes are always interior; `tri_count` and `first_tri` are unused and zero.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BvhGpuNode {
    /// Minimum corner of the left child's bounds.
    pub lmin: Vec3,
    /// Reference to the left child; see the type documentation for the encoding.
    pub left: u32,
    /// Maximum corner of the left child's bounds.
    pub lmax: Vec3,
    /// Reference to the right child.
    pub right: u32,
    /// Minimum corner of the right child's bounds.
    pub rmin: Vec3,
    /// Unused in this layout (leaves are encoded in the child references); always 0.
    pub tri_count: u32,
    /// Maximum corner of the right child's bounds.
    pub rmax: Vec3,
    /// Unused in this layout; always 0.
    pub first_tri: u32,
}

impl BvhGpuNode {
    /// The bit that marks a child reference as an inline leaf.
    pub const LEAF_FLAG: u32 = 0x8000_0000;

    /// Whether the node is a leaf, as tinybvh defines it (`tri_count > 0`). Leaves of this layout
    /// live in the child references, so this is `false` for every node tinybvh produces.
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.tri_count > 0
    }

    /// Bounds of the left child as `(min, max)`.
    #[inline]
    pub fn left_bounds(&self) -> (Vec3, Vec3) {
        (self.lmin, self.lmax)
    }

    /// Bounds of the right child as `(min, max)`.
    #[inline]
    pub fn right_bounds(&self) -> (Vec3, Vec3) {
        (self.rmin, self.rmax)
    }

    /// The two child references as `(left, right)`.
    #[inline]
    pub fn children(&self) -> (u32, u32) {
        (self.left, self.right)
    }

    /// Whether a child reference encodes a leaf.
    #[inline]
    pub fn child_is_leaf(child: u32) -> bool {
        child & Self::LEAF_FLAG != 0
    }

    /// The node index a child reference points to; `None` for a leaf.
    #[inline]
    pub fn child_node(child: u32) -> Option<u32> {
        if Self::child_is_leaf(child) {
            None
        } else {
            Some(child)
        }
    }

    /// The range into [`BvhGpu::prim_indices`] covered by a leaf reference; empty for a node
    /// reference (and for the empty leaf tinybvh uses to pad a single-leaf tree).
    #[inline]
    pub fn child_prim_range(child: u32) -> core::ops::Range<usize> {
        if Self::child_is_leaf(child) {
            let first = (child & 0x00ff_ffff) as usize;
            let count = ((child >> 24) & 127) as usize;
            first..first + count
        } else {
            0..0
        }
    }
}

const _: () =
    assert!(core::mem::size_of::<BvhGpuNode>() == core::mem::size_of::<ffi::tbvh_bvh_gpu_node>());

/// What a [`BvhGpu`] was built over; keeps the referenced data alive.
enum GpuSource {
    /// Nothing built yet.
    Empty,
    /// Triangles owned by the structure.
    Triangles(Geometry),
    /// A TLAS over instances of other structures.
    Tlas {
        instances: Vec<BlasInstance>,
        blasses: Vec<Arc<dyn Blas>>,
        blas_ptrs: Vec<*mut ffi::tbvh_base>,
    },
    /// Converted from a [`Bvh`], which tinybvh keeps referencing.
    Converted(Bvh),
}

/// The Aila & Laine GPU layout (tinybvh's `BVH_GPU`): 64-byte nodes that store the bounds of
/// both children, with leaves encoded inline in the child references.
///
/// The structure builds a [`Bvh`] internally and converts it, or takes an existing one through
/// [`from_bvh`](Self::from_bvh). Upload [`nodes`](Self::nodes), [`prim_indices`](Self::prim_indices)
/// and your vertex data to trace it on the GPU; [`intersect`](Self::intersect) runs tinybvh's
/// reference CPU traversal of the same data, for validation.
///
/// A `BvhGpu` can also be built as a TLAS ([`build_tlas`](Self::build_tlas)) to obtain GPU node
/// data over instances; tinybvh has no CPU traversal for such a tree, so queries on it panic.
///
/// # Example
///
/// ```
/// use glam::{Vec3, Vec4};
/// use tinybvh::{BvhGpu, BvhGpuNode, Ray};
///
/// let tri = |z: f32| [Vec4::new(-1.0, -1.0, z, 0.0), Vec4::new(1.0, -1.0, z, 0.0), Vec4::new(0.0, 1.0, z, 0.0)];
/// let mut vertices = Vec::new();
/// for i in 0..8 {
///     vertices.extend(tri(2.0 + i as f32));
/// }
///
/// let mut bvh = BvhGpu::new();
/// bvh.build(vertices);
/// assert!(!bvh.nodes().is_empty());
/// let root = bvh.nodes()[0];
/// assert!(BvhGpuNode::child_is_leaf(root.left) || BvhGpuNode::child_node(root.left).is_some());
///
/// let mut ray = Ray::new(Vec3::ZERO, Vec3::Z);
/// bvh.intersect(&mut ray);
/// assert_eq!(ray.hit.prim, 0);
/// assert!((ray.hit.t - 2.0).abs() < 1e-5);
/// ```
pub struct BvhGpu {
    raw: NonNull<ffi::tbvh_bvh_gpu>,
    source: GpuSource,
    /// False after a conversion from an external tree: tinybvh would optimize that tree in place.
    rebuildable: bool,
    /// Set when the source tree was a single leaf, which tinybvh wraps in one interior node
    /// without producing the triangle data its CPU traversal reads; queries then run in Rust.
    leaf_root: bool,
}

impl_base_accessors!(BvhGpu, tbvh_bvh_gpu_base, tbvh_bvh_gpu_base_const);

impl BvhGpu {
    /// An empty structure with default [`BuildSettings`](crate::BuildSettings).
    pub fn new() -> Self {
        crate::install_fatal_trampoline();
        let raw = unsafe { ffi::tbvh_bvh_gpu_new() };
        BvhGpu {
            raw: NonNull::new(raw).expect("tinybvh: allocation failed"),
            source: GpuSource::Empty,
            rebuildable: true,
            leaf_root: false,
        }
    }

    // --- builders ------------------------------------------------------------------------------

    /// Builds over triangles with the builder selected by the [settings](Self::settings). Any
    /// previous content is replaced.
    ///
    /// Accepts a `Vec<Vec4>` with three vertices per triangle, a `(Vec<Vec4>, Vec<u32>)` pair for
    /// indexed triangles, or a [`Geometry`]. Panics if the geometry is empty, not a multiple of
    /// three, or has an index out of range.
    pub fn build(&mut self, geometry: impl Into<Geometry>) {
        self.build_with(geometry.into(), false);
    }

    /// Builds with spatial splits for a higher quality tree (tinybvh's `BuildHQ`).
    pub fn build_hq(&mut self, geometry: impl Into<Geometry>) {
        self.build_with(geometry.into(), true);
    }

    fn build_with(&mut self, geometry: Geometry, hq: bool) {
        geometry.validate();
        // The old source is dropped only after tinybvh stopped referencing it, i.e. after the build.
        let old = core::mem::replace(&mut self.source, GpuSource::Empty);
        let raw = self.raw.as_ptr();
        unsafe {
            match (geometry.index_args(), hq) {
                (None, false) => ffi::tbvh_bvh_gpu_build(
                    raw,
                    geometry.vert_ptr(),
                    geometry.vert_count(),
                    geometry::VEC4_STRIDE,
                ),
                (None, true) => ffi::tbvh_bvh_gpu_build_hq(
                    raw,
                    geometry.vert_ptr(),
                    geometry.vert_count(),
                    geometry::VEC4_STRIDE,
                ),
                (Some((idx, prims)), false) => ffi::tbvh_bvh_gpu_build_indexed(
                    raw,
                    geometry.vert_ptr(),
                    geometry.vert_count(),
                    geometry::VEC4_STRIDE,
                    idx,
                    prims,
                ),
                (Some((idx, prims)), true) => ffi::tbvh_bvh_gpu_build_hq_indexed(
                    raw,
                    geometry.vert_ptr(),
                    geometry.vert_count(),
                    geometry::VEC4_STRIDE,
                    idx,
                    prims,
                ),
            }
        }
        self.source = GpuSource::Triangles(geometry);
        self.rebuildable = true;
        self.leaf_root = self.detect_leaf_root();
        drop(old);
    }

    /// Builds a top-level acceleration structure over `instances`, each referring by
    /// [`blas_idx`](BlasInstance::blas_idx) to one of `blasses`, with the same validation as
    /// [`Bvh::build_tlas`]. The result is GPU node data over instances; tinybvh has no CPU
    /// traversal for it, so [`intersect`](Self::intersect) and [`is_occluded`](Self::is_occluded)
    /// panic on a TLAS.
    ///
    /// Panics if there are no instances, an instance refers to a BLAS index out of range, or a
    /// BLAS has not been built.
    pub fn build_tlas(&mut self, instances: Vec<BlasInstance>, blasses: Vec<Arc<dyn Blas>>) {
        assert!(
            !instances.is_empty(),
            "tinybvh: a TLAS needs at least one instance"
        );
        assert!(
            !blasses.is_empty(),
            "tinybvh: a TLAS needs at least one BLAS"
        );
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
        let blas_ptrs: Vec<*mut ffi::tbvh_base> = blasses
            .iter()
            .map(|b| b.blas_base_ptr() as *mut _)
            .collect();
        let old = core::mem::replace(&mut self.source, GpuSource::Empty);
        self.source = GpuSource::Tlas {
            instances,
            blasses,
            blas_ptrs,
        };
        self.run_tlas_build();
        self.rebuildable = true;
        self.leaf_root = false;
        drop(old);
    }

    /// Rebuilds the TLAS over its current instances, after transforms were changed through
    /// [`instances_mut`](Self::instances_mut) or a BLAS was refitted.
    pub fn rebuild_tlas(&mut self) -> Result<()> {
        match self.source {
            GpuSource::Tlas { .. } => {
                self.run_tlas_build();
                Ok(())
            }
            _ => Err(Error::NotTlas),
        }
    }

    fn run_tlas_build(&mut self) {
        let raw = self.raw.as_ptr();
        if let GpuSource::Tlas {
            instances,
            blas_ptrs,
            ..
        } = &mut self.source
        {
            unsafe {
                ffi::tbvh_bvh_gpu_build_tlas(
                    raw,
                    instances.as_mut_ptr() as *mut ffi::tbvh_blas_instance,
                    instances.len() as u32,
                    blas_ptrs.as_mut_ptr(),
                    blas_ptrs.len() as u32,
                );
            }
        }
    }

    /// Converts a built [`Bvh`] over triangles, taking ownership of it: tinybvh references its
    /// nodes, primitive indices and vertices rather than copying them. The source stays
    /// accessible through [`bvh`](Self::bvh). The result cannot be [optimized](Self::optimize),
    /// since that would rewrite the source tree in place; optimize the `Bvh` beforehand instead.
    ///
    /// Panics if `source` has not been built, or is a TLAS or a tree over custom primitives (the
    /// conversion only produces traversable data for triangles).
    pub fn from_bvh(source: Bvh) -> Self {
        assert!(source.is_built(), "tinybvh: the Bvh has not been built");
        assert!(
            source.vertices().is_some() && !source.is_tlas(),
            "tinybvh: BvhGpu::from_bvh needs a Bvh built over triangles"
        );
        let mut gpu = BvhGpu::new();
        unsafe { ffi::tbvh_bvh_gpu_convert_from(gpu.raw.as_ptr(), source.raw_ptr(), true) };
        gpu.source = GpuSource::Converted(source);
        gpu.rebuildable = false;
        gpu.leaf_root = gpu.detect_leaf_root();
        gpu
    }

    /// Loads a [`Bvh`] saved by [`Bvh::save`] with its geometry and converts it, as
    /// [`from_bvh`](Self::from_bvh) does; the loaded `Bvh` is kept, so the primitive indices and
    /// vertices the nodes refer to remain available. Fails if the file cannot be read or carries
    /// no geometry.
    pub fn load_self_contained(path: impl AsRef<Path>) -> Result<BvhGpu> {
        let bvh = Bvh::load_self_contained(path)?;
        Ok(BvhGpu::from_bvh(bvh))
    }

    /// tinybvh wraps a single-leaf source tree in one interior node whose right child is an
    /// empty leaf, and skips building the reordered triangle data its CPU traversal reads
    /// (tinybvh 1.9.1, `BVH_GPU::ConvertFrom`). Such a tree is recognizable by that empty leaf.
    fn detect_leaf_root(&self) -> bool {
        self.nodes()
            .first()
            .is_some_and(|root| root.right == BvhGpuNode::LEAF_FLAG)
    }

    // --- queries ---------------------------------------------------------------------------------

    /// Whether a tree is available.
    pub fn is_built(&self) -> bool {
        self.used_nodes() > 0 && !matches!(self.source, GpuSource::Empty)
    }

    /// Whether this structure is a top-level acceleration structure.
    pub fn is_tlas(&self) -> bool {
        matches!(self.source, GpuSource::Tlas { .. })
    }

    #[inline]
    fn assert_traversable(&self) {
        assert!(self.is_built(), "tinybvh: the BvhGpu has not been built");
        assert!(!self.is_tlas(), "tinybvh: BVH_GPU has no CPU traversal for a TLAS; trace it on the GPU or use a Bvh TLAS");
    }

    /// The triangles tinybvh's conversion left out of its traversal data; see `leaf_root`.
    fn fallback_triangles(&self) -> Option<(&[Vec4], Option<&[u32]>)> {
        match &self.source {
            GpuSource::Triangles(g) => Some((&g.vertices, g.indices.as_deref())),
            GpuSource::Converted(bvh) => bvh.vertices().map(|v| (v, bvh.indices())),
            _ => None,
        }
    }

    /// Finds the closest hit along `ray` with tinybvh's reference traversal of the GPU data and
    /// records it in `ray.hit`. Returns the traversal cost.
    ///
    /// A tree whose source was a single leaf (one or two triangles, typically) is wrapped by
    /// tinybvh without the triangle data its traversal needs; such a tree is intersected in Rust
    /// instead, with the same result. Panics if the structure has not been built or is a TLAS.
    pub fn intersect(&self, ray: &mut Ray) -> u32 {
        self.assert_traversable();
        ray.assert_finite();
        if self.leaf_root {
            let (verts, indices) = self
                .fallback_triangles()
                .expect("tinybvh: no triangle data for a single-leaf tree");
            let count = self.prim_count() as usize;
            intersect_triangles(verts, indices, count, ray, false);
            return count as u32;
        }
        let steps = unsafe { ffi::tbvh_bvh_gpu_intersect(self.raw.as_ptr(), ray.as_ffi_mut()) };
        steps.max(0) as u32
    }

    /// Whether anything blocks `ray` before `ray.hit.t`. Panics like [`intersect`](Self::intersect).
    pub fn is_occluded(&self, ray: &Ray) -> bool {
        self.assert_traversable();
        ray.assert_finite();
        if self.leaf_root {
            let (verts, indices) = self
                .fallback_triangles()
                .expect("tinybvh: no triangle data for a single-leaf tree");
            let mut probe = *ray;
            return intersect_triangles(
                verts,
                indices,
                self.prim_count() as usize,
                &mut probe,
                true,
            );
        }
        unsafe { ffi::tbvh_bvh_gpu_is_occluded(self.raw.as_ptr(), ray.as_ffi()) }
    }

    // --- maintenance -----------------------------------------------------------------------------

    /// Improves the internal tree with insertion-based tree rotations and converts it again.
    /// `iterations` of 25 is the tinybvh default. Fails when nothing is built, or after
    /// [`from_bvh`](Self::from_bvh) / [`load_self_contained`](Self::load_self_contained).
    pub fn optimize(&mut self, iterations: u32) -> Result<()> {
        self.optimize_with(iterations, false)
    }

    /// [`optimize`](Self::optimize) with tinybvh's `extreme` knob, which considers every node as
    /// a candidate for reinsertion.
    pub fn optimize_with(&mut self, iterations: u32, extreme: bool) -> Result<()> {
        if !self.is_built() {
            return Err(Error::NotBuilt);
        }
        if !self.rebuildable || !self.is_rebuildable() {
            return Err(Error::NotRebuildable);
        }
        unsafe { ffi::tbvh_bvh_gpu_optimize(self.raw.as_ptr(), iterations, extreme) };
        self.leaf_root = self.detect_leaf_root();
        Ok(())
    }

    // --- metrics ---------------------------------------------------------------------------------

    /// SAH cost of the subtree rooted at `node` (0 for the whole tree): lower is better. The cost
    /// is evaluated on the binary tree this layout was converted from, so `node` indexes that
    /// tree: the source `Bvh` after [`from_bvh`](Self::from_bvh), or the internal tree, of which
    /// tinybvh exposes no node count; for the latter only indices below
    /// [`used_nodes`](Self::used_nodes) are accepted.
    ///
    /// Panics if the structure has not been built or `node` is out of range.
    pub fn sah_cost(&self, node: u32) -> f32 {
        assert!(self.is_built(), "tinybvh: the BvhGpu has not been built");
        assert!(
            node < self.internal_node_count(),
            "tinybvh: node index out of range"
        );
        // The C function takes a mutable handle because tinybvh declares BVH_GPU::SAHCost without
        // const; it only reads the tree, so handing it the pointer from `&self` is sound.
        unsafe { ffi::tbvh_bvh_gpu_sah_cost(self.raw.as_ptr(), node) }
    }

    /// Node count of the binary tree the SAH is evaluated on: the source `Bvh` after a
    /// conversion, or the internal tree, which has at least as many nodes as the GPU layout.
    fn internal_node_count(&self) -> u32 {
        match &self.source {
            GpuSource::Converted(bvh) => bvh.used_nodes(),
            _ => self.used_nodes(),
        }
    }

    // --- data access -----------------------------------------------------------------------------

    /// The nodes in use. Node 0 is the root.
    pub fn nodes(&self) -> &[BvhGpuNode] {
        let mut count = 0u32;
        let ptr = unsafe { ffi::tbvh_bvh_gpu_nodes(self.raw.as_ptr(), &mut count) };
        if ptr.is_null() || count == 0 {
            return &[];
        }
        unsafe { core::slice::from_raw_parts(ptr as *const BvhGpuNode, count as usize) }
    }

    /// The primitive index array that leaf references point into; for a TLAS the entries are
    /// instance indices. Empty when nothing is built.
    pub fn prim_indices(&self) -> &[u32] {
        let mut count = 0u32;
        let ptr = unsafe { ffi::tbvh_bvh_gpu_prim_indices(self.raw.as_ptr(), &mut count) };
        if ptr.is_null() || count == 0 {
            return &[];
        }
        unsafe { core::slice::from_raw_parts(ptr, count as usize) }
    }

    /// The owned triangle geometry, if the tree was built over triangles from a [`Geometry`]
    /// (directly, or through the source `Bvh`).
    pub fn geometry(&self) -> Option<&Geometry> {
        match &self.source {
            GpuSource::Triangles(g) => Some(g),
            GpuSource::Converted(bvh) => bvh.geometry(),
            _ => None,
        }
    }

    /// The vertices the tree references, if any.
    pub fn vertices(&self) -> Option<&[Vec4]> {
        match &self.source {
            GpuSource::Triangles(g) => Some(&g.vertices),
            GpuSource::Converted(bvh) => bvh.vertices(),
            _ => None,
        }
    }

    /// The vertex indices for indexed geometry.
    pub fn indices(&self) -> Option<&[u32]> {
        match &self.source {
            GpuSource::Triangles(g) => g.indices.as_deref(),
            GpuSource::Converted(bvh) => bvh.indices(),
            _ => None,
        }
    }

    /// The source tree after [`from_bvh`](Self::from_bvh) or
    /// [`load_self_contained`](Self::load_self_contained).
    pub fn bvh(&self) -> Option<&Bvh> {
        match &self.source {
            GpuSource::Converted(bvh) => Some(bvh),
            _ => None,
        }
    }

    /// The instances of a TLAS.
    pub fn instances(&self) -> &[BlasInstance] {
        match &self.source {
            GpuSource::Tlas { instances, .. } => instances,
            _ => &[],
        }
    }

    /// Mutable access to the instances of a TLAS, to change transforms or masks. Call
    /// [`rebuild_tlas`](Self::rebuild_tlas) afterwards; until then the node data is stale.
    pub fn instances_mut(&mut self) -> &mut [BlasInstance] {
        match &mut self.source {
            GpuSource::Tlas { instances, .. } => instances,
            _ => &mut [],
        }
    }

    /// The bottom-level structures of a TLAS.
    pub fn blasses(&self) -> &[Arc<dyn Blas>] {
        match &self.source {
            GpuSource::Tlas { blasses, .. } => blasses,
            _ => &[],
        }
    }

    /// Releases everything the structure holds and returns it to the empty state.
    pub fn clear(&mut self) {
        let old = core::mem::replace(&mut self.source, GpuSource::Empty);
        unsafe {
            ffi::tbvh_bvh_gpu_free(self.raw.as_ptr());
            self.raw = NonNull::new(ffi::tbvh_bvh_gpu_new()).expect("tinybvh: allocation failed");
        }
        self.rebuildable = true;
        self.leaf_root = false;
        drop(old);
    }

    /// Takes the owned triangle geometry out of the structure, leaving it empty.
    pub fn into_geometry(mut self) -> Option<Geometry> {
        let source = core::mem::replace(&mut self.source, GpuSource::Empty);
        unsafe {
            ffi::tbvh_bvh_gpu_free(self.raw.as_ptr());
            self.raw = NonNull::new(ffi::tbvh_bvh_gpu_new()).expect("tinybvh: allocation failed");
        }
        match source {
            GpuSource::Triangles(g) => Some(g),
            GpuSource::Converted(bvh) => bvh.into_geometry(),
            _ => None,
        }
    }

    /// Takes the source [`Bvh`] back out after [`from_bvh`](Self::from_bvh), leaving the
    /// structure empty.
    pub fn into_bvh(mut self) -> Option<Bvh> {
        let source = core::mem::replace(&mut self.source, GpuSource::Empty);
        unsafe {
            ffi::tbvh_bvh_gpu_free(self.raw.as_ptr());
            self.raw = NonNull::new(ffi::tbvh_bvh_gpu_new()).expect("tinybvh: allocation failed");
        }
        match source {
            GpuSource::Converted(bvh) => Some(bvh),
            _ => None,
        }
    }
}

impl Default for BvhGpu {
    fn default() -> Self {
        BvhGpu::new()
    }
}

impl Drop for BvhGpu {
    fn drop(&mut self) {
        // The C++ object goes first: it references the geometry, the source Bvh and the instances
        // that are dropped afterwards.
        unsafe { ffi::tbvh_bvh_gpu_free(self.raw.as_ptr()) };
    }
}

impl core::fmt::Debug for BvhGpu {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BvhGpu")
            .field("built", &self.is_built())
            .field("prim_count", &self.prim_count())
            .field("used_nodes", &self.used_nodes())
            .field("is_tlas", &self.is_tlas())
            .field("rebuildable", &self.rebuildable)
            .finish()
    }
}

// Queries take &self and only read the tree; builds take &mut self. The referenced geometry and
// the source Bvh are owned here, and BLASes are shared via Arc.
unsafe impl Send for BvhGpu {}
unsafe impl Sync for BvhGpu {}

// ---------------------------------------------------------------------------------------------
// Block-based layouts: Bvh4Gpu, Bvh4AmdHw, Bvh8Cwbvh
// ---------------------------------------------------------------------------------------------

/// What a block-based GPU layout was built over.
enum BlockSource<M> {
    /// Nothing built yet.
    Empty,
    /// Triangles owned by the structure.
    Triangles(Geometry),
    /// Converted from a wide intermediate layout, which tinybvh keeps referencing.
    Converted(M),
    /// Loaded from a file; the triangle data is embedded in the blocks.
    Loaded,
}

/// Generates the parts shared by the three block-based GPU layouts: construction, the triangle
/// builders, optimization, SAH, self-contained loading, geometry accessors, `clear`, `Debug`,
/// `Drop` and the thread-safety markers.
macro_rules! impl_block_layout {
    (
        $ty:ident, $raw:ident, $mbvh:ident,
        new = $new:ident, free = $free:ident,
        build = $build:ident, build_indexed = $build_indexed:ident,
        build_hq = $build_hq:ident, build_hq_indexed = $build_hq_indexed:ident,
        optimize = $optimize:ident, sah_cost = $sah_cost:ident, load = $load:ident,
        blocks = $blocks:ident
    ) => {
        impl $ty {
            /// An empty structure with default [`BuildSettings`](crate::BuildSettings).
            pub fn new() -> Self {
                crate::install_fatal_trampoline();
                let raw = unsafe { ffi::$new() };
                $ty {
                    raw: NonNull::new(raw).expect("tinybvh: allocation failed"),
                    source: BlockSource::Empty,
                    rebuildable: true,
                }
            }

            /// Builds over triangles with the builder selected by the [settings](Self::settings).
            /// Any previous content is replaced.
            ///
            /// Accepts a `Vec<Vec4>` with three vertices per triangle, a `(Vec<Vec4>, Vec<u32>)`
            /// pair for indexed triangles, or a [`Geometry`]. Panics if the geometry is empty, not
            /// a multiple of three, or has an index out of range.
            pub fn build(&mut self, geometry: impl Into<Geometry>) {
                self.build_with(geometry.into(), false);
            }

            /// Builds with spatial splits for a higher quality tree (tinybvh's `BuildHQ`).
            pub fn build_hq(&mut self, geometry: impl Into<Geometry>) {
                self.build_with(geometry.into(), true);
            }

            fn build_with(&mut self, geometry: Geometry, hq: bool) {
                geometry.validate();
                // The old source is dropped only after tinybvh stopped referencing it.
                let old = core::mem::replace(&mut self.source, BlockSource::Empty);
                let raw = self.raw.as_ptr();
                unsafe {
                    match (geometry.index_args(), hq) {
                        (None, false) => ffi::$build(
                            raw,
                            geometry.vert_ptr(),
                            geometry.vert_count(),
                            geometry::VEC4_STRIDE,
                        ),
                        (None, true) => ffi::$build_hq(
                            raw,
                            geometry.vert_ptr(),
                            geometry.vert_count(),
                            geometry::VEC4_STRIDE,
                        ),
                        (Some((idx, prims)), false) => ffi::$build_indexed(
                            raw,
                            geometry.vert_ptr(),
                            geometry.vert_count(),
                            geometry::VEC4_STRIDE,
                            idx,
                            prims,
                        ),
                        (Some((idx, prims)), true) => ffi::$build_hq_indexed(
                            raw,
                            geometry.vert_ptr(),
                            geometry.vert_count(),
                            geometry::VEC4_STRIDE,
                            idx,
                            prims,
                        ),
                    }
                }
                self.source = BlockSource::Triangles(geometry);
                self.rebuildable = true;
                drop(old);
            }

            /// Loads a [`Bvh`] file saved with its geometry by [`Bvh::save`] and converts it to
            /// this layout; the triangle data ends up embedded in the blocks. The intermediate
            /// trees are discarded afterwards, so the result can neither be
            /// [optimized](Self::optimize) nor report a [SAH cost](Self::sah_cost).
            pub fn load_self_contained(path: impl AsRef<Path>) -> Result<Self> {
                let cpath = geometry::path_to_cstring(path.as_ref())?;
                let mut layout = $ty::new();
                let ok = unsafe { ffi::$load(layout.raw.as_ptr(), cpath.as_ptr()) };
                if !ok {
                    return Err(Error::LoadFailed(path.as_ref().display().to_string()));
                }
                layout.source = BlockSource::Loaded;
                layout.rebuildable = false;
                Ok(layout)
            }

            fn used_blocks(&self) -> u32 {
                let mut used = 0u32;
                unsafe { ffi::$blocks(self.raw.as_ptr(), &mut used) };
                used
            }

            /// Whether a tree is available. tinybvh fills the block count for every path that
            /// produces this layout, including loads, so that is what decides.
            pub fn is_built(&self) -> bool {
                self.used_blocks() > 0 && !matches!(self.source, BlockSource::Empty)
            }

            /// Improves the internal tree with insertion-based tree rotations and converts it
            /// again. `iterations` of 25 is the tinybvh default. Fails when nothing is built, or
            /// when the structure was converted from an external tree or loaded from a file.
            pub fn optimize(&mut self, iterations: u32) -> Result<()> {
                self.optimize_with(iterations, false)
            }

            /// [`optimize`](Self::optimize) with tinybvh's `extreme` knob, which considers every
            /// node as a candidate for reinsertion.
            pub fn optimize_with(&mut self, iterations: u32, extreme: bool) -> Result<()> {
                if !self.is_built() {
                    return Err(Error::NotBuilt);
                }
                if !self.rebuildable || !self.is_rebuildable() {
                    return Err(Error::NotRebuildable);
                }
                unsafe { ffi::$optimize(self.raw.as_ptr(), iterations, extreme) };
                Ok(())
            }

            /// SAH cost of the subtree rooted at `node` of the intermediate wide tree (0 for the
            /// whole tree): lower is better. tinybvh does not expose the node count of the tree it
            /// builds internally, so only the root can be queried for a tree built here; after a
            /// conversion from an `Mbvh`, `node` indexes that structure's nodes.
            ///
            /// Panics if the structure has not been built, `node` is out of range, or the
            /// structure was loaded from a file: a load discards the intermediate tree.
            pub fn sah_cost(&self, node: u32) -> f32 {
                assert!(
                    self.is_built(),
                    concat!("tinybvh: the ", stringify!($ty), " has not been built")
                );
                assert!(
                    !matches!(self.source, BlockSource::Loaded),
                    "tinybvh: a loaded structure keeps no intermediate tree to evaluate the SAH on"
                );
                let limit = match &self.source {
                    BlockSource::Converted(m) => m.used_nodes(),
                    _ => 1,
                };
                assert!(node < limit, "tinybvh: node index out of range");
                unsafe { ffi::$sah_cost(self.raw.as_ptr(), node) }
            }

            /// The owned triangle geometry, if the tree was built from a [`Geometry`].
            pub fn geometry(&self) -> Option<&Geometry> {
                match &self.source {
                    BlockSource::Triangles(g) => Some(g),
                    BlockSource::Converted(m) => m.geometry(),
                    _ => None,
                }
            }

            /// The vertices the tree was built over, if available.
            pub fn vertices(&self) -> Option<&[Vec4]> {
                self.geometry().map(|g| &g.vertices[..])
            }

            /// The vertex indices for indexed geometry.
            pub fn indices(&self) -> Option<&[u32]> {
                self.geometry().and_then(|g| g.indices.as_deref())
            }

            /// Releases everything the structure holds and returns it to the empty state.
            pub fn clear(&mut self) {
                let old = core::mem::replace(&mut self.source, BlockSource::Empty);
                unsafe {
                    ffi::$free(self.raw.as_ptr());
                    self.raw = NonNull::new(ffi::$new()).expect("tinybvh: allocation failed");
                }
                self.rebuildable = true;
                drop(old);
            }

            /// Takes the owned triangle geometry out of the structure, leaving it empty.
            pub fn into_geometry(mut self) -> Option<Geometry> {
                let source = core::mem::replace(&mut self.source, BlockSource::Empty);
                unsafe {
                    ffi::$free(self.raw.as_ptr());
                    self.raw = NonNull::new(ffi::$new()).expect("tinybvh: allocation failed");
                }
                match source {
                    BlockSource::Triangles(g) => Some(g),
                    BlockSource::Converted(m) => m.into_geometry(),
                    _ => None,
                }
            }
        }

        impl Default for $ty {
            fn default() -> Self {
                $ty::new()
            }
        }

        impl Drop for $ty {
            fn drop(&mut self) {
                // The C++ object goes first: it references the geometry dropped afterwards.
                unsafe { ffi::$free(self.raw.as_ptr()) };
            }
        }

        impl core::fmt::Debug for $ty {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.debug_struct(stringify!($ty))
                    .field("built", &self.is_built())
                    .field("prim_count", &self.prim_count())
                    .field("used_blocks", &self.used_blocks())
                    .field("rebuildable", &self.rebuildable)
                    .finish()
            }
        }

        // Queries take &self and only read the data; builds take &mut self. The referenced
        // geometry and intermediate layout are owned here.
        unsafe impl Send for $ty {}
        unsafe impl Sync for $ty {}
    };
}

/// The quantized 4-wide GPU layout (tinybvh's `BVH4_GPU`): 64-byte nodes holding the parent
/// bounds in full precision and the four child bounds as 8-bit offsets, with the triangles
/// stored right after the leaves in the same buffer. Compact, and the format tinybvh's own
/// OpenCL kernels consume.
///
/// [`data`](Self::data) is the whole buffer in 16-byte blocks: upload it as is. Each node starts
/// at a block index returned in the child info words; see tinybvh's `BVH4_GPU::BVHNode` for the
/// bit layout.
///
/// # Example
///
/// ```
/// use glam::{Vec3, Vec4};
/// use tinybvh::{Bvh4Gpu, Ray};
///
/// let tri = |z: f32| [Vec4::new(-1.0, -1.0, z, 0.0), Vec4::new(1.0, -1.0, z, 0.0), Vec4::new(0.0, 1.0, z, 0.0)];
/// let mut vertices = Vec::new();
/// for i in 0..8 {
///     vertices.extend(tri(2.0 + i as f32));
/// }
///
/// let mut bvh = Bvh4Gpu::new();
/// bvh.build(vertices);
/// assert!(bvh.data().len() >= 4);
///
/// let mut ray = Ray::new(Vec3::ZERO, Vec3::Z);
/// bvh.intersect(&mut ray);
/// assert_eq!(ray.hit.prim, 0);
/// assert!((ray.hit.t - 2.0).abs() < 1e-5);
/// ```
pub struct Bvh4Gpu {
    raw: NonNull<ffi::tbvh_bvh4_gpu>,
    source: BlockSource<Mbvh4>,
    rebuildable: bool,
}

impl_base_accessors!(Bvh4Gpu, tbvh_bvh4_gpu_base, tbvh_bvh4_gpu_base_const);
impl_block_layout!(
    Bvh4Gpu,
    tbvh_bvh4_gpu,
    Mbvh4,
    new = tbvh_bvh4_gpu_new,
    free = tbvh_bvh4_gpu_free,
    build = tbvh_bvh4_gpu_build,
    build_indexed = tbvh_bvh4_gpu_build_indexed,
    build_hq = tbvh_bvh4_gpu_build_hq,
    build_hq_indexed = tbvh_bvh4_gpu_build_hq_indexed,
    optimize = tbvh_bvh4_gpu_optimize,
    sah_cost = tbvh_bvh4_gpu_sah_cost,
    load = tbvh_bvh4_gpu_load_self_contained,
    blocks = tbvh_bvh4_gpu_data
);

impl Bvh4Gpu {
    /// Converts a built [`Mbvh4`], taking ownership of it: tinybvh references its nodes and
    /// geometry. The result cannot be [optimized](Self::optimize), since that would rewrite the
    /// source tree in place. Panics if `source` has not been built.
    pub fn from_mbvh(source: Mbvh4) -> Self {
        assert!(source.is_built(), "tinybvh: the Mbvh4 has not been built");
        let mut layout = Bvh4Gpu::new();
        unsafe { ffi::tbvh_bvh4_gpu_convert_from(layout.raw.as_ptr(), source.raw_ptr(), true) };
        layout.source = BlockSource::Converted(source);
        layout.rebuildable = false;
        layout
    }

    #[inline]
    fn assert_built(&self) {
        assert!(self.is_built(), "tinybvh: the Bvh4Gpu has not been built");
    }

    /// Finds the closest hit along `ray` with tinybvh's reference traversal of the GPU data and
    /// records it in `ray.hit`. Returns the traversal cost. Panics if nothing is built.
    pub fn intersect(&self, ray: &mut Ray) -> u32 {
        self.assert_built();
        ray.assert_finite();
        let steps = unsafe { ffi::tbvh_bvh4_gpu_intersect(self.raw.as_ptr(), ray.as_ffi_mut()) };
        steps.max(0) as u32
    }

    /// Whether anything blocks `ray` before `ray.hit.t`. Panics if nothing is built.
    pub fn is_occluded(&self, ray: &Ray) -> bool {
        self.assert_built();
        ray.assert_finite();
        unsafe { ffi::tbvh_bvh4_gpu_is_occluded(self.raw.as_ptr(), ray.as_ffi()) }
    }

    /// The node and triangle data in 16-byte blocks, ready for upload. Empty when nothing is
    /// built.
    pub fn data(&self) -> &[Vec4] {
        let mut used = 0u32;
        let ptr = unsafe { ffi::tbvh_bvh4_gpu_data(self.raw.as_ptr(), &mut used) };
        unsafe { block_slice(ptr, used) }
    }

    /// The source tree after [`from_mbvh`](Self::from_mbvh).
    pub fn mbvh(&self) -> Option<&Mbvh4> {
        match &self.source {
            BlockSource::Converted(m) => Some(m),
            _ => None,
        }
    }
}

/// AMD's hardware BVH format (tinybvh's `BVH4_AMD_HW`): 128-byte BOX32 nodes with four children
/// and 64-byte triangle nodes holding one triangle each, as consumed by the ray tracing units of
/// RDNA GPUs (see Mesa's `bvh.h`). It is built from an [`Mbvh4`] whose leaves hold exactly one
/// triangle.
///
/// This layout only exists to be uploaded: tinybvh has no CPU traversal for it, so there is no
/// `intersect`. [`data`](Self::data) is the packed buffer in 16-byte blocks.
///
/// # Example
///
/// ```
/// use glam::Vec4;
/// use tinybvh::Bvh4AmdHw;
///
/// let tri = |z: f32| [Vec4::new(-1.0, -1.0, z, 0.0), Vec4::new(1.0, -1.0, z, 0.0), Vec4::new(0.0, 1.0, z, 0.0)];
/// let mut vertices = Vec::new();
/// for i in 0..8 {
///     vertices.extend(tri(2.0 + i as f32));
/// }
///
/// let mut bvh = Bvh4AmdHw::new();
/// bvh.build(vertices);
/// // Nodes are 64-byte aligned: a multiple of four blocks.
/// assert!(!bvh.data().is_empty() && bvh.data().len() % 4 == 0);
/// ```
pub struct Bvh4AmdHw {
    raw: NonNull<ffi::tbvh_bvh4_amd_hw>,
    source: BlockSource<Mbvh4>,
    rebuildable: bool,
}

impl_base_accessors!(
    Bvh4AmdHw,
    tbvh_bvh4_amd_hw_base,
    tbvh_bvh4_amd_hw_base_const
);
impl_block_layout!(
    Bvh4AmdHw,
    tbvh_bvh4_amd_hw,
    Mbvh4,
    new = tbvh_bvh4_amd_hw_new,
    free = tbvh_bvh4_amd_hw_free,
    build = tbvh_bvh4_amd_hw_build,
    build_indexed = tbvh_bvh4_amd_hw_build_indexed,
    build_hq = tbvh_bvh4_amd_hw_build_hq,
    build_hq_indexed = tbvh_bvh4_amd_hw_build_hq_indexed,
    optimize = tbvh_bvh4_amd_hw_optimize,
    sah_cost = tbvh_bvh4_amd_hw_sah_cost,
    load = tbvh_bvh4_amd_hw_load_self_contained,
    blocks = tbvh_bvh4_amd_hw_data
);

impl Bvh4AmdHw {
    /// Converts a built [`Mbvh4`] whose leaves each hold exactly one triangle, taking ownership
    /// of it. The result cannot be [optimized](Self::optimize). Panics if `source` has not been
    /// built or has a leaf with more than one triangle (tinybvh treats that as a fatal error).
    pub fn from_mbvh(source: Mbvh4) -> Self {
        assert!(source.is_built(), "tinybvh: the Mbvh4 has not been built");
        assert!(
            source.nodes().iter().all(|n| !n.is_leaf() || n.tri_count == 1),
            "tinybvh: Bvh4AmdHw needs one triangle per leaf; split the leaves of the source Bvh first"
        );
        let mut layout = Bvh4AmdHw::new();
        unsafe { ffi::tbvh_bvh4_amd_hw_convert_from(layout.raw.as_ptr(), source.raw_ptr()) };
        layout.source = BlockSource::Converted(source);
        layout.rebuildable = false;
        layout
    }

    /// The packed node and triangle data in 16-byte blocks, ready for upload. Empty when nothing
    /// is built.
    pub fn data(&self) -> &[Vec4] {
        let mut used = 0u32;
        let ptr = unsafe { ffi::tbvh_bvh4_amd_hw_data(self.raw.as_ptr(), &mut used) };
        unsafe { block_slice(ptr, used) }
    }

    /// The source tree after [`from_mbvh`](Self::from_mbvh).
    pub fn mbvh(&self) -> Option<&Mbvh4> {
        match &self.source {
            BlockSource::Converted(m) => Some(m),
            _ => None,
        }
    }
}

/// The compressed wide BVH (tinybvh's `BVH8_CWBVH`, Ylitie et al. 2017): 80-byte 8-wide nodes with
/// quantized child bounds and separate triangle data, the layout of choice for GPU path tracers.
///
/// [`nodes`](Self::nodes) and [`triangles`](Self::triangles) are the two buffers to upload, in
/// 16-byte blocks. The layout has its own cache format: [`save`](Self::save) and
/// [`load`](Self::load) store the converted data directly.
///
/// # Example
///
/// ```
/// use glam::{Vec3, Vec4};
/// use tinybvh::{Bvh8Cwbvh, Ray};
///
/// let tri = |z: f32| [Vec4::new(-1.0, -1.0, z, 0.0), Vec4::new(1.0, -1.0, z, 0.0), Vec4::new(0.0, 1.0, z, 0.0)];
/// let mut vertices = Vec::new();
/// for i in 0..8 {
///     vertices.extend(tri(2.0 + i as f32));
/// }
///
/// let mut bvh = Bvh8Cwbvh::new();
/// bvh.build(vertices);
/// assert!(!bvh.nodes().is_empty() && !bvh.triangles().is_empty());
///
/// let mut ray = Ray::new(Vec3::ZERO, Vec3::Z);
/// bvh.intersect(&mut ray);
/// assert_eq!(ray.hit.prim, 0);
/// assert!((ray.hit.t - 2.0).abs() < 1e-5);
/// ```
pub struct Bvh8Cwbvh {
    raw: NonNull<ffi::tbvh_bvh8_cwbvh>,
    source: BlockSource<Mbvh8>,
    rebuildable: bool,
}

impl_base_accessors!(Bvh8Cwbvh, tbvh_bvh8_cwbvh_base, tbvh_bvh8_cwbvh_base_const);
impl_block_layout!(
    Bvh8Cwbvh,
    tbvh_bvh8_cwbvh,
    Mbvh8,
    new = tbvh_bvh8_cwbvh_new,
    free = tbvh_bvh8_cwbvh_free,
    build = tbvh_bvh8_cwbvh_build,
    build_indexed = tbvh_bvh8_cwbvh_build_indexed,
    build_hq = tbvh_bvh8_cwbvh_build_hq,
    build_hq_indexed = tbvh_bvh8_cwbvh_build_hq_indexed,
    optimize = tbvh_bvh8_cwbvh_optimize,
    sah_cost = tbvh_bvh8_cwbvh_sah_cost,
    load = tbvh_bvh8_cwbvh_load_self_contained,
    blocks = tbvh_bvh8_cwbvh_nodes
);

impl Bvh8Cwbvh {
    /// Converts a built [`Mbvh8`], taking ownership of it. A CWBVH leaf slot holds at most three
    /// triangles and the root must be an interior node; tinybvh treats violations as fatal
    /// errors, so this panics if `source` has not been built, has a leaf with more than three
    /// triangles, or consists of a single leaf. The result cannot be
    /// [optimized](Self::optimize).
    pub fn from_mbvh(source: Mbvh8) -> Self {
        assert!(source.is_built(), "tinybvh: the Mbvh8 has not been built");
        let nodes = source.nodes();
        assert!(
            nodes.first().is_some_and(|root| !root.is_leaf()),
            "tinybvh: Bvh8Cwbvh cannot be converted from a single-leaf Mbvh8"
        );
        assert!(
            nodes.iter().all(|n| !n.is_leaf() || n.tri_count <= 3),
            "tinybvh: Bvh8Cwbvh needs at most three triangles per leaf; split the leaves of the source Bvh first"
        );
        let mut layout = Bvh8Cwbvh::new();
        unsafe { ffi::tbvh_bvh8_cwbvh_convert_from(layout.raw.as_ptr(), source.raw_ptr(), true) };
        layout.source = BlockSource::Converted(source);
        layout.rebuildable = false;
        layout
    }

    #[inline]
    fn assert_built(&self) {
        assert!(self.is_built(), "tinybvh: the Bvh8Cwbvh has not been built");
    }

    /// Finds the closest hit along `ray` with tinybvh's reference traversal of the GPU data and
    /// records it in `ray.hit`. tinybvh's CWBVH traversal does not count its steps, so the
    /// returned cost is always 0. Panics if nothing is built.
    pub fn intersect(&self, ray: &mut Ray) -> u32 {
        self.assert_built();
        ray.assert_finite();
        let steps = unsafe { ffi::tbvh_bvh8_cwbvh_intersect(self.raw.as_ptr(), ray.as_ffi_mut()) };
        steps.max(0) as u32
    }

    /// Whether anything blocks `ray` before `ray.hit.t`. Panics if nothing is built.
    pub fn is_occluded(&self, ray: &Ray) -> bool {
        self.assert_built();
        ray.assert_finite();
        unsafe { ffi::tbvh_bvh8_cwbvh_is_occluded(self.raw.as_ptr(), ray.as_ffi()) }
    }

    /// The node data in 16-byte blocks (five per node), ready for upload. Empty when nothing is
    /// built.
    pub fn nodes(&self) -> &[Vec4] {
        let mut used = 0u32;
        let ptr = unsafe { ffi::tbvh_bvh8_cwbvh_nodes(self.raw.as_ptr(), &mut used) };
        unsafe { block_slice(ptr, used) }
    }

    /// The triangle data in 16-byte blocks (three per triangle: a vertex and two edges, with the
    /// original primitive index in the first `w`), ready for upload. Empty when nothing is built.
    pub fn triangles(&self) -> &[Vec4] {
        let mut used = 0u32;
        let ptr = unsafe { ffi::tbvh_bvh8_cwbvh_tris(self.raw.as_ptr(), &mut used) };
        unsafe { block_slice(ptr, used) }
    }

    /// Writes the converted data to a file in tinybvh's CWBVH cache format, which
    /// [`load`](Self::load) and [`load_self_contained`](Self::load_self_contained) read back.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        if !self.is_built() {
            return Err(Error::NotBuilt);
        }
        let cpath = geometry::path_to_cstring(path.as_ref())?;
        unsafe { ffi::tbvh_bvh8_cwbvh_save(self.raw.as_ptr(), cpath.as_ptr()) };
        Ok(())
    }

    /// Loads a file written by [`save`](Self::save). `geometry` must be the data the tree was
    /// built over: the file is checked against its triangle count and the geometry is kept for
    /// reference, while traversal uses the triangle data embedded in the file. Any previous
    /// content is replaced; on failure the structure is left empty. The result can neither be
    /// [optimized](Self::optimize) nor report a [SAH cost](Self::sah_cost).
    pub fn load(&mut self, path: impl AsRef<Path>, geometry: impl Into<Geometry>) -> Result<()> {
        let geometry = geometry.into();
        geometry.validate();
        let cpath = geometry::path_to_cstring(path.as_ref())?;
        // A fresh object: tinybvh's Load overwrites the object without releasing a previous build.
        self.clear();
        let ok = unsafe {
            ffi::tbvh_bvh8_cwbvh_load(
                self.raw.as_ptr(),
                cpath.as_ptr(),
                geometry.triangle_count() as u32,
            )
        };
        if ok {
            self.source = BlockSource::Triangles(geometry);
            self.rebuildable = false;
            Ok(())
        } else {
            Err(Error::LoadFailed(path.as_ref().display().to_string()))
        }
    }

    /// The source tree after [`from_mbvh`](Self::from_mbvh).
    pub fn mbvh(&self) -> Option<&Mbvh8> {
        match &self.source {
            BlockSource::Converted(m) => Some(m),
            _ => None,
        }
    }
}
