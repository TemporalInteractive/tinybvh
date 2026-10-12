//! [`BvhVerbose`]: nodes with explicit child and parent links, for tree surgery and the optimizer.

use core::ptr::NonNull;

use glam::{Vec3, Vec4};

use crate::bvh::Bvh;
use crate::error::{Error, Result};
use crate::ffi;

/// A node of a [`BvhVerbose`] (tinybvh's `BVH_Verbose::BVHNode`): 64 bytes.
///
/// Unlike [`BvhNode`](crate::BvhNode), both children and the parent are stored explicitly, which
/// is what the insertion-based optimizer and the leaf surgery operations need. The root has
/// `parent == u32::MAX`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BvhVerboseNode {
    /// Minimum corner of the node bounds.
    pub aabb_min: Vec3,
    /// Index of the left child; unused for a leaf.
    pub left: u32,
    /// Maximum corner of the node bounds.
    pub aabb_max: Vec3,
    /// Index of the right child; unused for a leaf.
    pub right: u32,
    /// Number of primitives for a leaf; 0 for an interior node.
    pub tri_count: u32,
    /// First slot in the [primitive index array](BvhVerbose::prim_indices) for a leaf.
    pub first_tri: u32,
    /// Index of the parent node; `u32::MAX` for the root.
    pub parent: u32,
    /// Padding to 64 bytes.
    pub dummy: [f32; 5],
}

impl BvhVerboseNode {
    /// Sentinel stored in `parent` for the root node.
    pub const NO_PARENT: u32 = u32::MAX;

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
            Some((self.left, self.right))
        }
    }

    /// The parent node index; `None` for the root.
    #[inline]
    pub fn parent(&self) -> Option<u32> {
        if self.parent == Self::NO_PARENT {
            None
        } else {
            Some(self.parent)
        }
    }

    /// The range into [`BvhVerbose::prim_indices`] covered by a leaf; empty for an interior node.
    #[inline]
    pub fn prim_range(&self) -> core::ops::Range<usize> {
        if self.is_leaf() {
            self.first_tri as usize..(self.first_tri + self.tri_count) as usize
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

const _: () = assert!(core::mem::size_of::<BvhVerboseNode>() == core::mem::size_of::<ffi::tbvh_bvh_verbose_node>());

/// The verbose layout (tinybvh's `BVH_Verbose`): 64-byte nodes with parent links, used by the
/// tree-rotation optimizer and for leaf surgery. It is not traversed directly: convert a [`Bvh`]
/// with [`from_bvh`](Self::from_bvh), operate on it, and turn it back with
/// [`into_bvh`](Self::into_bvh).
///
/// The verbose tree references the primitive indices and vertices of its source `Bvh`, which is
/// why the conversion takes ownership of it; the source stays accessible through
/// [`bvh`](Self::bvh), and its vertices can be moved through [`vertices_mut`](Self::vertices_mut)
/// before a [refit](Self::refit).
///
/// # Example
///
/// ```
/// use glam::{Vec3, Vec4};
/// use tinybvh::{Bvh, BvhVerbose, Ray};
///
/// let tri = |z: f32| [Vec4::new(-1.0, -1.0, z, 0.0), Vec4::new(1.0, -1.0, z, 0.0), Vec4::new(0.0, 1.0, z, 0.0)];
/// let mut vertices = Vec::new();
/// for i in 0..64 {
///     vertices.extend(tri(2.0 + i as f32));
/// }
/// let mut bvh = Bvh::new();
/// bvh.build(vertices);
///
/// let mut verbose = BvhVerbose::from_bvh(bvh);
/// let before = verbose.sah_cost(0);
/// verbose.optimize(10);
/// assert!(verbose.sah_cost(0) <= before * 1.001);
/// verbose.split_leafs(1);
/// assert!(verbose.nodes().iter().all(|n| !n.is_leaf() || n.tri_count == 1));
///
/// let bvh = verbose.into_bvh();
/// let mut ray = Ray::new(Vec3::ZERO, Vec3::Z);
/// bvh.intersect(&mut ray);
/// assert!((ray.hit.t - 2.0).abs() < 1e-5);
/// ```
pub struct BvhVerbose {
    raw: NonNull<ffi::tbvh_bvh_verbose>,
    /// The tree this was converted from; tinybvh references its indices and vertices.
    source: Option<Bvh>,
}

impl_base_accessors!(BvhVerbose, tbvh_bvh_verbose_base, tbvh_bvh_verbose_base_const);

impl BvhVerbose {
    /// An empty structure; fill it with [`from_bvh`](Self::from_bvh).
    pub fn new() -> Self {
        crate::install_fatal_trampoline();
        let raw = unsafe { ffi::tbvh_bvh_verbose_new() };
        BvhVerbose { raw: NonNull::new(raw).expect("tinybvh: allocation failed"), source: None }
    }

    /// Converts a built [`Bvh`], taking ownership of it. Every kind of `Bvh` can be converted;
    /// operations that need triangle geometry ([`refit`](Self::refit)) report
    /// [`NoGeometry`](Error::NoGeometry) for the others. Panics if `source` has not been built.
    pub fn from_bvh(source: Bvh) -> Self {
        assert!(source.is_built(), "tinybvh: the Bvh has not been built");
        let mut verbose = BvhVerbose::new();
        unsafe { ffi::tbvh_bvh_verbose_convert_from(verbose.raw.as_ptr(), source.raw_ptr(), true) };
        verbose.source = Some(source);
        verbose
    }

    /// Converts the tree back into its source [`Bvh`] and returns it. The nodes are written
    /// depth-first into the source, replacing its previous tree; the geometry is untouched.
    /// Panics if the structure is empty.
    pub fn into_bvh(mut self) -> Bvh {
        self.assert_built();
        let mut bvh = self.source.take().expect("tinybvh: the BvhVerbose has no source");
        unsafe { ffi::tbvh_bvh_convert_from_verbose(bvh.raw_ptr_mut(), self.raw.as_ptr(), true) };
        // `self` is dropped here, which frees the verbose node pool only.
        bvh
    }

    // --- queries ---------------------------------------------------------------------------------

    /// Whether a tree is available.
    pub fn is_built(&self) -> bool {
        self.used_nodes() > 0 && self.source.is_some()
    }

    #[inline]
    fn assert_built(&self) {
        assert!(self.is_built(), "tinybvh: the BvhVerbose has not been built");
    }

    /// SAH cost of the subtree rooted at `node` (0 for the whole tree): lower is better.
    pub fn sah_cost(&self, node: u32) -> f32 {
        self.assert_built();
        assert!(node < self.used_nodes(), "tinybvh: node index out of range");
        unsafe { ffi::tbvh_bvh_verbose_sah_cost(self.raw.as_ptr(), node) }
    }

    /// Number of reachable nodes in the tree.
    pub fn node_count(&self) -> u32 {
        if !self.is_built() {
            return 0;
        }
        unsafe { ffi::tbvh_bvh_verbose_node_count(self.raw.as_ptr()) }.max(0) as u32
    }

    // --- maintenance -----------------------------------------------------------------------------

    /// Recomputes the node bounds bottom-up after the vertices changed through
    /// [`vertices_mut`](Self::vertices_mut). With `skip_leafs` the leaf bounds are kept and only
    /// the interior nodes are updated, which is what the optimizer needs after moving subtrees.
    ///
    /// Leaf refitting reads the vertices through the primitive indices, so it fails for trees
    /// without triangle geometry ([`NoGeometry`](Error::NoGeometry)), and for trees built with
    /// spatial splits or over indexed triangles ([`NotRefittable`](Error::NotRefittable)); pass
    /// `skip_leafs` for those.
    pub fn refit(&mut self, skip_leafs: bool) -> Result<()> {
        if !self.is_built() {
            return Err(Error::NotBuilt);
        }
        if !skip_leafs {
            if self.vertices().is_none() {
                return Err(Error::NoGeometry);
            }
            if !self.is_refittable() || self.is_over_indices() {
                return Err(Error::NotRefittable);
            }
        }
        unsafe { ffi::tbvh_bvh_verbose_refit(self.raw.as_ptr(), 0, skip_leafs) };
        Ok(())
    }

    /// Removes unused nodes and orders the nodes depth-first. Required before
    /// [`optimize`](Self::optimize) after [`merge_leafs`](Self::merge_leafs); done automatically.
    pub fn compact(&mut self) {
        self.assert_built();
        unsafe { ffi::tbvh_bvh_verbose_compact(self.raw.as_ptr()) };
    }

    /// Reorders the primitive index array so that the primitives of every subtree are
    /// contiguous, as a freshly built tree has them. The optimizer leaves the order scrambled;
    /// tinybvh does this before converting an optimized tree back.
    pub fn sort_indices(&mut self) {
        self.assert_built();
        unsafe { ffi::tbvh_bvh_verbose_sort_indices(self.raw.as_ptr()) };
    }

    /// Splits leaves until none holds more than `max_prims` primitives.
    ///
    /// The new leaf bounds come from the per-primitive bounds tinybvh keeps from the build; a
    /// source `Bvh` that was loaded from a file has none, and the call then fails with
    /// [`NotRebuildable`](crate::Error::NotRebuildable).
    pub fn split_leafs(&mut self, max_prims: u32) -> Result<()> {
        if !self.is_built() {
            return Err(crate::Error::NotBuilt);
        }
        assert!(max_prims > 0, "tinybvh: max_prims must be at least 1");
        let source = self.source.as_ref().expect("tinybvh: the BvhVerbose has no source");
        if !source.has_fragments() {
            return Err(crate::Error::NotRebuildable);
        }
        unsafe { ffi::tbvh_bvh_verbose_split_leafs(self.raw.as_ptr(), max_prims) };
        Ok(())
    }

    /// Merges leaves where the SAH says it pays off. The node array has holes afterwards; they
    /// are removed by [`compact`](Self::compact), which [`optimize`](Self::optimize) and
    /// [`into_bvh`](Self::into_bvh) handle on their own.
    pub fn merge_leafs(&mut self) {
        self.assert_built();
        unsafe { ffi::tbvh_bvh_verbose_merge_leafs(self.raw.as_ptr()) };
    }

    /// Improves the tree with insertion-based tree rotations (Bittner et al.), lowering the SAH
    /// cost. `iterations` of 25 is the tinybvh default. Trees with fewer than six nodes are left
    /// alone.
    pub fn optimize(&mut self, iterations: u32) {
        self.optimize_with(iterations, false, false);
    }

    /// [`optimize`](Self::optimize) with tinybvh's extra knobs: `extreme` considers every node as
    /// a candidate for reinsertion; `stochastic` randomizes candidate selection.
    pub fn optimize_with(&mut self, iterations: u32, extreme: bool, stochastic: bool) {
        self.assert_built();
        if self.may_have_holes() {
            self.compact();
        }
        unsafe { ffi::tbvh_bvh_verbose_optimize(self.raw.as_ptr(), iterations, extreme, stochastic) };
    }

    // --- data access -----------------------------------------------------------------------------

    /// The nodes in use. Node 0 is the root. After [`merge_leafs`](Self::merge_leafs) some
    /// entries are unreachable until the next [`compact`](Self::compact).
    pub fn nodes(&self) -> &[BvhVerboseNode] {
        let mut count = 0u32;
        let ptr = unsafe { ffi::tbvh_bvh_verbose_nodes(self.raw.as_ptr(), &mut count) };
        if ptr.is_null() || count == 0 {
            return &[];
        }
        unsafe { core::slice::from_raw_parts(ptr as *const BvhVerboseNode, count as usize) }
    }

    /// The primitive index array that leaves refer to; shared with the source `Bvh`.
    pub fn prim_indices(&self) -> &[u32] {
        let mut count = 0u32;
        let ptr = unsafe { ffi::tbvh_bvh_verbose_prim_indices(self.raw.as_ptr(), &mut count) };
        if ptr.is_null() || count == 0 {
            return &[];
        }
        unsafe { core::slice::from_raw_parts(ptr, count as usize) }
    }

    /// The source tree. Its node array is stale while this structure is being modified; use
    /// [`into_bvh`](Self::into_bvh) to bring it up to date.
    pub fn bvh(&self) -> Option<&Bvh> {
        self.source.as_ref()
    }

    /// The vertices the tree references, if it has triangle geometry.
    pub fn vertices(&self) -> Option<&[Vec4]> {
        self.source.as_ref().and_then(Bvh::vertices)
    }

    /// Mutable access to the vertices of the source, for animation followed by
    /// [`refit`](Self::refit). The number of vertices cannot change. Only available for geometry
    /// the source owns.
    pub fn vertices_mut(&mut self) -> Option<&mut [Vec4]> {
        self.source.as_mut().and_then(Bvh::vertices_mut)
    }

    /// Releases the tree and the source `Bvh`, returning the structure to the empty state.
    pub fn clear(&mut self) {
        unsafe {
            ffi::tbvh_bvh_verbose_free(self.raw.as_ptr());
            self.raw = NonNull::new(ffi::tbvh_bvh_verbose_new()).expect("tinybvh: allocation failed");
        }
        self.source = None;
    }
}

impl Default for BvhVerbose {
    fn default() -> Self {
        BvhVerbose::new()
    }
}

impl Drop for BvhVerbose {
    fn drop(&mut self) {
        // The verbose object only owns its node pool, but it references the source Bvh's indices
        // and vertices, so it goes first; the source is dropped afterwards.
        unsafe { ffi::tbvh_bvh_verbose_free(self.raw.as_ptr()) };
    }
}

impl core::fmt::Debug for BvhVerbose {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BvhVerbose")
            .field("built", &self.is_built())
            .field("prim_count", &self.prim_count())
            .field("used_nodes", &self.used_nodes())
            .field("may_have_holes", &self.may_have_holes())
            .finish()
    }
}

// Queries take &self; modifications take &mut self. The source Bvh is owned here.
unsafe impl Send for BvhVerbose {}
unsafe impl Sync for BvhVerbose {}
