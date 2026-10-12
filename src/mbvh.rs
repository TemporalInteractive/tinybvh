//! [`Mbvh4`] and [`Mbvh8`]: tinybvh's wide intermediate layouts (`MBVH<4>` / `MBVH<8>`).

use core::ptr::NonNull;
use std::path::Path;

use glam::{Vec3, Vec4};

use crate::bvh::Bvh;
use crate::error::{Error, Result};
use crate::ffi;
use crate::geometry::Geometry;

/// A node of an [`Mbvh4`] (tinybvh's `MBVH<4>::MBVHNode`): 64 bytes.
///
/// Interior nodes list up to four child indices in `child[..child_count]`; leaves store the first
/// slot in the [primitive index array](Mbvh4::prim_indices) and the number of primitives.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mbvh4Node {
    /// Minimum corner of the node bounds.
    pub aabb_min: Vec3,
    /// First primitive index slot for a leaf; unused for an interior node.
    pub first_tri: u32,
    /// Maximum corner of the node bounds.
    pub aabb_max: Vec3,
    /// Number of primitives for a leaf; 0 for an interior node.
    pub tri_count: u32,
    /// Child node indices; only the first `child_count` are valid.
    pub child: [u32; 4],
    /// Number of valid entries in `child`.
    pub child_count: u32,
    /// Padding to 64 bytes.
    pub dummy: [u32; 3],
}

/// A node of an [`Mbvh8`] (tinybvh's `MBVH<8>::MBVHNode`): 80 bytes.
///
/// Interior nodes list up to eight child indices in `child[..child_count]`; leaves store the first
/// slot in the [primitive index array](Mbvh8::prim_indices) and the number of primitives.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mbvh8Node {
    /// Minimum corner of the node bounds.
    pub aabb_min: Vec3,
    /// First primitive index slot for a leaf; unused for an interior node.
    pub first_tri: u32,
    /// Maximum corner of the node bounds.
    pub aabb_max: Vec3,
    /// Number of primitives for a leaf; 0 for an interior node.
    pub tri_count: u32,
    /// Child node indices; only the first `child_count` are valid.
    pub child: [u32; 8],
    /// Number of valid entries in `child`.
    pub child_count: u32,
    /// Padding to 80 bytes.
    pub dummy: [u32; 3],
}

macro_rules! impl_mbvh_node {
    ($node:ident, $width:literal) => {
        impl $node {
            /// Whether the node is a leaf. Empty leaves do not exist, so this is `tri_count > 0`.
            #[inline]
            pub fn is_leaf(&self) -> bool {
                self.tri_count > 0
            }

            /// The children of an interior node, in slot order; empty for a leaf.
            #[inline]
            pub fn children(&self) -> &[u32] {
                if self.is_leaf() {
                    &[]
                } else {
                    &self.child[..(self.child_count as usize).min($width)]
                }
            }

            /// The range into the primitive index array covered by a leaf; empty for an interior
            /// node.
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
    };
}

impl_mbvh_node!(Mbvh4Node, 4);
impl_mbvh_node!(Mbvh8Node, 8);

const _: () = {
    assert!(core::mem::size_of::<Mbvh4Node>() == core::mem::size_of::<ffi::tbvh_mbvh4_node>());
    assert!(core::mem::size_of::<Mbvh4Node>() == 64);
    assert!(core::mem::size_of::<Mbvh8Node>() == core::mem::size_of::<ffi::tbvh_mbvh8_node>());
    assert!(core::mem::size_of::<Mbvh8Node>() == 80);
};

/// What the structure was collapsed from; keeps the referenced data alive.
enum Source {
    /// Nothing built yet.
    Empty,
    /// The binary tree, whose nodes, indices and geometry tinybvh keeps referencing. Builds
    /// create it here; [`from_bvh`](Mbvh4::from_bvh) takes it from the caller.
    Bvh(Bvh),
}

macro_rules! impl_mbvh {
    (
        $(#[$meta:meta])*
        $ty:ident, $node:ident, $raw:ident, $name:literal,
        new = $new:ident, free = $free:ident, base = $base:ident, base_const = $base_const:ident,
        convert_from = $convert_from:ident, refit = $refit:ident,
        leaf_count = $leaf_count:ident, sah_cost = $sah_cost:ident,
        nodes = $nodes:ident, prim_indices = $prim_indices:ident
    ) => {
        $(#[$meta])*
        pub struct $ty {
            raw: NonNull<ffi::$raw>,
            source: Source,
            /// Set once the C++ object has been handed to a CPU layout: the conversion may have
            /// reallocated its node array behind its back, so it must not run its destructor
            /// anymore and its nodes are no longer read.
            detached: bool,
        }

        impl_base_accessors!($ty, $base, $base_const);

        impl $ty {
            /// An empty structure with default [`BuildSettings`](crate::BuildSettings).
            pub fn new() -> Self {
                crate::install_fatal_trampoline();
                let raw = unsafe { ffi::$new() };
                $ty { raw: NonNull::new(raw).expect("tinybvh: allocation failed"), source: Source::Empty, detached: false }
            }

            // --- builders --------------------------------------------------------------------

            /// Builds a [`Bvh`] with the builder selected by the [settings](Self::settings) and
            /// collapses it into the wide layout. Any previous content is replaced.
            ///
            /// Accepts a `Vec<Vec4>` with three vertices per triangle, a `(Vec<Vec4>, Vec<u32>)`
            /// pair for indexed triangles, or a [`Geometry`]. Panics if the geometry is empty, not
            /// a multiple of three, or has an index out of range.
            pub fn build(&mut self, geometry: impl Into<Geometry>) {
                let mut bvh = self.binary_tree_builder();
                bvh.build(geometry);
                self.collapse(bvh);
            }

            /// Builds with spatial splits for a higher quality tree (tinybvh's `BuildHQ`). The
            /// result cannot be [refitted](Self::refit).
            pub fn build_hq(&mut self, geometry: impl Into<Geometry>) {
                let mut bvh = self.binary_tree_builder();
                bvh.build_hq(geometry);
                self.collapse(bvh);
            }

            /// A `Bvh` that builds with this structure's settings and SAH costs.
            fn binary_tree_builder(&self) -> Bvh {
                let mut bvh = Bvh::new().with_settings(&self.settings());
                bvh.set_c_trav(self.c_trav());
                bvh.set_c_int(self.c_int());
                bvh
            }

            /// Collapses `bvh` into this structure and keeps it alive. The previous source is
            /// released only afterwards, since tinybvh referenced it until now.
            fn collapse(&mut self, mut bvh: Bvh) {
                unsafe { ffi::$convert_from(self.raw.as_ptr(), bvh.raw_ptr_mut(), true) };
                self.source = Source::Bvh(bvh);
            }

            /// Collapses the stored binary tree again, after it changed.
            fn recollapse(&mut self) {
                let raw = self.raw.as_ptr();
                if let Source::Bvh(bvh) = &mut self.source {
                    unsafe { ffi::$convert_from(raw, bvh.raw_ptr_mut(), true) };
                }
            }

            /// Collapses a built [`Bvh`] into the wide layout, consuming it. The source must have
            /// been built over triangles (not custom primitives or instances); its nodes are
            /// compacted first when they are not in depth-first order.
            ///
            /// tinybvh keeps referencing the binary tree and its geometry, so the `Bvh` lives on
            /// inside the result; [`source_bvh`](Self::source_bvh) gives access to it.
            ///
            /// Panics if the `Bvh` has not been built or has no triangle geometry.
            pub fn from_bvh(source: Bvh) -> Self {
                assert!(source.is_built(), concat!("tinybvh: ", $name, "::from_bvh: the Bvh has not been built"));
                assert!(
                    source.vertices().is_some(),
                    concat!("tinybvh: ", $name, "::from_bvh needs a Bvh built over triangles")
                );
                let mut mbvh = $ty::new();
                mbvh.collapse(source);
                mbvh
            }

            /// Loads a [`Bvh`] file written by [`Bvh::save`] with `include_geometry` and collapses
            /// it into the wide layout, as [`from_bvh`](Self::from_bvh) does. Fails if the file
            /// cannot be read or carries no geometry.
            pub fn load_self_contained(path: impl AsRef<Path>) -> Result<Self> {
                let bvh = Bvh::load_self_contained(path)?;
                Ok($ty::from_bvh(bvh))
            }

            // --- queries ---------------------------------------------------------------------

            /// Whether a tree is available.
            pub fn is_built(&self) -> bool {
                !self.detached && self.used_nodes() > 0 && !matches!(self.source, Source::Empty)
            }

            #[inline]
            fn assert_built(&self) {
                assert!(self.is_built(), concat!("tinybvh: the ", $name, " has not been built"));
            }

            // --- maintenance -----------------------------------------------------------------

            /// Updates the node bounds after the vertices changed through
            /// [`vertices_mut`](Self::vertices_mut), keeping the topology. The binary tree is
            /// refitted too, so a later conversion into a CPU layout sees the new bounds.
            ///
            /// Fails for trees built with spatial splits or presplitting, and for trees that may
            /// contain unused nodes.
            pub fn refit(&mut self) -> Result<()> {
                if !self.is_built() {
                    return Err(Error::NotBuilt);
                }
                if !self.is_refittable() {
                    return Err(Error::NotRefittable);
                }
                if self.may_have_holes() {
                    return Err(Error::HasHoles);
                }
                let bvh = match &mut self.source {
                    Source::Bvh(bvh) => bvh,
                    Source::Empty => return Err(Error::NoGeometry),
                };
                // A compaction inside the Bvh refit would move its node array, which the wide
                // tree points into; collapse again in that case instead of refitting in place.
                let relocates = bvh.may_have_holes() || !bvh.has_ordered_nodes();
                bvh.refit()?;
                if relocates {
                    self.recollapse();
                } else {
                    unsafe { ffi::$refit(self.raw.as_ptr(), 0) };
                }
                Ok(())
            }

            /// Improves the binary tree with tree rotations (Bittner et al.) and collapses it
            /// again. `iterations` of 25 is the tinybvh default.
            pub fn optimize(&mut self, iterations: u32) -> Result<()> {
                self.optimize_with(iterations, false)
            }

            /// [`optimize`](Self::optimize) with tinybvh's `extreme` knob, which considers every
            /// node as a candidate for reinsertion.
            pub fn optimize_with(&mut self, iterations: u32, extreme: bool) -> Result<()> {
                if !self.is_built() {
                    return Err(Error::NotBuilt);
                }
                match &mut self.source {
                    Source::Bvh(bvh) => bvh.optimize_with(iterations, extreme, false),
                    Source::Empty => return Err(Error::NoGeometry),
                }
                self.recollapse();
                Ok(())
            }

            // --- metrics ---------------------------------------------------------------------

            /// SAH cost of the subtree rooted at `node` (0 for the whole tree): lower is better.
            pub fn sah_cost(&self, node: u32) -> f32 {
                self.assert_built();
                assert!(node < self.used_nodes(), "tinybvh: node index out of range");
                unsafe { ffi::$sah_cost(self.raw.as_ptr(), node) }
            }

            /// Number of leaves in the subtree rooted at `node` (0 for the whole tree).
            pub fn leaf_count(&self, node: u32) -> u32 {
                self.assert_built();
                assert!(node < self.used_nodes(), "tinybvh: node index out of range");
                unsafe { ffi::$leaf_count(self.raw.as_ptr(), node) }
            }

            // --- data access -----------------------------------------------------------------

            /// The nodes in use. Node 0 is the root.
            pub fn nodes(&self) -> &[$node] {
                if self.detached {
                    return &[];
                }
                let mut count = 0u32;
                let ptr = unsafe { ffi::$nodes(self.raw.as_ptr(), &mut count) };
                if ptr.is_null() || count == 0 {
                    return &[];
                }
                unsafe { core::slice::from_raw_parts(ptr as *const $node, count as usize) }
            }

            /// The primitive index array that leaves refer to.
            pub fn prim_indices(&self) -> &[u32] {
                if self.detached {
                    return &[];
                }
                let mut count = 0u32;
                let ptr = unsafe { ffi::$prim_indices(self.raw.as_ptr(), &mut count) };
                if ptr.is_null() || count == 0 {
                    return &[];
                }
                unsafe { core::slice::from_raw_parts(ptr, count as usize) }
            }

            /// The triangle geometry the tree was built over, if it is owned (not loaded from a
            /// file).
            pub fn geometry(&self) -> Option<&Geometry> {
                self.source_bvh().and_then(Bvh::geometry)
            }

            /// The vertices the tree references.
            pub fn vertices(&self) -> Option<&[Vec4]> {
                self.source_bvh().and_then(Bvh::vertices)
            }

            /// Mutable access to the vertices, for animation followed by [`refit`](Self::refit).
            /// The number of vertices cannot change. Not available for geometry loaded from a
            /// file.
            pub fn vertices_mut(&mut self) -> Option<&mut [Vec4]> {
                match &mut self.source {
                    Source::Bvh(b) => b.vertices_mut(),
                    Source::Empty => None,
                }
            }

            /// The vertex indices for indexed geometry.
            pub fn indices(&self) -> Option<&[u32]> {
                self.source_bvh().and_then(Bvh::indices)
            }

            /// The binary tree this structure was collapsed from: the one built by
            /// [`build`](Self::build), or the one given to [`from_bvh`](Self::from_bvh).
            pub fn source_bvh(&self) -> Option<&Bvh> {
                match &self.source {
                    Source::Bvh(b) => Some(b),
                    Source::Empty => None,
                }
            }

            /// Releases everything the structure holds, including the geometry, and returns it
            /// to the empty state.
            pub fn clear(&mut self) {
                self.free_raw();
                self.raw = NonNull::new(unsafe { ffi::$new() }).expect("tinybvh: allocation failed");
                self.detached = false;
                self.source = Source::Empty;
            }

            /// Takes the binary tree out of the structure, leaving it empty.
            pub fn into_bvh(mut self) -> Option<Bvh> {
                let source = core::mem::replace(&mut self.source, Source::Empty);
                self.free_raw();
                self.raw = NonNull::new(unsafe { ffi::$new() }).expect("tinybvh: allocation failed");
                self.detached = false;
                match source {
                    Source::Bvh(b) => Some(b),
                    Source::Empty => None,
                }
            }

            /// Takes the owned triangle geometry out of the structure, leaving it empty.
            pub fn into_geometry(self) -> Option<Geometry> {
                self.into_bvh().and_then(Bvh::into_geometry)
            }

            /// Frees the C++ object. After a CPU layout took the tree over the object holds no
            /// buffers any more (the shim released them), so this is always safe.
            fn free_raw(&mut self) {
                unsafe { ffi::$free(self.raw.as_ptr()) };
            }

            /// Marks the wide tree as taken over by a CPU layout: the C++ object is empty from
            /// now on and its node array is no longer read. See `Bvh4Cpu::from_mbvh`.
            pub(crate) fn detach_raw(&mut self) {
                self.detached = true;
            }

            #[doc(hidden)]
            pub fn raw_ptr(&self) -> *const ffi::$raw {
                self.raw.as_ptr()
            }

            #[doc(hidden)]
            pub fn raw_ptr_mut(&mut self) -> *mut ffi::$raw {
                self.raw.as_ptr()
            }
        }

        impl Default for $ty {
            fn default() -> Self {
                $ty::new()
            }
        }

        impl Drop for $ty {
            fn drop(&mut self) {
                // The C++ object goes first: it references the Bvh dropped afterwards.
                self.free_raw();
            }
        }

        impl core::fmt::Debug for $ty {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.debug_struct($name)
                    .field("built", &self.is_built())
                    .field("prim_count", &self.prim_count())
                    .field("used_nodes", &self.used_nodes())
                    .finish()
            }
        }

        // Queries take &self and only read the tree; builds take &mut self. The referenced
        // binary tree and its geometry are owned here.
        unsafe impl Send for $ty {}
        unsafe impl Sync for $ty {}
    };
}

impl_mbvh!(
    /// A 4-wide BVH (tinybvh's `MBVH<4>`): the intermediate layout that
    /// [`Bvh4Cpu`](crate::Bvh4Cpu), [`Bvh4Gpu`](crate::Bvh4Gpu) and
    /// [`Bvh4AmdHw`](crate::Bvh4AmdHw) are converted from.
    ///
    /// An `Mbvh4` has no traversal of its own; it exposes its [nodes](Mbvh4::nodes) for
    /// inspection and conversion, and can be [refitted](Mbvh4::refit) and
    /// [optimized](Mbvh4::optimize). It always sits on top of a [`Bvh`]: [`build`](Mbvh4::build)
    /// creates one, [`from_bvh`](Mbvh4::from_bvh) collapses one that was tuned by the caller.
    ///
    /// # Example
    ///
    /// ```
    /// use glam::Vec4;
    /// use tinybvh::Mbvh4;
    ///
    /// let tri = |z: f32| [Vec4::new(-1.0, -1.0, z, 0.0), Vec4::new(1.0, -1.0, z, 0.0), Vec4::new(0.0, 1.0, z, 0.0)];
    /// let mut vertices = Vec::new();
    /// vertices.extend(tri(5.0));
    /// vertices.extend(tri(3.0));
    ///
    /// let mut mbvh = Mbvh4::new();
    /// mbvh.build(vertices);
    /// assert_eq!(mbvh.prim_count(), 2);
    /// let root = mbvh.nodes()[0];
    /// for &child in root.children() {
    ///     assert!((child as usize) < mbvh.nodes().len());
    /// }
    /// assert!(mbvh.sah_cost(0) > 0.0);
    /// ```
    Mbvh4, Mbvh4Node, tbvh_mbvh4, "Mbvh4",
    new = tbvh_mbvh4_new, free = tbvh_mbvh4_free, base = tbvh_mbvh4_base, base_const = tbvh_mbvh4_base_const,
    convert_from = tbvh_mbvh4_convert_from, refit = tbvh_mbvh4_refit,
    leaf_count = tbvh_mbvh4_leaf_count, sah_cost = tbvh_mbvh4_sah_cost,
    nodes = tbvh_mbvh4_nodes, prim_indices = tbvh_mbvh4_prim_indices
);

impl_mbvh!(
    /// An 8-wide BVH (tinybvh's `MBVH<8>`): the intermediate layout that
    /// [`Bvh8Cpu`](crate::Bvh8Cpu) and [`Bvh8Cwbvh`](crate::Bvh8Cwbvh) are converted from.
    ///
    /// An `Mbvh8` has no traversal of its own; it exposes its [nodes](Mbvh8::nodes) for
    /// inspection and conversion, and can be [refitted](Mbvh8::refit) and
    /// [optimized](Mbvh8::optimize). It always sits on top of a [`Bvh`]: [`build`](Mbvh8::build)
    /// creates one, [`from_bvh`](Mbvh8::from_bvh) collapses one that was tuned by the caller.
    ///
    /// # Example
    ///
    /// ```
    /// use glam::Vec4;
    /// use tinybvh::{Bvh, Mbvh8};
    ///
    /// let tri = |z: f32| [Vec4::new(-1.0, -1.0, z, 0.0), Vec4::new(1.0, -1.0, z, 0.0), Vec4::new(0.0, 1.0, z, 0.0)];
    /// let mut vertices = Vec::new();
    /// vertices.extend(tri(5.0));
    /// vertices.extend(tri(3.0));
    ///
    /// let mut bvh = Bvh::new();
    /// bvh.build(vertices);
    /// let mbvh = Mbvh8::from_bvh(bvh);
    /// assert_eq!(mbvh.prim_count(), 2);
    /// assert!(mbvh.leaf_count(0) >= 1);
    /// assert_eq!(mbvh.nodes()[0].aabb_min, mbvh.aabb().0);
    /// ```
    Mbvh8, Mbvh8Node, tbvh_mbvh8, "Mbvh8",
    new = tbvh_mbvh8_new, free = tbvh_mbvh8_free, base = tbvh_mbvh8_base, base_const = tbvh_mbvh8_base_const,
    convert_from = tbvh_mbvh8_convert_from, refit = tbvh_mbvh8_refit,
    leaf_count = tbvh_mbvh8_leaf_count, sah_cost = tbvh_mbvh8_sah_cost,
    nodes = tbvh_mbvh8_nodes, prim_indices = tbvh_mbvh8_prim_indices
);
