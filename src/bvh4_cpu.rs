//! [`Bvh4Cpu`]: tinybvh's `BVH4_CPU`, the fastest CPU traversal layout.

use crate::ffi;
use crate::mbvh::Mbvh4;
use crate::ray::Ray;

/// Generates a SIMD CPU layout wrapper (`BVH4_CPU` / `BVH8_CPU`); the two share every operation
/// except the bundle queries, which only `Bvh4Cpu` has.
///
/// The target type owns its geometry, or the wide intermediate structure it was converted from,
/// and tracks whether the intermediate trees that refitting, optimizing and the SAH metric need
/// still exist.
macro_rules! impl_cpu_layout {
    (
        $(#[$meta:meta])*
        $ty:ident, $mbvh:ident, $raw:ident, $name:literal,
        new = $new:ident, free = $free:ident, base = $base:ident, base_const = $base_const:ident,
        build = $build:ident, build_indexed = $build_indexed:ident,
        build_hq = $build_hq:ident, build_hq_indexed = $build_hq_indexed:ident,
        convert_from = $convert_from:ident, optimize = $optimize:ident, refit = $refit:ident,
        sah_cost = $sah_cost:ident, intersect = $intersect:ident, is_occluded = $is_occluded:ident,
        save = $save:ident, load = $load:ident, load_self_contained = $load_self_contained:ident,
        data = $data:ident
    ) => {
        /// What the structure was built over; keeps the referenced data alive.
        enum Source {
            /// Nothing built yet.
            Empty,
            /// Triangles owned by the structure.
            Triangles($crate::geometry::Geometry),
            /// Converted from a wide tree, which tinybvh keeps referencing.
            Mbvh($mbvh),
            /// Loaded from a self-contained file; the leaf blocks carry the triangles.
            Loaded,
        }

        $(#[$meta])*
        pub struct $ty {
            raw: core::ptr::NonNull<$crate::ffi::$raw>,
            source: Source,
            opacity_map: Option<Vec<u32>>,
            /// Whether the intermediate binary and wide trees still exist. They are gone after a
            /// build with `discard_intermediates` and after a file load; tinybvh raises a fatal
            /// error when refitting, optimizing or computing the SAH cost without them.
            intermediates: bool,
        }

        impl_base_accessors!($ty, $base, $base_const);
        impl_opacity_maps!($ty);

        impl $ty {
            /// An empty structure with default [`BuildSettings`](crate::BuildSettings).
            pub fn new() -> Self {
                $crate::install_fatal_trampoline();
                let raw = unsafe { $crate::ffi::$new() };
                $ty {
                    raw: core::ptr::NonNull::new(raw).expect("tinybvh: allocation failed"),
                    source: Source::Empty,
                    opacity_map: None,
                    intermediates: false,
                }
            }

            // --- builders --------------------------------------------------------------------

            /// Builds over triangles with the builder selected by the [settings](Self::settings)
            /// (Bonsai by default), then converts the tree into this layout. Any previous
            /// content is replaced, and an attached opacity micro map is detached.
            ///
            /// Accepts a `Vec<Vec4>` with three vertices per triangle, a `(Vec<Vec4>, Vec<u32>)`
            /// pair for indexed triangles, or a [`Geometry`](crate::Geometry). Panics if the
            /// geometry is empty, not a multiple of three, or has an index out of range.
            pub fn build(&mut self, geometry: impl Into<$crate::geometry::Geometry>) {
                let geometry = geometry.into();
                geometry.validate();
                let discard = self.settings().discard_intermediates;
                unsafe {
                    match geometry.index_args() {
                        None => $crate::ffi::$build(
                            self.raw.as_ptr(),
                            geometry.vert_ptr(),
                            geometry.vert_count(),
                            $crate::geometry::VEC4_STRIDE,
                        ),
                        Some((idx, prims)) => $crate::ffi::$build_indexed(
                            self.raw.as_ptr(),
                            geometry.vert_ptr(),
                            geometry.vert_count(),
                            $crate::geometry::VEC4_STRIDE,
                            idx,
                            prims,
                        ),
                    }
                }
                self.finish_build(Source::Triangles(geometry), !discard);
            }

            /// Builds with spatial splits for a higher quality tree (tinybvh's `BuildHQ`). The
            /// result cannot be [refitted](Self::refit).
            pub fn build_hq(&mut self, geometry: impl Into<$crate::geometry::Geometry>) {
                let geometry = geometry.into();
                geometry.validate();
                let discard = self.settings().discard_intermediates;
                unsafe {
                    match geometry.index_args() {
                        None => $crate::ffi::$build_hq(
                            self.raw.as_ptr(),
                            geometry.vert_ptr(),
                            geometry.vert_count(),
                            $crate::geometry::VEC4_STRIDE,
                        ),
                        Some((idx, prims)) => $crate::ffi::$build_hq_indexed(
                            self.raw.as_ptr(),
                            geometry.vert_ptr(),
                            geometry.vert_count(),
                            $crate::geometry::VEC4_STRIDE,
                            idx,
                            prims,
                        ),
                    }
                }
                self.finish_build(Source::Triangles(geometry), !discard);
            }

            /// Converts a built wide tree into this layout, consuming it. Use this to trace a
            /// tree that was tuned as an [`Mbvh4`](crate::Mbvh4) / [`Mbvh8`](crate::Mbvh8), or
            /// collapsed from a [`Bvh`](crate::Bvh).
            ///
            /// The result is not [rebuildable](Self::is_rebuildable): it references the source's
            /// buffers, so [`refit`](Self::refit) and [`optimize`](Self::optimize) are refused.
            /// A later [`build`](Self::build) releases the source.
            ///
            /// The wide tree's buffers are taken over by this structure, so the consumed source
            /// is kept only for its geometry.
            ///
            /// Panics if the source has not been built.
            pub fn from_mbvh(mut source: $mbvh) -> Self {
                assert!(source.is_built(), concat!("tinybvh: ", $name, "::from_mbvh: the wide tree has not been built"));
                let mut bvh = $ty::new();
                unsafe { $crate::ffi::$convert_from(bvh.raw.as_ptr(), source.raw_ptr_mut(), false) };
                source.detach_raw();
                bvh.finish_build(Source::Mbvh(source), true);
                bvh
            }

            /// Installs the new source after a build or conversion. The old source is released
            /// only here, after tinybvh stopped referencing it.
            fn finish_build(&mut self, source: Source, intermediates: bool) {
                self.source = source;
                self.intermediates = intermediates;
                // The conversion copies the (empty) opacity map pointer of the intermediate tree.
                self.opacity_map = None;
            }

            /// Re-attaches the opacity micro map after an operation that rebuilt the layout from
            /// the intermediate tree, which copies the (empty) map pointer of that tree.
            fn reapply_opacity_map(&mut self, n: u32) {
                if let Some(map) = &self.opacity_map {
                    if n > 0 {
                        let ptr = map.as_ptr();
                        unsafe { $crate::ffi::tbvh_base_set_opacity_micro_maps(self.base_ptr_mut(), ptr, n) };
                    }
                }
            }

            // --- queries ---------------------------------------------------------------------

            /// Whether a tree is available for queries.
            pub fn is_built(&self) -> bool {
                self.used_blocks() > 0 && !matches!(self.source, Source::Empty)
            }

            #[inline]
            fn assert_built(&self) {
                assert!(self.is_built(), concat!("tinybvh: the ", $name, " has not been built"));
            }

            /// Finds the closest hit along `ray` and records it in `ray.hit`. Returns the number
            /// of traversal steps when tinybvh counts them, which its SIMD kernels only do in
            /// debug builds; otherwise 0.
            ///
            /// Panics if the structure has not been built.
            pub fn intersect(&self, ray: &mut Ray) -> u32 {
                self.assert_built();
                ray.assert_finite();
                let steps = unsafe { $crate::ffi::$intersect(self.raw.as_ptr(), ray.as_ffi_mut()) };
                steps.max(0) as u32
            }

            /// Whether anything blocks `ray` strictly before `ray.hit.t`. Cheaper than
            /// [`intersect`](Self::intersect) for shadow rays.
            ///
            /// The current `hit.t` is the search limit: a fresh ray searches up to its maximum
            /// distance, while a ray that already records a hit at `t` reports the geometry at
            /// that very distance as *not* occluding. Reset the ray or shorten `hit.t` by an
            /// epsilon when testing a surface point against its own hit.
            pub fn is_occluded(&self, ray: &Ray) -> bool {
                self.assert_built();
                ray.assert_finite();
                unsafe { $crate::ffi::$is_occluded(self.raw.as_ptr(), ray.as_ffi()) }
            }

            // --- maintenance -----------------------------------------------------------------

            /// Whether the intermediate trees that [`refit`](Self::refit),
            /// [`optimize`](Self::optimize) and [`sah_cost`](Self::sah_cost) need still exist.
            /// False after a build with `discard_intermediates` and after a file load.
            pub fn has_intermediates(&self) -> bool {
                self.intermediates
            }

            /// Updates the node bounds after the vertices changed through
            /// [`vertices_mut`](Self::vertices_mut), keeping the topology. The intermediate tree
            /// is refitted and converted again.
            ///
            /// Fails with [`NotRebuildable`](crate::Error::NotRebuildable) after
            /// [`from_mbvh`](Self::from_mbvh), a file load, or a build with
            /// `discard_intermediates`; with [`NotRefittable`](crate::Error::NotRefittable) for
            /// trees built with spatial splits or presplitting.
            pub fn refit(&mut self) -> Result<(), $crate::Error> {
                if !self.is_built() {
                    return Err($crate::Error::NotBuilt);
                }
                if !self.intermediates || !self.is_rebuildable() {
                    return Err($crate::Error::NotRebuildable);
                }
                if !matches!(self.source, Source::Triangles(_)) {
                    return Err($crate::Error::NoGeometry);
                }
                if !self.is_refittable() {
                    return Err($crate::Error::NotRefittable);
                }
                let n = self.opacity_micro_map_n();
                unsafe { $crate::ffi::$refit(self.raw.as_ptr()) };
                self.reapply_opacity_map(n);
                Ok(())
            }

            /// Improves the intermediate tree with tree rotations (Bittner et al.) and converts
            /// it again, lowering the SAH cost. `iterations` of 25 is the tinybvh default.
            ///
            /// Fails with [`NotRebuildable`](crate::Error::NotRebuildable) after
            /// [`from_mbvh`](Self::from_mbvh), a file load, or a build with
            /// `discard_intermediates`.
            pub fn optimize(&mut self, iterations: u32) -> Result<(), $crate::Error> {
                self.optimize_with(iterations, false)
            }

            /// [`optimize`](Self::optimize) with tinybvh's `extreme` knob, which considers every
            /// node as a candidate for reinsertion.
            pub fn optimize_with(&mut self, iterations: u32, extreme: bool) -> Result<(), $crate::Error> {
                if !self.is_built() {
                    return Err($crate::Error::NotBuilt);
                }
                if !self.intermediates || !self.is_rebuildable() {
                    return Err($crate::Error::NotRebuildable);
                }
                let n = self.opacity_micro_map_n();
                unsafe { $crate::ffi::$optimize(self.raw.as_ptr(), iterations, extreme) };
                self.reapply_opacity_map(n);
                Ok(())
            }

            // --- metrics ---------------------------------------------------------------------

            /// SAH cost of the intermediate wide tree: lower is better. tinybvh does not report
            /// the node count of that tree for this layout, so only the whole-tree cost is
            /// available here; inspect an [`Mbvh4`](crate::Mbvh4) / [`Mbvh8`](crate::Mbvh8) for
            /// subtree costs.
            ///
            /// Panics if the structure has not been built or the
            /// [intermediate trees](Self::has_intermediates) are gone.
            pub fn sah_cost(&self) -> f32 {
                self.assert_built();
                assert!(
                    self.intermediates,
                    concat!("tinybvh: ", $name, "::sah_cost needs the intermediate tree, which was discarded or not loaded")
                );
                unsafe { $crate::ffi::$sah_cost(self.raw.as_ptr(), 0) }
            }

            /// Number of 64-byte blocks the layout occupies; see [`data`](Self::data).
            pub fn used_blocks(&self) -> u32 {
                let mut blocks = 0u32;
                let ptr = unsafe { $crate::ffi::$data(self.raw.as_ptr(), &mut blocks) };
                if ptr.is_null() {
                    0
                } else {
                    blocks
                }
            }

            // --- serialization ---------------------------------------------------------------

            /// Writes the layout to a file. The leaf blocks carry the triangle data, so the
            /// file is self-contained; the intermediate trees are not saved.
            pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<(), $crate::Error> {
                if !self.is_built() {
                    return Err($crate::Error::NotBuilt);
                }
                let cpath = $crate::geometry::path_to_cstring(path.as_ref())?;
                unsafe { $crate::ffi::$save(self.raw.as_ptr(), cpath.as_ptr()) };
                Ok(())
            }

            /// Loads a layout saved with [`save`](Self::save) and binds it to `geometry`, which
            /// must be the data the tree was built over. tinybvh only checks the triangle count
            /// against the file; the triangles themselves come from the file. Any previous
            /// content is replaced; on failure the structure is left empty.
            ///
            /// The loaded structure has no [intermediate trees](Self::has_intermediates), so
            /// it can neither be refitted nor optimized.
            pub fn load(&mut self, path: impl AsRef<std::path::Path>, geometry: impl Into<$crate::geometry::Geometry>) -> Result<(), $crate::Error> {
                let geometry = geometry.into();
                geometry.validate();
                let cpath = $crate::geometry::path_to_cstring(path.as_ref())?;
                let expected = geometry.triangle_count() as u32;
                let ok = unsafe { $crate::ffi::$load(self.raw.as_ptr(), cpath.as_ptr(), expected) };
                if ok {
                    self.finish_build(Source::Triangles(geometry), false);
                    self.clear_stale_opacity_map();
                    Ok(())
                } else {
                    // tinybvh leaves the object untouched on failure; the previous source stays
                    // referenced, so nothing changes here.
                    Err($crate::Error::LoadFailed(path.as_ref().display().to_string()))
                }
            }

            /// Loads a layout from a file written by [`save`](Self::save), or from a
            /// [`Bvh`](crate::Bvh) file written with geometry, which is converted on the way.
            /// The triangle data comes from the file.
            ///
            /// The loaded structure has no [intermediate trees](Self::has_intermediates), so
            /// it can neither be refitted nor optimized.
            pub fn load_self_contained(path: impl AsRef<std::path::Path>) -> Result<Self, $crate::Error> {
                let cpath = $crate::geometry::path_to_cstring(path.as_ref())?;
                let mut bvh = $ty::new();
                let ok = unsafe { $crate::ffi::$load_self_contained(bvh.raw.as_ptr(), cpath.as_ptr()) };
                if !ok {
                    return Err($crate::Error::LoadFailed(path.as_ref().display().to_string()));
                }
                bvh.finish_build(Source::Loaded, false);
                bvh.clear_stale_opacity_map();
                Ok(bvh)
            }

            /// A file stores the object verbatim, including a pointer to the opacity micro map
            /// it had when saved; that pointer means nothing in this process.
            fn clear_stale_opacity_map(&mut self) {
                unsafe { $crate::ffi::tbvh_base_set_opacity_micro_maps(self.base_ptr_mut(), core::ptr::null(), 0) };
                self.opacity_map = None;
            }

            // --- data access -----------------------------------------------------------------

            /// The interleaved node and leaf blocks of the layout, 64 bytes each, as tinybvh
            /// traverses them. Empty before a build.
            pub fn data(&self) -> &[u8] {
                let mut blocks = 0u32;
                let ptr = unsafe { $crate::ffi::$data(self.raw.as_ptr(), &mut blocks) };
                if ptr.is_null() || blocks == 0 {
                    return &[];
                }
                unsafe { core::slice::from_raw_parts(ptr, blocks as usize * 64) }
            }

            /// The triangle geometry the tree was built over, if any.
            pub fn geometry(&self) -> Option<&$crate::geometry::Geometry> {
                match &self.source {
                    Source::Triangles(g) => Some(g),
                    Source::Mbvh(m) => m.geometry(),
                    _ => None,
                }
            }

            /// The vertices the tree was built over. `None` for a self-contained file load.
            pub fn vertices(&self) -> Option<&[glam::Vec4]> {
                match &self.source {
                    Source::Triangles(g) => Some(&g.vertices),
                    Source::Mbvh(m) => m.vertices(),
                    _ => None,
                }
            }

            /// Mutable access to the vertices, for animation followed by [`refit`](Self::refit).
            /// The number of vertices cannot change. Only available for owned geometry.
            pub fn vertices_mut(&mut self) -> Option<&mut [glam::Vec4]> {
                match &mut self.source {
                    Source::Triangles(g) => Some(&mut g.vertices),
                    _ => None,
                }
            }

            /// The vertex indices for indexed geometry.
            pub fn indices(&self) -> Option<&[u32]> {
                match &self.source {
                    Source::Triangles(g) => g.indices.as_deref(),
                    Source::Mbvh(m) => m.indices(),
                    _ => None,
                }
            }

            /// Releases everything the structure holds, including the geometry, and returns it
            /// to the empty state.
            pub fn clear(&mut self) {
                unsafe {
                    $crate::ffi::$free(self.raw.as_ptr());
                    self.raw = core::ptr::NonNull::new($crate::ffi::$new()).expect("tinybvh: allocation failed");
                }
                self.source = Source::Empty;
                self.opacity_map = None;
                self.intermediates = false;
            }

            /// Takes the owned triangle geometry out of the structure, leaving it empty.
            pub fn into_geometry(mut self) -> Option<$crate::geometry::Geometry> {
                let source = core::mem::replace(&mut self.source, Source::Empty);
                unsafe {
                    $crate::ffi::$free(self.raw.as_ptr());
                    self.raw = core::ptr::NonNull::new($crate::ffi::$new()).expect("tinybvh: allocation failed");
                }
                self.opacity_map = None;
                self.intermediates = false;
                match source {
                    Source::Triangles(g) => Some(g),
                    Source::Mbvh(m) => m.into_geometry(),
                    _ => None,
                }
            }

            #[doc(hidden)]
            pub fn raw_ptr(&self) -> *const $crate::ffi::$raw {
                self.raw.as_ptr()
            }

            #[doc(hidden)]
            pub fn raw_ptr_mut(&mut self) -> *mut $crate::ffi::$raw {
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
                // The C++ object goes first: it references the geometry or the wide tree dropped
                // afterwards.
                unsafe { $crate::ffi::$free(self.raw.as_ptr()) };
            }
        }

        impl core::fmt::Debug for $ty {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.debug_struct($name)
                    .field("built", &self.is_built())
                    .field("prim_count", &self.prim_count())
                    .field("used_blocks", &self.used_blocks())
                    .field("rebuildable", &(self.intermediates && self.is_rebuildable()))
                    .finish()
            }
        }

        // Queries take &self and only read the layout; builds take &mut self. The referenced
        // geometry or wide tree is owned here, and the opacity map is an owned Vec.
        unsafe impl Send for $ty {}
        unsafe impl Sync for $ty {}

        impl $crate::instance::sealed::Sealed for $ty {}
        impl $crate::instance::Blas for $ty {
            fn blas_base_ptr(&self) -> *const $crate::ffi::tbvh_base {
                self.base_ptr()
            }

            fn blas_is_ready(&self) -> bool {
                self.is_built()
            }
        }
    };
}

pub(crate) use impl_cpu_layout;

impl_cpu_layout!(
    /// The 4-wide SIMD CPU layout (tinybvh's `BVH4_CPU`): the fastest layout for ray queries
    /// on the CPU, with four child boxes tested per step and four triangles per leaf.
    ///
    /// Build it over triangles like a [`Bvh`](crate::Bvh), or convert a tuned
    /// [`Mbvh4`](crate::Mbvh4) with [`from_mbvh`](Bvh4Cpu::from_mbvh). The layout keeps an
    /// intermediate binary tree so that it can be [refitted](Bvh4Cpu::refit) for animation and
    /// [optimized](Bvh4Cpu::optimize); set `discard_intermediates` in the
    /// [settings](Bvh4Cpu::settings) to drop it and save memory. A `Bvh4Cpu` can serve as a
    /// bottom-level structure in a [`Bvh`](crate::Bvh) TLAS, and honours
    /// [opacity micro maps](Bvh4Cpu::set_opacity_micro_maps).
    ///
    /// The layout lives in 64-byte blocks ([`data`](Bvh4Cpu::data), [`used_blocks`](Bvh4Cpu::used_blocks));
    /// tinybvh does not maintain [`used_nodes`](Bvh4Cpu::used_nodes) for it, which reads 0.
    ///
    /// # Example
    ///
    /// ```
    /// use glam::{Vec3, Vec4};
    /// use tinybvh::{Bvh4Cpu, Ray};
    ///
    /// let tri = |z: f32| [Vec4::new(-1.0, -1.0, z, 0.0), Vec4::new(1.0, -1.0, z, 0.0), Vec4::new(0.0, 1.0, z, 0.0)];
    /// let mut vertices = Vec::new();
    /// vertices.extend(tri(5.0));
    /// vertices.extend(tri(3.0));
    ///
    /// let mut bvh = Bvh4Cpu::new();
    /// bvh.build(vertices);
    ///
    /// let mut ray = Ray::new(Vec3::ZERO, Vec3::Z);
    /// bvh.intersect(&mut ray);
    /// assert_eq!(ray.hit.prim, 1); // the closer triangle
    /// assert!((ray.hit.t - 3.0).abs() < 1e-5);
    /// assert!(bvh.is_occluded(&Ray::with_t_max(Vec3::ZERO, Vec3::Z, 4.0)));
    /// assert!(!bvh.is_occluded(&Ray::with_t_max(Vec3::ZERO, Vec3::Z, 2.0)));
    /// ```
    Bvh4Cpu, Mbvh4, tbvh_bvh4_cpu, "Bvh4Cpu",
    new = tbvh_bvh4_cpu_new, free = tbvh_bvh4_cpu_free, base = tbvh_bvh4_cpu_base, base_const = tbvh_bvh4_cpu_base_const,
    build = tbvh_bvh4_cpu_build, build_indexed = tbvh_bvh4_cpu_build_indexed,
    build_hq = tbvh_bvh4_cpu_build_hq, build_hq_indexed = tbvh_bvh4_cpu_build_hq_indexed,
    convert_from = tbvh_bvh4_cpu_convert_from, optimize = tbvh_bvh4_cpu_optimize, refit = tbvh_bvh4_cpu_refit,
    sah_cost = tbvh_bvh4_cpu_sah_cost, intersect = tbvh_bvh4_cpu_intersect, is_occluded = tbvh_bvh4_cpu_is_occluded,
    save = tbvh_bvh4_cpu_save, load = tbvh_bvh4_cpu_load, load_self_contained = tbvh_bvh4_cpu_load_self_contained,
    data = tbvh_bvh4_cpu_data
);

impl Bvh4Cpu {
    /// Traces a bundle of coherent rays at once (WiVeC traversal); `rays.len()` must equal
    /// [`bundle_rays`](crate::bundle_rays). Unlike [`Bvh::intersect_bundle`](crate::Bvh::intersect_bundle)
    /// this works on the structure itself, no TLAS needed. Returns the traversal cost as
    /// reported by the kernel, which may be 0 in release builds.
    ///
    /// Panics if the structure has not been built or the slice has the wrong length.
    pub fn intersect_bundle(&self, rays: &mut [Ray]) -> u32 {
        self.assert_built();
        let n = crate::bundle_rays();
        assert!(
            rays.len() == n,
            "tinybvh: a bundle holds exactly {n} rays, got {}",
            rays.len()
        );
        rays.iter().for_each(Ray::assert_finite);
        let steps = unsafe {
            ffi::tbvh_bvh4_cpu_intersect_bundle(
                self.raw.as_ptr(),
                rays.as_mut_ptr() as *mut ffi::tbvh_ray,
            )
        };
        steps.max(0) as u32
    }

    /// Occlusion test for a bundle of rays; `occluded[i]` receives the result for `rays[i]`,
    /// with the [`is_occluded`](Self::is_occluded) semantics. Both slices must have
    /// [`bundle_rays`](crate::bundle_rays) entries. Returns the traversal cost as reported by
    /// the kernel, which may be 0 in release builds.
    ///
    /// Panics if the structure has not been built or a slice has the wrong length.
    pub fn is_occluded_bundle(&self, rays: &mut [Ray], occluded: &mut [bool]) -> u32 {
        self.assert_built();
        let n = crate::bundle_rays();
        assert!(
            rays.len() == n && occluded.len() == n,
            "tinybvh: a bundle holds exactly {n} rays"
        );
        rays.iter().for_each(Ray::assert_finite);
        let steps = unsafe {
            ffi::tbvh_bvh4_cpu_is_occluded_bundle(
                self.raw.as_ptr(),
                rays.as_mut_ptr() as *mut ffi::tbvh_ray,
                occluded.as_mut_ptr(),
            )
        };
        steps.max(0) as u32
    }
}
