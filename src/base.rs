//! Accessors shared by every layout, generated from the `tbvh_base` / `tbvh_base_ex` C functions.

/// Generates the methods that every single precision layout exposes through its `BVHBase`.
///
/// The target type must have a `raw` field holding the handle, and `$base` / `$base_const` name
/// the C functions that upcast the handle.
macro_rules! impl_base_accessors {
    ($ty:ident, $base:ident, $base_const:ident) => {
        impl $ty {
            #[inline]
            pub(crate) fn base_ptr(&self) -> *const $crate::ffi::tbvh_base {
                unsafe { $crate::ffi::$base_const(self.raw.as_ptr()) }
            }

            #[inline]
            pub(crate) fn base_ptr_mut(&mut self) -> *mut $crate::ffi::tbvh_base {
                unsafe { $crate::ffi::$base(self.raw.as_ptr()) }
            }

            /// The memory layout of this structure.
            pub fn layout(&self) -> $crate::Layout {
                $crate::Layout::from_raw(unsafe { $crate::ffi::tbvh_base_layout(self.base_ptr()) })
            }

            /// The build settings that the next build will use.
            pub fn settings(&self) -> $crate::BuildSettings {
                let mut raw = $crate::BuildSettings::default().to_ffi();
                unsafe { $crate::ffi::tbvh_base_get_settings(self.base_ptr(), &mut raw) };
                $crate::BuildSettings::from_ffi(&raw)
            }

            /// Replaces the build settings. Takes effect at the next build.
            pub fn set_settings(&mut self, settings: &$crate::BuildSettings) {
                let raw = settings.to_ffi();
                unsafe { $crate::ffi::tbvh_base_set_settings(self.base_ptr_mut(), &raw) };
            }

            /// Builder-style [`set_settings`](Self::set_settings).
            pub fn with_settings(mut self, settings: &$crate::BuildSettings) -> Self {
                self.set_settings(settings);
                self
            }

            /// Estimated cost of a traversal step, used to steer SAH construction (default 1).
            pub fn c_trav(&self) -> f32 {
                unsafe { $crate::ffi::tbvh_base_get_c_trav(self.base_ptr()) }
            }

            /// Estimated cost of a primitive intersection, used to steer SAH construction.
            pub fn c_int(&self) -> f32 {
                unsafe { $crate::ffi::tbvh_base_get_c_int(self.base_ptr()) }
            }

            /// Sets the SAH traversal step cost.
            pub fn set_c_trav(&mut self, cost: f32) {
                unsafe { $crate::ffi::tbvh_base_set_c_trav(self.base_ptr_mut(), cost) }
            }

            /// Sets the SAH primitive intersection cost.
            pub fn set_c_int(&mut self, cost: f32) {
                unsafe { $crate::ffi::tbvh_base_set_c_int(self.base_ptr_mut(), cost) }
            }

            /// Number of bins used by the high quality (spatial split) builders.
            pub fn hq_bins(&self) -> u32 {
                unsafe { $crate::ffi::tbvh_base_get_hq_bins(self.base_ptr()) }
            }

            /// Sets the number of bins for the high quality builders, clamped to `1..=128`.
            pub fn set_hq_bins(&mut self, bins: u32) {
                unsafe { $crate::ffi::tbvh_base_set_hq_bins(self.base_ptr_mut(), bins.clamp(1, 128)) }
            }

            /// Whether odd tree levels use one extra bin in the high quality builders.
            pub fn hq_odd_even(&self) -> bool {
                unsafe { $crate::ffi::tbvh_base_get_hq_odd_even(self.base_ptr()) }
            }

            /// Sets whether odd tree levels use one extra bin in the high quality builders.
            pub fn set_hq_odd_even(&mut self, odd_even: bool) {
                unsafe { $crate::ffi::tbvh_base_set_hq_odd_even(self.base_ptr_mut(), odd_even) }
            }

            /// SBVH builder: nodes with at most this many primitives become leaves without being
            /// split; 0 leaves the decision to the SAH.
            pub fn hq_leaf_prims(&self) -> u32 {
                unsafe { $crate::ffi::tbvh_base_get_hq_leaf_prims(self.base_ptr()) }
            }

            /// Sets the SBVH leaf primitive threshold; see [`hq_leaf_prims`](Self::hq_leaf_prims).
            pub fn set_hq_leaf_prims(&mut self, prims: u32) {
                unsafe { $crate::ffi::tbvh_base_set_hq_leaf_prims(self.base_ptr_mut(), prims) }
            }

            /// Bounds of the root node, as `(min, max)`. Only meaningful after a build.
            pub fn aabb(&self) -> (glam::Vec3, glam::Vec3) {
                unsafe {
                    (
                        $crate::geometry::vec3_from_ffi($crate::ffi::tbvh_base_aabb_min(self.base_ptr())),
                        $crate::geometry::vec3_from_ffi($crate::ffi::tbvh_base_aabb_max(self.base_ptr())),
                    )
                }
            }

            /// Number of primitives (triangles, custom primitives or instances) in the structure.
            pub fn prim_count(&self) -> u32 {
                unsafe { $crate::ffi::tbvh_base_tri_count(self.base_ptr()) }
            }

            /// Number of primitive index entries; can exceed [`prim_count`](Self::prim_count)
            /// after a build with spatial splits.
            pub fn index_count(&self) -> u32 {
                unsafe { $crate::ffi::tbvh_base_idx_count(self.base_ptr()) }
            }

            /// Number of nodes in use.
            pub fn used_nodes(&self) -> u32 {
                unsafe { $crate::ffi::tbvh_base_used_nodes(self.base_ptr()) }
            }

            /// Number of nodes allocated.
            pub fn allocated_nodes(&self) -> u32 {
                unsafe { $crate::ffi::tbvh_base_allocated_nodes(self.base_ptr()) }
            }

            /// Whether the tree can be refitted (no spatial splits or presplitting).
            pub fn is_refittable(&self) -> bool {
                unsafe { $crate::ffi::tbvh_base_is_refittable(self.base_ptr()) }
            }

            /// Whether the tree can be refitted or optimized in place; false after a conversion
            /// from an external tree.
            pub fn is_rebuildable(&self) -> bool {
                unsafe { $crate::ffi::tbvh_base_is_rebuildable(self.base_ptr()) }
            }

            /// Whether the node array may contain unused entries (threaded builds, leaf merging).
            pub fn may_have_holes(&self) -> bool {
                unsafe { $crate::ffi::tbvh_base_may_have_holes(self.base_ptr()) }
            }

            /// Whether nodes are in depth-first order: children follow their parent and the
            /// primitives of a subtree are contiguous.
            pub fn has_ordered_nodes(&self) -> bool {
                unsafe { $crate::ffi::tbvh_base_has_ordered_nodes(self.base_ptr()) }
            }

            /// Whether the tree was built over boxes rather than triangles.
            pub fn is_over_aabbs(&self) -> bool {
                unsafe { $crate::ffi::tbvh_base_is_over_aabbs(self.base_ptr()) }
            }

            /// Whether the tree was built over indexed triangles.
            pub fn is_over_indices(&self) -> bool {
                unsafe { $crate::ffi::tbvh_base_is_over_indices(self.base_ptr()) }
            }
        }
    };
}

/// Double precision counterpart of [`impl_base_accessors`].
macro_rules! impl_base_accessors_ex {
    ($ty:ident, $base:ident, $base_const:ident) => {
        impl $ty {
            #[inline]
            pub(crate) fn base_ptr(&self) -> *const $crate::ffi::tbvh_base_ex {
                unsafe { $crate::ffi::$base_const(self.raw.as_ptr()) }
            }

            #[inline]
            pub(crate) fn base_ptr_mut(&mut self) -> *mut $crate::ffi::tbvh_base_ex {
                unsafe { $crate::ffi::$base(self.raw.as_ptr()) }
            }

            /// The memory layout of this structure.
            pub fn layout(&self) -> $crate::Layout {
                $crate::Layout::from_raw(unsafe { $crate::ffi::tbvh_base_ex_layout(self.base_ptr()) })
            }

            /// The build settings that the next build will use.
            pub fn settings(&self) -> $crate::BuildSettings {
                let mut raw = $crate::BuildSettings::default().to_ffi();
                unsafe { $crate::ffi::tbvh_base_ex_get_settings(self.base_ptr(), &mut raw) };
                $crate::BuildSettings::from_ffi(&raw)
            }

            /// Replaces the build settings. Takes effect at the next build.
            pub fn set_settings(&mut self, settings: &$crate::BuildSettings) {
                let raw = settings.to_ffi();
                unsafe { $crate::ffi::tbvh_base_ex_set_settings(self.base_ptr_mut(), &raw) };
            }

            /// Builder-style [`set_settings`](Self::set_settings).
            pub fn with_settings(mut self, settings: &$crate::BuildSettings) -> Self {
                self.set_settings(settings);
                self
            }

            /// Estimated cost of a traversal step, used to steer SAH construction.
            pub fn c_trav(&self) -> f32 {
                unsafe { $crate::ffi::tbvh_base_ex_get_c_trav(self.base_ptr()) }
            }

            /// Estimated cost of a primitive intersection, used to steer SAH construction.
            pub fn c_int(&self) -> f32 {
                unsafe { $crate::ffi::tbvh_base_ex_get_c_int(self.base_ptr()) }
            }

            /// Sets the SAH traversal step cost.
            pub fn set_c_trav(&mut self, cost: f32) {
                unsafe { $crate::ffi::tbvh_base_ex_set_c_trav(self.base_ptr_mut(), cost) }
            }

            /// Sets the SAH primitive intersection cost.
            pub fn set_c_int(&mut self, cost: f32) {
                unsafe { $crate::ffi::tbvh_base_ex_set_c_int(self.base_ptr_mut(), cost) }
            }

            /// Number of bins used by the high quality builders.
            pub fn hq_bins(&self) -> u32 {
                unsafe { $crate::ffi::tbvh_base_ex_get_hq_bins(self.base_ptr()) }
            }

            /// Sets the number of bins for the high quality builders, clamped to `1..=128`.
            pub fn set_hq_bins(&mut self, bins: u32) {
                unsafe { $crate::ffi::tbvh_base_ex_set_hq_bins(self.base_ptr_mut(), bins.clamp(1, 128)) }
            }

            /// Whether odd tree levels use one extra bin in the high quality builders.
            pub fn hq_odd_even(&self) -> bool {
                unsafe { $crate::ffi::tbvh_base_ex_get_hq_odd_even(self.base_ptr()) }
            }

            /// Sets whether odd tree levels use one extra bin in the high quality builders.
            pub fn set_hq_odd_even(&mut self, odd_even: bool) {
                unsafe { $crate::ffi::tbvh_base_ex_set_hq_odd_even(self.base_ptr_mut(), odd_even) }
            }

            /// SBVH leaf primitive threshold; 0 leaves the decision to the SAH.
            pub fn hq_leaf_prims(&self) -> u32 {
                unsafe { $crate::ffi::tbvh_base_ex_get_hq_leaf_prims(self.base_ptr()) }
            }

            /// Sets the SBVH leaf primitive threshold.
            pub fn set_hq_leaf_prims(&mut self, prims: u32) {
                unsafe { $crate::ffi::tbvh_base_ex_set_hq_leaf_prims(self.base_ptr_mut(), prims) }
            }

            /// Bounds of the root node, as `(min, max)`. Only meaningful after a build.
            pub fn aabb(&self) -> (glam::DVec3, glam::DVec3) {
                unsafe {
                    (
                        $crate::geometry::dvec3_from_ffi($crate::ffi::tbvh_base_ex_aabb_min(self.base_ptr())),
                        $crate::geometry::dvec3_from_ffi($crate::ffi::tbvh_base_ex_aabb_max(self.base_ptr())),
                    )
                }
            }

            /// Number of primitives in the structure.
            pub fn prim_count(&self) -> u64 {
                unsafe { $crate::ffi::tbvh_base_ex_tri_count(self.base_ptr()) }
            }

            /// Number of primitive index entries.
            pub fn index_count(&self) -> u64 {
                unsafe { $crate::ffi::tbvh_base_ex_idx_count(self.base_ptr()) }
            }

            /// Number of nodes in use.
            pub fn used_nodes(&self) -> u64 {
                unsafe { $crate::ffi::tbvh_base_ex_used_nodes(self.base_ptr()) }
            }

            /// Number of nodes allocated.
            pub fn allocated_nodes(&self) -> u64 {
                unsafe { $crate::ffi::tbvh_base_ex_allocated_nodes(self.base_ptr()) }
            }

            /// Whether the tree can be refitted.
            pub fn is_refittable(&self) -> bool {
                unsafe { $crate::ffi::tbvh_base_ex_is_refittable(self.base_ptr()) }
            }

            /// Whether the tree can be refitted or optimized in place.
            pub fn is_rebuildable(&self) -> bool {
                unsafe { $crate::ffi::tbvh_base_ex_is_rebuildable(self.base_ptr()) }
            }

            /// Whether the node array may contain unused entries.
            pub fn may_have_holes(&self) -> bool {
                unsafe { $crate::ffi::tbvh_base_ex_may_have_holes(self.base_ptr()) }
            }

            /// Whether nodes are in depth-first order.
            pub fn has_ordered_nodes(&self) -> bool {
                unsafe { $crate::ffi::tbvh_base_ex_has_ordered_nodes(self.base_ptr()) }
            }

            /// Whether the tree was built over boxes rather than triangles.
            pub fn is_over_aabbs(&self) -> bool {
                unsafe { $crate::ffi::tbvh_base_ex_is_over_aabbs(self.base_ptr()) }
            }

            /// Whether the tree was built over indexed triangles.
            pub fn is_over_indices(&self) -> bool {
                unsafe { $crate::ffi::tbvh_base_ex_is_over_indices(self.base_ptr()) }
            }
        }
    };
}

/// Generates opacity micro map support for a single precision layout that honours it.
/// The target type must have an `opacity_map: Option<Vec<u32>>` field.
macro_rules! impl_opacity_maps {
    ($ty:ident) => {
        impl $ty {
            /// Attaches an opacity micro map: `n * n` bits per triangle, `(n * n + 31) / 32`
            /// `u32` words per triangle, in triangle order. Ray hits on micro-triangles whose bit
            /// is clear are ignored. The map must cover every triangle of the current build.
            pub fn set_opacity_micro_maps(&mut self, map: Vec<u32>, n: u32) {
                $crate::geometry::validate_opacity_map(&map, n, self.prim_count() as u64);
                self.opacity_map = Some(map);
                let ptr = self.opacity_map.as_ref().map(|m| m.as_ptr()).unwrap_or(core::ptr::null());
                unsafe { $crate::ffi::tbvh_base_set_opacity_micro_maps(self.base_ptr_mut(), ptr, n) };
            }

            /// Detaches the opacity micro map, if any.
            pub fn clear_opacity_micro_maps(&mut self) {
                unsafe { $crate::ffi::tbvh_base_set_opacity_micro_maps(self.base_ptr_mut(), core::ptr::null(), 0) };
                self.opacity_map = None;
            }

            /// Subdivision level of the attached opacity micro map; 0 when none is attached.
            pub fn opacity_micro_map_n(&self) -> u32 {
                unsafe { $crate::ffi::tbvh_base_opacity_micro_map_n(self.base_ptr()) }
            }
        }
    };
}

