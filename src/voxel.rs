//! [`VoxelSet`]: a brick-based voxel object with DDA traversal, usable as a BLAS.

use core::ptr::NonNull;

use glam::Vec3;

use crate::ffi;
use crate::geometry;
use crate::instance::{sealed, Blas};
use crate::ray::Ray;

/// A 256³ voxel object (tinybvh's `VoxelSet`) occupying the unit cube `[0, 1]³`.
///
/// Voxels are stored in 8³ bricks under a two-level grid and traversed with a 3D DDA; a hit
/// reports the entry distance and the voxel's value in [`hit.prim`](crate::Intersection::prim).
/// Voxel `(x, y, z)` covers `[x / 256, (x + 1) / 256)` on each axis. Placing the object elsewhere,
/// or at another size, is done by instancing it in a [`Bvh`](crate::Bvh) TLAS with a transform.
///
/// After a batch of [`set`](Self::set) calls, [`update_top_grid`](Self::update_top_grid) must run
/// before tracing: the coarse grid the traversal starts from is rebuilt by that call, and voxels
/// added since are invisible until then. Queries on a stale top grid panic.
///
/// A `VoxelSet` can only be traversed with single precision rays.
///
/// # Example
///
/// ```
/// use std::sync::Arc;
/// use glam::{Mat4, Vec3};
/// use tinybvh::{Blas, BlasInstance, Bvh, Ray, VoxelSet};
///
/// // A solid floor slab: the lowest 16 voxel layers.
/// let mut voxels = VoxelSet::new();
/// for z in 0..256 {
///     for x in 0..256 {
///         for y in 0..16 {
///             voxels.set(x, y, z, 1);
///         }
///     }
/// }
/// voxels.update_top_grid();
///
/// // Object space is the unit cube; scale it to 10 units and shift it down.
/// let transform = Mat4::from_translation(Vec3::new(-5.0, -2.0, -5.0)) * Mat4::from_scale(Vec3::splat(10.0));
/// let blasses: Vec<Arc<dyn Blas>> = vec![Arc::new(voxels)];
/// let mut tlas = Bvh::new();
/// tlas.build_tlas(vec![BlasInstance::with_transform(0, transform)], blasses);
///
/// // Looking straight down from the origin: the slab's top is 16 / 256 * 10 - 2 = -1.375.
/// let mut ray = Ray::new(Vec3::ZERO, -Vec3::Y);
/// tlas.intersect(&mut ray);
/// assert!(ray.hit.is_hit());
/// assert!((ray.hit.t - 1.375).abs() < 1e-3);
/// assert_eq!(ray.hit.prim, 1); // the voxel value
/// ```
pub struct VoxelSet {
    raw: NonNull<ffi::tbvh_voxelset>,
    /// Set by `set`, cleared by `update_top_grid`.
    dirty: bool,
}

impl_base_accessors!(VoxelSet, tbvh_voxelset_base, tbvh_voxelset_base_const);

impl VoxelSet {
    /// An empty voxel object.
    pub fn new() -> Self {
        crate::install_fatal_trampoline();
        let raw = unsafe { ffi::tbvh_voxelset_new() };
        VoxelSet { raw: NonNull::new(raw).expect("tinybvh: allocation failed"), dirty: false }
    }

    /// Number of voxels along each axis (tinybvh's `VoxelSet::objectDim`, 256).
    pub fn object_dim() -> u32 {
        unsafe { ffi::tbvh_voxelset_object_dim() }
    }

    /// Sets voxel `(x, y, z)` to `value`; 0 clears it. Marks the top grid stale. Panics if a
    /// coordinate is not below [`object_dim`](Self::object_dim).
    pub fn set(&mut self, x: u32, y: u32, z: u32, value: u32) {
        let dim = Self::object_dim();
        assert!(x < dim && y < dim && z < dim, "tinybvh: voxel ({x}, {y}, {z}) is outside the {dim}^3 object");
        unsafe { ffi::tbvh_voxelset_set(self.raw.as_ptr(), x, y, z, value) };
        self.dirty = true;
    }

    /// Rebuilds the coarse grid the traversal starts from. Call it after a batch of
    /// [`set`](Self::set) calls and before tracing.
    pub fn update_top_grid(&mut self) {
        unsafe { ffi::tbvh_voxelset_update_top_grid(self.raw.as_ptr()) };
        self.dirty = false;
    }

    /// Whether voxels were changed since the last [`update_top_grid`](Self::update_top_grid).
    pub fn is_top_grid_stale(&self) -> bool {
        self.dirty
    }

    #[inline]
    fn assert_current(&self) {
        assert!(!self.dirty, "tinybvh: the VoxelSet was modified; call update_top_grid() before tracing");
    }

    /// Finds the closest voxel along `ray` and records its entry distance in `ray.hit.t` and
    /// its value in `ray.hit.prim`. Returns the number of voxel steps taken.
    ///
    /// Panics if the top grid is stale or the ray contains NaN.
    pub fn intersect(&self, ray: &mut Ray) -> u32 {
        self.assert_current();
        ray.assert_finite();
        let steps = unsafe { ffi::tbvh_voxelset_intersect(self.raw.as_ptr(), ray.as_ffi_mut()) };
        steps.max(0) as u32
    }

    /// Whether any voxel blocks `ray` before `ray.hit.t`. Panics like [`intersect`](Self::intersect).
    pub fn is_occluded(&self, ray: &Ray) -> bool {
        self.assert_current();
        ray.assert_finite();
        unsafe { ffi::tbvh_voxelset_is_occluded(self.raw.as_ptr(), ray.as_ffi()) }
    }

    /// The face normal of the voxel at the ray's hit point (`origin + direction * hit.t`), in
    /// object space: one of the six axis directions, facing against the ray. Only meaningful for
    /// a ray that [`intersect`](Self::intersect) reported a hit for.
    pub fn normal(&self, ray: &Ray) -> Vec3 {
        ray.assert_finite();
        geometry::vec3_from_ffi(unsafe { ffi::tbvh_voxelset_normal(self.raw.as_ptr(), ray.as_ffi()) })
    }
}

impl Default for VoxelSet {
    fn default() -> Self {
        VoxelSet::new()
    }
}

impl Drop for VoxelSet {
    fn drop(&mut self) {
        unsafe { ffi::tbvh_voxelset_free(self.raw.as_ptr()) };
    }
}

impl core::fmt::Debug for VoxelSet {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("VoxelSet").field("object_dim", &Self::object_dim()).field("top_grid_stale", &self.dirty).finish()
    }
}

// Queries take &self and only read the grids; `set` and `update_top_grid` take &mut self. The
// object owns all of its allocations.
unsafe impl Send for VoxelSet {}
unsafe impl Sync for VoxelSet {}

impl sealed::Sealed for VoxelSet {}
impl Blas for VoxelSet {
    fn blas_base_ptr(&self) -> *const ffi::tbvh_base {
        self.base_ptr()
    }

    /// A voxel object is always traversable (an empty one simply reports no hits), so this is
    /// always `true`; a stale top grid is caught by the queries.
    fn blas_is_ready(&self) -> bool {
        true
    }
}
