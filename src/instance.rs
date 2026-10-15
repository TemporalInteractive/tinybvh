//! BLAS instances for top-level acceleration structures, and the traits that mark which layouts
//! can serve as a bottom-level structure.

use core::fmt;

use glam::{DMat4, DVec3, Mat4, Vec3};

use crate::ffi;
use crate::ray::{DBL_FAR, FAR, MASK_ALL};

/// Hidden implementation detail of [`Blas`] and [`BlasEx`]; prevents implementations outside
/// this crate, which could hand tinybvh an arbitrary pointer.
pub mod sealed {
    /// Marker trait for the layouts that tinybvh can traverse from a TLAS.
    pub trait Sealed {}
}

/// A single precision layout that can be instanced in a [`Bvh`](crate::Bvh) TLAS.
///
/// tinybvh dispatches on the layout at run time when a ray enters an instance, and supports
/// [`Bvh`](crate::Bvh), [`Bvh4Cpu`](crate::Bvh4Cpu), [`Bvh8Cpu`](crate::Bvh8Cpu) and
/// [`VoxelSet`](crate::VoxelSet). The trait is sealed.
pub trait Blas: sealed::Sealed + Send + Sync {
    /// Pointer to the underlying `BVHBase`, for the TLAS builder.
    #[doc(hidden)]
    fn blas_base_ptr(&self) -> *const ffi::tbvh_base;
    /// Whether the structure holds a traversable tree.
    #[doc(hidden)]
    fn blas_is_ready(&self) -> bool;
}

/// A double precision layout that can be instanced in a [`BvhDouble`](crate::BvhDouble) TLAS:
/// [`BvhDouble`](crate::BvhDouble) and [`Bvh4Double`](crate::Bvh4Double). The trait is sealed.
pub trait BlasEx: sealed::Sealed + Send + Sync {
    /// Pointer to the underlying `BVHBaseEx`, for the TLAS builder.
    #[doc(hidden)]
    fn blas_base_ptr(&self) -> *const ffi::tbvh_base_ex;
    /// Whether the structure holds a traversable tree.
    #[doc(hidden)]
    fn blas_is_ready(&self) -> bool;
}

/// An instance of a bottom-level acceleration structure (tinybvh's `BLASInstance`): which BLAS,
/// an object-to-world transform, and a 16-bit mask.
///
/// The world-space bounds and the inverse transform are computed by the TLAS build, so after
/// changing the transform the TLAS must be rebuilt (see [`Bvh::rebuild_tlas`](crate::Bvh::rebuild_tlas)).
///
/// The struct is 192 bytes, 64-byte aligned, and stored by the TLAS exactly as tinybvh expects.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct BlasInstance {
    transform: [f32; 16],
    inv_transform: [f32; 16],
    aabb_min: Vec3,
    /// Index into the BLAS list the TLAS was built over.
    pub blas_idx: u32,
    aabb_max: Vec3,
    /// 16-bit mask matched against [`Ray::mask`](crate::Ray::mask); the instance is skipped when
    /// the masks do not overlap.
    pub mask: u32,
}

const IDENTITY: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
];
const IDENTITY_D: [f64; 16] = [
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
];

impl BlasInstance {
    /// An instance of BLAS `blas_idx` with the identity transform and a mask that matches all rays.
    pub fn new(blas_idx: u32) -> Self {
        BlasInstance {
            transform: IDENTITY,
            inv_transform: IDENTITY,
            aabb_min: Vec3::splat(FAR),
            blas_idx,
            aabb_max: Vec3::splat(-FAR),
            mask: MASK_ALL,
        }
    }

    /// An instance with the given object-to-world transform.
    pub fn with_transform(blas_idx: u32, transform: Mat4) -> Self {
        let mut inst = BlasInstance::new(blas_idx);
        inst.set_transform(transform);
        inst
    }

    /// Sets the mask (low 16 bits) and returns the instance.
    pub fn with_mask(mut self, mask: u32) -> Self {
        self.mask = mask & MASK_ALL;
        self
    }

    /// The object-to-world transform.
    pub fn transform(&self) -> Mat4 {
        Mat4::from_cols_array(&self.transform).transpose()
    }

    /// Sets the object-to-world transform. The inverse and the world bounds are refreshed by the
    /// next TLAS build or by [`update`](Self::update).
    pub fn set_transform(&mut self, transform: Mat4) {
        self.transform = transform.transpose().to_cols_array();
    }

    /// The world-to-object transform, valid after a TLAS build, [`update`](Self::update) or
    /// [`invert_transform`](Self::invert_transform).
    pub fn inverse_transform(&self) -> Mat4 {
        Mat4::from_cols_array(&self.inv_transform).transpose()
    }

    /// World-space bounds of the instance, valid after a TLAS build or [`update`](Self::update).
    pub fn aabb(&self) -> (Vec3, Vec3) {
        (self.aabb_min, self.aabb_max)
    }

    /// Recomputes the inverse transform from the transform.
    pub fn invert_transform(&mut self) {
        unsafe { ffi::tbvh_blas_instance_invert_transform(self.as_ffi_mut()) }
    }

    /// Recomputes the inverse transform and the world-space bounds from the transform and the
    /// bounds of `blas`, as the TLAS build does.
    pub fn update(&mut self, blas: &dyn Blas) {
        unsafe { ffi::tbvh_blas_instance_update(self.as_ffi_mut(), blas.blas_base_ptr()) }
    }

    #[inline]
    pub(crate) fn as_ffi_mut(&mut self) -> *mut ffi::tbvh_blas_instance {
        self as *mut BlasInstance as *mut ffi::tbvh_blas_instance
    }
}

impl Default for BlasInstance {
    fn default() -> Self {
        BlasInstance::new(0)
    }
}

impl fmt::Debug for BlasInstance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BlasInstance")
            .field("blas_idx", &self.blas_idx)
            .field("mask", &self.mask)
            .field("transform", &self.transform())
            .field("aabb_min", &self.aabb_min)
            .field("aabb_max", &self.aabb_max)
            .finish()
    }
}

/// A double precision BLAS instance (tinybvh's `BLASInstanceEx`), 320 bytes, 64-byte aligned.
/// See [`BlasInstance`].
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct BlasInstanceEx {
    transform: [f64; 16],
    inv_transform: [f64; 16],
    aabb_min: DVec3,
    /// Index into the BLAS list the TLAS was built over.
    pub blas_idx: u64,
    aabb_max: DVec3,
    /// 16-bit mask matched against [`RayEx::mask`](crate::RayEx::mask).
    pub mask: u32,
}

impl BlasInstanceEx {
    /// An instance of BLAS `blas_idx` with the identity transform and a mask that matches all rays.
    pub fn new(blas_idx: u64) -> Self {
        BlasInstanceEx {
            transform: IDENTITY_D,
            inv_transform: IDENTITY_D,
            aabb_min: DVec3::splat(DBL_FAR),
            blas_idx,
            aabb_max: DVec3::splat(-DBL_FAR),
            mask: MASK_ALL,
        }
    }

    /// An instance with the given object-to-world transform.
    pub fn with_transform(blas_idx: u64, transform: DMat4) -> Self {
        let mut inst = BlasInstanceEx::new(blas_idx);
        inst.set_transform(transform);
        inst
    }

    /// Sets the mask (low 16 bits) and returns the instance.
    pub fn with_mask(mut self, mask: u32) -> Self {
        self.mask = mask & MASK_ALL;
        self
    }

    /// The object-to-world transform.
    pub fn transform(&self) -> DMat4 {
        DMat4::from_cols_array(&self.transform).transpose()
    }

    /// Sets the object-to-world transform.
    pub fn set_transform(&mut self, transform: DMat4) {
        self.transform = transform.transpose().to_cols_array();
    }

    /// The world-to-object transform, valid after a TLAS build or [`update`](Self::update).
    pub fn inverse_transform(&self) -> DMat4 {
        DMat4::from_cols_array(&self.inv_transform).transpose()
    }

    /// World-space bounds, valid after a TLAS build or [`update`](Self::update).
    pub fn aabb(&self) -> (DVec3, DVec3) {
        (self.aabb_min, self.aabb_max)
    }

    /// Recomputes the inverse transform from the transform.
    pub fn invert_transform(&mut self) {
        unsafe { ffi::tbvh_blas_instance_ex_invert_transform(self.as_ffi_mut()) }
    }

    /// Recomputes the inverse transform and the world-space bounds from the bounds of `blas`.
    pub fn update(&mut self, blas: &dyn BlasEx) {
        unsafe { ffi::tbvh_blas_instance_ex_update(self.as_ffi_mut(), blas.blas_base_ptr()) }
    }

    #[inline]
    pub(crate) fn as_ffi_mut(&mut self) -> *mut ffi::tbvh_blas_instance_ex {
        self as *mut BlasInstanceEx as *mut ffi::tbvh_blas_instance_ex
    }
}

impl Default for BlasInstanceEx {
    fn default() -> Self {
        BlasInstanceEx::new(0)
    }
}

impl fmt::Debug for BlasInstanceEx {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BlasInstanceEx")
            .field("blas_idx", &self.blas_idx)
            .field("mask", &self.mask)
            .field("transform", &self.transform())
            .field("aabb_min", &self.aabb_min)
            .field("aabb_max", &self.aabb_max)
            .finish()
    }
}

const _: () = {
    use core::mem::{align_of, size_of};
    assert!(size_of::<BlasInstance>() == size_of::<ffi::tbvh_blas_instance>());
    assert!(align_of::<BlasInstance>() == align_of::<ffi::tbvh_blas_instance>());
    assert!(core::mem::offset_of!(BlasInstance, blas_idx) == 140);
    assert!(core::mem::offset_of!(BlasInstance, mask) == 156);
    assert!(size_of::<BlasInstanceEx>() == size_of::<ffi::tbvh_blas_instance_ex>());
    assert!(align_of::<BlasInstanceEx>() == align_of::<ffi::tbvh_blas_instance_ex>());
    assert!(core::mem::offset_of!(BlasInstanceEx, blas_idx) == 280);
    assert!(core::mem::offset_of!(BlasInstanceEx, mask) == 312);
};
