//! Rays and hit records, in single and double precision.

use core::ffi::c_void;

use glam::{DVec3, Vec3};

use crate::ffi;

/// The 'no hit' distance of a single precision ray (tinybvh's `BVH_FAR`).
pub const FAR: f32 = 1e30;
/// The 'no hit' distance of a double precision ray (tinybvh's `BVH_DBL_FAR`).
pub const DBL_FAR: f64 = 1e300;
/// Ray / instance mask that matches everything (16 bits set).
pub const MASK_ALL: u32 = 0xFFFF;

/// Reciprocal of a zero direction component (tinybvh's `BVH_RCP_FAR`, 2^100).
const RCP_FAR: f32 = 1.2676506e30;
/// Double precision counterpart (`BVH_DBL_RCP_FAR`, 2^1000).
const DBL_RCP_FAR: f64 = 1.0715086071862673e301;

#[inline]
fn safe_rcp(x: f32) -> f32 {
    let r = 1.0 / x;
    // NaN (0 * inf) also takes this branch, as in tinybvh.
    if r.is_nan() || r.abs() > RCP_FAR {
        RCP_FAR.copysign(x)
    } else {
        r
    }
}

#[inline]
fn safe_rcp_d(x: f64) -> f64 {
    let r = 1.0 / x;
    if r.is_nan() || r.abs() > DBL_RCP_FAR {
        DBL_RCP_FAR.copysign(x)
    } else {
        r
    }
}

/// Result of a closest-hit query (tinybvh's `Intersection`).
///
/// A fresh ray has `t` set to its maximum distance ([`FAR`] by default); a hit lowers `t` and fills
/// in `prim`, `u`, `v` and, for TLAS traversal, `inst`. With the primitive index and the original
/// triangle data, every other shading attribute can be reconstructed.
///
/// The record is 84 bytes with 4-byte alignment, exactly as in tinybvh, and sits at offset 44 of a
/// [`Ray`].
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Intersection {
    /// Index of the [`BlasInstance`](crate::BlasInstance) that was hit, for TLAS traversal.
    /// Zero for a plain BLAS query.
    pub inst: u32,
    /// Distance along the ray, in units of the (not necessarily normalized) direction.
    pub t: f32,
    /// First barycentric coordinate of the hit on the triangle.
    pub u: f32,
    /// Second barycentric coordinate of the hit on the triangle.
    pub v: f32,
    /// Index of the primitive (triangle or custom primitive) that was hit.
    pub prim: u32,
    aux_data: [u8; ffi::PTR_SIZE],
    /// 56 bytes of payload that tinybvh never touches. Custom intersection callbacks can store
    /// shading data here, for example.
    pub user: [u32; 14],
}

impl Intersection {
    /// A record that has not recorded any hit, with `t` set to `t_max`.
    #[inline]
    pub fn none(t_max: f32) -> Self {
        Intersection { inst: 0, t: t_max, u: 0.0, v: 0.0, prim: 0, aux_data: [0; ffi::PTR_SIZE], user: [0; 14] }
    }

    /// True when `t` is below [`FAR`], which is how tinybvh signals a hit for a ray that started
    /// with the default maximum distance. For a ray created with a custom maximum distance,
    /// compare `t` against that distance instead.
    #[inline]
    pub fn is_hit(&self) -> bool {
        self.t < FAR
    }

    /// The user pointer stored in the record. tinybvh never reads or writes it.
    #[inline]
    pub fn aux_data(&self) -> *mut c_void {
        usize::from_ne_bytes(self.aux_data) as *mut c_void
    }

    /// Stores a user pointer in the record.
    #[inline]
    pub fn set_aux_data(&mut self, ptr: *mut c_void) {
        self.aux_data = (ptr as usize).to_ne_bytes();
    }
}

impl Default for Intersection {
    fn default() -> Self {
        Intersection::none(FAR)
    }
}

/// A single precision ray (tinybvh's `Ray`): origin, direction, reciprocal direction, a 16-bit
/// instance mask and the hit record.
///
/// The direction is not normalized by the constructor, so `hit.t` is expressed in units of the
/// direction length; call [`normalize`](Ray::normalize) if you need world units. The reciprocal
/// direction is maintained by the setters, which is why `direction` is not a public field.
///
/// `Ray` is 128 bytes, aligned to 64 bytes, and can be passed to the C library as is.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct Ray {
    /// Ray origin.
    pub origin: Vec3,
    /// 16-bit mask matched against [`BlasInstance::mask`](crate::BlasInstance::mask) during
    /// TLAS traversal; an instance is skipped when the masks do not overlap.
    pub mask: u32,
    direction: Vec3,
    inst_idx: u32,
    rcp_direction: Vec3,
    /// The hit record. Initially 'no hit' with `t` at the maximum distance.
    pub hit: Intersection,
}

impl Ray {
    /// A ray with the default maximum distance ([`FAR`]) and a mask that matches every instance.
    #[inline]
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Ray::with_t_max(origin, direction, FAR)
    }

    /// A ray that only reports hits closer than `t_max` (in units of the direction length).
    #[inline]
    pub fn with_t_max(origin: Vec3, direction: Vec3, t_max: f32) -> Self {
        Ray {
            origin,
            mask: MASK_ALL,
            direction,
            inst_idx: 0,
            rcp_direction: Vec3::new(safe_rcp(direction.x), safe_rcp(direction.y), safe_rcp(direction.z)),
            hit: Intersection::none(t_max),
        }
    }

    /// Sets the instance mask (only the low 16 bits are used) and returns the ray.
    #[inline]
    pub fn with_mask(mut self, mask: u32) -> Self {
        self.mask = mask & MASK_ALL;
        self
    }

    /// The ray direction.
    #[inline]
    pub fn direction(&self) -> Vec3 {
        self.direction
    }

    /// The component-wise reciprocal of the direction, as used by the slab test.
    #[inline]
    pub fn rcp_direction(&self) -> Vec3 {
        self.rcp_direction
    }

    /// Replaces the direction and updates the reciprocal direction.
    #[inline]
    pub fn set_direction(&mut self, direction: Vec3) {
        self.direction = direction;
        self.rcp_direction = Vec3::new(safe_rcp(direction.x), safe_rcp(direction.y), safe_rcp(direction.z));
    }

    /// Normalizes the direction (and updates the reciprocal), so that `hit.t` is a world distance.
    #[inline]
    pub fn normalize(&mut self) {
        let len = self.direction.length();
        if len > 0.0 {
            self.set_direction(self.direction / len);
        }
    }

    /// Index of the instance currently being traversed. Only meaningful inside custom geometry
    /// callbacks invoked through a TLAS; the final result is in [`hit.inst`](Intersection::inst).
    #[inline]
    pub fn inst_idx(&self) -> u32 {
        self.inst_idx
    }

    /// Maximum hit distance: the current `hit.t`.
    #[inline]
    pub fn t_max(&self) -> f32 {
        self.hit.t
    }

    /// The point `origin + direction * hit.t`.
    #[inline]
    pub fn hit_point(&self) -> Vec3 {
        self.origin + self.direction * self.hit.t
    }

    /// Clears the hit record and resets the maximum distance, keeping origin, direction and mask.
    #[inline]
    pub fn reset(&mut self, t_max: f32) {
        self.hit = Intersection::none(t_max);
        self.inst_idx = 0;
    }

    /// Panics when origin, direction or `hit.t` contain NaN: tinybvh treats such a ray as a fatal
    /// error and terminates the process, which is far less debuggable than a panic.
    #[inline]
    pub(crate) fn assert_finite(&self) {
        let probe = self.origin.x + self.origin.y + self.origin.z + self.direction.x + self.direction.y + self.direction.z + self.hit.t;
        assert!(!probe.is_nan(), "tinybvh: ray contains NaN components");
    }

    #[inline]
    pub(crate) fn as_ffi(&self) -> *const ffi::tbvh_ray {
        self as *const Ray as *const ffi::tbvh_ray
    }

    #[inline]
    pub(crate) fn as_ffi_mut(&mut self) -> *mut ffi::tbvh_ray {
        self as *mut Ray as *mut ffi::tbvh_ray
    }
}

// The aux pointer is an opaque user value that this crate neither dereferences nor hands out
// implicitly; a Ray is otherwise plain data.
unsafe impl Send for Ray {}
unsafe impl Sync for Ray {}
unsafe impl Send for Intersection {}
unsafe impl Sync for Intersection {}

/// Result of a double precision closest-hit query (tinybvh's `IntersectionEx`), 104 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct IntersectionEx {
    /// Index of the [`BlasInstanceEx`](crate::BlasInstanceEx) that was hit, for TLAS traversal.
    pub inst: u64,
    /// Distance along the ray, in units of the direction length.
    pub t: f64,
    /// First barycentric coordinate.
    pub u: f64,
    /// Second barycentric coordinate.
    pub v: f64,
    /// Index of the primitive that was hit.
    pub prim: u64,
    aux_data: [u8; ffi::PTR_SIZE],
    /// 56 bytes of user payload that tinybvh never touches.
    pub user: [u32; 14],
}

impl IntersectionEx {
    /// A record without a hit, with `t` set to `t_max`.
    #[inline]
    pub fn none(t_max: f64) -> Self {
        IntersectionEx { inst: 0, t: t_max, u: 0.0, v: 0.0, prim: 0, aux_data: [0; ffi::PTR_SIZE], user: [0; 14] }
    }

    /// True when `t` is below [`DBL_FAR`]; see [`Intersection::is_hit`] for the caveat.
    #[inline]
    pub fn is_hit(&self) -> bool {
        self.t < DBL_FAR
    }

    /// The user pointer stored in the record.
    #[inline]
    pub fn aux_data(&self) -> *mut c_void {
        usize::from_ne_bytes(self.aux_data) as *mut c_void
    }

    /// Stores a user pointer in the record.
    #[inline]
    pub fn set_aux_data(&mut self, ptr: *mut c_void) {
        self.aux_data = (ptr as usize).to_ne_bytes();
    }
}

impl Default for IntersectionEx {
    fn default() -> Self {
        IntersectionEx::none(DBL_FAR)
    }
}

/// A double precision ray (tinybvh's `RayEx`), 192 bytes, 64-byte aligned.
///
/// See [`Ray`] for the semantics; this is the same structure with `f64` components and 64-bit
/// indices, for use with [`BvhDouble`](crate::BvhDouble) and [`Bvh4Double`](crate::Bvh4Double).
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct RayEx {
    /// Ray origin.
    pub origin: DVec3,
    /// 16-bit instance mask.
    pub mask: u32,
    _pad0: u32,
    direction: DVec3,
    inst_idx: u64,
    rcp_direction: DVec3,
    /// The hit record.
    pub hit: IntersectionEx,
}

impl RayEx {
    /// A ray with the default maximum distance ([`DBL_FAR`]) and a mask that matches everything.
    #[inline]
    pub fn new(origin: DVec3, direction: DVec3) -> Self {
        RayEx::with_t_max(origin, direction, DBL_FAR)
    }

    /// A ray that only reports hits closer than `t_max`.
    #[inline]
    pub fn with_t_max(origin: DVec3, direction: DVec3, t_max: f64) -> Self {
        RayEx {
            origin,
            mask: MASK_ALL,
            _pad0: 0,
            direction,
            inst_idx: 0,
            rcp_direction: DVec3::new(safe_rcp_d(direction.x), safe_rcp_d(direction.y), safe_rcp_d(direction.z)),
            hit: IntersectionEx::none(t_max),
        }
    }

    /// Sets the instance mask (low 16 bits) and returns the ray.
    #[inline]
    pub fn with_mask(mut self, mask: u32) -> Self {
        self.mask = mask & MASK_ALL;
        self
    }

    /// The ray direction.
    #[inline]
    pub fn direction(&self) -> DVec3 {
        self.direction
    }

    /// The component-wise reciprocal of the direction.
    #[inline]
    pub fn rcp_direction(&self) -> DVec3 {
        self.rcp_direction
    }

    /// Replaces the direction and updates the reciprocal direction.
    #[inline]
    pub fn set_direction(&mut self, direction: DVec3) {
        self.direction = direction;
        self.rcp_direction = DVec3::new(safe_rcp_d(direction.x), safe_rcp_d(direction.y), safe_rcp_d(direction.z));
    }

    /// Normalizes the direction.
    #[inline]
    pub fn normalize(&mut self) {
        let len = self.direction.length();
        if len > 0.0 {
            self.set_direction(self.direction / len);
        }
    }

    /// Index of the instance currently being traversed; see [`Ray::inst_idx`].
    #[inline]
    pub fn inst_idx(&self) -> u64 {
        self.inst_idx
    }

    /// Maximum hit distance: the current `hit.t`.
    #[inline]
    pub fn t_max(&self) -> f64 {
        self.hit.t
    }

    /// The point `origin + direction * hit.t`.
    #[inline]
    pub fn hit_point(&self) -> DVec3 {
        self.origin + self.direction * self.hit.t
    }

    /// Clears the hit record and resets the maximum distance.
    #[inline]
    pub fn reset(&mut self, t_max: f64) {
        self.hit = IntersectionEx::none(t_max);
        self.inst_idx = 0;
    }

    /// Panics when origin, direction or `hit.t` contain NaN; see [`Ray::assert_finite`].
    #[inline]
    pub(crate) fn assert_finite(&self) {
        let probe = self.origin.x + self.origin.y + self.origin.z + self.direction.x + self.direction.y + self.direction.z + self.hit.t;
        assert!(!probe.is_nan(), "tinybvh: ray contains NaN components");
    }

    #[inline]
    pub(crate) fn as_ffi(&self) -> *const ffi::tbvh_ray_ex {
        self as *const RayEx as *const ffi::tbvh_ray_ex
    }

    #[inline]
    pub(crate) fn as_ffi_mut(&mut self) -> *mut ffi::tbvh_ray_ex {
        self as *mut RayEx as *mut ffi::tbvh_ray_ex
    }
}

unsafe impl Send for RayEx {}
unsafe impl Sync for RayEx {}
unsafe impl Send for IntersectionEx {}
unsafe impl Sync for IntersectionEx {}

const _: () = {
    use core::mem::{align_of, size_of};
    assert!(size_of::<Intersection>() == size_of::<ffi::tbvh_intersection>());
    assert!(align_of::<Intersection>() == align_of::<ffi::tbvh_intersection>());
    assert!(size_of::<Ray>() == size_of::<ffi::tbvh_ray>());
    assert!(align_of::<Ray>() == align_of::<ffi::tbvh_ray>());
    assert!(size_of::<IntersectionEx>() == size_of::<ffi::tbvh_intersection_ex>());
    assert!(size_of::<RayEx>() == size_of::<ffi::tbvh_ray_ex>());
    assert!(align_of::<RayEx>() == align_of::<ffi::tbvh_ray_ex>());
    // glam's vector types must have the plain layout tinybvh expects.
    assert!(size_of::<Vec3>() == 12 && align_of::<Vec3>() == 4);
    assert!(size_of::<DVec3>() == 24 && align_of::<DVec3>() == 8);
    assert!(core::mem::offset_of!(Ray, hit) == 44);
    assert!(core::mem::offset_of!(RayEx, hit) == 88);
};
