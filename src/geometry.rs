//! Geometry input: owned triangle storage, custom primitive callbacks, and the plumbing that keeps
//! panics from crossing the C boundary.

use core::ffi::{c_void, CStr};
use std::any::Any;
use std::ffi::CString;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use glam::{DVec3, Vec3, Vec4};

use crate::error::{Error, Result};
use crate::ffi;
use crate::ray::{Ray, RayEx};

// tinybvh reads vertices with aligned 16-byte loads; glam's Vec4 provides that alignment unless
// glam is built with its `scalar-math` feature, which this crate does not support.
const _: () = {
    assert!(
        core::mem::size_of::<Vec4>() == 16 && core::mem::align_of::<Vec4>() == 16,
        "tinybvh needs 16-byte aligned glam::Vec4 vertices; do not enable glam's `scalar-math` feature"
    );
};

/// Stride in bytes between consecutive vertices in a `&[Vec4]`.
pub(crate) const VEC4_STRIDE: u32 = 16;
/// Stride in bytes between consecutive vertices in a `&[DVec3]`.
pub(crate) const DVEC3_STRIDE: u64 = 24;

/// Triangle geometry owned by an acceleration structure.
///
/// tinybvh references vertex data instead of copying it, so the structures in this crate take
/// ownership of their input and keep it alive for as long as the tree exists. Vertices are
/// `glam::Vec4`, three per triangle; the `w` component is padding that tinybvh ignores.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Geometry {
    /// Vertex positions, three per triangle (or referenced by `indices` when present).
    pub vertices: Vec<Vec4>,
    /// Optional vertex indices, three per triangle.
    pub indices: Option<Vec<u32>>,
}

impl Geometry {
    /// A triangle soup: `vertices.len()` must be a non-zero multiple of three.
    pub fn new(vertices: Vec<Vec4>) -> Self {
        Geometry { vertices, indices: None }
    }

    /// Indexed triangles: `indices.len()` must be a non-zero multiple of three and every index
    /// must be below `vertices.len()`.
    pub fn indexed(vertices: Vec<Vec4>, indices: Vec<u32>) -> Self {
        Geometry { vertices, indices: Some(indices) }
    }

    /// Number of triangles.
    pub fn triangle_count(&self) -> usize {
        match &self.indices {
            Some(indices) => indices.len() / 3,
            None => self.vertices.len() / 3,
        }
    }

    /// The three vertex positions of triangle `tri`.
    pub fn triangle(&self, tri: usize) -> [Vec3; 3] {
        let [i0, i1, i2] = match &self.indices {
            Some(indices) => [indices[tri * 3] as usize, indices[tri * 3 + 1] as usize, indices[tri * 3 + 2] as usize],
            None => [tri * 3, tri * 3 + 1, tri * 3 + 2],
        };
        [self.vertices[i0].truncate(), self.vertices[i1].truncate(), self.vertices[i2].truncate()]
    }

    /// Panics with a clear message when the data is not valid input for tinybvh. Called before
    /// every build; this is also what keeps the C library from reading out of bounds.
    pub(crate) fn validate(&self) {
        assert!(!self.vertices.is_empty(), "tinybvh: geometry has no vertices");
        assert!(
            u32::try_from(self.vertices.len()).is_ok(),
            "tinybvh: single precision layouts support at most 2^32 - 1 vertices"
        );
        match &self.indices {
            None => assert!(
                self.vertices.len() % 3 == 0,
                "tinybvh: a triangle soup needs a multiple of three vertices, got {}",
                self.vertices.len()
            ),
            Some(indices) => {
                assert!(!indices.is_empty(), "tinybvh: indexed geometry has no indices");
                assert!(
                    indices.len() % 3 == 0,
                    "tinybvh: indexed geometry needs a multiple of three indices, got {}",
                    indices.len()
                );
                let n = self.vertices.len() as u32;
                if let Some(bad) = indices.iter().position(|&i| i >= n) {
                    panic!("tinybvh: index {} at position {bad} is out of range for {n} vertices", indices[bad]);
                }
            }
        }
    }

    #[inline]
    pub(crate) fn vert_ptr(&self) -> *const ffi::tbvh_vec4 {
        self.vertices.as_ptr() as *const ffi::tbvh_vec4
    }

    #[inline]
    pub(crate) fn vert_count(&self) -> u32 {
        self.vertices.len() as u32
    }

    /// Index pointer and primitive count for the indexed build functions; `None` for a soup.
    #[inline]
    pub(crate) fn index_args(&self) -> Option<(*const u32, u32)> {
        self.indices.as_ref().map(|i| (i.as_ptr(), (i.len() / 3) as u32))
    }
}

/// Double precision triangle geometry: `glam::DVec3` vertices and `u64` indices.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GeometryEx {
    /// Vertex positions, three per triangle (or referenced by `indices`).
    pub vertices: Vec<DVec3>,
    /// Optional vertex indices, three per triangle.
    pub indices: Option<Vec<u64>>,
}

impl GeometryEx {
    /// A triangle soup.
    pub fn new(vertices: Vec<DVec3>) -> Self {
        GeometryEx { vertices, indices: None }
    }

    /// Indexed triangles.
    pub fn indexed(vertices: Vec<DVec3>, indices: Vec<u64>) -> Self {
        GeometryEx { vertices, indices: Some(indices) }
    }

    /// Number of triangles.
    pub fn triangle_count(&self) -> usize {
        match &self.indices {
            Some(indices) => indices.len() / 3,
            None => self.vertices.len() / 3,
        }
    }

    /// The three vertex positions of triangle `tri`.
    pub fn triangle(&self, tri: usize) -> [DVec3; 3] {
        let [i0, i1, i2] = match &self.indices {
            Some(indices) => [indices[tri * 3] as usize, indices[tri * 3 + 1] as usize, indices[tri * 3 + 2] as usize],
            None => [tri * 3, tri * 3 + 1, tri * 3 + 2],
        };
        [self.vertices[i0], self.vertices[i1], self.vertices[i2]]
    }

    pub(crate) fn validate(&self) {
        assert!(!self.vertices.is_empty(), "tinybvh: geometry has no vertices");
        match &self.indices {
            None => assert!(
                self.vertices.len() % 3 == 0,
                "tinybvh: a triangle soup needs a multiple of three vertices, got {}",
                self.vertices.len()
            ),
            Some(indices) => {
                assert!(!indices.is_empty(), "tinybvh: indexed geometry has no indices");
                assert!(
                    indices.len() % 3 == 0,
                    "tinybvh: indexed geometry needs a multiple of three indices, got {}",
                    indices.len()
                );
                let n = self.vertices.len() as u64;
                if let Some(bad) = indices.iter().position(|&i| i >= n) {
                    panic!("tinybvh: index {} at position {bad} is out of range for {n} vertices", indices[bad]);
                }
            }
        }
    }

    #[inline]
    pub(crate) fn vert_ptr(&self) -> *const ffi::tbvh_dbl3 {
        self.vertices.as_ptr() as *const ffi::tbvh_dbl3
    }

    #[inline]
    pub(crate) fn vert_count(&self) -> u64 {
        self.vertices.len() as u64
    }

    #[inline]
    pub(crate) fn index_args(&self) -> Option<(*const u64, u64)> {
        self.indices.as_ref().map(|i| (i.as_ptr(), (i.len() / 3) as u64))
    }
}

/// User-defined primitives for a [`Bvh`](crate::Bvh) built with
/// [`build_custom`](crate::Bvh::build_custom).
///
/// tinybvh builds the tree over the bounds returned by [`aabb`](Self::aabb) and calls
/// [`intersect`](Self::intersect) / [`is_occluded`](Self::is_occluded) for the primitives a ray
/// reaches. Inside a TLAS the ray is in object space of the instance, and
/// [`Ray::inst_idx`] names the instance.
///
/// `aabb` may be called from several threads at once during a threaded build; the intersection
/// callbacks run on the thread that issued the query. A panic inside a callback is caught at the
/// boundary and re-raised on the calling thread once the C library returns.
pub trait CustomGeometry: Send + Sync {
    /// Number of primitives. Must be at least one.
    fn prim_count(&self) -> u32;
    /// Axis-aligned bounds of primitive `prim`, as `(min, max)`.
    fn aabb(&self, prim: u32) -> (Vec3, Vec3);
    /// Closest-hit test. On a hit closer than `ray.hit.t`, update `ray.hit` (at least `t`, and
    /// `prim` to the primitive index) and return `true`.
    fn intersect(&self, ray: &mut Ray, prim: u32) -> bool;
    /// Any-hit test: whether the primitive blocks the ray before `ray.hit.t`.
    fn is_occluded(&self, ray: &Ray, prim: u32) -> bool;
}

/// Double precision counterpart of [`CustomGeometry`], for
/// [`BvhDouble::build_custom`](crate::BvhDouble::build_custom).
pub trait CustomGeometryEx: Send + Sync {
    /// Number of primitives. Must be at least one.
    fn prim_count(&self) -> u64;
    /// Axis-aligned bounds of primitive `prim`, as `(min, max)`.
    fn aabb(&self, prim: u64) -> (DVec3, DVec3);
    /// Closest-hit test; see [`CustomGeometry::intersect`].
    fn intersect(&self, ray: &mut RayEx, prim: u64) -> bool;
    /// Any-hit test; see [`CustomGeometry::is_occluded`].
    fn is_occluded(&self, ray: &RayEx, prim: u64) -> bool;
}

/// A bounding box as `(min, max)`, usable as a simple custom primitive: a ray hits the box where
/// it enters it, and `u`, `v` are left at zero.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    /// Minimum corner.
    pub min: Vec3,
    /// Maximum corner.
    pub max: Vec3,
}

impl Aabb {
    /// A box from its corners.
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Aabb { min, max }
    }

    /// Slab test. Returns the entry distance when the ray hits the box before `t_max`.
    pub fn intersect_ray(&self, ray: &Ray, t_max: f32) -> Option<f32> {
        let rd = ray.rcp_direction();
        let t1 = (self.min - ray.origin) * rd;
        let t2 = (self.max - ray.origin) * rd;
        let tmin = t1.min(t2).max_element();
        let tmax = t1.max(t2).min_element();
        if tmax >= tmin && tmin < t_max && tmax >= 0.0 {
            Some(tmin.max(0.0))
        } else {
            None
        }
    }
}

/// A list of boxes as custom geometry, for collision queries and tests.
impl CustomGeometry for Vec<Aabb> {
    fn prim_count(&self) -> u32 {
        self.len() as u32
    }

    fn aabb(&self, prim: u32) -> (Vec3, Vec3) {
        let b = self[prim as usize];
        (b.min, b.max)
    }

    fn intersect(&self, ray: &mut Ray, prim: u32) -> bool {
        match self[prim as usize].intersect_ray(ray, ray.hit.t) {
            Some(t) => {
                ray.hit.t = t;
                ray.hit.prim = prim;
                true
            }
            None => false,
        }
    }

    fn is_occluded(&self, ray: &Ray, prim: u32) -> bool {
        self[prim as usize].intersect_ray(ray, ray.hit.t).is_some()
    }
}

// ---------------------------------------------------------------------------------------------
// Panic plumbing. Callbacks run under catch_unwind; the payload of the first panic is parked here
// and re-raised on the thread that called into the C library, once it returns.
// ---------------------------------------------------------------------------------------------

static PANIC_PENDING: AtomicBool = AtomicBool::new(false);
static PANIC_PAYLOAD: Mutex<Option<Box<dyn Any + Send>>> = Mutex::new(None);

fn park_panic(payload: Box<dyn Any + Send>) {
    let mut slot = PANIC_PAYLOAD.lock().unwrap_or_else(|e| e.into_inner());
    if slot.is_none() {
        *slot = Some(payload);
    }
    PANIC_PENDING.store(true, Ordering::Release);
}

/// Re-raises a panic that occurred in a callback during the preceding C call.
#[inline]
pub(crate) fn rethrow_pending_panic() {
    if PANIC_PENDING.load(Ordering::Acquire) {
        let payload = PANIC_PAYLOAD.lock().unwrap_or_else(|e| e.into_inner()).take();
        PANIC_PENDING.store(false, Ordering::Release);
        if let Some(payload) = payload {
            resume_unwind(payload);
        }
    }
}

/// Owner of a boxed custom geometry with a stable address that can be handed to C as `user`.
pub(crate) struct CustomHolder(Box<Box<dyn CustomGeometry>>);

impl CustomHolder {
    pub(crate) fn new(geometry: Box<dyn CustomGeometry>) -> Self {
        CustomHolder(Box::new(geometry))
    }

    pub(crate) fn geometry(&self) -> &dyn CustomGeometry {
        &**self.0
    }

    pub(crate) fn user_ptr(&self) -> *mut c_void {
        &*self.0 as *const Box<dyn CustomGeometry> as *mut c_void
    }

    pub(crate) fn aabb_fn() -> ffi::tbvh_aabb_fn {
        Some(custom_aabb)
    }

    pub(crate) fn intersect_fn() -> ffi::tbvh_intersect_fn {
        Some(custom_intersect)
    }

    pub(crate) fn occluded_fn() -> ffi::tbvh_occluded_fn {
        Some(custom_occluded)
    }
}

unsafe extern "C" fn custom_aabb(prim: u32, bmin: *mut ffi::tbvh_vec3, bmax: *mut ffi::tbvh_vec3, user: *mut c_void) {
    let geometry: &dyn CustomGeometry = &**(user as *const Box<dyn CustomGeometry>);
    match catch_unwind(AssertUnwindSafe(|| geometry.aabb(prim))) {
        Ok((mn, mx)) => {
            *bmin = ffi::tbvh_vec3 { x: mn.x, y: mn.y, z: mn.z };
            *bmax = ffi::tbvh_vec3 { x: mx.x, y: mx.y, z: mx.z };
        }
        Err(payload) => {
            park_panic(payload);
            *bmin = ffi::tbvh_vec3 { x: 0.0, y: 0.0, z: 0.0 };
            *bmax = ffi::tbvh_vec3 { x: 0.0, y: 0.0, z: 0.0 };
        }
    }
}

unsafe extern "C" fn custom_intersect(ray: *mut ffi::tbvh_ray, prim: u32, user: *mut c_void) -> bool {
    let geometry: &dyn CustomGeometry = &**(user as *const Box<dyn CustomGeometry>);
    let ray = &mut *(ray as *mut Ray);
    match catch_unwind(AssertUnwindSafe(|| geometry.intersect(ray, prim))) {
        Ok(hit) => hit,
        Err(payload) => {
            park_panic(payload);
            false
        }
    }
}

unsafe extern "C" fn custom_occluded(ray: *const ffi::tbvh_ray, prim: u32, user: *mut c_void) -> bool {
    let geometry: &dyn CustomGeometry = &**(user as *const Box<dyn CustomGeometry>);
    let ray = &*(ray as *const Ray);
    match catch_unwind(AssertUnwindSafe(|| geometry.is_occluded(ray, prim))) {
        Ok(hit) => hit,
        Err(payload) => {
            park_panic(payload);
            false
        }
    }
}

/// Double precision counterpart of [`CustomHolder`].
pub(crate) struct CustomHolderEx(Box<Box<dyn CustomGeometryEx>>);

impl CustomHolderEx {
    pub(crate) fn new(geometry: Box<dyn CustomGeometryEx>) -> Self {
        CustomHolderEx(Box::new(geometry))
    }

    pub(crate) fn geometry(&self) -> &dyn CustomGeometryEx {
        &**self.0
    }

    pub(crate) fn user_ptr(&self) -> *mut c_void {
        &*self.0 as *const Box<dyn CustomGeometryEx> as *mut c_void
    }

    pub(crate) fn aabb_fn() -> ffi::tbvh_aabb_ex_fn {
        Some(custom_aabb_ex)
    }

    pub(crate) fn intersect_fn() -> ffi::tbvh_intersect_ex_fn {
        Some(custom_intersect_ex)
    }

    pub(crate) fn occluded_fn() -> ffi::tbvh_occluded_ex_fn {
        Some(custom_occluded_ex)
    }
}

unsafe extern "C" fn custom_aabb_ex(prim: u64, bmin: *mut ffi::tbvh_dbl3, bmax: *mut ffi::tbvh_dbl3, user: *mut c_void) {
    let geometry: &dyn CustomGeometryEx = &**(user as *const Box<dyn CustomGeometryEx>);
    match catch_unwind(AssertUnwindSafe(|| geometry.aabb(prim))) {
        Ok((mn, mx)) => {
            *bmin = ffi::tbvh_dbl3 { x: mn.x, y: mn.y, z: mn.z };
            *bmax = ffi::tbvh_dbl3 { x: mx.x, y: mx.y, z: mx.z };
        }
        Err(payload) => {
            park_panic(payload);
            *bmin = ffi::tbvh_dbl3 { x: 0.0, y: 0.0, z: 0.0 };
            *bmax = ffi::tbvh_dbl3 { x: 0.0, y: 0.0, z: 0.0 };
        }
    }
}

unsafe extern "C" fn custom_intersect_ex(ray: *mut ffi::tbvh_ray_ex, prim: u64, user: *mut c_void) -> bool {
    let geometry: &dyn CustomGeometryEx = &**(user as *const Box<dyn CustomGeometryEx>);
    let ray = &mut *(ray as *mut RayEx);
    match catch_unwind(AssertUnwindSafe(|| geometry.intersect(ray, prim))) {
        Ok(hit) => hit,
        Err(payload) => {
            park_panic(payload);
            false
        }
    }
}

unsafe extern "C" fn custom_occluded_ex(ray: *const ffi::tbvh_ray_ex, prim: u64, user: *mut c_void) -> bool {
    let geometry: &dyn CustomGeometryEx = &**(user as *const Box<dyn CustomGeometryEx>);
    let ray = &*(ray as *const RayEx);
    match catch_unwind(AssertUnwindSafe(|| geometry.is_occluded(ray, prim))) {
        Ok(hit) => hit,
        Err(payload) => {
            park_panic(payload);
            false
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Misc helpers shared by the layout modules.
// ---------------------------------------------------------------------------------------------

/// Converts a path for the C library. Non-UTF-8 paths are passed as their lossy UTF-8 form.
pub(crate) fn path_to_cstring(path: &Path) -> Result<CString> {
    CString::new(path.to_string_lossy().as_bytes()).map_err(|_| Error::InvalidPath)
}

/// Owned copy of a C string, for error messages.
pub(crate) unsafe fn cstr_to_string(ptr: *const core::ffi::c_char) -> String {
    if ptr.is_null() {
        String::new()
    } else {
        CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}

#[inline]
pub(crate) fn vec3_from_ffi(v: ffi::tbvh_vec3) -> Vec3 {
    Vec3::new(v.x, v.y, v.z)
}

#[inline]
pub(crate) fn vec3_to_ffi(v: Vec3) -> ffi::tbvh_vec3 {
    ffi::tbvh_vec3 { x: v.x, y: v.y, z: v.z }
}

#[inline]
pub(crate) fn dvec3_from_ffi(v: ffi::tbvh_dbl3) -> DVec3 {
    DVec3::new(v.x, v.y, v.z)
}

#[inline]
pub(crate) fn dvec3_to_ffi(v: DVec3) -> ffi::tbvh_dbl3 {
    ffi::tbvh_dbl3 { x: v.x, y: v.y, z: v.z }
}

/// Validates an opacity micro map buffer against the triangle count.
pub(crate) fn validate_opacity_map(map: &[u32], n: u32, tri_count: u64) {
    assert!(n > 0, "tinybvh: opacity micro map subdivision must be at least 1");
    let words_per_tri = (n as u64 * n as u64).div_ceil(32);
    let needed = words_per_tri * tri_count;
    assert!(
        map.len() as u64 >= needed,
        "tinybvh: opacity micro map needs {needed} words for {tri_count} triangles at n = {n}, got {}",
        map.len()
    );
}
