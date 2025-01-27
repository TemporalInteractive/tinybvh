use glam::Vec3;

mod ffi;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct Intersection(ffi::tinybvh_Intersection);

impl Default for Intersection {
    fn default() -> Self {
        Self::new(0, 0.0, 0.0, 0.0, 0)
    }
}

impl Intersection {
    pub fn new(instance: u32, t: f32, u: f32, v: f32, primitive: u32) -> Self {
        Self(ffi::tinybvh_Intersection {
            inst: instance,
            t,
            u,
            v,
            prim: primitive,
        })
    }
}

#[repr(C)]
#[repr(align(16))]
#[derive(Copy, Clone)]
pub struct Ray(ffi::tinybvh_Ray);

impl Ray {
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self(unsafe { ffi::tinybvh_Ray_new(origin.into(), direction.into()) })
    }

    pub fn origin(&self) -> Vec3 {
        self.0.O.into()
    }

    pub fn direction(&self) -> Vec3 {
        self.0.D.into()
    }
}

#[repr(C)]
pub struct Bvh(ffi::tinybvh_BVH);

impl Default for Bvh {
    fn default() -> Self {
        Self::new()
    }
}

impl Bvh {
    pub fn new() -> Self {
        Self(unsafe { ffi::tinybvh_BVH_new() })
    }

    pub fn intersect(&self, ray: &mut Ray) {
        unsafe {
            ffi::tinybvh_BVH_Intersect(&self.0, &mut ray.0);
        }
    }
}
