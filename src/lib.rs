use glam::{Vec3, Vec4};

mod ffi;

pub type Intersection = ffi::tinybvh_Intersection;

impl Default for Intersection {
    fn default() -> Self {
        Self::new(0, 0.0, 0.0, 0.0, 0)
    }
}

impl Intersection {
    pub fn new(inst: u32, t: f32, u: f32, v: f32, prim: u32) -> Self {
        Self {
            inst,
            t,
            u,
            v,
            prim,
        }
    }
}

pub type Ray = ffi::tinybvh_Ray;

impl Ray {
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        unsafe { ffi::tinybvh_Ray_new(origin.into(), direction.into()) }
    }
}

pub type Bvh = ffi::tinybvh_BVH;

impl Default for Bvh {
    fn default() -> Self {
        Self::new()
    }
}

impl Bvh {
    pub fn new() -> Self {
        unsafe { ffi::tinybvh_BVH_new() }
    }

    pub fn build(&mut self, vertices: &[Vec4]) {
        unsafe {
            let vertices: &[ffi::tinybvh_bvhvec4] = std::mem::transmute(vertices);
            ffi::tinybvh_BVH_Build(self, &vertices[0], vertices.len() as u32);
        }
    }

    /// Intersects a ray against the bvh, the resulting distance is stored in `Ray.t` which is `INFINITE` when no intersection happened.
    /// Returns the cost of the intersection.
    pub fn intersect(&self, ray: &mut Ray) -> i32 {
        unsafe { ffi::tinybvh_BVH_Intersect(self, ray) }
    }
}
