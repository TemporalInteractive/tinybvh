use glam::{Vec3, Vec4};

mod ffi;
pub mod vec_helpers;

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

pub type BlasInstance = ffi::tinybvh_BLASInstance;

pub trait BvhBase {
    fn base(&self) -> &ffi::tinybvh_BVHBase;
    fn base_mut(&mut self) -> &mut ffi::tinybvh_BVHBase;
}

pub struct Bvh(Box<ffi::tinybvh_BVH>);

impl Default for Bvh {
    fn default() -> Self {
        Self::new()
    }
}

impl BvhBase for Bvh {
    fn base(&self) -> &ffi::tinybvh_BVHBase {
        &self.0._base
    }

    fn base_mut(&mut self) -> &mut ffi::tinybvh_BVHBase {
        &mut self.0._base
    }
}

impl Bvh {
    pub fn new() -> Self {
        Self(unsafe { Box::from_raw(ffi::tinybvh_BVH_new()) })
    }

    pub fn build(&mut self, vertices: &[Vec4]) {
        unsafe {
            let vertices: &[ffi::tinybvh_bvhvec4] = std::mem::transmute(vertices);
            ffi::tinybvh_BVH_Build(self.0.as_mut(), &vertices[0], vertices.len() as u32 / 3);
        }
    }

    pub fn build_with_indices(&mut self, vertices: &[Vec4], indices: &[u32]) {
        unsafe {
            let vertices: &[ffi::tinybvh_bvhvec4] = std::mem::transmute(vertices);
            ffi::tinybvh_BVH_Build2(
                self.0.as_mut(),
                &vertices[0],
                &indices[0],
                indices.len() as u32 / 3,
            );
        }
    }

    pub fn build_with_blas_instances(
        &mut self,
        blas_instances: &mut [BlasInstance],
        blasses: &mut [&mut dyn BvhBase],
    ) {
        // TODO: Is this a peformance hit we HAVE to take?
        let mut blas_bases: Vec<*mut ffi::tinybvh_BVHBase> = blasses
            .iter_mut()
            .map(|blas| blas.base_mut() as *mut ffi::tinybvh_BVHBase)
            .collect();

        unsafe {
            ffi::tinybvh_BVH_Build4(
                self.0.as_mut(),
                &mut blas_instances[0],
                blas_instances.len() as u32,
                blas_bases.as_mut_ptr(),
                blasses.len() as u32,
            );
        }
    }

    /// Intersects a ray against the bvh, the resulting distance is stored in `Ray.t` which is `INFINITE` when no intersection happened.
    /// Returns the cost of the intersection.
    pub fn intersect(&self, ray: &mut Ray) -> i32 {
        unsafe { ffi::tinybvh_BVH_Intersect(self.0.as_ref(), ray) }
    }

    /// Intersects a ray against the bvh, returning if any hit took place.
    pub fn is_occluded(&self, ray: &Ray) -> bool {
        unsafe { ffi::tinybvh_BVH_IsOccluded(self.0.as_ref(), ray) }
    }
}

impl Drop for Bvh {
    fn drop(&mut self) {
        unsafe {
            ffi::tinybvh_BVH_BVH_destructor(self.0.as_mut());
        }
    }
}
