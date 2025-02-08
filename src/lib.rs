use glam::{Mat4, Vec3, Vec4};
use std::sync::Arc;

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

impl BlasInstance {
    pub fn new(transform: Mat4, blas_idx: u32) -> Self {
        let mut blas_instance = ffi::tinybvh_BLASInstance {
            transform: transform.transpose().to_cols_array(),
            invTransform: [0.0; 16usize],
            aabbMin: Vec3::ZERO.into(),
            blasIdx: blas_idx,
            aabbMax: Vec3::ZERO.into(),
            dummy: [0; 9usize],
        };

        unsafe {
            ffi::tinybvh_BLASInstance_InvertTransform(&mut blas_instance);
        }

        blas_instance
    }
}

pub trait BvhBase {
    fn base(&self) -> &ffi::tinybvh_BVHBase;
    fn base_mut(&mut self) -> &mut ffi::tinybvh_BVHBase;
}

pub enum BvhBuildQuality {
    Low,
    High,
}

pub struct Bvh {
    bvh: Box<ffi::tinybvh_BVH>,
    vertices: Vec<Vec4>,
    indices: Vec<u32>,
    blas_instances: Vec<BlasInstance>,
    blasses: Vec<Arc<dyn BvhBase>>,
    blas_bases: Vec<*mut ffi::tinybvh_BVHBase>,
}

impl Bvh {
    pub fn new() -> Self {
        let bvh = unsafe { Box::from_raw(ffi::tinybvh_BVH_new()) };
        Self {
            bvh,
            vertices: vec![],
            indices: vec![],
            blas_instances: vec![],
            blasses: vec![],
            blas_bases: vec![],
        }
    }

    pub fn build(&mut self, vertices: Vec<Vec4>, quality: BvhBuildQuality) {
        self.vertices = vertices;

        unsafe {
            let vertices: &[ffi::tinybvh_bvhvec4] = std::mem::transmute(self.vertices.as_slice());

            match quality {
                BvhBuildQuality::Low => {
                    ffi::tinybvh_BVH_Build(
                        self.bvh.as_mut(),
                        &vertices[0],
                        vertices.len() as u32 / 3,
                    );
                }
                BvhBuildQuality::High => {
                    ffi::tinybvh_BVH_BuildHQ(
                        self.bvh.as_mut(),
                        &vertices[0],
                        vertices.len() as u32 / 3,
                    );
                }
            }
        }
    }

    pub fn build_with_indices(
        &mut self,
        vertices: Vec<Vec4>,
        indices: Vec<u32>,
        quality: BvhBuildQuality,
    ) {
        self.vertices = vertices;
        self.indices = indices;

        unsafe {
            let vertices: &[ffi::tinybvh_bvhvec4] = std::mem::transmute(self.vertices.as_slice());

            match quality {
                BvhBuildQuality::Low => {
                    ffi::tinybvh_BVH_Build2(
                        self.bvh.as_mut(),
                        &vertices[0],
                        &self.indices[0],
                        self.indices.len() as u32 / 3,
                    );
                }
                BvhBuildQuality::High => {
                    ffi::tinybvh_BVH_BuildHQ2(
                        self.bvh.as_mut(),
                        &vertices[0],
                        &self.indices[0],
                        self.indices.len() as u32 / 3,
                    );
                }
            }
        }
    }

    pub fn build_with_blas_instances(
        &mut self,
        blas_instances: Vec<BlasInstance>,
        blasses: Vec<Arc<dyn BvhBase>>,
    ) {
        self.blas_instances = blas_instances;
        self.blasses = blasses;

        // TODO: Is this a peformance hit we HAVE to take?
        self.blas_bases = self
            .blasses
            .iter_mut()
            .map(|blas| blas.base() as *const ffi::tinybvh_BVHBase as *mut ffi::tinybvh_BVHBase)
            .collect();

        let num_instances = self.blas_instances.len();

        unsafe {
            ffi::tinybvh_BVH_Build4(
                self.bvh.as_mut(),
                &mut self.blas_instances[0],
                num_instances as u32,
                self.blas_bases.as_mut_ptr(),
                self.blasses.len() as u32,
            );
        }
    }

    /// Intersects a ray against the bvh, the resulting distance is stored in `Ray.t` which is `INFINITE` when no intersection happened.
    /// Returns the cost of the intersection.
    pub fn intersect(&self, ray: &mut Ray) -> i32 {
        unsafe { ffi::tinybvh_BVH_Intersect(self.bvh.as_ref(), ray) }
    }

    /// Intersects a packet of 256 rays against the bvh, the resulting distance is stored in `Ray.t` which is `INFINITE` when no intersection happened.
    /// This is faster than using individual calls to intersect for a packet of coherent rays. For now only supported for BLASes.
    pub fn intersect_256(&self, rays: &mut [Ray]) {
        unsafe {
            ffi::tinybvh_BVH_Intersect256Rays(self.bvh.as_ref(), rays.as_mut_ptr());
        }
    }

    /// Intersects a ray against the bvh, returning if any hit took place.
    pub fn is_occluded(&self, ray: &Ray) -> bool {
        unsafe { ffi::tinybvh_BVH_IsOccluded(self.bvh.as_ref(), ray) }
    }
}

impl Default for Bvh {
    fn default() -> Self {
        Self::new()
    }
}

impl BvhBase for Bvh {
    fn base(&self) -> &ffi::tinybvh_BVHBase {
        &self.bvh._base
    }

    fn base_mut(&mut self) -> &mut ffi::tinybvh_BVHBase {
        &mut self.bvh._base
    }
}

impl Drop for Bvh {
    fn drop(&mut self) {
        unsafe {
            ffi::tinybvh_BVH_BVH_destructor(self.bvh.as_mut());
        }
    }
}

#[cfg(feature = "unsafe-send-sync")]
unsafe impl Send for Bvh {}
#[cfg(feature = "unsafe-send-sync")]
unsafe impl Sync for Bvh {}

pub struct BvhSoA {
    bvh: Box<ffi::tinybvh_BVH_SoA>,
    vertices: Vec<Vec4>,
    indices: Vec<u32>,
}

impl BvhSoA {
    pub fn new() -> Self {
        let bvh = unsafe { Box::from_raw(ffi::tinybvh_BVH_SoA_new()) };
        Self {
            bvh,
            vertices: vec![],
            indices: vec![],
        }
    }

    pub fn build(&mut self, vertices: Vec<Vec4>, quality: BvhBuildQuality) {
        self.vertices = vertices;

        unsafe {
            let vertices: &[ffi::tinybvh_bvhvec4] = std::mem::transmute(self.vertices.as_slice());

            match quality {
                BvhBuildQuality::Low => {
                    ffi::tinybvh_BVH_SoA_Build(
                        self.bvh.as_mut(),
                        &vertices[0],
                        vertices.len() as u32 / 3,
                    );
                }
                BvhBuildQuality::High => {
                    ffi::tinybvh_BVH_SoA_BuildHQ(
                        self.bvh.as_mut(),
                        &vertices[0],
                        vertices.len() as u32 / 3,
                    );
                }
            }
        }
    }

    pub fn build_with_indices(
        &mut self,
        vertices: Vec<Vec4>,
        indices: Vec<u32>,
        quality: BvhBuildQuality,
    ) {
        self.vertices = vertices;
        self.indices = indices;

        unsafe {
            let vertices: &[ffi::tinybvh_bvhvec4] = std::mem::transmute(self.vertices.as_slice());

            match quality {
                BvhBuildQuality::Low => {
                    ffi::tinybvh_BVH_SoA_Build2(
                        self.bvh.as_mut(),
                        &vertices[0],
                        &self.indices[0],
                        self.indices.len() as u32 / 3,
                    );
                }
                BvhBuildQuality::High => {
                    ffi::tinybvh_BVH_SoA_BuildHQ2(
                        self.bvh.as_mut(),
                        &vertices[0],
                        &self.indices[0],
                        self.indices.len() as u32 / 3,
                    );
                }
            }
        }
    }

    /// Intersects a ray against the bvh, the resulting distance is stored in `Ray.t` which is `INFINITE` when no intersection happened.
    /// Returns the cost of the intersection.
    pub fn intersect(&self, ray: &mut Ray) -> i32 {
        unsafe { ffi::tinybvh_BVH_SoA_Intersect(self.bvh.as_ref(), ray) }
    }

    /// Intersects a ray against the bvh, returning if any hit took place.
    pub fn is_occluded(&self, ray: &Ray) -> bool {
        unsafe { ffi::tinybvh_BVH_SoA_IsOccluded(self.bvh.as_ref(), ray) }
    }
}

impl Default for BvhSoA {
    fn default() -> Self {
        Self::new()
    }
}

impl BvhBase for BvhSoA {
    fn base(&self) -> &ffi::tinybvh_BVHBase {
        &self.bvh._base
    }

    fn base_mut(&mut self) -> &mut ffi::tinybvh_BVHBase {
        &mut self.bvh._base
    }
}

impl Drop for BvhSoA {
    fn drop(&mut self) {
        unsafe {
            ffi::tinybvh_BVH_SoA_BVH_SoA_destructor(self.bvh.as_mut());
        }
    }
}

#[cfg(feature = "unsafe-send-sync")]
unsafe impl Send for BvhSoA {}
#[cfg(feature = "unsafe-send-sync")]
unsafe impl Sync for BvhSoA {}
