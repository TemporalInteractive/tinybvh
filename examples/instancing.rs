//! A top-level acceleration structure: one triangle mesh instanced many times with different
//! transforms, plus a custom sphere primitive, traced through a single TLAS. Instances are then
//! animated by changing their transforms and rebuilding the (small) TLAS.

use std::sync::Arc;

use glam::{Mat4, Quat, Vec3, Vec4};
use tinybvh::{Blas, BlasInstance, Bvh, CustomGeometry, Ray};

/// A unit cube as 12 triangles.
fn cube() -> Vec<Vec4> {
    let p = |x: f32, y: f32, z: f32| Vec4::new(x - 0.5, y - 0.5, z - 0.5, 0.0);
    let faces = [
        [p(0., 0., 0.), p(1., 0., 0.), p(1., 1., 0.), p(0., 1., 0.)],
        [p(0., 0., 1.), p(1., 1., 1.), p(1., 0., 1.), p(0., 1., 1.)],
        [p(0., 0., 0.), p(0., 1., 0.), p(0., 1., 1.), p(0., 0., 1.)],
        [p(1., 0., 0.), p(1., 1., 1.), p(1., 1., 0.), p(1., 0., 1.)],
        [p(0., 0., 0.), p(0., 0., 1.), p(1., 0., 1.), p(1., 0., 0.)],
        [p(0., 1., 0.), p(1., 1., 1.), p(0., 1., 1.), p(1., 1., 0.)],
    ];
    let mut verts = Vec::new();
    for [a, b, c, d] in faces {
        verts.extend([a, b, c, a, c, d]);
    }
    verts
}

/// Spheres as custom primitives.
struct Spheres(Vec<(Vec3, f32)>);

impl Spheres {
    fn hit(&self, ray: &Ray, prim: u32) -> Option<f32> {
        let (center, radius) = self.0[prim as usize];
        let oc = ray.origin - center;
        let d = ray.direction();
        let a = d.dot(d);
        let b = 2.0 * oc.dot(d);
        let c = oc.dot(oc) - radius * radius;
        let disc = b * b - 4.0 * a * c;
        if disc < 0.0 {
            return None;
        }
        let t = (-b - disc.sqrt()) / (2.0 * a);
        (t > 0.0 && t < ray.hit.t).then_some(t)
    }
}

impl CustomGeometry for Spheres {
    fn prim_count(&self) -> u32 {
        self.0.len() as u32
    }
    fn aabb(&self, prim: u32) -> (Vec3, Vec3) {
        let (c, r) = self.0[prim as usize];
        (c - r, c + r)
    }
    fn intersect(&self, ray: &mut Ray, prim: u32) -> bool {
        match self.hit(ray, prim) {
            Some(t) => {
                ray.hit.t = t;
                ray.hit.prim = prim;
                true
            }
            None => false,
        }
    }
    fn is_occluded(&self, ray: &Ray, prim: u32) -> bool {
        self.hit(ray, prim).is_some()
    }
}

fn main() {
    // Bottom-level structures, shared with the TLAS through Arc.
    let mut cube_bvh = Bvh::new();
    cube_bvh.build(cube());
    let mut sphere_bvh = Bvh::new();
    sphere_bvh.build_custom(Spheres(vec![(Vec3::ZERO, 0.5)]));
    let blasses: Vec<Arc<dyn Blas>> = vec![Arc::new(cube_bvh), Arc::new(sphere_bvh)];

    // A 4 x 4 grid of cubes (BLAS 0) in the plane z = 5, and a sphere (BLAS 1) in front of it.
    let mut instances = Vec::new();
    for y in 0..4 {
        for x in 0..4 {
            let position = Vec3::new(x as f32 * 2.0 - 3.0, y as f32 * 2.0 - 3.0, 5.0);
            instances.push(BlasInstance::with_transform(
                0,
                Mat4::from_translation(position),
            ));
        }
    }
    instances.push(
        BlasInstance::with_transform(1, Mat4::from_translation(Vec3::new(1.0, 1.0, 2.0)))
            .with_mask(0b10),
    );

    let mut tlas = Bvh::new();
    tlas.build_tlas(instances, blasses);

    // Trace a few rays. The instance index tells which object was hit.
    for (origin, label) in [
        (Vec3::new(1.0, 1.0, 0.0), "towards the sphere"),
        (Vec3::new(-3.0, -3.0, 0.0), "towards a cube"),
    ] {
        let mut ray = Ray::new(origin, Vec3::Z);
        tlas.intersect(&mut ray);
        println!(
            "{label}: instance {} (BLAS {}), prim {}, t = {:.3}",
            ray.hit.inst,
            tlas.instances()[ray.hit.inst as usize].blas_idx,
            ray.hit.prim,
            ray.hit.t
        );
    }

    // Masks: a ray that only matches mask bit 0 ignores the sphere and sees the cube behind it.
    let mut ray = Ray::new(Vec3::new(1.0, 1.0, 0.0), Vec3::Z).with_mask(0b01);
    tlas.intersect(&mut ray);
    println!(
        "masked ray: instance {}, t = {:.3}",
        ray.hit.inst, ray.hit.t
    );

    // Animation: rotate the cubes and rebuild the TLAS each frame.
    for frame in 1..=3 {
        let angle = frame as f32 * 0.3;
        for inst in tlas.instances_mut().iter_mut().take(16) {
            let translation = inst.transform().w_axis.truncate();
            inst.set_transform(
                Mat4::from_translation(translation) * Mat4::from_quat(Quat::from_rotation_z(angle)),
            );
        }
        tlas.rebuild_tlas().unwrap();
        let mut ray = Ray::new(Vec3::new(-3.0, -2.4, 0.0), Vec3::Z);
        tlas.intersect(&mut ray);
        println!(
            "frame {frame}: {}",
            if ray.hit.is_hit() {
                format!("hit at t = {:.3}", ray.hit.t)
            } else {
                "miss".into()
            }
        );
    }
}
