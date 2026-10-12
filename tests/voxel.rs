//! Tests for `VoxelSet`: voxel placement, DDA traversal, normals, occlusion, the top grid
//! bookkeeping and use as a BLAS in a TLAS.

mod common;

use std::sync::Arc;

use approx::assert_relative_eq;
use glam::{Mat4, Vec3};
use tinybvh::{Blas, BlasInstance, Bvh, Layout, Ray, VoxelSet, FAR};

use common::triangle_at;

const DIM: f32 = 256.0;

/// Center of voxel `(x, y, z)` in object space.
fn center(x: u32, y: u32, z: u32) -> Vec3 {
    Vec3::new((x as f32 + 0.5) / DIM, (y as f32 + 0.5) / DIM, (z as f32 + 0.5) / DIM)
}

#[test]
fn object_dim_and_bounds() {
    assert_eq!(VoxelSet::object_dim(), 256);
    let voxels = VoxelSet::new();
    assert_eq!(voxels.layout(), Layout::VoxelSet);
    assert_eq!(voxels.aabb(), (Vec3::ZERO, Vec3::ONE));
    assert_eq!(voxels.prim_count(), 0);
    assert!(!voxels.is_top_grid_stale());
    assert!(format!("{voxels:?}").contains("VoxelSet"));
    // An empty object is traversable and reports nothing.
    let mut ray = Ray::new(Vec3::new(0.5, 0.5, -1.0), Vec3::Z);
    assert_eq!(voxels.intersect(&mut ray), 0);
    assert!(!ray.hit.is_hit());
    assert!(!voxels.is_occluded(&ray));
    let _ = VoxelSet::default();
}

#[test]
fn set_update_and_trace() {
    let mut voxels = VoxelSet::new();
    voxels.set(128, 128, 128, 7);
    assert!(voxels.is_top_grid_stale());
    voxels.update_top_grid();
    assert!(!voxels.is_top_grid_stale());

    // Voxel 128 starts at 0.5 on each axis; the ray from z = -1 enters it at t = 1.5.
    let c = center(128, 128, 128);
    let mut ray = Ray::new(Vec3::new(c.x, c.y, -1.0), Vec3::Z);
    let steps = voxels.intersect(&mut ray);
    assert!(steps > 0);
    assert!(ray.hit.is_hit());
    assert_relative_eq!(ray.hit.t, 1.5, epsilon = 1e-5);
    assert_eq!(ray.hit.prim, 7, "the voxel value is reported as the primitive");
    assert_eq!(ray.hit.inst, 0);

    // Rays that miss the voxel, and rays that miss the object entirely.
    let mut miss = Ray::new(Vec3::new(c.x + 0.1, c.y, -1.0), Vec3::Z);
    voxels.intersect(&mut miss);
    assert!(!miss.hit.is_hit());
    assert_eq!(miss.hit.t, FAR);
    let mut away = Ray::new(Vec3::new(c.x, c.y, -1.0), -Vec3::Z);
    voxels.intersect(&mut away);
    assert!(!away.hit.is_hit());
    let mut outside = Ray::new(Vec3::new(5.0, 5.0, 5.0), Vec3::X);
    assert_eq!(voxels.intersect(&mut outside), 0);
    assert!(!outside.hit.is_hit());

    // A ray starting inside the object finds the voxel too.
    let mut inside = Ray::new(Vec3::new(c.x, c.y, 0.25), Vec3::Z);
    voxels.intersect(&mut inside);
    assert_relative_eq!(inside.hit.t, 0.25, epsilon = 1e-5);

    // The direction is not normalized: t scales with its length.
    let mut scaled = Ray::new(Vec3::new(c.x, c.y, -1.0), Vec3::Z * 3.0);
    voxels.intersect(&mut scaled);
    assert_relative_eq!(scaled.hit.t, 0.5, epsilon = 1e-5);

    // Occlusion honours t_max.
    assert!(voxels.is_occluded(&Ray::new(Vec3::new(c.x, c.y, -1.0), Vec3::Z)));
    assert!(voxels.is_occluded(&Ray::with_t_max(Vec3::new(c.x, c.y, -1.0), Vec3::Z, 1.6)));
    assert!(!voxels.is_occluded(&Ray::with_t_max(Vec3::new(c.x, c.y, -1.0), Vec3::Z, 1.4)));
    assert!(!voxels.is_occluded(&Ray::new(Vec3::new(c.x + 0.1, c.y, -1.0), Vec3::Z)));
    assert!(!voxels.is_occluded(&Ray::new(Vec3::new(c.x, c.y, -1.0), -Vec3::Z)));

    // The closest of several voxels wins.
    voxels.set(128, 128, 64, 3);
    voxels.set(128, 128, 200, 9);
    voxels.update_top_grid();
    let mut ray = Ray::new(Vec3::new(c.x, c.y, -1.0), Vec3::Z);
    voxels.intersect(&mut ray);
    assert_eq!(ray.hit.prim, 3);
    assert_relative_eq!(ray.hit.t, 1.0 + 64.0 / DIM, epsilon = 1e-5);
    let mut back = Ray::new(Vec3::new(c.x, c.y, 2.0), -Vec3::Z);
    voxels.intersect(&mut back);
    assert_eq!(back.hit.prim, 9);
    assert_relative_eq!(back.hit.t, 2.0 - 201.0 / DIM, epsilon = 1e-5);
}

#[test]
fn entry_distance_per_axis_and_normals() {
    let mut voxels = VoxelSet::new();
    let (x, y, z) = (10u32, 20, 30);
    voxels.set(x, y, z, 1);
    voxels.update_top_grid();
    let c = center(x, y, z);
    let lo = Vec3::new(x as f32, y as f32, z as f32) / DIM;
    let hi = lo + 1.0 / DIM;
    let cases = [
        (Vec3::new(-1.0, c.y, c.z), Vec3::X, 1.0 + lo.x, -Vec3::X),
        (Vec3::new(2.0, c.y, c.z), -Vec3::X, 2.0 - hi.x, Vec3::X),
        (Vec3::new(c.x, -1.0, c.z), Vec3::Y, 1.0 + lo.y, -Vec3::Y),
        (Vec3::new(c.x, 2.0, c.z), -Vec3::Y, 2.0 - hi.y, Vec3::Y),
        (Vec3::new(c.x, c.y, -1.0), Vec3::Z, 1.0 + lo.z, -Vec3::Z),
        (Vec3::new(c.x, c.y, 2.0), -Vec3::Z, 2.0 - hi.z, Vec3::Z),
    ];
    for (origin, dir, expected_t, expected_normal) in cases {
        let mut ray = Ray::new(origin, dir);
        voxels.intersect(&mut ray);
        assert!(ray.hit.is_hit(), "ray from {origin} along {dir}");
        assert_relative_eq!(ray.hit.t, expected_t, epsilon = 1e-4);
        assert_eq!(ray.hit.prim, 1);
        let n = voxels.normal(&ray);
        assert!(n.abs_diff_eq(expected_normal, 1e-6), "normal {n} for ray along {dir}");
        // A ray that already records the hit is not occluded by it; a fresh one is.
        assert!(!voxels.is_occluded(&ray));
        assert!(voxels.is_occluded(&Ray::new(origin, dir)));
        assert!(n.dot(dir) < 0.0, "the normal faces the ray");
    }
    // The normal of a diagonal hit is the face that was entered.
    let mut diagonal = Ray::new(c - Vec3::new(0.5, 0.3, 0.1), Vec3::new(0.5, 0.3, 0.1));
    voxels.intersect(&mut diagonal);
    assert!(diagonal.hit.is_hit());
    assert_eq!(voxels.normal(&diagonal), -Vec3::X);
}

#[test]
fn clearing_a_voxel_removes_the_hit() {
    let mut voxels = VoxelSet::new();
    voxels.set(1, 2, 3, 5);
    voxels.set(255, 255, 255, 6);
    voxels.update_top_grid();
    let c = center(1, 2, 3);
    let probe = || Ray::new(Vec3::new(c.x, c.y, -1.0), Vec3::Z);
    let mut ray = probe();
    voxels.intersect(&mut ray);
    assert_eq!(ray.hit.prim, 5);
    let far = center(255, 255, 255);
    let mut ray_far = Ray::new(Vec3::new(far.x, far.y, 2.0), -Vec3::Z);
    voxels.intersect(&mut ray_far);
    assert_eq!(ray_far.hit.prim, 6);
    assert_relative_eq!(ray_far.hit.t, 1.0, epsilon = 1e-5);

    voxels.set(1, 2, 3, 0);
    voxels.update_top_grid();
    let mut ray = probe();
    voxels.intersect(&mut ray);
    assert!(!ray.hit.is_hit());
    assert!(!voxels.is_occluded(&probe()));
    // Setting it again brings it back with the new value.
    voxels.set(1, 2, 3, 42);
    voxels.update_top_grid();
    let mut ray = probe();
    voxels.intersect(&mut ray);
    assert_eq!(ray.hit.prim, 42);
}

#[test]
fn many_voxels_fill_bricks() {
    // A solid 64^3 block exercises brick allocation and the mid-level grid.
    let mut voxels = VoxelSet::new();
    for x in 64..128 {
        for y in 64..128 {
            for z in 64..128 {
                voxels.set(x, y, z, 1 + (x + y + z) % 3);
            }
        }
    }
    voxels.update_top_grid();
    let mut rng = common::Rng::new(5);
    let mut hits = 0;
    for _ in 0..200 {
        let o = Vec3::new(rng.unit(), rng.unit(), -0.5);
        let mut ray = Ray::new(o, Vec3::Z);
        voxels.intersect(&mut ray);
        let inside = (0.25..0.5).contains(&o.x) && (0.25..0.5).contains(&o.y);
        assert_eq!(ray.hit.is_hit(), inside, "ray at {o}");
        assert_eq!(voxels.is_occluded(&Ray::new(o, Vec3::Z)), inside);
        if inside {
            assert_relative_eq!(ray.hit.t, 0.75, epsilon = 1e-4);
            assert!((1..=3).contains(&ray.hit.prim));
            hits += 1;
        }
    }
    assert!(hits > 5);
}

#[test]
fn tlas_with_a_scaled_voxel_instance_and_a_triangle_blas() {
    // Voxels: a solid floor slab in the lowest 16 layers of the unit cube.
    let mut voxels = VoxelSet::new();
    for z in 0..256 {
        for x in 0..256 {
            for y in 0..16 {
                voxels.set(x, y, z, 2);
            }
        }
    }
    voxels.update_top_grid();
    assert!(voxels.blas_is_ready());
    // Triangles: two stacked triangles at z = 3 and z = 5 around the origin.
    let mut mesh = Bvh::new();
    let mut verts = Vec::new();
    verts.extend(triangle_at(5.0));
    verts.extend(triangle_at(3.0));
    mesh.build(verts);

    // The voxel object is scaled to 10 units and placed so that it spans [-5, 5] in x and z and
    // [-2, 8] in y; its slab top is then at y = -2 + 16 / 256 * 10 = -1.375.
    let voxel_transform = Mat4::from_translation(Vec3::new(-5.0, -2.0, -5.0)) * Mat4::from_scale(Vec3::splat(10.0));
    let instances = vec![
        BlasInstance::with_transform(0, voxel_transform).with_mask(1),
        BlasInstance::new(1).with_mask(2),
        BlasInstance::with_transform(0, Mat4::from_translation(Vec3::new(20.0, 0.0, 0.0)) * Mat4::from_scale(Vec3::splat(2.0))).with_mask(4),
    ];
    let blasses: Vec<Arc<dyn Blas>> = vec![Arc::new(voxels), Arc::new(mesh)];
    let mut tlas = Bvh::new();
    tlas.build_tlas(instances, blasses);
    assert_eq!(tlas.prim_count(), 3);
    let (mn, mx) = tlas.instances()[0].aabb();
    assert!(mn.abs_diff_eq(Vec3::new(-5.0, -2.0, -5.0), 1e-5) && mx.abs_diff_eq(Vec3::new(5.0, 8.0, 5.0), 1e-5));

    // Straight down: the voxel slab.
    let mut down = Ray::new(Vec3::ZERO, -Vec3::Y);
    tlas.intersect(&mut down);
    assert!(down.hit.is_hit());
    assert_eq!(down.hit.inst, 0);
    assert_eq!(down.hit.prim, 2, "voxel value");
    assert_relative_eq!(down.hit.t, 1.375, epsilon = 1e-3);
    assert!(tlas.is_occluded(&Ray::new(Vec3::ZERO, -Vec3::Y)));
    assert!(!tlas.is_occluded(&Ray::with_t_max(Vec3::ZERO, -Vec3::Y, 1.0)));

    // Along +Z: the triangle mesh.
    let mut forward = Ray::new(Vec3::ZERO, Vec3::Z);
    tlas.intersect(&mut forward);
    assert_eq!(forward.hit.inst, 1);
    assert_eq!(forward.hit.prim, 1);
    assert_relative_eq!(forward.hit.t, 3.0, epsilon = 1e-5);

    // Up: nothing.
    let mut up = Ray::new(Vec3::ZERO, Vec3::Y);
    tlas.intersect(&mut up);
    assert!(!up.hit.is_hit());

    // The second voxel instance, 2 units in size at x = 20..22: its slab top is at y = 0.125.
    let mut second = Ray::new(Vec3::new(21.0, 1.0, 1.0), -Vec3::Y);
    tlas.intersect(&mut second);
    assert_eq!(second.hit.inst, 2);
    assert_relative_eq!(second.hit.t, 0.875, epsilon = 1e-4);

    // Masks apply to voxel instances as to any other.
    let mut masked = Ray::new(Vec3::ZERO, -Vec3::Y).with_mask(2);
    tlas.intersect(&mut masked);
    assert!(!masked.hit.is_hit());
    assert!(!tlas.is_occluded(&masked));
    assert!(tlas.is_occluded(&Ray::new(Vec3::ZERO, -Vec3::Y).with_mask(1)));

    // The TLAS keeps the voxel set alive through the Arc.
    assert_eq!(tlas.blasses().len(), 2);
}

#[test]
fn stale_top_grid_is_detected() {
    // tinybvh traverses from the top grid: a voxel set without the matching update is simply
    // invisible. The wrapper refuses such queries instead of returning misses.
    let mut voxels = VoxelSet::new();
    voxels.set(3, 3, 3, 1);
    assert!(voxels.is_top_grid_stale());
    let c = center(3, 3, 3);
    let ray = Ray::new(Vec3::new(c.x, c.y, -1.0), Vec3::Z);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| voxels.is_occluded(&ray)));
    assert!(result.is_err());
    voxels.update_top_grid();
    assert!(voxels.is_occluded(&ray));
    voxels.set(3, 3, 3, 0);
    assert!(voxels.is_top_grid_stale());
    voxels.update_top_grid();
    assert!(!voxels.is_occluded(&ray));
}

#[test]
#[should_panic(expected = "update_top_grid")]
fn intersect_with_stale_top_grid_panics() {
    let mut voxels = VoxelSet::new();
    voxels.set(3, 3, 3, 1);
    let mut ray = Ray::new(Vec3::new(0.0, 0.0, -1.0), Vec3::Z);
    voxels.intersect(&mut ray);
}

#[test]
#[should_panic(expected = "outside the 256^3 object")]
fn out_of_range_set_panics() {
    VoxelSet::new().set(1, 256, 1, 1);
}

#[test]
#[should_panic(expected = "NaN")]
fn nan_ray_panics() {
    let mut voxels = VoxelSet::new();
    voxels.set(1, 1, 1, 1);
    voxels.update_top_grid();
    voxels.intersect(&mut Ray::new(Vec3::new(f32::NAN, 0.0, 0.0), Vec3::Z));
}

#[test]
fn shared_across_threads() {
    let mut voxels = VoxelSet::new();
    for i in 0..256 {
        voxels.set(i, i, i, 1);
    }
    voxels.update_top_grid();
    let voxels = Arc::new(voxels);
    let handles: Vec<_> = (0..4)
        .map(|t| {
            let voxels = voxels.clone();
            std::thread::spawn(move || {
                for i in (t..256).step_by(4) {
                    let c = center(i, i, i);
                    let mut ray = Ray::new(Vec3::new(c.x, c.y, -1.0), Vec3::Z);
                    voxels.intersect(&mut ray);
                    assert!(ray.hit.is_hit());
                    assert_relative_eq!(ray.hit.t, 1.0 + i as f32 / DIM, epsilon = 1e-4);
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
}
