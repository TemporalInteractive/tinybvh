//! Animating a mesh: move the vertices in place and refit the tree instead of rebuilding it, then
//! compare the cost of refitting against a full rebuild.

use std::time::Instant;

use glam::{Vec3, Vec4};
use tinybvh::{BuildSettings, Bvh, Ray};

fn main() {
    // A wavy grid of 200 x 200 quads.
    let n = 200;
    let mut vertices = Vec::with_capacity(n * n * 6);
    let at = |i: usize, j: usize, t: f32| {
        let x = i as f32 / n as f32 * 10.0 - 5.0;
        let z = j as f32 / n as f32 * 10.0 - 5.0;
        Vec4::new(x, (x * 1.3 + t).sin() * (z * 0.7 - t).cos(), z, 0.0)
    };
    for j in 0..n {
        for i in 0..n {
            vertices.extend([at(i, j, 0.0), at(i + 1, j, 0.0), at(i, j + 1, 0.0), at(i + 1, j, 0.0), at(i + 1, j + 1, 0.0), at(i, j + 1, 0.0)]);
        }
    }

    // Refitting needs a tree without spatial splits.
    let mut bvh = Bvh::new().with_settings(&BuildSettings::refittable());
    let start = Instant::now();
    bvh.build(vertices);
    println!("initial build of {} triangles: {:.2?}", bvh.prim_count(), start.elapsed());

    for frame in 1..=5 {
        let t = frame as f32 * 0.25;
        for (k, v) in bvh.vertices_mut().unwrap().iter_mut().enumerate() {
            // Recompute the height of every vertex for this frame.
            let quad = k / 6;
            let corner = k % 6;
            let (i, j) = (quad % n, quad / n);
            let (di, dj) = [(0, 0), (1, 0), (0, 1), (1, 0), (1, 1), (0, 1)][corner];
            *v = at(i + di, j + dj, t);
        }
        let start = Instant::now();
        bvh.refit().unwrap();
        let refit_time = start.elapsed();

        let mut ray = Ray::new(Vec3::new(0.3, 10.0, 0.3), -Vec3::Y);
        bvh.intersect(&mut ray);
        println!("frame {frame}: refit {:.2?}, surface at y = {:.3}, SAH cost {:.2}", refit_time, ray.hit_point().y, bvh.sah_cost(0));
    }

    // For comparison, a full rebuild of the final frame.
    let geometry = bvh.into_geometry().unwrap();
    let mut rebuilt = Bvh::new();
    let start = Instant::now();
    rebuilt.build(geometry);
    println!("full rebuild: {:.2?}, SAH cost {:.2}", start.elapsed(), rebuilt.sah_cost(0));
}
