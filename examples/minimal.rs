//! The smallest useful program: build a BVH over random triangles and trace one ray.
//! Mirrors tinybvh's `tiny_bvh_minimal.cpp`.

use glam::Vec3;
use tinybvh::{Bvh, Ray};

fn main() {
    // A simple deterministic generator, so the output is reproducible.
    let mut state = 0x9E3779B97F4A7C15u64;
    let mut rand = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u64 << 24) as f32
    };

    // 8192 small random triangles in the unit cube; three Vec4 per triangle, w unused.
    let mut vertices = Vec::with_capacity(8192 * 3);
    for _ in 0..8192 {
        let base = Vec3::new(rand(), rand(), rand());
        for _ in 0..3 {
            vertices.push((base + 0.1 * Vec3::new(rand(), rand(), rand())).extend(0.0));
        }
    }

    let mut bvh = Bvh::new();
    bvh.build(vertices);
    println!(
        "built a BVH over {} triangles: {} nodes, SAH cost {:.2}",
        bvh.prim_count(),
        bvh.node_count(),
        bvh.sah_cost(0)
    );

    let mut ray = Ray::new(Vec3::new(0.5, 0.5, -1.0), Vec3::new(0.1, 0.0, 2.0));
    let steps = bvh.intersect(&mut ray);
    if ray.hit.is_hit() {
        println!(
            "hit triangle {} at t = {:.4} ({steps} traversal steps)",
            ray.hit.prim, ray.hit.t
        );
    } else {
        println!("no hit ({steps} traversal steps)");
    }
}
