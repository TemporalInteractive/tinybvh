//! Shared helpers for the integration tests.
#![allow(dead_code)]

use glam::{DVec3, Vec3, Vec4};

/// Deterministic xorshift generator, so tests do not depend on an external crate.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.max(1))
    }

    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 32) as u32
    }

    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    pub fn unit_f64(&mut self) -> f64 {
        (self.next_u32() >> 8) as f64 / (1u32 << 24) as f64
    }
}

/// `count` small random triangles inside the unit cube, like tinybvh's minimal example.
pub fn random_triangles(count: usize, seed: u64) -> Vec<Vec4> {
    let mut rng = Rng::new(seed);
    let mut verts = Vec::with_capacity(count * 3);
    for _ in 0..count {
        let base = Vec3::new(rng.unit(), rng.unit(), rng.unit());
        for _ in 0..3 {
            let v = base + 0.1 * Vec3::new(rng.unit(), rng.unit(), rng.unit());
            verts.push(v.extend(0.0));
        }
    }
    verts
}

/// A single triangle in the plane z = `z`, spanning roughly [-1, 1] in x and y.
pub fn triangle_at(z: f32) -> [Vec4; 3] {
    [Vec4::new(-1.0, -1.0, z, 0.0), Vec4::new(1.0, -1.0, z, 0.0), Vec4::new(0.0, 1.0, z, 0.0)]
}

/// A regular grid of `n x n` quads (two triangles each) in the plane z = `z`, covering
/// [0, size] in x and y, as a triangle soup.
pub fn grid_soup(n: usize, size: f32, z: f32) -> Vec<Vec4> {
    let (verts, indices) = grid_indexed(n, size, z);
    indices.iter().map(|&i| verts[i as usize]).collect()
}

/// The same grid as indexed geometry.
pub fn grid_indexed(n: usize, size: f32, z: f32) -> (Vec<Vec4>, Vec<u32>) {
    let mut verts = Vec::with_capacity((n + 1) * (n + 1));
    for j in 0..=n {
        for i in 0..=n {
            verts.push(Vec4::new(i as f32 / n as f32 * size, j as f32 / n as f32 * size, z, 0.0));
        }
    }
    let mut indices = Vec::with_capacity(n * n * 6);
    let w = (n + 1) as u32;
    for j in 0..n as u32 {
        for i in 0..n as u32 {
            let a = j * w + i;
            indices.extend_from_slice(&[a, a + 1, a + w, a + 1, a + w + 1, a + w]);
        }
    }
    (verts, indices)
}

/// Double precision grid soup.
pub fn grid_soup_d(n: usize, size: f64, z: f64) -> Vec<DVec3> {
    let (verts, indices) = grid_indexed_d(n, size, z);
    indices.iter().map(|&i| verts[i as usize]).collect()
}

/// Double precision indexed grid.
pub fn grid_indexed_d(n: usize, size: f64, z: f64) -> (Vec<DVec3>, Vec<u64>) {
    let mut verts = Vec::with_capacity((n + 1) * (n + 1));
    for j in 0..=n {
        for i in 0..=n {
            verts.push(DVec3::new(i as f64 / n as f64 * size, j as f64 / n as f64 * size, z));
        }
    }
    let mut indices = Vec::with_capacity(n * n * 6);
    let w = (n + 1) as u64;
    for j in 0..n as u64 {
        for i in 0..n as u64 {
            let a = j * w + i;
            indices.extend_from_slice(&[a, a + 1, a + w, a + 1, a + w + 1, a + w]);
        }
    }
    (verts, indices)
}

/// Brute-force closest hit against a triangle soup (Möller–Trumbore), as a reference.
pub fn brute_force_hit(verts: &[Vec4], origin: Vec3, dir: Vec3) -> Option<(f32, usize)> {
    let mut best: Option<(f32, usize)> = None;
    for (tri, v) in verts.chunks_exact(3).enumerate() {
        let v0 = v[0].truncate();
        let e1 = v[1].truncate() - v0;
        let e2 = v[2].truncate() - v0;
        let h = dir.cross(e2);
        let a = e1.dot(h);
        if a.abs() < 1e-9 {
            continue;
        }
        let f = 1.0 / a;
        let s = origin - v0;
        let u = f * s.dot(h);
        if !(0.0..=1.0).contains(&u) {
            continue;
        }
        let q = s.cross(e1);
        let v = f * dir.dot(q);
        if v < 0.0 || u + v > 1.0 {
            continue;
        }
        let t = f * e2.dot(q);
        if t > 0.0 && best.map_or(true, |(bt, _)| t < bt) {
            best = Some((t, tri));
        }
    }
    best
}

/// Compares a layout's closest-hit results against the brute force reference for `rays` random
/// rays from outside the unit cube towards it. `intersect` receives the ray and must fill `hit`.
pub fn check_against_brute_force(
    verts: &[Vec4],
    rays: usize,
    seed: u64,
    mut intersect: impl FnMut(&mut tinybvh::Ray),
) {
    let mut rng = Rng::new(seed);
    let mut hits = 0;
    for _ in 0..rays {
        let origin = Vec3::new(rng.unit() * 2.0 - 0.5, rng.unit() * 2.0 - 0.5, -1.0);
        let target = Vec3::new(rng.unit(), rng.unit(), rng.unit());
        let dir = target - origin;
        let mut ray = tinybvh::Ray::new(origin, dir);
        intersect(&mut ray);
        let reference = brute_force_hit(verts, origin, dir);
        match reference {
            None => assert!(!ray.hit.is_hit(), "layout found a hit where brute force found none"),
            Some((t, _)) => {
                assert!(ray.hit.is_hit(), "layout missed a hit at t = {t}");
                assert!(
                    (ray.hit.t - t).abs() <= 1e-4 * t.max(1.0),
                    "distance mismatch: layout {} vs brute force {t}",
                    ray.hit.t
                );
                hits += 1;
            }
        }
    }
    assert!(hits > rays / 4, "too few hits ({hits} of {rays}) for the test to be meaningful");
}
