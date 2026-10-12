//! Tests for the double precision layouts `BvhDouble` and `Bvh4Double`: builders, queries,
//! maintenance, custom geometry, TLAS construction, serialization and input validation.

mod common;

use std::sync::Arc;

use approx::assert_relative_eq;
use glam::{DMat4, DVec3};
use tinybvh::{
    BlasEx, BlasInstanceEx, BuildSettings, Bvh4Double, BvhDouble, BvhNodeEx, CustomGeometryEx, Error, GeometryEx, Layout,
    RayEx, DBL_FAR,
};

use common::{grid_indexed_d, grid_soup_d, Rng};

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------

/// A single triangle in the plane z = `z`, spanning roughly [-1, 1] in x and y.
fn triangle_at(z: f64) -> [DVec3; 3] {
    [DVec3::new(-1.0, -1.0, z), DVec3::new(1.0, -1.0, z), DVec3::new(0.0, 1.0, z)]
}

fn two_triangles() -> Vec<DVec3> {
    let mut v = Vec::new();
    v.extend(triangle_at(5.0));
    v.extend(triangle_at(3.0));
    v
}

/// `count` small random triangles inside the unit cube.
fn random_triangles(count: usize, seed: u64) -> Vec<DVec3> {
    let mut rng = Rng::new(seed);
    let mut verts = Vec::with_capacity(count * 3);
    for _ in 0..count {
        let base = DVec3::new(rng.unit_f64(), rng.unit_f64(), rng.unit_f64());
        for _ in 0..3 {
            verts.push(base + 0.1 * DVec3::new(rng.unit_f64(), rng.unit_f64(), rng.unit_f64()));
        }
    }
    verts
}

/// Brute-force closest hit against a double precision triangle soup (Möller–Trumbore).
fn brute_force_hit(verts: &[DVec3], origin: DVec3, dir: DVec3) -> Option<(f64, usize)> {
    let mut best: Option<(f64, usize)> = None;
    for (tri, v) in verts.chunks_exact(3).enumerate() {
        let e1 = v[1] - v[0];
        let e2 = v[2] - v[0];
        let h = dir.cross(e2);
        let a = e1.dot(h);
        if a.abs() < 1e-18 {
            continue;
        }
        let f = 1.0 / a;
        let s = origin - v[0];
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
/// rays from outside the unit cube towards it.
fn check_against_brute_force(verts: &[DVec3], rays: usize, seed: u64, mut intersect: impl FnMut(&mut RayEx)) {
    let mut rng = Rng::new(seed);
    let mut hits = 0;
    for _ in 0..rays {
        let origin = DVec3::new(rng.unit_f64() * 2.0 - 0.5, rng.unit_f64() * 2.0 - 0.5, -1.0);
        let target = DVec3::new(rng.unit_f64(), rng.unit_f64(), rng.unit_f64());
        let dir = target - origin;
        let mut ray = RayEx::new(origin, dir);
        intersect(&mut ray);
        match brute_force_hit(verts, origin, dir) {
            None => assert!(!ray.hit.is_hit(), "layout found a hit where brute force found none"),
            Some((t, _)) => {
                assert!(ray.hit.is_hit(), "layout missed a hit at t = {t}");
                assert!((ray.hit.t - t).abs() <= 1e-9 * t.max(1.0), "distance mismatch: layout {} vs brute force {t}", ray.hit.t);
                hits += 1;
            }
        }
    }
    assert!(hits > rays / 4, "too few hits ({hits} of {rays}) for the test to be meaningful");
}

/// The message of a panic payload, whichever form `panic!` produced it in.
fn payload_str(payload: &Box<dyn std::any::Any + Send>) -> &str {
    payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("<non-string panic payload>")
}

/// A named build recipe for the builder comparison.
type Variant<T> = (&'static str, Box<dyn Fn(&mut T)>);

fn temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("tinybvh-rs-double-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

// ---------------------------------------------------------------------------------------------
// BvhDouble basics
// ---------------------------------------------------------------------------------------------

#[test]
fn node_layout_matches_tinybvh() {
    assert_eq!(core::mem::size_of::<BvhNodeEx>(), 64);
    assert_eq!(core::mem::align_of::<BvhNodeEx>(), 8);
    let leaf = BvhNodeEx { aabb_min: DVec3::ZERO, left_first: 4, aabb_max: DVec3::new(1.0, 2.0, 3.0), tri_count: 3 };
    assert!(leaf.is_leaf());
    assert_eq!(leaf.children(), None);
    assert_eq!(leaf.prim_range(), 4..7);
    assert_relative_eq!(leaf.half_area(), 2.0 + 6.0 + 3.0);
    let interior = BvhNodeEx { tri_count: 0, left_first: 8, ..leaf };
    assert!(!interior.is_leaf());
    assert_eq!(interior.children(), Some((8, 9)));
    assert_eq!(interior.prim_range(), 0..0);
}

#[test]
fn build_and_intersect_soup() {
    let mut bvh = BvhDouble::new();
    assert!(!bvh.is_built());
    assert!(format!("{bvh:?}").contains("BvhDouble"));
    bvh.build(two_triangles());
    assert!(bvh.is_built());
    assert_eq!(bvh.layout(), Layout::Bvh);
    assert_eq!(bvh.prim_count(), 2);
    assert!(!bvh.is_tlas() && !bvh.is_indexed() && !bvh.has_custom_geometry());
    assert!(!bvh.is_over_aabbs() && !bvh.is_over_indices());

    let mut ray = RayEx::new(DVec3::ZERO, DVec3::Z);
    let steps = bvh.intersect(&mut ray);
    assert!(steps > 0);
    assert!(ray.hit.is_hit());
    assert_relative_eq!(ray.hit.t, 3.0, epsilon = 1e-12);
    assert_eq!(ray.hit.prim, 1);
    assert_eq!(ray.hit.inst, 0);
    assert!(ray.hit_point().abs_diff_eq(DVec3::new(0.0, 0.0, 3.0), 1e-12));
    // Barycentrics: the hit at (0, 0) on the triangle (-1,-1), (1,-1), (0,1) has v = 0.5, u = 0.25.
    assert_relative_eq!(ray.hit.v, 0.5, epsilon = 1e-12);
    assert_relative_eq!(ray.hit.u, 0.25, epsilon = 1e-12);

    // A ray pointing away misses.
    let mut miss = RayEx::new(DVec3::ZERO, -DVec3::Z);
    bvh.intersect(&mut miss);
    assert!(!miss.hit.is_hit());
    assert_eq!(miss.hit.t, DBL_FAR);

    // t_max limits the search; is_occluded honours it too.
    let mut limited = RayEx::with_t_max(DVec3::ZERO, DVec3::Z, 2.0);
    bvh.intersect(&mut limited);
    assert_eq!(limited.hit.t, 2.0);
    assert!(bvh.is_occluded(&RayEx::new(DVec3::ZERO, DVec3::Z)));
    assert!(bvh.is_occluded(&RayEx::with_t_max(DVec3::ZERO, DVec3::Z, 4.0)));
    assert!(!bvh.is_occluded(&RayEx::with_t_max(DVec3::ZERO, DVec3::Z, 2.0)));
    assert!(!bvh.is_occluded(&RayEx::new(DVec3::ZERO, -DVec3::Z)));
    // hit.t is a strict upper bound: a ray that already holds the closest hit is not occluded.
    assert!(!bvh.is_occluded(&ray));
    let mut far = RayEx::with_t_max(DVec3::ZERO, DVec3::Z, 4.0);
    bvh.intersect(&mut far);
    assert_relative_eq!(far.hit.t, 3.0, epsilon = 1e-12);
    assert!(!bvh.is_occluded(&far));

    // The direction is not normalized: t scales with its length.
    let mut scaled = RayEx::new(DVec3::ZERO, DVec3::Z * 3.0);
    bvh.intersect(&mut scaled);
    assert_relative_eq!(scaled.hit.t, 1.0, epsilon = 1e-12);
    scaled.normalize();
    scaled.reset(DBL_FAR);
    bvh.intersect(&mut scaled);
    assert_relative_eq!(scaled.hit.t, 3.0, epsilon = 1e-12);

    // Precision beyond f32: triangles far from the origin still resolve sub-micron offsets.
    let far_off = 1e7;
    let mut precise = BvhDouble::new();
    let mut v = Vec::new();
    v.extend(triangle_at(far_off + 1e-6));
    v.extend(triangle_at(far_off));
    precise.build(v);
    let mut ray = RayEx::new(DVec3::ZERO, DVec3::Z);
    precise.intersect(&mut ray);
    assert_eq!(ray.hit.prim, 1);
    assert_relative_eq!(ray.hit.t, far_off, epsilon = 1e-9);
}

#[test]
fn soup_and_indexed_agree() {
    let (verts, indices) = grid_indexed_d(16, 4.0, 1.0);
    let soup = grid_soup_d(16, 4.0, 1.0);
    let mut a = BvhDouble::new();
    a.build(soup.clone());
    let mut b = BvhDouble::new();
    b.build((verts.clone(), indices.clone()));
    assert!(b.is_indexed());
    assert!(b.is_over_indices());
    assert_eq!(a.prim_count(), b.prim_count());
    assert_eq!(b.indices().unwrap(), &indices[..]);
    assert_eq!(b.geometry().unwrap().triangle_count(), 512);
    assert_eq!(b.geometry().unwrap().triangle(0), [verts[0], verts[1], verts[17]]);
    assert!(a.indices().is_none());

    let mut c = Bvh4Double::new();
    c.build(soup.clone());
    let mut d = Bvh4Double::new();
    d.build((verts.clone(), indices.clone()));
    assert_eq!(d.indices().unwrap(), &indices[..]);
    assert_eq!(c.prim_count(), 512);

    for i in 0..40 {
        let o = DVec3::new(0.1 + i as f64 * 0.09, 3.9 - i as f64 * 0.08, -2.0);
        let mut rays = [RayEx::new(o, DVec3::Z); 4];
        a.intersect(&mut rays[0]);
        b.intersect(&mut rays[1]);
        c.intersect(&mut rays[2]);
        d.intersect(&mut rays[3]);
        for r in &rays {
            assert!(r.hit.is_hit());
            assert_relative_eq!(r.hit.t, 3.0, epsilon = 1e-12);
            // A ray on a quad diagonal may legitimately report either triangle of the quad.
            assert_eq!(r.hit.prim / 2, rays[0].hit.prim / 2);
        }
        assert!(a.is_occluded(&RayEx::new(o, DVec3::Z)));
        assert!(b.is_occluded(&RayEx::new(o, DVec3::Z)));
        assert!(c.is_occluded(&RayEx::new(o, DVec3::Z)));
        assert!(d.is_occluded(&RayEx::new(o, DVec3::Z)));
        assert!(!c.is_occluded(&rays[2]));
        assert!(!d.is_occluded(&rays[3]));
    }

    // Indexed geometry can be taken back out.
    let geometry = b.into_geometry().unwrap();
    assert_eq!(geometry.vertices, verts);
    let geometry = d.into_geometry().unwrap();
    assert_eq!(geometry.indices.unwrap(), indices);
}

#[test]
fn builders_match_brute_force() {
    let verts = random_triangles(2000, 7);
    let variants: Vec<Variant<BvhDouble>> = vec![
        ("default", Box::new(|b: &mut BvhDouble| b.build(random_triangles(2000, 7)))),
        ("hq", Box::new(|b: &mut BvhDouble| b.build_hq(random_triangles(2000, 7)))),
        ("quick", Box::new(|b: &mut BvhDouble| b.build_quick(random_triangles(2000, 7)))),
        (
            "binned",
            Box::new(|b: &mut BvhDouble| {
                b.set_settings(&BuildSettings { binned_bvh: true, ..Default::default() });
                b.build(random_triangles(2000, 7))
            }),
        ),
        (
            "sbvh",
            Box::new(|b: &mut BvhDouble| {
                b.set_settings(&BuildSettings { binned_bvh: true, use_spatial_splits: true, ..Default::default() });
                b.build(random_triangles(2000, 7))
            }),
        ),
        (
            "high-quality-settings",
            Box::new(|b: &mut BvhDouble| {
                b.set_settings(&BuildSettings::high_quality());
                b.build(random_triangles(2000, 7))
            }),
        ),
        (
            "single-threaded",
            Box::new(|b: &mut BvhDouble| {
                b.set_settings(&BuildSettings { enable_threading: false, ..Default::default() });
                b.build(random_triangles(2000, 7))
            }),
        ),
        (
            "full-sweep",
            Box::new(|b: &mut BvhDouble| {
                b.set_settings(&BuildSettings { use_full_sweep: true, ..Default::default() });
                b.build(random_triangles(2000, 7))
            }),
        ),
        (
            "hploc",
            Box::new(|b: &mut BvhDouble| {
                b.set_settings(&BuildSettings { use_hploc: true, ..Default::default() });
                b.build(random_triangles(2000, 7))
            }),
        ),
        (
            "presplit+optimize",
            Box::new(|b: &mut BvhDouble| {
                b.set_settings(&BuildSettings { use_presplitting: true, post_optimize: true, ..Default::default() });
                b.build(random_triangles(2000, 7))
            }),
        ),
        ("indexed", Box::new(|b: &mut BvhDouble| b.build(indexed_copy(&random_triangles(2000, 7))))),
        ("indexed-hq", Box::new(|b: &mut BvhDouble| b.build_hq(indexed_copy(&random_triangles(2000, 7))))),
    ];
    for (name, build) in variants {
        let mut bvh = BvhDouble::new();
        build(&mut bvh);
        assert!(bvh.is_built(), "{name}");
        assert_eq!(bvh.prim_count(), 2000, "{name}");
        assert!(bvh.sah_cost(0) > 0.0, "{name}");
        assert!(bvh.index_count() >= 2000, "{name}");
        check_against_brute_force(&verts, 300, 11, |ray| {
            bvh.intersect(ray);
        });
        let steps = bvh.intersect(&mut RayEx::new(DVec3::new(0.5, 0.5, -1.0), DVec3::Z));
        assert!(steps > 0, "{name}");
    }

    let variants4: Vec<Variant<Bvh4Double>> = vec![
        ("default", Box::new(|b: &mut Bvh4Double| b.build(random_triangles(2000, 7)))),
        ("hq", Box::new(|b: &mut Bvh4Double| b.build_hq(random_triangles(2000, 7)))),
        ("indexed", Box::new(|b: &mut Bvh4Double| b.build(indexed_copy(&random_triangles(2000, 7))))),
        ("indexed-hq", Box::new(|b: &mut Bvh4Double| b.build_hq(indexed_copy(&random_triangles(2000, 7))))),
        (
            "sbvh",
            Box::new(|b: &mut Bvh4Double| {
                b.set_settings(&BuildSettings { binned_bvh: true, use_spatial_splits: true, ..Default::default() });
                b.build(random_triangles(2000, 7))
            }),
        ),
        (
            "discard-intermediates",
            Box::new(|b: &mut Bvh4Double| {
                b.set_settings(&BuildSettings { discard_intermediates: true, ..Default::default() });
                b.build(random_triangles(2000, 7))
            }),
        ),
    ];
    for (name, build) in variants4 {
        let mut bvh = Bvh4Double::new();
        build(&mut bvh);
        assert!(bvh.is_built(), "{name}");
        assert_eq!(bvh.layout(), Layout::Bvh4Cpu, "{name}");
        assert_eq!(bvh.prim_count(), 2000, "{name}");
        assert!(bvh.used_blocks() > 0, "{name}");
        assert_eq!(bvh.data().len(), bvh.used_blocks() as usize * 64, "{name}");
        check_against_brute_force(&verts, 300, 11, |ray| {
            bvh.intersect(ray);
        });
        // The 4-wide double precision kernel only counts traversal steps in debug builds of the
        // C++ library, so the step count is not asserted here.
        bvh.intersect(&mut RayEx::new(DVec3::new(0.5, 0.5, -1.0), DVec3::Z));
    }
}

/// The soup as indexed geometry with a shuffled vertex order.
fn indexed_copy(soup: &[DVec3]) -> GeometryEx {
    let n = soup.len();
    let mut verts: Vec<DVec3> = soup.iter().rev().copied().collect();
    let mut indices: Vec<u64> = (0..n as u64).map(|i| n as u64 - 1 - i).collect();
    // Duplicate the vertex that triangle 0 starts with, so the vertex count is not a multiple of
    // three either and the soup's triangles are still described exactly.
    verts.push(verts[n - 1]);
    indices[0] = n as u64;
    GeometryEx::indexed(verts, indices)
}

#[test]
fn large_threaded_build() {
    // Above tinybvh's threading threshold (50k primitives) the build runs on the pool.
    let verts = random_triangles(60_000, 99);
    let mut bvh = BvhDouble::new();
    bvh.build(verts.clone());
    assert_eq!(bvh.prim_count(), 60_000);
    assert!(bvh.node_count() > 1000);
    let mut bvh4 = Bvh4Double::new();
    bvh4.build(verts.clone());
    assert_eq!(bvh4.prim_count(), 60_000);

    let mut rng = Rng::new(5);
    for _ in 0..200 {
        let o = DVec3::new(rng.unit_f64(), rng.unit_f64(), -1.0);
        let d = DVec3::Z;
        let mut ray = RayEx::new(o, d);
        bvh.intersect(&mut ray);
        let mut ray4 = RayEx::new(o, d);
        bvh4.intersect(&mut ray4);
        let reference = brute_force_hit(&verts, o, d);
        assert_eq!(ray.hit.is_hit(), reference.is_some());
        assert_eq!(ray4.hit.is_hit(), reference.is_some());
        if let Some((t, prim)) = reference {
            assert_relative_eq!(ray.hit.t, t, epsilon = 1e-9);
            assert_relative_eq!(ray4.hit.t, t, epsilon = 1e-9);
            assert_eq!(ray.hit.prim as usize, prim);
            assert_eq!(ray4.hit.prim as usize, prim);
        }
    }
    // Refit works after a threaded build (compaction happens automatically for BvhDouble).
    bvh.refit().unwrap();
    bvh4.refit().unwrap();
    check_against_brute_force(&verts, 50, 12, |ray| {
        bvh4.intersect(ray);
    });
}

// ---------------------------------------------------------------------------------------------
// Maintenance
// ---------------------------------------------------------------------------------------------

#[test]
fn refit_follows_moved_vertices() {
    let mut bvh = BvhDouble::new().with_settings(&BuildSettings::refittable());
    bvh.build(two_triangles());
    assert!(bvh.is_refittable());
    for v in bvh.vertices_mut().unwrap() {
        v.z += 10.0;
    }
    bvh.refit().unwrap();
    let mut ray = RayEx::new(DVec3::ZERO, DVec3::Z);
    bvh.intersect(&mut ray);
    assert_relative_eq!(ray.hit.t, 13.0, epsilon = 1e-12);
    let (mn, mx) = bvh.aabb();
    assert_relative_eq!(mn.z, 13.0, epsilon = 1e-12);
    assert_relative_eq!(mx.z, 15.0, epsilon = 1e-12);
    assert_eq!(bvh.nodes()[0].aabb_min, mn);

    let mut bvh4 = Bvh4Double::new().with_settings(&BuildSettings::refittable());
    bvh4.build(two_triangles());
    assert!(bvh4.is_refittable() && bvh4.is_rebuildable());
    for v in bvh4.vertices_mut().unwrap() {
        v.z += 10.0;
    }
    bvh4.refit().unwrap();
    let mut ray = RayEx::new(DVec3::ZERO, DVec3::Z);
    bvh4.intersect(&mut ray);
    assert_relative_eq!(ray.hit.t, 13.0, epsilon = 1e-12);
    assert_eq!(ray.hit.prim, 1);
    let (mn, mx) = bvh4.aabb();
    assert_relative_eq!(mn.z, 13.0, epsilon = 1e-12);
    assert_relative_eq!(mx.z, 15.0, epsilon = 1e-12);

    // Indexed geometry refits too, for both layouts.
    let (verts, indices) = grid_indexed_d(8, 2.0, 1.0);
    let mut indexed = BvhDouble::new();
    indexed.build((verts.clone(), indices.clone()));
    for v in indexed.vertices_mut().unwrap() {
        v.z = 4.0;
    }
    indexed.refit().unwrap();
    let mut ray = RayEx::new(DVec3::new(1.0, 1.0, 0.0), DVec3::Z);
    indexed.intersect(&mut ray);
    assert_relative_eq!(ray.hit.t, 4.0, epsilon = 1e-12);
    let mut indexed4 = Bvh4Double::new();
    indexed4.build((verts, indices));
    for v in indexed4.vertices_mut().unwrap() {
        v.z = 4.0;
    }
    indexed4.refit().unwrap();
    let mut ray = RayEx::new(DVec3::new(1.0, 1.0, 0.0), DVec3::Z);
    indexed4.intersect(&mut ray);
    assert_relative_eq!(ray.hit.t, 4.0, epsilon = 1e-12);
}

#[test]
fn refit_is_refused_when_not_possible() {
    let mut empty = BvhDouble::new();
    assert_eq!(empty.refit(), Err(Error::NotBuilt));
    assert_eq!(empty.epo_cost(0), Err(Error::NotBuilt));
    assert_eq!(Bvh4Double::new().refit(), Err(Error::NotBuilt));
    assert_eq!(Bvh4Double::new().optimize(5), Err(Error::NotBuilt));

    let mut hq = BvhDouble::new();
    hq.set_settings(&BuildSettings { binned_bvh: true, use_spatial_splits: true, ..Default::default() });
    hq.build(random_triangles(500, 1));
    assert!(!hq.is_refittable());
    assert_eq!(hq.refit(), Err(Error::NotRefittable));

    let mut hq4 = Bvh4Double::new();
    hq4.build_hq(random_triangles(500, 1));
    assert!(!hq4.is_refittable());
    assert_eq!(hq4.refit(), Err(Error::NotRefittable));
    hq4.optimize(2).unwrap(); // optimizing does not need a refittable tree

    let mut custom = BvhDouble::new();
    custom.build_custom(Spheres(vec![(DVec3::ZERO, 1.0)]));
    assert_eq!(custom.refit(), Err(Error::NoGeometry));
    assert_eq!(custom.epo_cost(0), Err(Error::NoGeometry));

    let mut blas = BvhDouble::new();
    blas.build(two_triangles());
    let mut tlas = BvhDouble::new();
    tlas.build_tlas(vec![BlasInstanceEx::new(0)], vec![Arc::new(blas)]);
    assert_eq!(tlas.refit(), Err(Error::NoGeometry));

    let mut merged = BvhDouble::new();
    merged.build(random_triangles(500, 2));
    merged.merge_leafs();
    assert!(merged.may_have_holes());
    merged.refit().unwrap(); // compacts first, which clears the holes flag
    assert!(!merged.may_have_holes());
    merged.optimize(2);

    let mut discarded = Bvh4Double::new();
    discarded.set_settings(&BuildSettings { discard_intermediates: true, ..Default::default() });
    discarded.build(two_triangles());
    assert!(discarded.is_built());
    assert_eq!(discarded.refit(), Err(Error::NotRebuildable));
    assert_eq!(discarded.optimize(5), Err(Error::NotRebuildable));
    // Rebuilding with the default settings restores the intermediate tree.
    discarded.set_settings(&BuildSettings::default());
    discarded.build(two_triangles());
    discarded.refit().unwrap();
    discarded.optimize(5).unwrap();
}

#[test]
fn maintenance_operations() {
    let verts = random_triangles(3000, 21);
    let mut bvh = BvhDouble::new();
    bvh.set_settings(&BuildSettings { enable_threading: false, ..Default::default() });
    bvh.build(verts.clone());
    let nodes_before = bvh.node_count();
    let leaves_before = bvh.leaf_count();
    assert!(nodes_before > leaves_before);
    assert!(nodes_before <= bvh.used_nodes());
    assert_eq!(bvh.prim_count_in(0), 3000);
    let epo = bvh.epo_cost(0).unwrap();
    assert!(epo >= 0.0);

    let before = bvh.sah_cost(0);
    bvh.optimize(10);
    let after = bvh.sah_cost(0);
    assert!(after <= before * 1.001, "optimize made the SAH cost worse: {before} -> {after}");
    check_against_brute_force(&verts, 100, 2, |ray| {
        bvh.intersect(ray);
    });
    bvh.optimize_with(2, true, true);
    check_against_brute_force(&verts, 50, 5, |ray| {
        bvh.intersect(ray);
    });

    bvh.split_leafs(1).unwrap();
    for node in bvh.nodes() {
        if node.is_leaf() {
            assert_eq!(node.tri_count, 1);
        }
    }
    assert_eq!(bvh.leaf_count(), 3000);
    check_against_brute_force(&verts, 50, 3, |ray| {
        bvh.intersect(ray);
    });
    bvh.merge_leafs();
    assert!(bvh.leaf_count() < bvh.prim_count());
    assert!(bvh.may_have_holes());
    bvh.compact();
    assert!(bvh.has_ordered_nodes());
    assert!(!bvh.may_have_holes(), "compaction clears the holes flag left by merge_leafs");
    bvh.optimize(2);
    bvh.refit().unwrap();
    check_against_brute_force(&verts, 50, 4, |ray| {
        bvh.intersect(ray);
    });

    // Node data is consistent: the root covers everything, leaves reference valid primitives,
    // the subtree primitive counts add up.
    let nodes = bvh.nodes();
    let prim_indices = bvh.prim_indices();
    let (mn, mx) = bvh.aabb();
    assert_eq!(nodes[0].aabb_min, mn);
    assert_eq!(nodes[0].aabb_max, mx);
    let mut seen = vec![false; 3000];
    for (i, node) in nodes.iter().enumerate() {
        if i == 1 {
            continue; // padding slot
        }
        if node.is_leaf() {
            for &p in &prim_indices[node.prim_range()] {
                seen[p as usize] = true;
            }
        } else {
            let (l, r) = node.children().unwrap();
            assert!(l < nodes.len() as u64 && r < nodes.len() as u64);
            assert_eq!(bvh.prim_count_in(i as u64), bvh.prim_count_in(l) + bvh.prim_count_in(r));
            // Children are enclosed by their parent.
            for c in [l, r] {
                let child = &nodes[c as usize];
                assert!(child.aabb_min.cmpge(node.aabb_min).all() && child.aabb_max.cmple(node.aabb_max).all());
            }
        }
    }
    assert!(seen.iter().all(|&s| s), "every primitive is referenced by a leaf");
    assert!(bvh.sah_cost(nodes[0].children().unwrap().0) > 0.0);

    // The 4-wide layout optimizes through its intermediate tree.
    let mut bvh4 = Bvh4Double::new();
    bvh4.build(verts.clone());
    let before = bvh4.sah_cost();
    assert!(before > 0.0);
    bvh4.optimize(10).unwrap();
    let after = bvh4.sah_cost();
    assert!(after <= before * 1.001, "optimize made the SAH cost worse: {before} -> {after}");
    bvh4.optimize_with(2, true).unwrap();
    check_against_brute_force(&verts, 100, 6, |ray| {
        bvh4.intersect(ray);
    });
}

#[test]
fn sphere_query() {
    let mut bvh = BvhDouble::new();
    bvh.build(two_triangles());
    assert!(bvh.intersect_sphere(DVec3::new(0.0, 0.0, 3.2), 0.5));
    assert!(bvh.intersect_sphere(DVec3::new(0.0, 0.0, 4.9), 0.2));
    assert!(!bvh.intersect_sphere(DVec3::new(0.0, 0.0, 0.0), 0.5));
    assert!(!bvh.intersect_sphere(DVec3::new(5.0, 5.0, 3.0), 0.5));
    assert!(!bvh.intersect_sphere(DVec3::new(0.0, 0.0, 4.0), 0.5));
}

#[test]
fn settings_and_costs_round_trip() {
    let mut bvh = BvhDouble::new();
    assert_eq!(bvh.settings(), BuildSettings::default());
    let custom = BuildSettings { binned_bvh: true, enable_threading: false, optimize_iterations: 12, ..Default::default() };
    bvh.set_settings(&custom);
    assert_eq!(bvh.settings(), custom);
    bvh.set_c_trav(2.5);
    bvh.set_c_int(0.75);
    bvh.set_hq_bins(16);
    bvh.set_hq_odd_even(true);
    bvh.set_hq_leaf_prims(3);
    assert_eq!(bvh.c_trav(), 2.5);
    assert_eq!(bvh.c_int(), 0.75);
    assert_eq!(bvh.hq_bins(), 16);
    assert!(bvh.hq_odd_even());
    assert_eq!(bvh.hq_leaf_prims(), 3);
    bvh.set_hq_bins(1000);
    assert_eq!(bvh.hq_bins(), 128);
    assert_eq!(bvh.allocated_nodes(), 0);

    let mut bvh4 = Bvh4Double::new().with_settings(&custom);
    assert_eq!(bvh4.settings(), custom);
    bvh4.set_c_trav(1.5);
    assert_eq!(bvh4.c_trav(), 1.5);
    bvh4.build(two_triangles());
    assert!(bvh4.used_blocks() > 0);
    // tinybvh does not maintain the node counters for this layout.
    assert_eq!(bvh4.used_nodes(), 0);
    let (mn, mx) = bvh4.aabb();
    assert_eq!(mn, DVec3::new(-1.0, -1.0, 3.0));
    assert_eq!(mx, DVec3::new(1.0, 1.0, 5.0));
}

// ---------------------------------------------------------------------------------------------
// Custom geometry
// ---------------------------------------------------------------------------------------------

/// Spheres as custom primitives.
struct Spheres(Vec<(DVec3, f64)>);

impl Spheres {
    fn hit(&self, ray: &RayEx, prim: u64) -> Option<f64> {
        let (c, r) = self.0[prim as usize];
        let oc = ray.origin - c;
        let d = ray.direction();
        let a = d.dot(d);
        let b = 2.0 * oc.dot(d);
        let cc = oc.dot(oc) - r * r;
        let disc = b * b - 4.0 * a * cc;
        if disc < 0.0 {
            return None;
        }
        let t = (-b - disc.sqrt()) / (2.0 * a);
        (t > 0.0 && t < ray.hit.t).then_some(t)
    }
}

impl CustomGeometryEx for Spheres {
    fn prim_count(&self) -> u64 {
        self.0.len() as u64
    }
    fn aabb(&self, prim: u64) -> (DVec3, DVec3) {
        let (c, r) = self.0[prim as usize];
        (c - r, c + r)
    }
    fn intersect(&self, ray: &mut RayEx, prim: u64) -> bool {
        match self.hit(ray, prim) {
            Some(t) => {
                ray.hit.t = t;
                ray.hit.prim = prim;
                ray.hit.user[0] = 0xC0FFEE;
                ray.hit.user[1] = ray.inst_idx() as u32;
                ray.hit.user[2] = ray.mask;
                true
            }
            None => false,
        }
    }
    fn is_occluded(&self, ray: &RayEx, prim: u64) -> bool {
        self.hit(ray, prim).is_some()
    }
}

#[test]
fn custom_geometry() {
    let mut bvh = BvhDouble::new();
    bvh.build_custom(Spheres(vec![
        (DVec3::new(0.0, 0.0, 5.0), 1.0),
        (DVec3::new(8.0, 0.0, 0.0), 1.0),
        (DVec3::new(0.0, -7.0, 0.0), 2.0),
    ]));
    assert!(bvh.has_custom_geometry());
    assert!(bvh.is_over_aabbs());
    assert_eq!(bvh.prim_count(), 3);
    assert!(bvh.custom_geometry().is_some());
    assert!(bvh.geometry().is_none() && bvh.vertices().is_none());
    let (mn, mx) = bvh.aabb();
    assert_eq!(mn, DVec3::new(-2.0, -9.0, -2.0));
    assert_eq!(mx, DVec3::new(9.0, 1.0, 6.0));

    let mut ray = RayEx::new(DVec3::ZERO, DVec3::Z);
    bvh.intersect(&mut ray);
    assert_relative_eq!(ray.hit.t, 4.0, epsilon = 1e-12);
    assert_eq!(ray.hit.prim, 0);
    assert_eq!(ray.hit.user[0], 0xC0FFEE);
    assert_eq!(ray.hit.user[1], 0);

    let mut ray = RayEx::new(DVec3::ZERO, -DVec3::Y);
    bvh.intersect(&mut ray);
    assert_relative_eq!(ray.hit.t, 5.0, epsilon = 1e-12);
    assert_eq!(ray.hit.prim, 2);

    assert!(bvh.is_occluded(&RayEx::new(DVec3::ZERO, DVec3::X)));
    assert!(!bvh.is_occluded(&RayEx::with_t_max(DVec3::ZERO, DVec3::X, 6.0)));
    assert!(!bvh.is_occluded(&RayEx::new(DVec3::ZERO, DVec3::Y)));

    // Boxed geometry works the same way.
    let mut boxed = BvhDouble::new();
    boxed.build_custom_boxed(Box::new(Spheres(vec![(DVec3::new(0.0, 0.0, 5.0), 1.0)])));
    let mut ray = RayEx::new(DVec3::ZERO, DVec3::Z);
    boxed.intersect(&mut ray);
    assert_relative_eq!(ray.hit.t, 4.0, epsilon = 1e-12);

    // Rebuilding over triangles releases the custom geometry and its callbacks.
    boxed.build(two_triangles());
    assert!(!boxed.has_custom_geometry());
    assert!(boxed.custom_geometry().is_none());
    let mut ray = RayEx::new(DVec3::ZERO, DVec3::Z);
    boxed.intersect(&mut ray);
    assert_relative_eq!(ray.hit.t, 3.0, epsilon = 1e-12);
    assert_eq!(ray.hit.user[0], 0);

    // And back again.
    boxed.build_custom(Spheres(vec![(DVec3::new(0.0, 0.0, 7.0), 1.0)]));
    let mut ray = RayEx::new(DVec3::ZERO, DVec3::Z);
    boxed.intersect(&mut ray);
    assert_relative_eq!(ray.hit.t, 6.0, epsilon = 1e-12);
}

#[test]
fn panic_in_callback_propagates() {
    struct Bad;
    impl CustomGeometryEx for Bad {
        fn prim_count(&self) -> u64 {
            1
        }
        fn aabb(&self, _: u64) -> (DVec3, DVec3) {
            (DVec3::ZERO, DVec3::ONE)
        }
        fn intersect(&self, _: &mut RayEx, _: u64) -> bool {
            panic!("boom from intersect")
        }
        fn is_occluded(&self, _: &RayEx, _: u64) -> bool {
            false
        }
    }
    let mut bvh = BvhDouble::new();
    bvh.build_custom(Bad);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut ray = RayEx::new(DVec3::new(0.5, 0.5, -1.0), DVec3::Z);
        bvh.intersect(&mut ray);
    }));
    let err = result.expect_err("the panic must surface on the caller");
    assert_eq!(payload_str(&err), "boom from intersect");
    // The structure is still usable afterwards.
    assert!(!bvh.is_occluded(&RayEx::new(DVec3::new(0.5, 0.5, -1.0), DVec3::Z)));

    // A panic in the bounds callback surfaces from the build.
    struct BadBounds;
    impl CustomGeometryEx for BadBounds {
        fn prim_count(&self) -> u64 {
            2
        }
        fn aabb(&self, prim: u64) -> (DVec3, DVec3) {
            assert!(prim < 1, "boom from aabb");
            (DVec3::ZERO, DVec3::ONE)
        }
        fn intersect(&self, _: &mut RayEx, _: u64) -> bool {
            false
        }
        fn is_occluded(&self, _: &RayEx, _: u64) -> bool {
            false
        }
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut bvh = BvhDouble::new();
        bvh.build_custom(BadBounds);
    }));
    let err = result.expect_err("the panic must surface from the build");
    assert_eq!(payload_str(&err), "boom from aabb");
}

// ---------------------------------------------------------------------------------------------
// TLAS
// ---------------------------------------------------------------------------------------------

#[test]
fn tlas_with_instances_and_masks() {
    let mut blas = BvhDouble::new();
    blas.build(two_triangles());
    let blas: Arc<dyn BlasEx> = Arc::new(blas);
    let mut blas4 = Bvh4Double::new();
    blas4.build(vec![DVec3::new(-1.0, -1.0, 7.0), DVec3::new(1.0, -1.0, 7.0), DVec3::new(0.0, 1.0, 7.0)]);
    let blas4: Arc<dyn BlasEx> = Arc::new(blas4);

    let instances = vec![
        BlasInstanceEx::new(0),
        BlasInstanceEx::with_transform(0, DMat4::from_translation(DVec3::new(10.0, 0.0, 0.0))).with_mask(2),
        BlasInstanceEx::with_transform(0, DMat4::from_translation(DVec3::new(0.0, 0.0, -10.0))).with_mask(4),
        BlasInstanceEx::with_transform(1, DMat4::from_translation(DVec3::new(0.0, 20.0, 0.0))),
    ];
    let mut tlas = BvhDouble::new();
    tlas.build_tlas(instances, vec![blas.clone(), blas4.clone()]);
    assert!(tlas.is_tlas());
    assert!(tlas.is_over_aabbs());
    assert_eq!(tlas.prim_count(), 4);
    assert_eq!(tlas.instances().len(), 4);
    assert_eq!(tlas.blasses().len(), 2);
    assert_eq!(Arc::strong_count(&blas), 2);
    assert_eq!(Arc::strong_count(&blas4), 2);
    assert!(tlas.geometry().is_none() && tlas.vertices().is_none());

    // Instance bounds were computed by the build.
    let (mn, mx) = tlas.instances()[1].aabb();
    assert_relative_eq!(mn.x, 9.0, epsilon = 1e-12);
    assert_relative_eq!(mx.x, 11.0, epsilon = 1e-12);
    assert!(tlas.instances()[1].inverse_transform().abs_diff_eq(DMat4::from_translation(DVec3::new(-10.0, 0.0, 0.0)), 1e-12));
    let (mn, mx) = tlas.instances()[3].aabb();
    assert_eq!(mn, DVec3::new(-1.0, 19.0, 7.0));
    assert_eq!(mx, DVec3::new(1.0, 21.0, 7.0));
    let (mn, mx) = tlas.aabb();
    assert_eq!(mn, DVec3::new(-1.0, -1.0, -7.0));
    assert_eq!(mx, DVec3::new(11.0, 21.0, 7.0));

    let mut ray = RayEx::new(DVec3::ZERO, DVec3::Z);
    let steps = tlas.intersect(&mut ray);
    assert!(steps > 0);
    assert_relative_eq!(ray.hit.t, 3.0, epsilon = 1e-12);
    assert_eq!(ray.hit.inst, 0);
    assert_eq!(ray.hit.prim, 1);
    assert_relative_eq!(ray.hit.u, 0.25, epsilon = 1e-12);

    let mut ray = RayEx::new(DVec3::new(10.0, 0.0, 0.0), DVec3::Z);
    tlas.intersect(&mut ray);
    assert_eq!(ray.hit.inst, 1);
    assert_eq!(ray.hit.prim, 1);
    assert_relative_eq!(ray.hit.t, 3.0, epsilon = 1e-12);

    // The 4-wide BLAS is traversed through the TLAS too.
    let mut ray = RayEx::new(DVec3::new(0.0, 20.0, 0.0), DVec3::Z);
    tlas.intersect(&mut ray);
    assert_eq!(ray.hit.inst, 3);
    assert_eq!(ray.hit.prim, 0);
    assert_relative_eq!(ray.hit.t, 7.0, epsilon = 1e-12);
    assert!(tlas.is_occluded(&RayEx::new(DVec3::new(0.0, 20.0, 0.0), DVec3::Z)));
    assert!(!tlas.is_occluded(&RayEx::with_t_max(DVec3::new(0.0, 20.0, 0.0), DVec3::Z, 6.0)));

    // A ray that starts past everything misses.
    let mut ray = RayEx::new(DVec3::new(0.0, 0.0, 8.0), DVec3::Z);
    tlas.intersect(&mut ray);
    assert!(!ray.hit.is_hit());

    // Masks: instance 1 only answers rays with bit 1 set.
    let mut masked = RayEx::new(DVec3::new(10.0, 0.0, 0.0), DVec3::Z).with_mask(1);
    tlas.intersect(&mut masked);
    assert!(!masked.hit.is_hit());
    assert!(!tlas.is_occluded(&masked));
    assert!(tlas.is_occluded(&RayEx::new(DVec3::new(10.0, 0.0, 0.0), DVec3::Z).with_mask(2)));
    assert!(tlas.is_occluded(&RayEx::new(DVec3::new(10.0, 0.0, 0.0), DVec3::Z)));

    // Move an instance and rebuild.
    tlas.instances_mut()[0].set_transform(DMat4::from_translation(DVec3::new(0.0, 0.0, 20.0)));
    tlas.rebuild_tlas().unwrap();
    let mut ray = RayEx::new(DVec3::ZERO, DVec3::Z);
    tlas.intersect(&mut ray);
    assert_relative_eq!(ray.hit.t, 23.0, epsilon = 1e-12);
    assert_eq!(ray.hit.inst, 0);
    let (mn, _) = tlas.instances()[0].aabb();
    assert_relative_eq!(mn.z, 23.0, epsilon = 1e-12);

    // Instances can be refreshed by hand as well.
    let mut inst = BlasInstanceEx::with_transform(0, DMat4::from_scale(DVec3::splat(2.0)));
    inst.update(&*blas);
    assert_eq!(inst.aabb(), (DVec3::new(-2.0, -2.0, 6.0), DVec3::new(2.0, 2.0, 10.0)));

    // Operations that do not apply to a TLAS.
    let mut blas_only = BvhDouble::new();
    blas_only.build(two_triangles());
    assert_eq!(blas_only.rebuild_tlas(), Err(Error::NotTlas));
    assert_eq!(tlas.save("unused.bvh", true), Err(Error::NotTlas));
    assert_eq!(tlas.epo_cost(0), Err(Error::NoGeometry));
    assert!(tlas.sah_cost(0) > 0.0);
    assert_eq!(tlas.prim_count_in(0), 4);
    assert!(tlas.instances_mut().len() == 4 && blas_only.instances_mut().is_empty());

    // Dropping the TLAS releases the BLASes.
    drop(tlas);
    assert_eq!(Arc::strong_count(&blas), 1);
    assert_eq!(Arc::strong_count(&blas4), 1);
}

#[test]
fn tlas_over_custom_blas_passes_state() {
    let mut blas = BvhDouble::new();
    blas.build_custom(Spheres(vec![(DVec3::new(0.0, 0.0, 5.0), 1.0)]));
    let mut tlas = BvhDouble::new();
    tlas.build_tlas(
        vec![
            BlasInstanceEx::new(0).with_mask(1),
            BlasInstanceEx::with_transform(0, DMat4::from_translation(DVec3::X * 20.0)).with_mask(2),
        ],
        vec![Arc::new(blas)],
    );
    let mut ray = RayEx::new(DVec3::new(20.0, 0.0, 0.0), DVec3::Z).with_mask(2);
    tlas.intersect(&mut ray);
    assert_relative_eq!(ray.hit.t, 4.0, epsilon = 1e-12);
    assert_eq!(ray.hit.inst, 1);
    assert_eq!(ray.hit.prim, 0);
    // The callback saw the instance being traversed and the ray mask.
    assert_eq!(ray.hit.user[0], 0xC0FFEE);
    assert_eq!(ray.hit.user[1], 1);
    assert_eq!(ray.hit.user[2], 2);
    assert_eq!(ray.inst_idx(), 0, "the instance index is restored after traversal");

    let mut ray = RayEx::new(DVec3::ZERO, DVec3::Z).with_mask(1);
    tlas.intersect(&mut ray);
    assert_eq!(ray.hit.inst, 0);
    assert_eq!(ray.hit.user[1], 0);
    assert!(tlas.is_occluded(&RayEx::new(DVec3::ZERO, DVec3::Z).with_mask(1)));
    assert!(!tlas.is_occluded(&RayEx::new(DVec3::ZERO, DVec3::Z).with_mask(2)));
}

// ---------------------------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------------------------

#[test]
fn save_and_load_bvh_double() {
    let dir = temp_dir("bvh");
    let path = dir.join("tree.bvh");
    let verts = random_triangles(1000, 5);

    let mut bvh = BvhDouble::new();
    bvh.build(verts.clone());
    bvh.save(&path, true).unwrap();
    let reference_nodes = bvh.nodes().to_vec();

    // Self-contained: geometry comes from the file.
    let loaded = BvhDouble::load_self_contained(&path).unwrap();
    assert!(loaded.is_built());
    assert_eq!(loaded.prim_count(), 1000);
    assert_eq!(loaded.nodes(), &reference_nodes[..]);
    assert_eq!(loaded.vertices().unwrap(), &verts[..]);
    assert!(loaded.indices().is_none());
    assert!(loaded.geometry().is_none(), "loaded geometry is owned by tinybvh");
    check_against_brute_force(&verts, 100, 8, |ray| {
        loaded.intersect(ray);
    });
    assert!(loaded.intersect_sphere(verts[0], 0.01));
    assert!(loaded.epo_cost(0).unwrap() >= 0.0);

    // Bound to caller geometry.
    let mut bound = BvhDouble::new();
    bound.load(&path, verts.clone()).unwrap();
    assert_eq!(bound.nodes(), &reference_nodes[..]);
    assert_eq!(bound.geometry().unwrap().vertices, verts);
    check_against_brute_force(&verts, 100, 9, |ray| {
        bound.intersect(ray);
    });
    // ... and refittable afterwards.
    for v in bound.vertices_mut().unwrap() {
        v.z += 1.0;
    }
    bound.refit().unwrap();
    let (mn, _) = bound.aabb();
    assert!(mn.z >= 1.0);

    // Wrong geometry is rejected and leaves the structure empty.
    let mut wrong = BvhDouble::new();
    assert!(matches!(wrong.load(&path, random_triangles(10, 1)), Err(Error::LoadFailed(_))));
    assert!(!wrong.is_built());
    let (gv, gi) = grid_indexed_d(4, 1.0, 0.0);
    assert!(matches!(wrong.load(&path, (gv, gi)), Err(Error::LoadFailed(_))), "indexing must match");
    assert!(matches!(BvhDouble::load_self_contained(dir.join("missing.bvh")), Err(Error::LoadFailed(_))));
    assert!(matches!(wrong.load(dir.join("missing.bvh"), verts.clone()), Err(Error::LoadFailed(_))));

    // Without geometry a self-contained load is refused.
    let nodes_only = dir.join("nodes.bvh");
    bvh.save(&nodes_only, false).unwrap();
    assert!(std::fs::metadata(&nodes_only).unwrap().len() < std::fs::metadata(&path).unwrap().len());
    assert!(matches!(BvhDouble::load_self_contained(&nodes_only), Err(Error::LoadFailed(_))));
    let mut bound2 = BvhDouble::new();
    bound2.load(&nodes_only, verts.clone()).unwrap();
    assert_eq!(bound2.nodes(), &reference_nodes[..]);

    // Indexed geometry round trips with its indices.
    let (gv, gi) = grid_indexed_d(8, 2.0, 1.0);
    let mut indexed = BvhDouble::new();
    indexed.build((gv.clone(), gi.clone()));
    let indexed_path = dir.join("indexed.bvh");
    indexed.save(&indexed_path, true).unwrap();
    let loaded = BvhDouble::load_self_contained(&indexed_path).unwrap();
    assert!(loaded.is_indexed());
    assert_eq!(loaded.vertices().unwrap(), &gv[..]);
    assert_eq!(loaded.indices().unwrap(), &gi[..]);
    let mut ray = RayEx::new(DVec3::new(1.0, 1.0, 0.0), DVec3::Z);
    loaded.intersect(&mut ray);
    assert_relative_eq!(ray.hit.t, 1.0, epsilon = 1e-12);

    assert_eq!(BvhDouble::new().save(&path, true), Err(Error::NotBuilt));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn save_and_load_bvh4_double() {
    let dir = temp_dir("bvh4");
    let path = dir.join("tree.bvh4");
    let verts = random_triangles(1000, 5);

    let mut bvh = Bvh4Double::new();
    bvh.build(verts.clone());
    bvh.save(&path).unwrap();
    let reference = bvh.data().to_vec();
    assert!(!reference.is_empty());

    // Bound to caller geometry (only the triangle count is checked).
    let mut bound = Bvh4Double::new();
    bound.load(&path, verts.clone()).unwrap();
    assert!(bound.is_built());
    assert_eq!(bound.prim_count(), 1000);
    assert_eq!(bound.data(), &reference[..]);
    assert_eq!(bound.geometry().unwrap().vertices, verts);
    assert!(bound.vertices_mut().is_none(), "a loaded layout does not reference its geometry");
    assert_eq!(bound.refit(), Err(Error::NotRebuildable));
    assert_eq!(bound.optimize(5), Err(Error::NotRebuildable));
    check_against_brute_force(&verts, 100, 9, |ray| {
        bound.intersect(ray);
    });

    // Self-contained.
    let mut loaded = Bvh4Double::load_self_contained(&path).unwrap();
    assert!(loaded.is_built());
    assert_eq!(loaded.prim_count(), 1000);
    assert_eq!(loaded.data(), &reference[..]);
    assert!(loaded.geometry().is_none());
    assert_eq!(loaded.refit(), Err(Error::NotRebuildable));
    check_against_brute_force(&verts, 100, 8, |ray| {
        loaded.intersect(ray);
    });
    assert!(loaded.into_geometry().is_none());

    // The wrong triangle count or a missing file is rejected and leaves the structure empty.
    let mut wrong = Bvh4Double::new();
    wrong.build(two_triangles());
    assert!(matches!(wrong.load(&path, random_triangles(10, 1)), Err(Error::LoadFailed(_))));
    assert!(!wrong.is_built());
    assert!(wrong.data().is_empty());
    assert!(matches!(wrong.load(dir.join("missing.bvh4"), verts.clone()), Err(Error::LoadFailed(_))));
    assert!(matches!(Bvh4Double::load_self_contained(dir.join("missing.bvh4")), Err(Error::LoadFailed(_))));

    // A BvhDouble file is not a Bvh4Double file for `load`...
    let bvh_path = dir.join("binary.bvh");
    let mut binary = BvhDouble::new();
    binary.build(verts.clone());
    binary.save(&bvh_path, true).unwrap();
    assert!(matches!(wrong.load(&bvh_path, verts.clone()), Err(Error::LoadFailed(_))));
    // ... but `load_self_contained` converts a BvhDouble file saved with its geometry.
    let converted = Bvh4Double::load_self_contained(&bvh_path).unwrap();
    assert_eq!(converted.prim_count(), 1000);
    check_against_brute_force(&verts, 100, 10, |ray| {
        converted.intersect(ray);
    });
    let nodes_only = dir.join("nodes.bvh");
    binary.save(&nodes_only, false).unwrap();
    assert!(matches!(Bvh4Double::load_self_contained(&nodes_only), Err(Error::LoadFailed(_))));

    // A loaded layout can be rebuilt from scratch and then refitted again.
    let mut again = Bvh4Double::load_self_contained(&path).unwrap();
    again.build(two_triangles());
    again.refit().unwrap();

    assert_eq!(Bvh4Double::new().save(&path), Err(Error::NotBuilt));
    std::fs::remove_dir_all(&dir).ok();
}

// ---------------------------------------------------------------------------------------------
// Threads, lifecycle
// ---------------------------------------------------------------------------------------------

#[test]
fn shared_across_threads() {
    let verts = random_triangles(5000, 77);
    let mut bvh = BvhDouble::new();
    bvh.build(verts.clone());
    let bvh = Arc::new(bvh);
    let mut bvh4 = Bvh4Double::new();
    bvh4.build(verts.clone());
    let bvh4 = Arc::new(bvh4);
    let handles: Vec<_> = (0..4)
        .map(|t| {
            let bvh = bvh.clone();
            let bvh4 = bvh4.clone();
            let verts = verts.clone();
            std::thread::spawn(move || {
                check_against_brute_force(&verts, 100, 100 + t, |ray| {
                    bvh.intersect(ray);
                });
                check_against_brute_force(&verts, 100, 200 + t, |ray| {
                    bvh4.intersect(ray);
                });
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn clear_and_rebuild() {
    let mut bvh = BvhDouble::new();
    bvh.build(two_triangles());
    bvh.clear();
    assert!(!bvh.is_built());
    assert_eq!(bvh.node_count(), 0);
    assert_eq!(bvh.leaf_count(), 0);
    assert!(bvh.nodes().is_empty());
    assert!(bvh.prim_indices().is_empty());
    bvh.build(random_triangles(100, 3));
    assert!(bvh.is_built());
    assert_eq!(bvh.prim_count(), 100);
    bvh.build(two_triangles()); // rebuilding replaces the geometry
    assert_eq!(bvh.prim_count(), 2);
    assert_eq!(bvh.vertices().unwrap().len(), 6);
    assert!(format!("{bvh:?}").contains("BvhDouble"));
    let geometry = bvh.into_geometry().unwrap();
    assert_eq!(geometry.vertices.len(), 6);

    let mut bvh4 = Bvh4Double::new();
    assert!(!bvh4.is_built());
    assert!(bvh4.data().is_empty());
    assert_eq!(bvh4.used_blocks(), 0);
    bvh4.build(two_triangles());
    assert!(bvh4.is_built());
    bvh4.clear();
    assert!(!bvh4.is_built());
    assert!(bvh4.data().is_empty());
    bvh4.build(random_triangles(100, 3));
    assert_eq!(bvh4.prim_count(), 100);
    bvh4.build(two_triangles());
    assert_eq!(bvh4.prim_count(), 2);
    assert_eq!(bvh4.vertices().unwrap().len(), 6);
    assert!(format!("{bvh4:?}").contains("Bvh4Double"));
    assert_eq!(bvh4.into_geometry().unwrap().vertices.len(), 6);

    let d = BvhDouble::default();
    assert!(!d.is_built());
    let d4 = Bvh4Double::default();
    assert!(!d4.is_built());
}

// ---------------------------------------------------------------------------------------------
// Input validation
// ---------------------------------------------------------------------------------------------

#[test]
#[should_panic(expected = "no vertices")]
fn empty_geometry_panics() {
    BvhDouble::new().build(Vec::<DVec3>::new());
}

#[test]
#[should_panic(expected = "no vertices")]
fn empty_geometry_panics_bvh4() {
    Bvh4Double::new().build(Vec::<DVec3>::new());
}

#[test]
#[should_panic(expected = "multiple of three")]
fn partial_triangle_panics() {
    BvhDouble::new().build(vec![DVec3::ZERO, DVec3::ONE]);
}

#[test]
#[should_panic(expected = "multiple of three")]
fn partial_triangle_panics_bvh4() {
    Bvh4Double::new().build_hq(vec![DVec3::ZERO, DVec3::ONE, DVec3::X, DVec3::Y]);
}

#[test]
#[should_panic(expected = "out of range")]
fn bad_index_panics() {
    BvhDouble::new().build(GeometryEx::indexed(vec![DVec3::ZERO, DVec3::ONE, DVec3::X], vec![0, 1, 3]));
}

#[test]
#[should_panic(expected = "out of range")]
fn bad_index_panics_bvh4() {
    Bvh4Double::new().build((vec![DVec3::ZERO, DVec3::ONE, DVec3::X], vec![0, 1, 3]));
}

#[test]
#[should_panic(expected = "no indices")]
fn empty_indices_panics() {
    BvhDouble::new().build((vec![DVec3::ZERO, DVec3::ONE, DVec3::X], Vec::<u64>::new()));
}

#[test]
#[should_panic(expected = "does not support indexed")]
fn build_quick_indexed_panics() {
    BvhDouble::new().build_quick((vec![DVec3::ZERO, DVec3::ONE, DVec3::X], vec![0, 1, 2]));
}

#[test]
#[should_panic(expected = "no primitives")]
fn empty_custom_geometry_panics() {
    BvhDouble::new().build_custom(Spheres(Vec::new()));
}

#[test]
#[should_panic(expected = "has not been built")]
fn intersect_before_build_panics() {
    let bvh = BvhDouble::new();
    let mut ray = RayEx::new(DVec3::ZERO, DVec3::Z);
    bvh.intersect(&mut ray);
}

#[test]
#[should_panic(expected = "has not been built")]
fn intersect_before_build_panics_bvh4() {
    let bvh = Bvh4Double::new();
    let mut ray = RayEx::new(DVec3::ZERO, DVec3::Z);
    bvh.intersect(&mut ray);
}

#[test]
#[should_panic(expected = "has not been built")]
fn occluded_before_build_panics() {
    BvhDouble::new().is_occluded(&RayEx::new(DVec3::ZERO, DVec3::Z));
}

#[test]
#[should_panic(expected = "has not been built")]
fn sphere_before_build_panics() {
    BvhDouble::new().intersect_sphere(DVec3::ZERO, 1.0);
}

#[test]
#[should_panic(expected = "needs triangle geometry")]
fn sphere_on_custom_geometry_panics() {
    let mut bvh = BvhDouble::new();
    bvh.build_custom(Spheres(vec![(DVec3::ZERO, 1.0)]));
    bvh.intersect_sphere(DVec3::ZERO, 1.0);
}

#[test]
#[should_panic(expected = "no intermediate tree")]
fn sah_cost_on_loaded_bvh4_panics() {
    let dir = temp_dir("sah");
    let path = dir.join("tree.bvh4");
    let mut bvh = Bvh4Double::new();
    bvh.build(two_triangles());
    bvh.save(&path).unwrap();
    let loaded = Bvh4Double::load_self_contained(&path).unwrap();
    std::fs::remove_dir_all(&dir).ok();
    loaded.sah_cost();
}

#[test]
#[should_panic(expected = "node index out of range")]
fn bad_node_index_panics() {
    let mut bvh = BvhDouble::new();
    bvh.build(two_triangles());
    bvh.sah_cost(bvh.used_nodes());
}

#[test]
#[should_panic(expected = "refers to BLAS 1")]
fn tlas_bad_blas_index_panics() {
    let mut blas = BvhDouble::new();
    blas.build(two_triangles());
    BvhDouble::new().build_tlas(vec![BlasInstanceEx::new(1)], vec![Arc::new(blas)]);
}

#[test]
#[should_panic(expected = "has not been built")]
fn tlas_unbuilt_blas_panics() {
    BvhDouble::new().build_tlas(vec![BlasInstanceEx::new(0)], vec![Arc::new(Bvh4Double::new())]);
}

#[test]
#[should_panic(expected = "at least one instance")]
fn tlas_without_instances_panics() {
    let mut blas = BvhDouble::new();
    blas.build(two_triangles());
    BvhDouble::new().build_tlas(Vec::new(), vec![Arc::new(blas)]);
}

#[test]
#[should_panic(expected = "NaN")]
fn nan_ray_panics() {
    let mut bvh = BvhDouble::new();
    bvh.build(two_triangles());
    let mut ray = RayEx::new(DVec3::new(f64::NAN, 0.0, 0.0), DVec3::Z);
    bvh.intersect(&mut ray);
}

#[test]
#[should_panic(expected = "NaN")]
fn nan_ray_panics_bvh4() {
    let mut bvh = Bvh4Double::new();
    bvh.build(two_triangles());
    bvh.is_occluded(&RayEx::with_t_max(DVec3::ZERO, DVec3::Z, f64::NAN));
}

