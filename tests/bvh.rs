//! Tests for the standard `Bvh` layout: builders, queries, maintenance, custom geometry, TLAS
//! construction, serialization and input validation.

mod common;

use std::sync::Arc;

use approx::assert_relative_eq;
use glam::{Mat4, Vec3, Vec4};
use tinybvh::{
    Aabb, BlasInstance, BuildSettings, Bvh, CustomGeometry, Error, Geometry, Layout, Ray, FAR,
};

use common::{check_against_brute_force, grid_indexed, grid_soup, random_triangles, triangle_at};

fn two_triangles() -> Vec<Vec4> {
    let mut v = Vec::new();
    v.extend(triangle_at(5.0));
    v.extend(triangle_at(3.0));
    v
}

#[test]
fn build_and_intersect_soup() {
    let mut bvh = Bvh::new();
    assert!(!bvh.is_built());
    bvh.build(two_triangles());
    assert!(bvh.is_built());
    assert_eq!(bvh.layout(), Layout::Bvh);
    assert_eq!(bvh.prim_count(), 2);
    assert!(!bvh.is_tlas() && !bvh.is_indexed() && !bvh.has_custom_geometry());

    let mut ray = Ray::new(Vec3::ZERO, Vec3::Z);
    let steps = bvh.intersect(&mut ray);
    assert!(steps > 0);
    assert!(ray.hit.is_hit());
    assert_relative_eq!(ray.hit.t, 3.0, epsilon = 1e-5);
    assert_eq!(ray.hit.prim, 1);
    assert_eq!(ray.hit.inst, 0);
    assert!(ray.hit_point().abs_diff_eq(Vec3::new(0.0, 0.0, 3.0), 1e-5));

    // Barycentrics: the hit at (0, 0) on the triangle (-1,-1), (1,-1), (0,1) has v = 0.5, u = 0.25.
    assert_relative_eq!(ray.hit.v, 0.5, epsilon = 1e-5);
    assert_relative_eq!(ray.hit.u, 0.25, epsilon = 1e-5);

    // A ray pointing away misses.
    let mut miss = Ray::new(Vec3::ZERO, -Vec3::Z);
    bvh.intersect(&mut miss);
    assert!(!miss.hit.is_hit());
    assert_eq!(miss.hit.t, FAR);

    // t_max limits the search; is_occluded honours it too.
    let mut limited = Ray::with_t_max(Vec3::ZERO, Vec3::Z, 2.0);
    bvh.intersect(&mut limited);
    assert_eq!(limited.hit.t, 2.0);
    assert!(bvh.is_occluded(&Ray::new(Vec3::ZERO, Vec3::Z)));
    assert!(!bvh.is_occluded(&Ray::with_t_max(Vec3::ZERO, Vec3::Z, 2.0)));
    assert!(!bvh.is_occluded(&Ray::new(Vec3::ZERO, -Vec3::Z)));

    // The direction is not normalized: t scales with its length.
    let mut scaled = Ray::new(Vec3::ZERO, Vec3::Z * 3.0);
    bvh.intersect(&mut scaled);
    assert_relative_eq!(scaled.hit.t, 1.0, epsilon = 1e-5);
    scaled.normalize();
    scaled.reset(FAR);
    bvh.intersect(&mut scaled);
    assert_relative_eq!(scaled.hit.t, 3.0, epsilon = 1e-5);
}

#[test]
fn soup_and_indexed_agree() {
    let (verts, indices) = grid_indexed(16, 4.0, 1.0);
    let soup = grid_soup(16, 4.0, 1.0);
    let mut a = Bvh::new();
    a.build(soup.clone());
    let mut b = Bvh::new();
    b.build((verts.clone(), indices.clone()));
    assert!(b.is_indexed());
    assert_eq!(a.prim_count(), b.prim_count());
    assert_eq!(b.indices().unwrap(), &indices[..]);
    assert_eq!(b.geometry().unwrap().triangle_count(), 512);

    for i in 0..40 {
        let o = Vec3::new(0.1 + i as f32 * 0.09, 3.9 - i as f32 * 0.08, -2.0);
        let mut ra = Ray::new(o, Vec3::Z);
        let mut rb = Ray::new(o, Vec3::Z);
        a.intersect(&mut ra);
        b.intersect(&mut rb);
        assert!(ra.hit.is_hit() && rb.hit.is_hit());
        assert_relative_eq!(ra.hit.t, 3.0, epsilon = 1e-4);
        assert_eq!(ra.hit.prim, rb.hit.prim);
        assert!(a.is_occluded(&ra) == b.is_occluded(&rb));
    }

    // Indexed geometry can be taken back out.
    let geometry = b.into_geometry().unwrap();
    assert_eq!(geometry.vertices, verts);
}

#[test]
fn builders_match_brute_force() {
    let verts = random_triangles(2000, 7);
    type Variant<'a> = (&'a str, Box<dyn Fn(&mut Bvh)>);
    let variants: Vec<Variant> = vec![
        (
            "default",
            Box::new(|b: &mut Bvh| b.build(random_triangles(2000, 7))),
        ),
        (
            "hq",
            Box::new(|b: &mut Bvh| b.build_hq(random_triangles(2000, 7))),
        ),
        (
            "quick",
            Box::new(|b: &mut Bvh| b.build_quick(random_triangles(2000, 7))),
        ),
        (
            "binned",
            Box::new(|b: &mut Bvh| {
                b.set_settings(&BuildSettings {
                    binned_bvh: true,
                    ..Default::default()
                });
                b.build(random_triangles(2000, 7))
            }),
        ),
        (
            "sbvh",
            Box::new(|b: &mut Bvh| {
                b.set_settings(&BuildSettings {
                    binned_bvh: true,
                    use_spatial_splits: true,
                    ..Default::default()
                });
                b.build(random_triangles(2000, 7))
            }),
        ),
        (
            "scalar",
            Box::new(|b: &mut Bvh| {
                b.set_settings(&BuildSettings {
                    use_simd_if_available: false,
                    ..Default::default()
                });
                b.build(random_triangles(2000, 7))
            }),
        ),
        (
            "single-threaded",
            Box::new(|b: &mut Bvh| {
                b.set_settings(&BuildSettings {
                    enable_threading: false,
                    ..Default::default()
                });
                b.build(random_triangles(2000, 7))
            }),
        ),
        (
            "full-sweep",
            Box::new(|b: &mut Bvh| {
                b.set_settings(&BuildSettings {
                    use_full_sweep: true,
                    ..Default::default()
                });
                b.build(random_triangles(2000, 7))
            }),
        ),
        (
            "hploc",
            Box::new(|b: &mut Bvh| {
                b.set_settings(&BuildSettings {
                    use_hploc: true,
                    ..Default::default()
                });
                b.build(random_triangles(2000, 7))
            }),
        ),
        (
            "presplit+optimize",
            Box::new(|b: &mut Bvh| {
                b.set_settings(&BuildSettings {
                    use_presplitting: true,
                    post_optimize: true,
                    ..Default::default()
                });
                b.build(random_triangles(2000, 7))
            }),
        ),
    ];
    for (name, build) in variants {
        let mut bvh = Bvh::new();
        build(&mut bvh);
        assert!(bvh.is_built(), "{name}");
        assert_eq!(bvh.prim_count(), 2000, "{name}");
        assert!(bvh.sah_cost(0) > 0.0, "{name}");
        check_against_brute_force(&verts, 300, 11, |ray| {
            bvh.intersect(ray);
        });
        let steps = bvh.intersect(&mut Ray::new(Vec3::new(0.5, 0.5, -1.0), Vec3::Z));
        assert!(steps > 0, "{name}");
    }
    let mut simd = Bvh::new();
    if tinybvh::simd_builder_available() {
        simd.build_simd(verts.clone()).unwrap();
        check_against_brute_force(&verts, 100, 3, |ray| {
            simd.intersect(ray);
        });
    } else {
        assert_eq!(
            simd.build_simd(verts.clone()),
            Err(Error::SimdBuilderUnavailable)
        );
    }
}

#[test]
fn large_threaded_build() {
    // Above tinybvh's threading threshold (50k primitives) the build runs on the pool.
    let verts = random_triangles(80_000, 99);
    let mut bvh = Bvh::new();
    bvh.build(verts.clone());
    assert_eq!(bvh.prim_count(), 80_000);
    assert!(bvh.node_count() > 1000);
    let mut rng = common::Rng::new(5);
    for _ in 0..200 {
        let o = Vec3::new(rng.unit(), rng.unit(), -1.0);
        let d = Vec3::new(0.0, 0.0, 1.0);
        let mut ray = Ray::new(o, d);
        bvh.intersect(&mut ray);
        let reference = common::brute_force_hit(&verts, o, d);
        assert_eq!(ray.hit.is_hit(), reference.is_some());
        if let Some((t, _)) = reference {
            assert_relative_eq!(ray.hit.t, t, epsilon = 1e-4);
        }
    }
    // Refit works after a threaded build (compaction happens automatically).
    bvh.refit().unwrap();
}

#[test]
fn refit_follows_moved_vertices() {
    let mut bvh = Bvh::new().with_settings(&BuildSettings::refittable());
    bvh.build(two_triangles());
    assert!(bvh.is_refittable());
    for v in bvh.vertices_mut().unwrap() {
        v.z += 10.0;
    }
    // Before the refit the node bounds are stale; after it the ray hits at the new distance.
    bvh.refit().unwrap();
    let mut ray = Ray::new(Vec3::ZERO, Vec3::Z);
    bvh.intersect(&mut ray);
    assert_relative_eq!(ray.hit.t, 13.0, epsilon = 1e-4);
    let (mn, mx) = bvh.aabb();
    assert_relative_eq!(mn.z, 13.0, epsilon = 1e-5);
    assert_relative_eq!(mx.z, 15.0, epsilon = 1e-5);
}

#[test]
fn refit_is_refused_when_not_possible() {
    let mut empty = Bvh::new();
    assert_eq!(empty.refit(), Err(Error::NotBuilt));

    let mut hq = Bvh::new();
    hq.set_settings(&BuildSettings {
        binned_bvh: true,
        use_spatial_splits: true,
        ..Default::default()
    });
    hq.build(random_triangles(500, 1));
    assert!(!hq.is_refittable());
    assert_eq!(hq.refit(), Err(Error::NotRefittable));

    let mut custom = Bvh::new();
    custom.build_custom(vec![Aabb::new(Vec3::ZERO, Vec3::ONE)]);
    assert_eq!(custom.refit(), Err(Error::NoGeometry));
}

#[test]
fn maintenance_operations() {
    let verts = random_triangles(3000, 21);
    let mut bvh = Bvh::new();
    bvh.set_settings(&BuildSettings {
        enable_threading: false,
        ..Default::default()
    });
    bvh.build(verts.clone());
    let nodes_before = bvh.node_count();
    let leaves_before = bvh.leaf_count();
    assert!(nodes_before > leaves_before);
    assert_eq!(bvh.prim_count_in(0), 3000);
    assert_eq!(bvh.collect_nodes(0).len() as u32, nodes_before);
    let epo = bvh.epo_cost(0).unwrap();
    assert!(epo >= 0.0);

    let before = bvh.sah_cost(0);
    bvh.optimize(10);
    let after = bvh.sah_cost(0);
    assert!(
        after <= before * 1.001,
        "optimize made the SAH cost worse: {before} -> {after}"
    );
    check_against_brute_force(&verts, 100, 2, |ray| {
        bvh.intersect(ray);
    });

    bvh.split_leafs(1).unwrap();
    for node in bvh.nodes() {
        if node.is_leaf() {
            assert_eq!(node.tri_count, 1);
        }
    }
    check_against_brute_force(&verts, 50, 3, |ray| {
        bvh.intersect(ray);
    });
    bvh.merge_leafs();
    assert!(bvh.leaf_count() < bvh.prim_count());
    bvh.compact();
    assert!(bvh.has_ordered_nodes());
    assert!(
        !bvh.may_have_holes(),
        "compaction clears the holes flag left by merge_leafs"
    );
    bvh.optimize(2);
    bvh.refit().unwrap();
    check_against_brute_force(&verts, 50, 4, |ray| {
        bvh.intersect(ray);
    });

    // Node data is consistent: the root covers everything, leaves reference valid primitives.
    let nodes = bvh.nodes();
    let prim_indices = bvh.prim_indices();
    let (mn, mx) = bvh.aabb();
    assert_eq!(nodes[0].aabb_min, mn);
    assert_eq!(nodes[0].aabb_max, mx);
    let mut seen = vec![false; 3000];
    for node in nodes {
        if node.is_leaf() {
            for &p in &prim_indices[node.prim_range()] {
                seen[p as usize] = true;
            }
        } else {
            let (l, r) = node.children().unwrap();
            assert!(l < nodes.len() as u32 && r < nodes.len() as u32);
        }
    }
    assert!(
        seen.iter().all(|&s| s),
        "every primitive is referenced by a leaf"
    );
}

#[test]
fn sphere_query() {
    let mut bvh = Bvh::new();
    bvh.build(two_triangles());
    assert!(bvh.intersect_sphere(Vec3::new(0.0, 0.0, 3.2), 0.5));
    assert!(!bvh.intersect_sphere(Vec3::new(0.0, 0.0, 0.0), 0.5));
    assert!(!bvh.intersect_sphere(Vec3::new(5.0, 5.0, 3.0), 0.5));
}

/// Spheres as custom primitives.
struct Spheres(Vec<(Vec3, f32)>);

impl Spheres {
    fn hit(&self, ray: &Ray, prim: u32) -> Option<f32> {
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
                ray.hit.user[0] = 0xC0FFEE;
                true
            }
            None => false,
        }
    }
    fn is_occluded(&self, ray: &Ray, prim: u32) -> bool {
        self.hit(ray, prim).is_some()
    }
}

#[test]
fn custom_geometry() {
    let mut bvh = Bvh::new();
    bvh.build_custom(Spheres(vec![
        (Vec3::new(0.0, 0.0, 5.0), 1.0),
        (Vec3::new(8.0, 0.0, 0.0), 1.0),
        (Vec3::new(0.0, -7.0, 0.0), 2.0),
    ]));
    assert!(bvh.has_custom_geometry());
    assert_eq!(bvh.prim_count(), 3);
    assert!(bvh.custom_geometry().is_some());

    let mut ray = Ray::new(Vec3::ZERO, Vec3::Z);
    bvh.intersect(&mut ray);
    assert_relative_eq!(ray.hit.t, 4.0, epsilon = 1e-5);
    assert_eq!(ray.hit.prim, 0);
    assert_eq!(ray.hit.user[0], 0xC0FFEE);

    let mut ray = Ray::new(Vec3::ZERO, -Vec3::Y);
    bvh.intersect(&mut ray);
    assert_relative_eq!(ray.hit.t, 5.0, epsilon = 1e-5);
    assert_eq!(ray.hit.prim, 2);

    assert!(bvh.is_occluded(&Ray::new(Vec3::ZERO, Vec3::X)));
    assert!(!bvh.is_occluded(&Ray::with_t_max(Vec3::ZERO, Vec3::X, 6.0)));
    assert!(!bvh.is_occluded(&Ray::new(Vec3::ZERO, Vec3::Y)));

    // Boxes via the built-in Aabb geometry.
    let mut boxes = Bvh::new();
    boxes.build_custom(vec![
        Aabb::new(Vec3::new(-1.0, -1.0, 2.0), Vec3::new(1.0, 1.0, 3.0)),
        Aabb::new(Vec3::splat(10.0), Vec3::splat(11.0)),
    ]);
    let mut ray = Ray::new(Vec3::ZERO, Vec3::Z);
    boxes.intersect(&mut ray);
    assert_relative_eq!(ray.hit.t, 2.0, epsilon = 1e-6);
    assert_eq!(ray.hit.prim, 0);
    let mut ray = Ray::new(Vec3::splat(9.0), Vec3::ONE);
    boxes.intersect(&mut ray);
    assert_eq!(ray.hit.prim, 1);
    assert_relative_eq!(ray.hit.t, 1.0, epsilon = 1e-6);

    // Rebuilding over triangles releases the custom geometry.
    boxes.build(two_triangles());
    assert!(!boxes.has_custom_geometry());
    assert!(boxes.custom_geometry().is_none());
}

#[test]
fn panic_in_callback_propagates() {
    struct Bad;
    impl CustomGeometry for Bad {
        fn prim_count(&self) -> u32 {
            1
        }
        fn aabb(&self, _: u32) -> (Vec3, Vec3) {
            (Vec3::ZERO, Vec3::ONE)
        }
        fn intersect(&self, _: &mut Ray, _: u32) -> bool {
            panic!("boom from intersect")
        }
        fn is_occluded(&self, _: &Ray, _: u32) -> bool {
            false
        }
    }
    let mut bvh = Bvh::new();
    bvh.build_custom(Bad);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut ray = Ray::new(Vec3::new(0.5, 0.5, -1.0), Vec3::Z);
        bvh.intersect(&mut ray);
    }));
    let err = result.expect_err("the panic must surface on the caller");
    assert_eq!(err.downcast_ref::<&str>(), Some(&"boom from intersect"));
    // The structure is still usable afterwards.
    assert!(!bvh.is_occluded(&Ray::new(Vec3::new(0.5, 0.5, -1.0), Vec3::Z)));
}

#[test]
fn tlas_with_instances_and_masks() {
    let mut blas = Bvh::new();
    blas.build(two_triangles());
    let blas: Arc<dyn tinybvh::Blas> = Arc::new(blas);

    let instances = vec![
        BlasInstance::new(0),
        BlasInstance::with_transform(0, Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0)))
            .with_mask(2),
        BlasInstance::with_transform(0, Mat4::from_translation(Vec3::new(0.0, 0.0, -10.0)))
            .with_mask(4),
    ];
    let mut tlas = Bvh::new();
    tlas.build_tlas(instances, vec![blas.clone()]);
    assert!(tlas.is_tlas());
    assert_eq!(tlas.prim_count(), 3);
    assert_eq!(tlas.instances().len(), 3);
    assert_eq!(tlas.blasses().len(), 1);
    assert_eq!(Arc::strong_count(&blas), 2);

    // Instance bounds were computed by the build.
    let (mn, mx) = tlas.instances()[1].aabb();
    assert_relative_eq!(mn.x, 9.0, epsilon = 1e-5);
    assert_relative_eq!(mx.x, 11.0, epsilon = 1e-5);
    assert!(tlas.instances()[1]
        .inverse_transform()
        .abs_diff_eq(Mat4::from_translation(Vec3::new(-10.0, 0.0, 0.0)), 1e-6));

    let mut ray = Ray::new(Vec3::ZERO, Vec3::Z);
    tlas.intersect(&mut ray);
    assert_relative_eq!(ray.hit.t, 3.0, epsilon = 1e-5);
    assert_eq!(ray.hit.inst, 0);
    assert_eq!(ray.hit.prim, 1);

    let mut ray = Ray::new(Vec3::new(10.0, 0.0, 0.0), Vec3::Z);
    tlas.intersect(&mut ray);
    assert_eq!(ray.hit.inst, 1);
    assert_relative_eq!(ray.hit.t, 3.0, epsilon = 1e-5);

    // Masks: instance 1 only answers rays with bit 1 set.
    let mut masked = Ray::new(Vec3::new(10.0, 0.0, 0.0), Vec3::Z).with_mask(1);
    tlas.intersect(&mut masked);
    assert!(!masked.hit.is_hit());
    assert!(!tlas.is_occluded(&masked));
    assert!(tlas.is_occluded(&Ray::new(Vec3::new(10.0, 0.0, 0.0), Vec3::Z).with_mask(2)));

    // Move an instance and rebuild.
    tlas.instances_mut()[0].set_transform(Mat4::from_translation(Vec3::new(0.0, 0.0, 20.0)));
    tlas.rebuild_tlas().unwrap();
    let mut ray = Ray::new(Vec3::ZERO, Vec3::Z);
    tlas.intersect(&mut ray);
    assert_relative_eq!(ray.hit.t, 23.0, epsilon = 1e-4);

    // Bundles are available on a TLAS.
    let n = tinybvh::bundle_rays();
    let mut rays: Vec<Ray> = (0..n)
        .map(|i| Ray::new(Vec3::new(i as f32 * 0.005, 0.0, 0.0), Vec3::Z))
        .collect();
    tlas.intersect_bundle(&mut rays).unwrap();
    for ray in &rays {
        assert_relative_eq!(ray.hit.t, 23.0, epsilon = 1e-3);
    }
    let mut occluded = vec![false; n];
    let mut rays: Vec<Ray> = (0..n)
        .map(|i| Ray::new(Vec3::new(i as f32 * 0.005, 0.0, 0.0), Vec3::Z))
        .collect();
    tlas.is_occluded_bundle(&mut rays, &mut occluded).unwrap();
    assert!(occluded.iter().all(|&o| o));
    let mut blas_only = Bvh::new();
    blas_only.build(two_triangles());
    assert_eq!(blas_only.intersect_bundle(&mut rays), Err(Error::NotTlas));
    assert_eq!(blas_only.rebuild_tlas(), Err(Error::NotTlas));
    assert_eq!(tlas.save("unused.bvh", true), Err(Error::NotTlas));

    // Dropping the TLAS releases the BLAS.
    drop(tlas);
    assert_eq!(Arc::strong_count(&blas), 1);
}

#[test]
fn tlas_over_custom_blas_passes_state() {
    let mut blas = Bvh::new();
    blas.build_custom(Spheres(vec![(Vec3::new(0.0, 0.0, 5.0), 1.0)]));
    let mut tlas = Bvh::new();
    tlas.build_tlas(
        vec![
            BlasInstance::new(0).with_mask(1),
            BlasInstance::with_transform(0, Mat4::from_translation(Vec3::X * 20.0)).with_mask(2),
        ],
        vec![Arc::new(blas)],
    );
    let mut ray = Ray::new(Vec3::new(20.0, 0.0, 0.0), Vec3::Z).with_mask(2);
    tlas.intersect(&mut ray);
    assert_relative_eq!(ray.hit.t, 4.0, epsilon = 1e-5);
    assert_eq!(ray.hit.inst, 1);
    assert_eq!(ray.hit.prim, 0);
}

#[test]
fn save_and_load() {
    let dir = std::env::temp_dir().join(format!("tinybvh-rs-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("tree.bvh");
    let verts = random_triangles(1000, 5);

    let mut bvh = Bvh::new();
    bvh.build(verts.clone());
    bvh.save(&path, true).unwrap();
    let reference_nodes = bvh.nodes().to_vec();

    // Self-contained: geometry comes from the file.
    let loaded = Bvh::load_self_contained(&path).unwrap();
    assert_eq!(loaded.prim_count(), 1000);
    assert_eq!(loaded.nodes(), &reference_nodes[..]);
    assert_eq!(loaded.vertices().unwrap(), &verts[..]);
    check_against_brute_force(&verts, 100, 8, |ray| {
        loaded.intersect(ray);
    });

    // Bound to caller geometry.
    let mut bound = Bvh::new();
    bound.load(&path, verts.clone()).unwrap();
    assert_eq!(bound.nodes(), &reference_nodes[..]);
    check_against_brute_force(&verts, 100, 9, |ray| {
        bound.intersect(ray);
    });

    // Wrong geometry is rejected and leaves the structure empty.
    let mut wrong = Bvh::new();
    assert!(matches!(
        wrong.load(&path, random_triangles(10, 1)),
        Err(Error::LoadFailed(_))
    ));
    assert!(!wrong.is_built());
    assert!(matches!(
        Bvh::load_self_contained(dir.join("missing.bvh")),
        Err(Error::LoadFailed(_))
    ));

    // Without geometry a self-contained load is refused.
    let nodes_only = dir.join("nodes.bvh");
    bvh.save(&nodes_only, false).unwrap();
    assert!(matches!(
        Bvh::load_self_contained(&nodes_only),
        Err(Error::LoadFailed(_))
    ));
    let mut bound2 = Bvh::new();
    bound2.load(&nodes_only, verts.clone()).unwrap();
    assert_eq!(bound2.nodes(), &reference_nodes[..]);

    assert_eq!(Bvh::new().save(&path, true), Err(Error::NotBuilt));
    let mut loaded = Bvh::load_self_contained(&path).unwrap();
    assert_eq!(
        loaded.split_leafs(1),
        Err(Error::NotRebuildable),
        "a loaded tree has no primitive bounds"
    );
    loaded.optimize(2); // but can still be optimized and refitted
    loaded.refit().unwrap();
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn opacity_micro_maps() {
    // One triangle; a 1x1 map with the bit clear makes it transparent.
    let mut bvh = Bvh::new();
    bvh.build(triangle_at(2.0).to_vec());
    assert_eq!(bvh.opacity_micro_map_n(), 0);
    bvh.set_opacity_micro_maps(vec![0u32], 1);
    assert_eq!(bvh.opacity_micro_map_n(), 1);
    let mut ray = Ray::new(Vec3::ZERO, Vec3::Z);
    bvh.intersect(&mut ray);
    assert!(
        !ray.hit.is_hit(),
        "a fully transparent map hides the triangle"
    );
    bvh.set_opacity_micro_maps(vec![1u32], 1);
    let mut ray = Ray::new(Vec3::ZERO, Vec3::Z);
    bvh.intersect(&mut ray);
    assert!(ray.hit.is_hit());
    bvh.clear_opacity_micro_maps();
    assert_eq!(bvh.opacity_micro_map_n(), 0);
}

#[test]
fn shared_across_threads() {
    let verts = random_triangles(5000, 77);
    let mut bvh = Bvh::new();
    bvh.build(verts.clone());
    let bvh = Arc::new(bvh);
    let handles: Vec<_> = (0..4)
        .map(|t| {
            let bvh = bvh.clone();
            let verts = verts.clone();
            std::thread::spawn(move || {
                check_against_brute_force(&verts, 100, 100 + t, |ray| {
                    bvh.intersect(ray);
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
    let mut bvh = Bvh::new();
    bvh.build(two_triangles());
    bvh.clear();
    assert!(!bvh.is_built());
    assert_eq!(bvh.node_count(), 0);
    assert!(bvh.nodes().is_empty());
    bvh.build(random_triangles(100, 3));
    assert!(bvh.is_built());
    assert_eq!(bvh.prim_count(), 100);
    bvh.build(two_triangles()); // rebuilding replaces the geometry
    assert_eq!(bvh.prim_count(), 2);
    assert_eq!(bvh.vertices().unwrap().len(), 6);
    assert!(format!("{bvh:?}").contains("Bvh"));
}

#[test]
#[should_panic(expected = "no vertices")]
fn empty_geometry_panics() {
    Bvh::new().build(Vec::<Vec4>::new());
}

#[test]
#[should_panic(expected = "multiple of three")]
fn partial_triangle_panics() {
    Bvh::new().build(vec![Vec4::ZERO, Vec4::ONE]);
}

#[test]
#[should_panic(expected = "out of range")]
fn bad_index_panics() {
    Bvh::new().build(Geometry::indexed(
        vec![Vec4::ZERO, Vec4::ONE, Vec4::X],
        vec![0, 1, 3],
    ));
}

#[test]
#[should_panic(expected = "has not been built")]
fn intersect_before_build_panics() {
    let bvh = Bvh::new();
    let mut ray = Ray::new(Vec3::ZERO, Vec3::Z);
    bvh.intersect(&mut ray);
}

#[test]
#[should_panic(expected = "refers to BLAS 1")]
fn tlas_bad_blas_index_panics() {
    let mut blas = Bvh::new();
    blas.build(two_triangles());
    Bvh::new().build_tlas(vec![BlasInstance::new(1)], vec![Arc::new(blas)]);
}

#[test]
#[should_panic(expected = "has not been built")]
fn tlas_unbuilt_blas_panics() {
    Bvh::new().build_tlas(vec![BlasInstance::new(0)], vec![Arc::new(Bvh::new())]);
}

#[test]
#[should_panic(expected = "NaN")]
fn nan_ray_panics() {
    let mut bvh = Bvh::new();
    bvh.build(two_triangles());
    let mut ray = Ray::new(Vec3::new(f32::NAN, 0.0, 0.0), Vec3::Z);
    bvh.intersect(&mut ray);
}

#[test]
#[should_panic(expected = "needs triangle geometry")]
fn sphere_query_on_custom_panics() {
    let mut bvh = Bvh::new();
    bvh.build_custom(vec![Aabb::new(Vec3::ZERO, Vec3::ONE)]);
    bvh.intersect_sphere(Vec3::ZERO, 1.0);
}
