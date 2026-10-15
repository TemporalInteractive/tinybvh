//! Tests for the GPU layouts `BvhGpu`, `Bvh4Gpu`, `Bvh8Cwbvh` and `Bvh4AmdHw`: builds, the
//! reference CPU traversal, serialization, conversions, the tiny-mesh cases and input validation.

mod common;

use std::sync::Arc;

use approx::assert_relative_eq;
use glam::{Mat4, Vec3, Vec4};
use tinybvh::{
    Blas, BlasInstance, BuildSettings, Bvh, Bvh4AmdHw, Bvh4Gpu, Bvh8Cwbvh, BvhGpu, BvhGpuNode,
    Error, Layout, Mbvh4, Mbvh8, Ray, FAR,
};

use common::{check_against_brute_force, random_triangles, triangle_at};

/// Named build variants of one layout, run against the same reference.
type Variants<T> = Vec<(&'static str, Box<dyn Fn(&mut T)>)>;

/// `n` stacked triangles at z = 2, 3, ...; a ray along +Z from the origin hits triangle 0 at t = 2.
fn stacked(n: usize) -> Vec<Vec4> {
    let mut v = Vec::new();
    for i in 0..n {
        v.extend(triangle_at(2.0 + i as f32));
    }
    v
}

/// The soup as indexed geometry with an identity index buffer, so brute force stays valid.
fn as_indexed(verts: &[Vec4]) -> (Vec<Vec4>, Vec<u32>) {
    (verts.to_vec(), (0..verts.len() as u32).collect())
}

/// Bit-exact comparison of packed GPU data, where some lanes hold integers that read as NaN.
fn same_bits(a: &[Vec4], b: &[Vec4]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x.to_array().map(f32::to_bits) == y.to_array().map(f32::to_bits))
}

/// A ray from `z = -1` towards the cube that the brute force reference says hits something.
fn hitting_ray(verts: &[Vec4]) -> Ray {
    (0..100)
        .map(|i| Ray::new(Vec3::new(0.3 + i as f32 * 0.004, 0.5, -1.0), Vec3::Z))
        .find(|r| common::brute_force_hit(verts, r.origin, r.direction()).is_some())
        .expect("some probe ray hits the triangles")
}

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("tinybvh-rs-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

// --- tiny meshes: the cases tinybvh's conversions get wrong ----------------------------------------

#[test]
fn bvh_gpu_tiny_meshes_are_traversable() {
    // tinybvh wraps a single-leaf tree in one interior node with an empty right leaf and leaves
    // its triangle data unbuilt; the wrapper traces such a tree itself.
    for n in [1usize, 2] {
        for hq in [false, true] {
            let mut bvh = BvhGpu::new();
            if hq {
                bvh.build_hq(stacked(n));
            } else {
                bvh.build(stacked(n));
            }
            assert!(bvh.is_built(), "n = {n}, hq = {hq}");
            assert_eq!(bvh.prim_count(), n as u32);
            let root = bvh.nodes()[0];
            assert_eq!(
                root.right,
                BvhGpuNode::LEAF_FLAG,
                "the single leaf is wrapped"
            );
            assert!(BvhGpuNode::child_prim_range(root.right).is_empty());
            assert_eq!(BvhGpuNode::child_prim_range(root.left).len(), n);

            let mut ray = Ray::new(Vec3::ZERO, Vec3::Z);
            let steps = bvh.intersect(&mut ray);
            assert!(steps > 0);
            assert!(ray.hit.is_hit());
            assert_relative_eq!(ray.hit.t, 2.0, epsilon = 1e-5);
            assert_eq!(ray.hit.prim, 0);
            assert_relative_eq!(ray.hit.v, 0.5, epsilon = 1e-5);
            assert_relative_eq!(ray.hit.u, 0.25, epsilon = 1e-5);

            let mut miss = Ray::new(Vec3::ZERO, -Vec3::Z);
            bvh.intersect(&mut miss);
            assert!(!miss.hit.is_hit());
            assert!(bvh.is_occluded(&Ray::new(Vec3::ZERO, Vec3::Z)));
            assert!(!bvh.is_occluded(&Ray::with_t_max(Vec3::ZERO, Vec3::Z, 1.5)));
            assert!(!bvh.is_occluded(&Ray::new(Vec3::ZERO, -Vec3::Z)));
            // Indexed geometry takes the same path.
            let mut indexed = BvhGpu::new();
            indexed.build(as_indexed(&stacked(n)));
            let mut ray = Ray::new(Vec3::ZERO, Vec3::Z);
            indexed.intersect(&mut ray);
            assert_relative_eq!(ray.hit.t, 2.0, epsilon = 1e-5);
            assert_eq!(ray.hit.prim, 0);
        }
    }
    // Converting a single-leaf Bvh and loading one go through the same guard.
    let mut source = Bvh::new();
    source.build(stacked(2));
    assert!(source.nodes()[0].is_leaf());
    let converted = BvhGpu::from_bvh(source);
    let mut ray = Ray::new(Vec3::ZERO, Vec3::Z);
    converted.intersect(&mut ray);
    assert_relative_eq!(ray.hit.t, 2.0, epsilon = 1e-5);
    let dir = temp_dir("gpu-tiny");
    let path = dir.join("tiny.bvh");
    converted.bvh().unwrap().save(&path, true).unwrap();
    let loaded = BvhGpu::load_self_contained(&path).unwrap();
    let mut ray = Ray::new(Vec3::ZERO, Vec3::Z);
    loaded.intersect(&mut ray);
    assert_relative_eq!(ray.hit.t, 2.0, epsilon = 1e-5);
    assert!(loaded.is_occluded(&Ray::new(Vec3::ZERO, Vec3::Z)));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn block_layouts_handle_tiny_meshes() {
    // The wide layouts add a level above a single leaf themselves; make sure 1 and 2 triangles
    // build, trace and report a SAH without hitting tinybvh's fatal errors.
    for n in [1usize, 2, 3] {
        let mut b4 = Bvh4Gpu::new();
        b4.build(stacked(n));
        let mut ray = Ray::new(Vec3::ZERO, Vec3::Z);
        b4.intersect(&mut ray);
        assert_relative_eq!(ray.hit.t, 2.0, epsilon = 1e-5);
        assert_eq!(ray.hit.prim, 0);
        assert!(b4.is_occluded(&Ray::new(Vec3::ZERO, Vec3::Z)));
        assert!(!b4.is_occluded(&Ray::with_t_max(Vec3::ZERO, Vec3::Z, 1.5)));
        assert!(b4.sah_cost(0) > 0.0);
        assert!(!b4.data().is_empty());

        let mut cw = Bvh8Cwbvh::new();
        cw.build(stacked(n));
        let mut ray = Ray::new(Vec3::ZERO, Vec3::Z);
        cw.intersect(&mut ray);
        assert_relative_eq!(ray.hit.t, 2.0, epsilon = 1e-5);
        assert_eq!(ray.hit.prim, 0);
        assert!(cw.is_occluded(&Ray::new(Vec3::ZERO, Vec3::Z)));
        assert!(!cw.is_occluded(&Ray::with_t_max(Vec3::ZERO, Vec3::Z, 1.5)));
        assert!(cw.sah_cost(0) > 0.0);
        assert!(!cw.nodes().is_empty() && !cw.triangles().is_empty());
        let mut cw_hq = Bvh8Cwbvh::new();
        cw_hq.build_hq(stacked(n));
        let mut ray = Ray::new(Vec3::ZERO, Vec3::Z);
        cw_hq.intersect(&mut ray);
        assert_relative_eq!(ray.hit.t, 2.0, epsilon = 1e-5);

        let mut amd = Bvh4AmdHw::new();
        amd.build(stacked(n));
        assert!(!amd.data().is_empty());
        assert_eq!(amd.data().len() % 4, 0);
        assert!(amd.sah_cost(0) > 0.0);
    }
}

// --- traversal against brute force -----------------------------------------------------------------

#[test]
fn bvh_gpu_matches_brute_force() {
    let verts = random_triangles(2000, 7);
    let variants: Variants<BvhGpu> = vec![
        (
            "soup",
            Box::new(|b: &mut BvhGpu| b.build(random_triangles(2000, 7))),
        ),
        (
            "indexed",
            Box::new(|b: &mut BvhGpu| b.build(as_indexed(&random_triangles(2000, 7)))),
        ),
        (
            "hq",
            Box::new(|b: &mut BvhGpu| b.build_hq(random_triangles(2000, 7))),
        ),
        (
            "hq indexed",
            Box::new(|b: &mut BvhGpu| b.build_hq(as_indexed(&random_triangles(2000, 7)))),
        ),
        (
            "binned",
            Box::new(|b: &mut BvhGpu| {
                b.set_settings(&BuildSettings {
                    binned_bvh: true,
                    ..Default::default()
                });
                b.build(random_triangles(2000, 7))
            }),
        ),
    ];
    for (name, build) in variants {
        let mut bvh = BvhGpu::new();
        build(&mut bvh);
        assert!(bvh.is_built(), "{name}");
        assert_eq!(bvh.layout(), Layout::BvhGpu, "{name}");
        assert_eq!(bvh.prim_count(), 2000, "{name}");
        assert!(bvh.sah_cost(0) > 0.0, "{name}");
        assert!(!bvh.nodes().is_empty(), "{name}");
        assert_eq!(
            bvh.prim_indices().len(),
            bvh.index_count() as usize,
            "{name}"
        );
        check_against_brute_force(&verts, 300, 11, |ray| {
            bvh.intersect(ray);
        });
        // Occlusion agrees with the closest hit, and the traversal reports its cost.
        let mut ray = hitting_ray(&verts);
        let steps = bvh.intersect(&mut ray);
        assert!(steps > 0 && ray.hit.is_hit(), "{name}");
        assert!(
            bvh.is_occluded(&Ray::new(ray.origin, ray.direction())),
            "{name}"
        );
        assert!(
            !bvh.is_occluded(&Ray::with_t_max(
                ray.origin,
                ray.direction(),
                ray.hit.t * 0.99
            )),
            "{name}"
        );
        assert!(!bvh.is_occluded(&Ray::new(ray.origin, -Vec3::Z)), "{name}");
    }
}

#[test]
fn bvh4_gpu_matches_brute_force() {
    let verts = random_triangles(2000, 8);
    let variants: Variants<Bvh4Gpu> = vec![
        (
            "soup",
            Box::new(|b: &mut Bvh4Gpu| b.build(random_triangles(2000, 8))),
        ),
        (
            "indexed",
            Box::new(|b: &mut Bvh4Gpu| b.build(as_indexed(&random_triangles(2000, 8)))),
        ),
        (
            "hq",
            Box::new(|b: &mut Bvh4Gpu| b.build_hq(random_triangles(2000, 8))),
        ),
        (
            "hq indexed",
            Box::new(|b: &mut Bvh4Gpu| b.build_hq(as_indexed(&random_triangles(2000, 8)))),
        ),
    ];
    for (name, build) in variants {
        let mut bvh = Bvh4Gpu::new();
        build(&mut bvh);
        assert!(bvh.is_built(), "{name}");
        assert_eq!(bvh.layout(), Layout::Bvh4Gpu, "{name}");
        assert_eq!(bvh.prim_count(), 2000, "{name}");
        assert!(bvh.sah_cost(0) > 0.0, "{name}");
        // Every node is four blocks and every triangle three; the buffer holds all of them.
        assert!(bvh.data().len() >= 4 + 3 * 2000, "{name}");
        check_against_brute_force(&verts, 300, 12, |ray| {
            bvh.intersect(ray);
        });
        let mut ray = hitting_ray(&verts);
        bvh.intersect(&mut ray);
        assert!(ray.hit.is_hit(), "{name}");
        assert!(
            bvh.is_occluded(&Ray::new(ray.origin, ray.direction())),
            "{name}"
        );
        assert!(
            !bvh.is_occluded(&Ray::with_t_max(
                ray.origin,
                ray.direction(),
                ray.hit.t * 0.99
            )),
            "{name}"
        );
    }
}

#[test]
fn bvh8_cwbvh_matches_brute_force() {
    let verts = random_triangles(2000, 9);
    let variants: Variants<Bvh8Cwbvh> = vec![
        (
            "soup",
            Box::new(|b: &mut Bvh8Cwbvh| b.build(random_triangles(2000, 9))),
        ),
        (
            "indexed",
            Box::new(|b: &mut Bvh8Cwbvh| b.build(as_indexed(&random_triangles(2000, 9)))),
        ),
        (
            "hq",
            Box::new(|b: &mut Bvh8Cwbvh| b.build_hq(random_triangles(2000, 9))),
        ),
        (
            "hq indexed",
            Box::new(|b: &mut Bvh8Cwbvh| b.build_hq(as_indexed(&random_triangles(2000, 9)))),
        ),
    ];
    for (name, build) in variants {
        let mut bvh = Bvh8Cwbvh::new();
        build(&mut bvh);
        assert!(bvh.is_built(), "{name}");
        assert_eq!(bvh.layout(), Layout::Cwbvh, "{name}");
        assert_eq!(bvh.prim_count(), 2000, "{name}");
        assert!(bvh.sah_cost(0) > 0.0, "{name}");
        assert_eq!(bvh.nodes().len() % 5, 0, "{name}: five blocks per node");
        // three 16-byte blocks per referenced triangle (spatial splits may leave unused slots in index_count)
        assert!(
            bvh.triangles().len() >= 3 * bvh.prim_count() as usize,
            "{name}"
        );
        check_against_brute_force(&verts, 300, 13, |ray| {
            bvh.intersect(ray);
        });
        let mut ray = hitting_ray(&verts);
        bvh.intersect(&mut ray);
        assert!(ray.hit.is_hit(), "{name}");
        assert!(
            bvh.is_occluded(&Ray::new(ray.origin, ray.direction())),
            "{name}"
        );
        assert!(
            !bvh.is_occluded(&Ray::with_t_max(
                ray.origin,
                ray.direction(),
                ray.hit.t * 0.99
            )),
            "{name}"
        );
    }
}

#[test]
fn is_occluded_honours_t_max() {
    let verts = stacked(2); // triangles at z = 2 and z = 3
    let mut gpu = BvhGpu::new();
    gpu.build(stacked(8)); // more triangles, so tinybvh traverses this one itself
    let mut b4 = Bvh4Gpu::new();
    b4.build(verts.clone());
    let mut cw = Bvh8Cwbvh::new();
    cw.build(verts);
    type Occluded<'a> = Box<dyn Fn(&Ray) -> bool + 'a>;
    let checks: Vec<(&str, Occluded)> = vec![
        ("BvhGpu", Box::new(|r| gpu.is_occluded(r))),
        ("Bvh4Gpu", Box::new(|r| b4.is_occluded(r))),
        ("Bvh8Cwbvh", Box::new(|r| cw.is_occluded(r))),
    ];
    for (name, occluded) in checks {
        assert!(occluded(&Ray::new(Vec3::ZERO, Vec3::Z)), "{name}");
        assert!(
            occluded(&Ray::with_t_max(Vec3::ZERO, Vec3::Z, 2.5)),
            "{name}"
        );
        assert!(
            !occluded(&Ray::with_t_max(Vec3::ZERO, Vec3::Z, 1.5)),
            "{name}"
        );
        assert!(!occluded(&Ray::new(Vec3::ZERO, -Vec3::Z)), "{name}");
        assert!(
            !occluded(&Ray::new(Vec3::new(5.0, 5.0, 0.0), Vec3::Z)),
            "{name}"
        );
        // A ray starting between the triangles only sees the second one.
        assert!(
            occluded(&Ray::new(Vec3::new(0.0, 0.0, 2.5), Vec3::Z)),
            "{name}"
        );
        assert!(
            !occluded(&Ray::with_t_max(Vec3::new(0.0, 0.0, 2.5), Vec3::Z, 0.4)),
            "{name}"
        );
    }
}

// --- node data -------------------------------------------------------------------------------------

#[test]
fn bvh_gpu_node_invariants() {
    let mut bvh = BvhGpu::new();
    bvh.build(random_triangles(2000, 21));
    let nodes = bvh.nodes();
    let prims = bvh.prim_indices();
    let (mn, mx) = bvh.aabb();
    assert_eq!(nodes.len(), bvh.used_nodes() as usize);
    let inside = |lo: Vec3, hi: Vec3| {
        lo.cmpge(mn - 1e-6).all() && hi.cmple(mx + 1e-6).all() && lo.cmple(hi).all()
    };
    let mut seen = vec![0u32; 2000];
    let mut referenced_nodes = vec![false; nodes.len()];
    referenced_nodes[0] = true;
    for node in nodes {
        assert!(
            !node.is_leaf(),
            "nodes of this layout are interior; leaves live in the child references"
        );
        assert_eq!((node.tri_count, node.first_tri), (0, 0));
        let (lmin, lmax) = node.left_bounds();
        let (rmin, rmax) = node.right_bounds();
        assert!(
            inside(lmin, lmax) && inside(rmin, rmax),
            "child bounds lie inside the root bounds"
        );
        for child in [node.left, node.right] {
            match BvhGpuNode::child_node(child) {
                Some(idx) => {
                    assert!((idx as usize) < nodes.len(), "child index in range");
                    assert!(!referenced_nodes[idx as usize], "every node has one parent");
                    referenced_nodes[idx as usize] = true;
                }
                None => {
                    let range = BvhGpuNode::child_prim_range(child);
                    assert!(!range.is_empty(), "no empty leaves in a real tree");
                    assert!(range.end <= prims.len());
                    for &p in &prims[range] {
                        seen[p as usize] += 1;
                    }
                }
            }
        }
    }
    assert!(
        referenced_nodes.iter().all(|&r| r),
        "every node is reachable"
    );
    assert!(
        seen.iter().all(|&s| s == 1),
        "every primitive is referenced exactly once"
    );
    // The root's children cover the root bounds.
    let root = nodes[0];
    let union_min = root.lmin.min(root.rmin);
    let union_max = root.lmax.max(root.rmax);
    assert!(union_min.abs_diff_eq(mn, 1e-5) && union_max.abs_diff_eq(mx, 1e-5));
    assert_eq!(root.children(), (root.left, root.right));
}

#[test]
fn amd_hw_build_gives_64_byte_nodes() {
    for (name, geometry) in [
        ("soup", random_triangles(500, 4).into()),
        ("indexed", as_indexed(&random_triangles(500, 4)).into()),
    ] {
        let geometry: tinybvh::Geometry = geometry;
        let mut amd = Bvh4AmdHw::new();
        amd.build(geometry.clone());
        assert!(amd.is_built(), "{name}");
        assert_eq!(amd.layout(), Layout::Bvh4AmdHw, "{name}");
        assert_eq!(amd.prim_count(), 500, "{name}");
        let data = amd.data();
        assert!(!data.is_empty(), "{name}");
        assert_eq!(data.len() % 4, 0, "{name}: nodes are 64-byte aligned");
        // One 64-byte triangle node per triangle plus at least one 128-byte box node.
        assert!(data.len() >= 4 * 500 + 8, "{name}");
        assert!(amd.sah_cost(0) > 0.0, "{name}");
        assert_eq!(amd.geometry(), Some(&geometry), "{name}");
        let mut hq = Bvh4AmdHw::new();
        hq.build_hq(geometry);
        assert!(!hq.data().is_empty() && hq.data().len() % 4 == 0, "{name}");
    }
}

// --- maintenance -----------------------------------------------------------------------------------

#[test]
fn optimize_does_not_worsen_the_sah() {
    let verts = random_triangles(3000, 31);
    let mut gpu = BvhGpu::new();
    gpu.build(verts.clone());
    let before = gpu.sah_cost(0);
    gpu.optimize(10).unwrap();
    assert!(gpu.sah_cost(0) <= before * 1.001);
    check_against_brute_force(&verts, 100, 2, |ray| {
        gpu.intersect(ray);
    });

    let mut b4 = Bvh4Gpu::new();
    b4.build(verts.clone());
    let before = b4.sah_cost(0);
    b4.optimize_with(10, false).unwrap();
    assert!(b4.sah_cost(0) <= before * 1.001);
    check_against_brute_force(&verts, 100, 3, |ray| {
        b4.intersect(ray);
    });

    let mut cw = Bvh8Cwbvh::new();
    cw.build(verts.clone());
    let before = cw.sah_cost(0);
    cw.optimize(10).unwrap();
    assert!(cw.sah_cost(0) <= before * 1.001);
    check_against_brute_force(&verts, 100, 4, |ray| {
        cw.intersect(ray);
    });

    let mut amd = Bvh4AmdHw::new();
    amd.build(verts.clone());
    let before = amd.sah_cost(0);
    amd.optimize(5).unwrap();
    assert!(amd.sah_cost(0) <= before * 1.001);
    assert_eq!(amd.data().len() % 4, 0);

    // Refused when nothing is built.
    assert_eq!(BvhGpu::new().optimize(5), Err(Error::NotBuilt));
    assert_eq!(Bvh4Gpu::new().optimize(5), Err(Error::NotBuilt));
    assert_eq!(Bvh8Cwbvh::new().optimize(5), Err(Error::NotBuilt));
    assert_eq!(Bvh4AmdHw::new().optimize(5), Err(Error::NotBuilt));
}

// --- conversions -----------------------------------------------------------------------------------

#[test]
fn from_bvh_equals_direct_build() {
    let verts = random_triangles(1500, 17);
    let settings = BuildSettings {
        enable_threading: false,
        ..Default::default()
    };
    let mut direct = BvhGpu::new().with_settings(&settings);
    direct.build(verts.clone());

    let mut source = Bvh::new().with_settings(&settings);
    source.build(verts.clone());
    let source_nodes = source.used_nodes();
    let converted = BvhGpu::from_bvh(source);
    assert!(converted.is_built());
    assert_eq!(converted.nodes(), direct.nodes());
    assert_eq!(converted.prim_indices(), direct.prim_indices());
    assert_eq!(converted.prim_count(), 1500);
    assert_eq!(converted.vertices().unwrap(), &verts[..]);
    assert!(converted.bvh().is_some());
    assert_relative_eq!(converted.sah_cost(0), direct.sah_cost(0), epsilon = 1e-3);
    // Converted trees cannot be optimized in place.
    let mut converted = converted;
    assert_eq!(converted.optimize(5), Err(Error::NotRebuildable));
    assert!(converted.sah_cost(source_nodes - 1) >= 0.0);
    for i in 0..50 {
        let o = Vec3::new(i as f32 / 50.0, 0.5, -1.0);
        let mut a = Ray::new(o, Vec3::Z);
        let mut b = Ray::new(o, Vec3::Z);
        direct.intersect(&mut a);
        converted.intersect(&mut b);
        assert_eq!(a.hit.t, b.hit.t);
        assert_eq!(a.hit.prim, b.hit.prim);
    }
    // The source comes back out and still works.
    let bvh = converted.into_bvh().unwrap();
    assert!(bvh.is_built());
    check_against_brute_force(&verts, 50, 5, |ray| {
        bvh.intersect(ray);
    });
    // A rebuild after a conversion owns its tree again.
    let mut again = BvhGpu::from_bvh(bvh);
    again.build(verts.clone());
    again.optimize(5).unwrap();
    assert!(again.bvh().is_none());
    check_against_brute_force(&verts, 50, 6, |ray| {
        again.intersect(ray);
    });
}

#[test]
fn conversions_from_mbvh() {
    let verts = random_triangles(1200, 41);
    let mut m4 = Mbvh4::new();
    m4.build(verts.clone());
    let m4_nodes = m4.used_nodes();
    let mut b4 = Bvh4Gpu::from_mbvh(m4);
    assert!(b4.is_built());
    assert_eq!(b4.prim_count(), 1200);
    assert!(b4.mbvh().is_some());
    assert_eq!(b4.vertices().unwrap(), &verts[..]);
    assert!(b4.sah_cost(0) > 0.0);
    assert!(b4.sah_cost(m4_nodes - 1) >= 0.0);
    assert_eq!(b4.optimize(5), Err(Error::NotRebuildable));
    check_against_brute_force(&verts, 200, 42, |ray| {
        b4.intersect(ray);
    });

    let mut m8 = Mbvh8::new();
    m8.build(verts.clone());
    let mut cw = Bvh8Cwbvh::from_mbvh(m8);
    assert!(cw.is_built());
    assert!(cw.mbvh().is_some());
    assert_eq!(cw.nodes().len() % 5, 0);
    assert_eq!(cw.optimize(5), Err(Error::NotRebuildable));
    check_against_brute_force(&verts, 200, 43, |ray| {
        cw.intersect(ray);
    });

    // The AMD format needs one triangle per leaf.
    let mut bvh = Bvh::new();
    bvh.build(verts.clone());
    bvh.split_leafs(1).unwrap();
    let m4 = Mbvh4::from_bvh(bvh);
    assert!(m4.nodes().iter().all(|n| !n.is_leaf() || n.tri_count == 1));
    let mut amd = Bvh4AmdHw::from_mbvh(m4);
    assert!(amd.is_built());
    assert!(amd.data().len() >= 4 * 1200 && amd.data().len() % 4 == 0);
    assert!(amd.mbvh().is_some());
    assert_eq!(amd.optimize(5), Err(Error::NotRebuildable));
    let geometry = amd.into_geometry().unwrap();
    assert_eq!(geometry.vertices, verts);
}

#[test]
#[should_panic(expected = "one triangle per leaf")]
fn amd_hw_rejects_multi_primitive_leaves() {
    let mut m4 = Mbvh4::new();
    m4.build(random_triangles(500, 4));
    assert!(m4.nodes().iter().any(|n| n.is_leaf() && n.tri_count > 1));
    Bvh4AmdHw::from_mbvh(m4);
}

#[test]
#[should_panic(expected = "single-leaf")]
fn cwbvh_rejects_single_leaf_mbvh() {
    // An Mbvh8 built here wraps the leaf itself, so the guard is exercised through a converted
    // tiny Bvh whose Mbvh8 keeps the root as a leaf.
    let mut bvh = Bvh::new();
    bvh.build(stacked(2));
    let m8 = Mbvh8::from_bvh(bvh);
    if m8.nodes().first().is_some_and(|root| !root.is_leaf()) {
        panic!("single-leaf case not reproducible: tinybvh wrapped the leaf; single-leaf guard untested");
    }
    Bvh8Cwbvh::from_mbvh(m8);
}

// --- serialization ---------------------------------------------------------------------------------

#[test]
fn load_self_contained_from_bvh_file() {
    let dir = temp_dir("gpu-load");
    let path = dir.join("tree.bvh");
    let verts = random_triangles(1000, 5);
    let mut bvh = Bvh::new();
    bvh.build(verts.clone());
    bvh.save(&path, true).unwrap();
    let rays: Vec<Ray> = (0..100)
        .map(|i| {
            Ray::new(
                Vec3::new(i as f32 / 100.0, 0.5, -1.0),
                Vec3::new(0.0, 0.1, 1.0),
            )
        })
        .collect();
    let reference: Vec<(f32, u32)> = rays
        .iter()
        .map(|r| {
            let mut r = *r;
            bvh.intersect(&mut r);
            (r.hit.t, r.hit.prim)
        })
        .collect();
    let compare = |name: &str, intersect: &dyn Fn(&mut Ray)| {
        for (ray, &(t, prim)) in rays.iter().zip(&reference) {
            let mut ray = *ray;
            intersect(&mut ray);
            assert_relative_eq!(ray.hit.t, t, epsilon = 1e-4);
            if t < FAR {
                assert_eq!(ray.hit.prim, prim, "{name}");
            }
        }
    };

    let mut gpu = BvhGpu::load_self_contained(&path).unwrap();
    assert!(gpu.is_built());
    assert_eq!(gpu.prim_count(), 1000);
    assert_eq!(gpu.prim_indices().len(), 1000);
    assert_eq!(gpu.vertices().unwrap(), &verts[..]);
    assert!(gpu.bvh().is_some() && gpu.geometry().is_none());
    assert!(gpu.sah_cost(0) > 0.0);
    assert_eq!(gpu.optimize(5), Err(Error::NotRebuildable));
    compare("BvhGpu", &|r| {
        gpu.intersect(r);
    });

    let mut b4 = Bvh4Gpu::load_self_contained(&path).unwrap();
    assert!(b4.is_built());
    assert_eq!(b4.prim_count(), 1000);
    assert!(b4.data().len() >= 4 + 3 * 1000);
    assert!(b4.geometry().is_none());
    assert_eq!(b4.optimize(5), Err(Error::NotRebuildable));
    compare("Bvh4Gpu", &|r| {
        b4.intersect(r);
    });

    let mut cw = Bvh8Cwbvh::load_self_contained(&path).unwrap();
    assert!(cw.is_built());
    assert_eq!(cw.prim_count(), 1000);
    assert_eq!(cw.nodes().len() % 5, 0);
    assert_eq!(cw.optimize(5), Err(Error::NotRebuildable));
    compare("Bvh8Cwbvh", &|r| {
        cw.intersect(r);
    });

    let mut amd = Bvh4AmdHw::load_self_contained(&path).unwrap();
    assert!(amd.is_built());
    assert_eq!(amd.prim_count(), 1000);
    assert!(amd.data().len() >= 4 * 1000 && amd.data().len() % 4 == 0);
    assert_eq!(amd.optimize(5), Err(Error::NotRebuildable));

    // Missing files are reported.
    let missing = dir.join("missing.bvh");
    assert!(matches!(
        BvhGpu::load_self_contained(&missing),
        Err(Error::LoadFailed(_))
    ));
    assert!(matches!(
        Bvh4Gpu::load_self_contained(&missing),
        Err(Error::LoadFailed(_))
    ));
    assert!(matches!(
        Bvh8Cwbvh::load_self_contained(&missing),
        Err(Error::LoadFailed(_))
    ));
    assert!(matches!(
        Bvh4AmdHw::load_self_contained(&missing),
        Err(Error::LoadFailed(_))
    ));
    // A file without geometry cannot be converted.
    let nodes_only = dir.join("nodes.bvh");
    bvh.save(&nodes_only, false).unwrap();
    assert!(matches!(
        BvhGpu::load_self_contained(&nodes_only),
        Err(Error::LoadFailed(_))
    ));
    assert!(matches!(
        Bvh4Gpu::load_self_contained(&nodes_only),
        Err(Error::LoadFailed(_))
    ));
    assert!(matches!(
        Bvh8Cwbvh::load_self_contained(&nodes_only),
        Err(Error::LoadFailed(_))
    ));
    assert!(matches!(
        Bvh4AmdHw::load_self_contained(&nodes_only),
        Err(Error::LoadFailed(_))
    ));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[should_panic(expected = "no intermediate tree")]
fn sah_cost_after_load_panics() {
    let dir = temp_dir("gpu-sah");
    let path = dir.join("tree.bvh");
    let mut bvh = Bvh::new();
    bvh.build(random_triangles(200, 5));
    bvh.save(&path, true).unwrap();
    let loaded = Bvh4Gpu::load_self_contained(&path).unwrap();
    std::fs::remove_dir_all(&dir).ok();
    loaded.sah_cost(0);
}

#[test]
fn cwbvh_save_and_load_round_trip() {
    let dir = temp_dir("cwbvh");
    let path = dir.join("tree.cwbvh");
    let verts = random_triangles(1000, 23);
    let mut cw = Bvh8Cwbvh::new();
    cw.build(verts.clone());
    cw.save(&path).unwrap();
    let nodes = cw.nodes().to_vec();
    let tris = cw.triangles().to_vec();

    let mut loaded = Bvh8Cwbvh::new();
    loaded.load(&path, verts.clone()).unwrap();
    assert!(loaded.is_built());
    assert_eq!(loaded.prim_count(), 1000);
    // Packed node bytes can form NaN patterns, so compare bit-exact rather than with ==.
    assert!(same_bits(loaded.nodes(), &nodes) && same_bits(loaded.triangles(), &tris));
    assert_eq!(loaded.vertices().unwrap(), &verts[..]);
    assert_eq!(loaded.optimize(5), Err(Error::NotRebuildable));
    check_against_brute_force(&verts, 100, 24, |ray| {
        loaded.intersect(ray);
    });
    // Loading over an existing build replaces it.
    let mut reused = Bvh8Cwbvh::new();
    reused.build(random_triangles(50, 1));
    reused.load(&path, verts.clone()).unwrap();
    assert_eq!(reused.prim_count(), 1000);
    assert!(same_bits(reused.nodes(), &nodes));
    // The cache file is also accepted by load_self_contained.
    let self_contained = Bvh8Cwbvh::load_self_contained(&path).unwrap();
    assert!(
        same_bits(self_contained.nodes(), &nodes) && same_bits(self_contained.triangles(), &tris)
    );
    // A triangle count mismatch is rejected and leaves the structure empty.
    let mut wrong = Bvh8Cwbvh::new();
    assert!(matches!(
        wrong.load(&path, random_triangles(10, 1)),
        Err(Error::LoadFailed(_))
    ));
    assert!(!wrong.is_built());
    assert!(matches!(
        wrong.load(dir.join("missing.cwbvh"), verts.clone()),
        Err(Error::LoadFailed(_))
    ));
    assert_eq!(Bvh8Cwbvh::new().save(&path), Err(Error::NotBuilt));
    std::fs::remove_dir_all(&dir).ok();
}

// --- TLAS ------------------------------------------------------------------------------------------

#[test]
fn bvh_gpu_tlas_node_data() {
    let mut blas = Bvh::new();
    blas.build(stacked(2));
    let blas: Arc<dyn Blas> = Arc::new(blas);
    let offsets = [
        Vec3::ZERO,
        Vec3::new(10.0, 0.0, 0.0),
        Vec3::new(0.0, 10.0, 0.0),
        Vec3::new(0.0, 0.0, -10.0),
    ];
    let instances: Vec<BlasInstance> = offsets
        .iter()
        .enumerate()
        .map(|(i, &o)| BlasInstance::with_transform(0, Mat4::from_translation(o)).with_mask(1 << i))
        .collect();
    let mut tlas = BvhGpu::new();
    tlas.build_tlas(instances, vec![blas.clone()]);
    assert!(tlas.is_tlas() && tlas.is_built() && tlas.is_over_aabbs());
    assert_eq!(tlas.prim_count(), 4);
    assert_eq!(tlas.instances().len(), 4);
    assert_eq!(tlas.blasses().len(), 1);
    assert_eq!(Arc::strong_count(&blas), 2);
    assert!(tlas.geometry().is_none() && tlas.vertices().is_none());

    // Instance bounds were computed by the build.
    let (mn, mx) = tlas.instances()[1].aabb();
    assert_relative_eq!(mn.x, 9.0, epsilon = 1e-5);
    assert_relative_eq!(mx.x, 11.0, epsilon = 1e-5);
    assert_eq!(tlas.instances()[2].mask, 4);

    // The nodes cover the instances: every instance index appears exactly once in a leaf, and
    // every child box lies inside the root bounds.
    let nodes = tlas.nodes();
    let prims = tlas.prim_indices();
    assert_eq!(prims.len(), 4);
    let (root_min, root_max) = tlas.aabb();
    assert_relative_eq!(root_min.z, -8.0, epsilon = 1e-5); // instance 3: triangles at z = 2..3, shifted by -10
    assert_relative_eq!(root_max.x, 11.0, epsilon = 1e-5);
    let mut seen = [0u32; 4];
    for node in nodes {
        for (child, lo, hi) in [
            (node.left, node.lmin, node.lmax),
            (node.right, node.rmin, node.rmax),
        ] {
            assert!(lo.cmpge(root_min - 1e-5).all() && hi.cmple(root_max + 1e-5).all());
            if BvhGpuNode::child_is_leaf(child) {
                for &p in &prims[BvhGpuNode::child_prim_range(child)] {
                    seen[p as usize] += 1;
                }
            } else {
                assert!((child as usize) < nodes.len());
            }
        }
    }
    assert_eq!(seen, [1, 1, 1, 1]);

    // Moving an instance and rebuilding updates the bounds.
    tlas.instances_mut()[3].set_transform(Mat4::from_translation(Vec3::new(0.0, 0.0, 30.0)));
    tlas.rebuild_tlas().unwrap();
    let (root_min, root_max) = tlas.aabb();
    assert_relative_eq!(root_min.z, 2.0, epsilon = 1e-5);
    assert_relative_eq!(root_max.z, 33.0, epsilon = 1e-5);
    assert_eq!(BvhGpu::new().rebuild_tlas(), Err(Error::NotTlas));
    assert!(format!("{tlas:?}").contains("is_tlas: true"));

    drop(tlas);
    assert_eq!(Arc::strong_count(&blas), 1);
}

#[test]
#[should_panic(expected = "no CPU traversal for a TLAS")]
fn bvh_gpu_tlas_intersect_panics() {
    let mut blas = Bvh::new();
    blas.build(stacked(2));
    let mut tlas = BvhGpu::new();
    tlas.build_tlas(
        vec![BlasInstance::new(0), BlasInstance::new(0)],
        vec![Arc::new(blas)],
    );
    let mut ray = Ray::new(Vec3::ZERO, Vec3::Z);
    tlas.intersect(&mut ray);
}

#[test]
#[should_panic(expected = "refers to BLAS 1")]
fn bvh_gpu_tlas_bad_blas_index_panics() {
    let mut blas = Bvh::new();
    blas.build(stacked(2));
    BvhGpu::new().build_tlas(vec![BlasInstance::new(1)], vec![Arc::new(blas)]);
}

#[test]
#[should_panic(expected = "has not been built")]
fn bvh_gpu_tlas_unbuilt_blas_panics() {
    BvhGpu::new().build_tlas(vec![BlasInstance::new(0)], vec![Arc::new(Bvh::new())]);
}

// --- housekeeping ----------------------------------------------------------------------------------

#[test]
fn clear_rebuild_and_debug() {
    let verts = random_triangles(100, 3);

    let mut gpu = BvhGpu::new();
    assert!(!gpu.is_built() && gpu.nodes().is_empty() && gpu.prim_indices().is_empty());
    gpu.build(verts.clone());
    gpu.clear();
    assert!(!gpu.is_built() && gpu.nodes().is_empty());
    gpu.build(stacked(8));
    assert_eq!(gpu.prim_count(), 8);
    gpu.build(verts.clone()); // rebuilding replaces the geometry
    assert_eq!(gpu.prim_count(), 100);
    assert!(format!("{gpu:?}").contains("BvhGpu"));
    assert_eq!(gpu.into_geometry().unwrap().vertices, verts);

    let mut b4 = Bvh4Gpu::new();
    assert!(!b4.is_built() && b4.data().is_empty());
    b4.build(verts.clone());
    assert_eq!(b4.indices(), None);
    b4.clear();
    assert!(!b4.is_built() && b4.data().is_empty());
    b4.build(as_indexed(&verts));
    assert_eq!(b4.indices().unwrap().len(), 300);
    assert!(format!("{b4:?}").contains("Bvh4Gpu"));
    assert_eq!(b4.into_geometry().unwrap().vertices, verts);

    let mut cw = Bvh8Cwbvh::new();
    assert!(!cw.is_built() && cw.nodes().is_empty() && cw.triangles().is_empty());
    cw.build(verts.clone());
    cw.clear();
    assert!(!cw.is_built());
    cw.build(verts.clone());
    assert!(format!("{cw:?}").contains("Bvh8Cwbvh"));
    assert_eq!(cw.into_geometry().unwrap().vertices, verts);

    let mut amd = Bvh4AmdHw::new();
    assert!(!amd.is_built() && amd.data().is_empty());
    amd.build(verts.clone());
    amd.clear();
    assert!(!amd.is_built() && amd.data().is_empty());
    amd.build(verts.clone());
    assert!(format!("{amd:?}").contains("Bvh4AmdHw"));
    assert_eq!(amd.into_geometry().unwrap().vertices, verts);

    let _ = (
        BvhGpu::default(),
        Bvh4Gpu::default(),
        Bvh8Cwbvh::default(),
        Bvh4AmdHw::default(),
    );
}

#[test]
fn shared_across_threads() {
    let verts = random_triangles(3000, 77);
    let mut cw = Bvh8Cwbvh::new();
    cw.build(verts.clone());
    let cw = Arc::new(cw);
    let handles: Vec<_> = (0..4)
        .map(|t| {
            let cw = cw.clone();
            let verts = verts.clone();
            std::thread::spawn(move || {
                check_against_brute_force(&verts, 100, 100 + t, |ray| {
                    cw.intersect(ray);
                });
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
}

// --- validation ------------------------------------------------------------------------------------

#[test]
#[should_panic(expected = "no vertices")]
fn bvh_gpu_empty_geometry_panics() {
    BvhGpu::new().build(Vec::<Vec4>::new());
}

#[test]
#[should_panic(expected = "no vertices")]
fn bvh4_gpu_empty_geometry_panics() {
    Bvh4Gpu::new().build(Vec::<Vec4>::new());
}

#[test]
#[should_panic(expected = "multiple of three")]
fn cwbvh_partial_triangle_panics() {
    Bvh8Cwbvh::new().build(vec![Vec4::ZERO, Vec4::ONE]);
}

#[test]
#[should_panic(expected = "out of range")]
fn amd_hw_bad_index_panics() {
    Bvh4AmdHw::new().build(tinybvh::Geometry::indexed(
        vec![Vec4::ZERO, Vec4::ONE, Vec4::X],
        vec![0, 1, 3],
    ));
}

#[test]
#[should_panic(expected = "has not been built")]
fn bvh_gpu_intersect_before_build_panics() {
    BvhGpu::new().intersect(&mut Ray::new(Vec3::ZERO, Vec3::Z));
}

#[test]
#[should_panic(expected = "has not been built")]
fn bvh4_gpu_intersect_before_build_panics() {
    Bvh4Gpu::new().is_occluded(&Ray::new(Vec3::ZERO, Vec3::Z));
}

#[test]
#[should_panic(expected = "has not been built")]
fn cwbvh_intersect_before_build_panics() {
    Bvh8Cwbvh::new().intersect(&mut Ray::new(Vec3::ZERO, Vec3::Z));
}

#[test]
#[should_panic(expected = "has not been built")]
fn from_bvh_unbuilt_panics() {
    BvhGpu::from_bvh(Bvh::new());
}

#[test]
#[should_panic(expected = "built over triangles")]
fn from_bvh_custom_geometry_panics() {
    let mut bvh = Bvh::new();
    bvh.build_custom(vec![tinybvh::Aabb::new(Vec3::ZERO, Vec3::ONE)]);
    BvhGpu::from_bvh(bvh);
}

#[test]
#[should_panic(expected = "NaN")]
fn nan_ray_panics() {
    let mut cw = Bvh8Cwbvh::new();
    cw.build(stacked(4));
    cw.intersect(&mut Ray::new(Vec3::new(f32::NAN, 0.0, 0.0), Vec3::Z));
}
