//! Tests for `BvhVerbose`: conversion from and back to `Bvh`, the optimizer, leaf surgery,
//! refitting, node invariants and input validation.

mod common;

use approx::assert_relative_eq;
use glam::{Vec3, Vec4};
use tinybvh::{Aabb, BuildSettings, Bvh, BvhVerbose, BvhVerboseNode, Error, Layout, Ray};

use common::{check_against_brute_force, grid_indexed, random_triangles, triangle_at};

fn two_triangles() -> Vec<Vec4> {
    let mut v = Vec::new();
    v.extend(triangle_at(5.0));
    v.extend(triangle_at(3.0));
    v
}

fn unthreaded() -> BuildSettings {
    BuildSettings { enable_threading: false, ..Default::default() }
}

/// Checks the structural invariants of a verbose tree: parent links agree with child links,
/// child bounds lie inside their parent, and the leaves reference every primitive once.
fn check_invariants(verbose: &BvhVerbose) -> u32 {
    let nodes = verbose.nodes();
    let prims = verbose.prim_indices();
    assert_eq!(nodes[0].parent, BvhVerboseNode::NO_PARENT);
    assert_eq!(nodes[0].parent(), None);
    let mut seen = vec![0u32; verbose.prim_count() as usize];
    let mut stack = vec![0u32];
    let mut reachable = 0;
    while let Some(idx) = stack.pop() {
        reachable += 1;
        let node = nodes[idx as usize];
        assert!(node.aabb_min.cmple(node.aabb_max).all());
        match node.children() {
            Some((l, r)) => {
                assert!(l != r && (l as usize) < nodes.len() && (r as usize) < nodes.len());
                for c in [l, r] {
                    let child = nodes[c as usize];
                    assert_eq!(child.parent, idx, "parent link of node {c}");
                    assert_eq!(child.parent(), Some(idx));
                    assert!(child.aabb_min.cmpge(node.aabb_min - 1e-5).all() && child.aabb_max.cmple(node.aabb_max + 1e-5).all());
                }
                stack.push(l);
                stack.push(r);
            }
            None => {
                let range = node.prim_range();
                assert!(!range.is_empty() && range.end <= prims.len());
                for &p in &prims[range] {
                    seen[p as usize] += 1;
                }
                assert!(node.half_area() >= 0.0);
            }
        }
    }
    assert!(seen.iter().all(|&s| s == 1), "every primitive is referenced exactly once");
    reachable
}

#[test]
fn from_bvh_matches_source() {
    let verts = random_triangles(2000, 5);
    let mut bvh = Bvh::new().with_settings(&unthreaded());
    bvh.build(verts.clone());
    let node_count = bvh.node_count();
    let used_nodes = bvh.used_nodes();
    let sah = bvh.sah_cost(0);
    let (mn, mx) = bvh.aabb();
    let prim_indices = bvh.prim_indices().to_vec();

    let verbose = BvhVerbose::from_bvh(bvh);
    assert!(verbose.is_built());
    assert_eq!(verbose.layout(), Layout::BvhVerbose);
    assert_eq!(verbose.node_count(), node_count);
    assert_eq!(verbose.used_nodes(), used_nodes);
    assert_eq!(verbose.prim_count(), 2000);
    assert_relative_eq!(verbose.sah_cost(0), sah, epsilon = 1e-3);
    assert_eq!(verbose.aabb(), (mn, mx));
    assert_eq!(verbose.prim_indices(), &prim_indices[..]);
    assert_eq!(verbose.vertices().unwrap(), &verts[..]);
    assert!(verbose.bvh().is_some());
    assert_eq!(check_invariants(&verbose), node_count);
    assert_eq!(verbose.nodes()[0].aabb_min, mn);
    assert_eq!(verbose.nodes()[0].aabb_max, mx);
    assert!(format!("{verbose:?}").contains("BvhVerbose"));
    // Subtree costs are positive and the root dominates.
    let (l, r) = verbose.nodes()[0].children().unwrap();
    assert!(verbose.sah_cost(l) > 0.0 && verbose.sah_cost(r) > 0.0);
}

#[test]
fn optimize_lowers_or_keeps_the_sah() {
    let verts = random_triangles(3000, 9);
    let mut bvh = Bvh::new().with_settings(&unthreaded());
    bvh.build(verts.clone());
    let mut verbose = BvhVerbose::from_bvh(bvh);
    let before = verbose.sah_cost(0);
    verbose.optimize(10);
    let after = verbose.sah_cost(0);
    assert!(after <= before * 1.001, "optimize made the SAH cost worse: {before} -> {after}");
    check_invariants(&verbose);
    verbose.optimize_with(5, true, true);
    assert!(verbose.sah_cost(0) <= before * 1.001);
    check_invariants(&verbose);
    verbose.sort_indices();
    check_invariants(&verbose);

    let bvh = verbose.into_bvh();
    assert!(bvh.is_built());
    assert_relative_eq!(bvh.sah_cost(0), after, max_relative = 0.01);
    assert!(bvh.sah_cost(0) <= before * 1.001);
    check_against_brute_force(&verts, 300, 10, |ray| {
        bvh.intersect(ray);
    });
}

#[test]
fn split_and_merge_leafs() {
    let verts = random_triangles(2000, 13);
    let mut bvh = Bvh::new().with_settings(&unthreaded());
    bvh.build(verts.clone());
    let mut verbose = BvhVerbose::from_bvh(bvh);
    let nodes_before = verbose.node_count();

    verbose.split_leafs(1).unwrap();
    assert!(verbose.node_count() > nodes_before);
    let reachable = check_invariants(&verbose);
    assert_eq!(reachable, verbose.node_count());
    let mut stack = vec![0u32];
    while let Some(idx) = stack.pop() {
        let node = verbose.nodes()[idx as usize];
        match node.children() {
            Some((l, r)) => {
                stack.push(l);
                stack.push(r);
            }
            None => assert_eq!(node.tri_count, 1),
        }
    }
    assert_eq!(verbose.node_count(), 2 * 2000 - 1, "a full binary tree over single-primitive leaves");

    verbose.merge_leafs();
    assert!(verbose.may_have_holes());
    let reachable = check_invariants(&verbose);
    assert!(reachable < 2 * 2000 - 1);
    assert_eq!(reachable, verbose.node_count());
    verbose.compact();
    assert!(!verbose.may_have_holes());
    assert!(verbose.used_nodes() <= reachable + 1, "compact leaves at most the conventional unused slot 1");
    check_invariants(&verbose);
    // Optimizing after merging compacts on its own.
    verbose.merge_leafs();
    verbose.optimize(5);
    assert!(!verbose.may_have_holes());
    check_invariants(&verbose);

    let bvh = verbose.into_bvh();
    assert!(bvh.used_nodes() <= bvh.node_count() + 1, "compacted: only the conventional unused slot 1 remains");
    check_against_brute_force(&verts, 200, 14, |ray| {
        bvh.intersect(ray);
    });
}

#[test]
fn refit_follows_moved_vertices() {
    let mut bvh = Bvh::new().with_settings(&BuildSettings::refittable());
    bvh.build(random_triangles(500, 2));
    let mut verbose = BvhVerbose::from_bvh(bvh);
    assert!(verbose.is_refittable());
    for v in verbose.vertices_mut().unwrap() {
        v.z += 10.0;
    }
    verbose.refit(false).unwrap();
    let (mn, mx) = verbose.aabb();
    assert!(mn.z >= 10.0 && mx.z <= 11.2);
    check_invariants(&verbose);
    let moved: Vec<Vec4> = verbose.vertices().unwrap().to_vec();
    let bvh = verbose.into_bvh();
    let (mn, mx) = bvh.aabb();
    assert!(mn.z >= 10.0 && mx.z <= 11.2);
    for i in 0..50 {
        let o = Vec3::new(i as f32 / 50.0, 0.5, 0.0);
        let mut ray = Ray::new(o, Vec3::Z);
        bvh.intersect(&mut ray);
        let reference = common::brute_force_hit(&moved, o, Vec3::Z);
        assert_eq!(ray.hit.is_hit(), reference.is_some());
        if let Some((t, _)) = reference {
            assert_relative_eq!(ray.hit.t, t, epsilon = 1e-4);
        }
    }

    // Interior-only refit keeps the leaves and is always allowed on triangles.
    let mut bvh = Bvh::new();
    bvh.build(two_triangles());
    let mut verbose = BvhVerbose::from_bvh(bvh);
    verbose.refit(true).unwrap();
    assert_eq!(verbose.aabb().0.z, 3.0);
}

#[test]
fn refit_is_refused_when_not_possible() {
    let mut empty = BvhVerbose::new();
    assert!(!empty.is_built());
    assert_eq!(empty.node_count(), 0);
    assert_eq!(empty.refit(false), Err(Error::NotBuilt));
    assert!(empty.nodes().is_empty() && empty.prim_indices().is_empty() && empty.bvh().is_none());

    let mut sbvh = Bvh::new();
    sbvh.set_settings(&BuildSettings { binned_bvh: true, use_spatial_splits: true, ..Default::default() });
    sbvh.build(random_triangles(500, 1));
    let mut verbose = BvhVerbose::from_bvh(sbvh);
    assert!(!verbose.is_refittable());
    assert_eq!(verbose.refit(false), Err(Error::NotRefittable));
    verbose.refit(true).unwrap();

    let (verts, indices) = grid_indexed(8, 2.0, 1.0);
    let mut indexed = Bvh::new();
    indexed.build((verts, indices));
    let mut verbose = BvhVerbose::from_bvh(indexed);
    assert!(verbose.is_over_indices());
    assert_eq!(verbose.refit(false), Err(Error::NotRefittable));
    verbose.refit(true).unwrap();

    let mut custom = Bvh::new();
    custom.build_custom(vec![Aabb::new(Vec3::ZERO, Vec3::ONE), Aabb::new(Vec3::splat(2.0), Vec3::splat(3.0))]);
    let mut verbose = BvhVerbose::from_bvh(custom);
    assert_eq!(verbose.refit(false), Err(Error::NoGeometry));
    assert!(verbose.vertices().is_none() && verbose.vertices_mut().is_none());
    verbose.refit(true).unwrap();
    verbose.split_leafs(1).unwrap(); // custom builds keep primitive bounds
    check_invariants(&verbose);
    let bvh = verbose.into_bvh();
    assert!(bvh.has_custom_geometry());
    let mut ray = Ray::new(Vec3::new(0.5, 0.5, -1.0), Vec3::Z);
    bvh.intersect(&mut ray);
    assert_relative_eq!(ray.hit.t, 1.0, epsilon = 1e-6);
}

#[test]
fn into_bvh_round_trip() {
    let verts = random_triangles(1000, 33);
    let mut bvh = Bvh::new().with_settings(&unthreaded());
    bvh.build(verts.clone());
    let node_count = bvh.node_count();
    let sah = bvh.sah_cost(0);
    let verbose = BvhVerbose::from_bvh(bvh);
    let bvh = verbose.into_bvh();
    // The conversion back writes the nodes depth-first, so the array order may change but the
    // tree does not.
    assert_eq!(bvh.node_count(), node_count);
    assert_relative_eq!(bvh.sah_cost(0), sah, epsilon = 1e-3);
    assert!(bvh.has_ordered_nodes());
    assert_eq!(bvh.vertices().unwrap(), &verts[..]);
    check_against_brute_force(&verts, 200, 34, |ray| {
        bvh.intersect(ray);
    });
    // Several round trips with surgery in between.
    let mut verbose = BvhVerbose::from_bvh(bvh);
    verbose.split_leafs(2).unwrap();
    verbose.optimize(3);
    verbose.merge_leafs();
    let mut bvh = verbose.into_bvh();
    check_against_brute_force(&verts, 200, 35, |ray| {
        bvh.intersect(ray);
    });
    assert!(bvh.refit().is_ok());
    let mut verbose = BvhVerbose::from_bvh(bvh);
    verbose.optimize(3);
    let bvh = verbose.into_bvh();
    check_against_brute_force(&verts, 100, 36, |ray| {
        bvh.intersect(ray);
    });
    let _ = BvhVerbose::default();
}

#[test]
fn loaded_bvh_converts_but_keeps_no_primitive_bounds() {
    let dir = std::env::temp_dir().join(format!("tinybvh-rs-verbose-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("tree.bvh");
    let verts = random_triangles(2000, 8);
    let mut bvh = Bvh::new();
    bvh.build(verts.clone());
    bvh.save(&path, true).unwrap();
    let loaded = Bvh::load_self_contained(&path).unwrap();
    std::fs::remove_dir_all(&dir).ok();
    let mut verbose = BvhVerbose::from_bvh(loaded);
    check_invariants(&verbose);
    verbose.optimize(3);
    verbose.refit(true).unwrap();
    assert_eq!(verbose.split_leafs(1), Err(tinybvh::Error::NotRebuildable), "split_leafs is refused for a loaded tree");
    let bvh = verbose.into_bvh();
    check_against_brute_force(&verts, 100, 9, |ray| {
        bvh.intersect(ray);
    });
}

#[test]
fn clear_releases_the_source() {
    let mut bvh = Bvh::new();
    bvh.build(two_triangles());
    let mut verbose = BvhVerbose::from_bvh(bvh);
    assert!(verbose.is_built());
    verbose.clear();
    assert!(!verbose.is_built() && verbose.bvh().is_none() && verbose.nodes().is_empty());
    assert_eq!(verbose.node_count(), 0);
}

#[test]
#[should_panic(expected = "has not been built")]
fn from_unbuilt_bvh_panics() {
    BvhVerbose::from_bvh(Bvh::new());
}

#[test]
#[should_panic(expected = "has not been built")]
fn sah_cost_on_empty_panics() {
    BvhVerbose::new().sah_cost(0);
}

#[test]
#[should_panic(expected = "has not been built")]
fn into_bvh_on_empty_panics() {
    BvhVerbose::new().into_bvh();
}

#[test]
#[should_panic(expected = "out of range")]
fn sah_cost_out_of_range_panics() {
    let mut bvh = Bvh::new();
    bvh.build(two_triangles());
    let verbose = BvhVerbose::from_bvh(bvh);
    verbose.sah_cost(verbose.used_nodes());
}

#[test]
#[should_panic(expected = "at least 1")]
fn split_leafs_zero_panics() {
    let mut bvh = Bvh::new();
    bvh.build(two_triangles());
    let _ = BvhVerbose::from_bvh(bvh).split_leafs(0);
}
