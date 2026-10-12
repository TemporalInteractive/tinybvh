//! Tests for the wide intermediate layouts `Mbvh4` and `Mbvh8`: builders, node invariants,
//! metrics, refitting, optimization, conversion from a `Bvh`, loading, and conversion into the
//! CPU traversal layouts.

mod common;

use std::sync::Arc;

use approx::assert_relative_eq;
use glam::{Vec3, Vec4};
use tinybvh::{BuildSettings, Bvh, Bvh4Cpu, Bvh8Cpu, Error, Geometry, Layout, Mbvh4, Mbvh4Node, Mbvh8, Mbvh8Node, Ray};

use common::{grid_indexed, random_triangles, triangle_at};

fn two_triangles() -> Vec<Vec4> {
    let mut v = Vec::new();
    v.extend(triangle_at(5.0));
    v.extend(triangle_at(3.0));
    v
}

/// A node of either width, for the shared invariant checks.
trait WideNode: Copy {
    const WIDTH: usize;
    fn is_leaf(&self) -> bool;
    fn children(&self) -> &[u32];
    fn child_count(&self) -> u32;
    fn prim_range(&self) -> std::ops::Range<usize>;
    fn bounds(&self) -> (Vec3, Vec3);
}

impl WideNode for Mbvh4Node {
    const WIDTH: usize = 4;
    fn is_leaf(&self) -> bool {
        Mbvh4Node::is_leaf(self)
    }
    fn children(&self) -> &[u32] {
        Mbvh4Node::children(self)
    }
    fn child_count(&self) -> u32 {
        self.child_count
    }
    fn prim_range(&self) -> std::ops::Range<usize> {
        Mbvh4Node::prim_range(self)
    }
    fn bounds(&self) -> (Vec3, Vec3) {
        (self.aabb_min, self.aabb_max)
    }
}

impl WideNode for Mbvh8Node {
    const WIDTH: usize = 8;
    fn is_leaf(&self) -> bool {
        Mbvh8Node::is_leaf(self)
    }
    fn children(&self) -> &[u32] {
        Mbvh8Node::children(self)
    }
    fn child_count(&self) -> u32 {
        self.child_count
    }
    fn prim_range(&self) -> std::ops::Range<usize> {
        Mbvh8Node::prim_range(self)
    }
    fn bounds(&self) -> (Vec3, Vec3) {
        (self.aabb_min, self.aabb_max)
    }
}

/// The operations the two widths share.
trait Wide: Sized + Send + Sync + 'static {
    type Node: WideNode;
    const NAME: &'static str;
    fn new() -> Self;
    fn with_settings(self, settings: &BuildSettings) -> Self;
    fn build(&mut self, geometry: Geometry);
    fn build_hq(&mut self, geometry: Geometry);
    fn from_bvh(bvh: Bvh) -> Self;
    fn load_self_contained(path: &std::path::Path) -> Result<Self, Error>;
    fn refit(&mut self) -> Result<(), Error>;
    fn optimize(&mut self, iterations: u32) -> Result<(), Error>;
    fn sah_cost(&self, node: u32) -> f32;
    fn leaf_count(&self, node: u32) -> u32;
    fn nodes(&self) -> &[Self::Node];
    fn source_bvh(&self) -> Option<&Bvh>;
    fn prim_indices(&self) -> &[u32];
    fn is_built(&self) -> bool;
    fn is_refittable(&self) -> bool;
    fn may_have_holes(&self) -> bool;
    fn prim_count(&self) -> u32;
    fn used_nodes(&self) -> u32;
    fn layout(&self) -> Layout;
    fn aabb(&self) -> (Vec3, Vec3);
    fn vertices(&self) -> Option<&[Vec4]>;
    fn vertices_mut(&mut self) -> Option<&mut [Vec4]>;
    fn indices(&self) -> Option<&[u32]>;
    fn geometry(&self) -> Option<&Geometry>;
    fn into_geometry(self) -> Option<Geometry>;
    fn clear(&mut self);
    fn debug(&self) -> String;
    /// Converts into the matching CPU layout and traces `ray`.
    fn trace_as_cpu(self, rays: &mut [Ray]) -> u32;
}

macro_rules! impl_wide {
    ($ty:ident, $node:ident, $cpu:ident) => {
        impl Wide for $ty {
            type Node = $node;
            const NAME: &'static str = stringify!($ty);
            fn new() -> Self {
                $ty::new()
            }
            fn with_settings(self, settings: &BuildSettings) -> Self {
                $ty::with_settings(self, settings)
            }
            fn build(&mut self, geometry: Geometry) {
                $ty::build(self, geometry)
            }
            fn build_hq(&mut self, geometry: Geometry) {
                $ty::build_hq(self, geometry)
            }
            fn from_bvh(bvh: Bvh) -> Self {
                $ty::from_bvh(bvh)
            }
            fn load_self_contained(path: &std::path::Path) -> Result<Self, Error> {
                $ty::load_self_contained(path)
            }
            fn refit(&mut self) -> Result<(), Error> {
                $ty::refit(self)
            }
            fn optimize(&mut self, iterations: u32) -> Result<(), Error> {
                $ty::optimize(self, iterations)
            }
            fn sah_cost(&self, node: u32) -> f32 {
                $ty::sah_cost(self, node)
            }
            fn leaf_count(&self, node: u32) -> u32 {
                $ty::leaf_count(self, node)
            }
            fn nodes(&self) -> &[$node] {
                $ty::nodes(self)
            }
            fn source_bvh(&self) -> Option<&Bvh> {
                $ty::source_bvh(self)
            }
            fn prim_indices(&self) -> &[u32] {
                $ty::prim_indices(self)
            }
            fn is_built(&self) -> bool {
                $ty::is_built(self)
            }
            fn is_refittable(&self) -> bool {
                $ty::is_refittable(self)
            }
            fn may_have_holes(&self) -> bool {
                $ty::may_have_holes(self)
            }
            fn prim_count(&self) -> u32 {
                $ty::prim_count(self)
            }
            fn used_nodes(&self) -> u32 {
                $ty::used_nodes(self)
            }
            fn layout(&self) -> Layout {
                $ty::layout(self)
            }
            fn aabb(&self) -> (Vec3, Vec3) {
                $ty::aabb(self)
            }
            fn vertices(&self) -> Option<&[Vec4]> {
                $ty::vertices(self)
            }
            fn vertices_mut(&mut self) -> Option<&mut [Vec4]> {
                $ty::vertices_mut(self)
            }
            fn indices(&self) -> Option<&[u32]> {
                $ty::indices(self)
            }
            fn geometry(&self) -> Option<&Geometry> {
                $ty::geometry(self)
            }
            fn into_geometry(self) -> Option<Geometry> {
                $ty::into_geometry(self)
            }
            fn clear(&mut self) {
                $ty::clear(self)
            }
            fn debug(&self) -> String {
                format!("{self:?}")
            }
            fn trace_as_cpu(self, rays: &mut [Ray]) -> u32 {
                let cpu = $cpu::from_mbvh(self);
                assert!(cpu.is_built());
                let mut steps = 0;
                for ray in rays {
                    steps += cpu.intersect(ray);
                }
                steps
            }
        }
    };
}

impl_wide!(Mbvh4, Mbvh4Node, Bvh4Cpu);
impl_wide!(Mbvh8, Mbvh8Node, Bvh8Cpu);

/// Checks the structural invariants of a wide tree: root bounds, child indices, child counts,
/// leaf ranges, and (for `exact_coverage`) that every primitive is referenced exactly once.
fn check_invariants<W: Wide>(mbvh: &W, exact_coverage: bool) {
    let nodes = mbvh.nodes();
    let prim_indices = mbvh.prim_indices();
    assert!(!nodes.is_empty(), "{}", W::NAME);
    assert_eq!(nodes.len(), mbvh.used_nodes() as usize);
    let (mn, mx) = mbvh.aabb();
    assert_eq!(nodes[0].bounds(), (mn, mx), "{}: root bounds equal aabb()", W::NAME);

    let prim_count = mbvh.prim_count() as usize;
    let mut references = vec![0u32; prim_count];
    let mut reachable = vec![false; nodes.len()];
    reachable[0] = true;
    for (i, node) in nodes.iter().enumerate() {
        let (nmin, nmax) = node.bounds();
        assert!(nmin.cmple(nmax).all(), "{}: node {i} has inverted bounds", W::NAME);
        if node.is_leaf() {
            assert!(node.children().is_empty());
            let range = node.prim_range();
            assert!(!range.is_empty());
            assert!(range.end <= prim_indices.len(), "{}: leaf {i} range {range:?} exceeds {} indices", W::NAME, prim_indices.len());
            for &p in &prim_indices[range] {
                assert!((p as usize) < prim_count, "{}: primitive index {p} out of range", W::NAME);
                references[p as usize] += 1;
            }
        } else {
            assert!(node.child_count() as usize <= W::Node::WIDTH, "{}: node {i} has {} children", W::NAME, node.child_count());
            assert!(node.child_count() >= 1, "{}: interior node {i} without children", W::NAME);
            assert_eq!(node.children().len(), node.child_count() as usize);
            assert!(node.prim_range().is_empty());
            for &c in node.children() {
                assert!((c as usize) < nodes.len(), "{}: child {c} of node {i} out of range", W::NAME);
                assert!(c as usize > i, "{}: child {c} of node {i} does not follow its parent", W::NAME);
                assert!(!reachable[c as usize], "{}: node {c} has two parents", W::NAME);
                reachable[c as usize] = true;
            }
            // Children lie inside their parent.
            for &c in node.children() {
                let (cmin, cmax) = nodes[c as usize].bounds();
                assert!(cmin.cmpge(nmin - 1e-5).all() && cmax.cmple(nmax + 1e-5).all(), "{}: child {c} sticks out of node {i}", W::NAME);
            }
        }
    }
    if !mbvh.may_have_holes() {
        assert!(reachable.iter().all(|&r| r), "{}: every node is reachable", W::NAME);
    }
    if exact_coverage {
        assert!(references.iter().all(|&r| r == 1), "{}: every primitive is referenced exactly once", W::NAME);
    } else {
        assert!(references.iter().all(|&r| r >= 1), "{}: every primitive is referenced", W::NAME);
    }
}

fn build_variants<W: Wide>() {
    let verts = random_triangles(2000, 7);
    let mut mbvh = W::new();
    assert!(!mbvh.is_built());
    assert!(mbvh.nodes().is_empty());
    assert!(mbvh.prim_indices().is_empty());
    mbvh.build(Geometry::new(verts.clone()));
    assert!(mbvh.is_built());
    assert_eq!(mbvh.layout(), Layout::Mbvh);
    assert_eq!(mbvh.prim_count(), 2000);
    assert_eq!(mbvh.prim_indices().len(), 2000);
    assert_eq!(mbvh.vertices().unwrap(), &verts[..]);
    assert!(mbvh.indices().is_none());
    assert!(mbvh.is_refittable());
    check_invariants(&mbvh, true);
    assert!(mbvh.leaf_count(0) > 0);
    assert!(mbvh.leaf_count(0) <= 2000);
    assert!(mbvh.sah_cost(0) > 0.0);
    // Leaf counts of the root's children add up to the total.
    let root = mbvh.nodes()[0];
    let sum: u32 = root.children().iter().map(|&c| mbvh.leaf_count(c)).sum();
    assert_eq!(sum, mbvh.leaf_count(0));
    assert!(mbvh.debug().contains(W::NAME));

    // Indexed geometry.
    let (gv, gi) = grid_indexed(16, 4.0, 1.0);
    let mut indexed = W::new();
    indexed.build(Geometry::indexed(gv.clone(), gi.clone()));
    assert_eq!(indexed.prim_count(), 512);
    assert_eq!(indexed.indices().unwrap(), &gi[..]);
    check_invariants(&indexed, true);
    let geometry = indexed.into_geometry().unwrap();
    assert_eq!(geometry.vertices, gv);

    // Spatial splits: primitives may be referenced more than once, and the tree is not refittable.
    let mut hq = W::new();
    hq.build_hq(Geometry::new(verts.clone()));
    assert!(hq.is_built());
    assert!(!hq.is_refittable());
    assert!(hq.prim_indices().len() >= 2000);
    check_invariants(&hq, false);
    assert_eq!(hq.refit(), Err(Error::NotRefittable));

    // Settings variants.
    let mut single = W::new().with_settings(&BuildSettings { enable_threading: false, binned_bvh: true, ..Default::default() });
    single.build(Geometry::new(verts.clone()));
    check_invariants(&single, true);
    let mut hq_indexed = W::new();
    hq_indexed.build_hq(Geometry::indexed(gv, gi));
    check_invariants(&hq_indexed, false);
}

fn large_build<W: Wide>() {
    let verts = random_triangles(60_000, 99);
    let mut mbvh = W::new();
    mbvh.build(Geometry::new(verts));
    assert_eq!(mbvh.prim_count(), 60_000);
    assert!(mbvh.used_nodes() > 1000);
    check_invariants(&mbvh, true);
    mbvh.refit().unwrap();
    check_invariants(&mbvh, true);
}

fn refit_moves_bounds<W: Wide>() {
    let mut mbvh = W::new().with_settings(&BuildSettings::refittable());
    mbvh.build(Geometry::new(two_triangles()));
    let (mn, mx) = mbvh.aabb();
    assert_relative_eq!(mn.z, 3.0, epsilon = 1e-6);
    assert_relative_eq!(mx.z, 5.0, epsilon = 1e-6);
    for v in mbvh.vertices_mut().unwrap() {
        v.z += 10.0;
    }
    mbvh.refit().unwrap();
    let (mn, mx) = mbvh.aabb();
    assert_relative_eq!(mn.z, 13.0, epsilon = 1e-5);
    assert_relative_eq!(mx.z, 15.0, epsilon = 1e-5);
    assert_eq!(mbvh.nodes()[0].bounds(), (mn, mx));
    check_invariants(&mbvh, true);
    for node in mbvh.nodes() {
        let (nmin, nmax) = node.bounds();
        assert!(nmin.z >= 13.0 - 1e-5 && nmax.z <= 15.0 + 1e-5);
    }
    // The refitted tree still converts into a working CPU layout.
    let mut ray = Ray::new(Vec3::ZERO, Vec3::Z);
    mbvh.trace_as_cpu(std::slice::from_mut(&mut ray));
    assert_relative_eq!(ray.hit.t, 13.0, epsilon = 1e-4);

    let mut empty = W::new();
    assert_eq!(empty.refit(), Err(Error::NotBuilt));
    assert_eq!(empty.optimize(1), Err(Error::NotBuilt));
}

fn optimize_not_worse<W: Wide>() {
    let verts = random_triangles(3000, 21);
    let mut mbvh = W::new().with_settings(&BuildSettings { enable_threading: false, ..Default::default() });
    mbvh.build(Geometry::new(verts.clone()));
    let before = mbvh.sah_cost(0);
    mbvh.optimize(10).unwrap();
    let after = mbvh.sah_cost(0);
    assert!(after <= before * 1.001, "{}: optimize made the SAH cost worse: {before} -> {after}", W::NAME);
    assert_eq!(mbvh.prim_count(), 3000);
    check_invariants(&mbvh, true);
    mbvh.refit().unwrap();
    check_invariants(&mbvh, true);
    let mut rng = common::Rng::new(22);
    let mut rays: Vec<Ray> = (0..200)
        .map(|_| {
            let o = Vec3::new(rng.unit() * 2.0 - 0.5, rng.unit() * 2.0 - 0.5, -1.0);
            let d = Vec3::new(rng.unit(), rng.unit(), rng.unit()) - o;
            Ray::new(o, d)
        })
        .collect();
    let reference: Vec<Ray> = rays.clone();
    mbvh.trace_as_cpu(&mut rays);
    for (ray, fresh) in rays.iter().zip(&reference) {
        let expected = common::brute_force_hit(&verts, fresh.origin, fresh.direction());
        assert_eq!(ray.hit.is_hit(), expected.is_some());
        if let Some((t, _)) = expected {
            assert!((ray.hit.t - t).abs() <= 1e-4 * t.max(1.0));
        }
    }
}

fn from_bvh_conversion<W: Wide>() {
    let verts = random_triangles(2000, 41);
    let mut bvh = Bvh::new();
    bvh.build(verts.clone());
    bvh.optimize(5);
    let prim_count = bvh.prim_count();
    let (bmin, bmax) = bvh.aabb();
    let mbvh = W::from_bvh(bvh);
    assert!(mbvh.is_built());
    assert_eq!(mbvh.prim_count(), prim_count);
    assert_eq!(mbvh.prim_indices().len(), prim_count as usize);
    assert_eq!(mbvh.aabb(), (bmin, bmax));
    assert_eq!(mbvh.vertices().unwrap(), &verts[..]);
    assert!(mbvh.geometry().is_some());
    check_invariants(&mbvh, true);
    assert!(mbvh.leaf_count(0) > 0);
    assert!(mbvh.sah_cost(0) > 0.0);

    // The collapsed tree can be refitted (its vertices are the Bvh's) and optimized, which
    // collapses the Bvh again.
    let mut mbvh = mbvh;
    for v in mbvh.vertices_mut().unwrap() {
        v.z += 1.0;
    }
    mbvh.refit().unwrap();
    let (mn, mx) = mbvh.aabb();
    assert_relative_eq!(mn.z, bmin.z + 1.0, epsilon = 1e-5);
    assert_relative_eq!(mx.z, bmax.z + 1.0, epsilon = 1e-5);
    assert_eq!(mbvh.source_bvh().unwrap().aabb(), (mn, mx), "{}: the Bvh was refitted too", W::NAME);
    check_invariants(&mbvh, true);
    mbvh.optimize(3).unwrap();
    assert_eq!(mbvh.aabb(), (mn, mx));
    check_invariants(&mbvh, true);

    // Converted into a CPU layout, it traces the moved geometry.
    let moved: Vec<Vec4> = verts.iter().map(|v| *v + Vec4::new(0.0, 0.0, 1.0, 0.0)).collect();
    let mut rng = common::Rng::new(42);
    let mut rays: Vec<Ray> = (0..300)
        .map(|_| {
            let o = Vec3::new(rng.unit() * 2.0 - 0.5, rng.unit() * 2.0 - 0.5, -1.0);
            let d = Vec3::new(rng.unit(), rng.unit(), rng.unit() + 1.0) - o;
            Ray::new(o, d)
        })
        .collect();
    let fresh = rays.clone();
    mbvh.trace_as_cpu(&mut rays);
    let mut hits = 0;
    for (ray, f) in rays.iter().zip(&fresh) {
        let expected = common::brute_force_hit(&moved, f.origin, f.direction());
        assert_eq!(ray.hit.is_hit(), expected.is_some());
        if let Some((t, prim)) = expected {
            assert!((ray.hit.t - t).abs() <= 1e-4 * t.max(1.0));
            assert_eq!(ray.hit.prim as usize, prim);
            hits += 1;
        }
    }
    assert!(hits > 50);

    // A rebuild over new geometry releases the Bvh.
    let mut rebuilt = W::from_bvh({
        let mut b = Bvh::new();
        b.build(two_triangles());
        b
    });
    rebuilt.build(Geometry::new(random_triangles(100, 3)));
    assert_eq!(rebuilt.prim_count(), 100);
    rebuilt.optimize(2).unwrap();
    check_invariants(&rebuilt, true);
}

fn load_self_contained<W: Wide>() {
    let dir = std::env::temp_dir().join(format!("tinybvh-rs-mbvh-{}-{}", W::NAME.to_lowercase(), std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let verts = random_triangles(1000, 5);
    let mut bvh = Bvh::new();
    bvh.build(verts.clone());
    let with_geometry = dir.join("with.bvh");
    let without_geometry = dir.join("without.bvh");
    bvh.save(&with_geometry, true).unwrap();
    bvh.save(&without_geometry, false).unwrap();

    let loaded = W::load_self_contained(&with_geometry).unwrap();
    assert!(loaded.is_built());
    assert_eq!(loaded.prim_count(), 1000);
    assert_eq!(loaded.vertices().unwrap(), &verts[..]);
    check_invariants(&loaded, true);
    let mut loaded = loaded;
    loaded.refit().unwrap();
    check_invariants(&loaded, true);
    let mut rng = common::Rng::new(6);
    let mut rays: Vec<Ray> = (0..200)
        .map(|_| {
            let o = Vec3::new(rng.unit() * 2.0 - 0.5, rng.unit() * 2.0 - 0.5, -1.0);
            let d = Vec3::new(rng.unit(), rng.unit(), rng.unit()) - o;
            Ray::new(o, d)
        })
        .collect();
    let fresh = rays.clone();
    loaded.trace_as_cpu(&mut rays);
    for (ray, f) in rays.iter().zip(&fresh) {
        let expected = common::brute_force_hit(&verts, f.origin, f.direction());
        assert_eq!(ray.hit.is_hit(), expected.is_some());
        if let Some((t, _)) = expected {
            assert!((ray.hit.t - t).abs() <= 1e-4 * t.max(1.0));
        }
    }

    assert!(matches!(W::load_self_contained(&without_geometry), Err(Error::LoadFailed(_))));
    assert!(matches!(W::load_self_contained(&dir.join("missing.bvh")), Err(Error::LoadFailed(_))));
    std::fs::remove_dir_all(&dir).ok();
}

fn to_cpu_layout_matches_brute_force<W: Wide>() {
    let verts = random_triangles(2000, 61);
    let mut mbvh = W::new();
    mbvh.build(Geometry::new(verts.clone()));
    let mut rng = common::Rng::new(62);
    let mut rays: Vec<Ray> = (0..300)
        .map(|_| {
            let o = Vec3::new(rng.unit() * 2.0 - 0.5, rng.unit() * 2.0 - 0.5, -1.0);
            let d = Vec3::new(rng.unit(), rng.unit(), rng.unit()) - o;
            Ray::new(o, d)
        })
        .collect();
    let fresh = rays.clone();
    mbvh.trace_as_cpu(&mut rays);
    let mut hits = 0;
    for (ray, f) in rays.iter().zip(&fresh) {
        let expected = common::brute_force_hit(&verts, f.origin, f.direction());
        assert_eq!(ray.hit.is_hit(), expected.is_some());
        if let Some((t, prim)) = expected {
            assert!((ray.hit.t - t).abs() <= 1e-4 * t.max(1.0));
            assert_eq!(ray.hit.prim as usize, prim);
            hits += 1;
        }
    }
    assert!(hits > 75);

    // The HQ variant too.
    let mut hq = W::new();
    hq.build_hq(Geometry::new(verts.clone()));
    let mut rays = fresh.clone();
    hq.trace_as_cpu(&mut rays);
    for (ray, f) in rays.iter().zip(&fresh) {
        let expected = common::brute_force_hit(&verts, f.origin, f.direction());
        assert_eq!(ray.hit.is_hit(), expected.is_some());
        if let Some((t, _)) = expected {
            assert!((ray.hit.t - t).abs() <= 1e-4 * t.max(1.0));
        }
    }
}

fn clear_and_threads<W: Wide>() {
    let mut mbvh = W::new();
    mbvh.build(Geometry::new(two_triangles()));
    mbvh.clear();
    assert!(!mbvh.is_built());
    assert!(mbvh.nodes().is_empty());
    assert!(mbvh.vertices().is_none());
    mbvh.build(Geometry::new(random_triangles(500, 3)));
    assert!(mbvh.is_built());
    let shared = Arc::new(mbvh);
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let shared = shared.clone();
            std::thread::spawn(move || {
                check_invariants(&*shared, true);
                shared.sah_cost(0)
            })
        })
        .collect();
    let costs: Vec<f32> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert!(costs.windows(2).all(|w| w[0] == w[1]));
}

#[test]
fn mbvh4_build_variants() {
    build_variants::<Mbvh4>();
}
#[test]
fn mbvh8_build_variants() {
    build_variants::<Mbvh8>();
}

#[test]
fn mbvh4_large_build() {
    large_build::<Mbvh4>();
}
#[test]
fn mbvh8_large_build() {
    large_build::<Mbvh8>();
}

#[test]
fn mbvh4_refit_moves_bounds() {
    refit_moves_bounds::<Mbvh4>();
}
#[test]
fn mbvh8_refit_moves_bounds() {
    refit_moves_bounds::<Mbvh8>();
}

#[test]
fn mbvh4_optimize_not_worse() {
    optimize_not_worse::<Mbvh4>();
}
#[test]
fn mbvh8_optimize_not_worse() {
    optimize_not_worse::<Mbvh8>();
}

#[test]
fn mbvh4_from_bvh_conversion() {
    from_bvh_conversion::<Mbvh4>();
}
#[test]
fn mbvh8_from_bvh_conversion() {
    from_bvh_conversion::<Mbvh8>();
}

#[test]
fn mbvh4_load_self_contained() {
    load_self_contained::<Mbvh4>();
}
#[test]
fn mbvh8_load_self_contained() {
    load_self_contained::<Mbvh8>();
}

#[test]
fn mbvh4_to_bvh4_cpu_matches_brute_force() {
    to_cpu_layout_matches_brute_force::<Mbvh4>();
}
#[test]
fn mbvh8_to_bvh8_cpu_matches_brute_force() {
    to_cpu_layout_matches_brute_force::<Mbvh8>();
}

#[test]
fn mbvh4_clear_and_threads() {
    clear_and_threads::<Mbvh4>();
}
#[test]
fn mbvh8_clear_and_threads() {
    clear_and_threads::<Mbvh8>();
}

#[test]
fn node_helpers() {
    let leaf = Mbvh4Node { aabb_min: Vec3::ZERO, first_tri: 7, aabb_max: Vec3::ONE, tri_count: 3, child: [9; 4], child_count: 0, dummy: [0; 3] };
    assert!(leaf.is_leaf());
    assert!(leaf.children().is_empty());
    assert_eq!(leaf.prim_range(), 7..10);
    assert_eq!(leaf.half_area(), 3.0);
    let interior = Mbvh4Node { aabb_min: Vec3::ZERO, first_tri: 0, aabb_max: Vec3::ONE, tri_count: 0, child: [1, 2, 3, 0], child_count: 3, dummy: [0; 3] };
    assert!(!interior.is_leaf());
    assert_eq!(interior.children(), &[1, 2, 3]);
    assert!(interior.prim_range().is_empty());
    let wide = Mbvh8Node { aabb_min: Vec3::ZERO, first_tri: 0, aabb_max: Vec3::new(2.0, 1.0, 1.0), tri_count: 0, child: [1, 2, 3, 4, 5, 6, 7, 8], child_count: 8, dummy: [0; 3] };
    assert_eq!(wide.children().len(), 8);
    assert_eq!(wide.half_area(), 5.0);
    assert_eq!(std::mem::size_of::<Mbvh4Node>(), 64);
    assert_eq!(std::mem::size_of::<Mbvh8Node>(), 80);
}

#[test]
#[should_panic(expected = "no vertices")]
fn mbvh4_empty_geometry_panics() {
    Mbvh4::new().build(Geometry::new(Vec::new()));
}

#[test]
#[should_panic(expected = "multiple of three")]
fn mbvh8_partial_triangle_panics() {
    Mbvh8::new().build_hq(Geometry::new(vec![Vec4::ZERO]));
}

#[test]
#[should_panic(expected = "has not been built")]
fn mbvh4_sah_cost_before_build_panics() {
    Mbvh4::new().sah_cost(0);
}

#[test]
#[should_panic(expected = "out of range")]
fn mbvh8_leaf_count_out_of_range_panics() {
    let mut mbvh = Mbvh8::new();
    mbvh.build(Geometry::new(two_triangles()));
    mbvh.leaf_count(mbvh.used_nodes());
}

#[test]
#[should_panic(expected = "has not been built")]
fn mbvh4_from_unbuilt_bvh_panics() {
    Mbvh4::from_bvh(Bvh::new());
}

#[test]
#[should_panic(expected = "built over triangles")]
fn mbvh8_from_custom_bvh_panics() {
    let mut bvh = Bvh::new();
    bvh.build_custom(vec![tinybvh::Aabb::new(Vec3::ZERO, Vec3::ONE)]);
    Mbvh8::from_bvh(bvh);
}
