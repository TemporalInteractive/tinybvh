//! Tests for the SIMD CPU layouts `Bvh4Cpu` and `Bvh8Cpu`: builders, queries, refitting,
//! optimization, serialization, opacity micro maps, bundles, instancing and input validation.

mod common;

use std::sync::Arc;

use approx::assert_relative_eq;
use glam::{Mat4, Vec3, Vec4};
use tinybvh::{Blas, BlasInstance, BuildSettings, Bvh, Bvh4Cpu, Bvh8Cpu, Error, Geometry, Layout, Mbvh4, Mbvh8, Ray};

use common::{check_against_brute_force, grid_indexed, grid_soup, random_triangles, triangle_at};

fn two_triangles() -> Vec<Vec4> {
    let mut v = Vec::new();
    v.extend(triangle_at(5.0));
    v.extend(triangle_at(3.0));
    v
}

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("tinybvh-rs-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The operations the two layouts share, so each test runs against both.
trait CpuLayout: Blas + Sized + 'static {
    const NAME: &'static str;
    const LAYOUT: Layout;
    fn new() -> Self;
    fn with_settings(self, settings: &BuildSettings) -> Self;
    fn build(&mut self, geometry: Geometry);
    fn build_hq(&mut self, geometry: Geometry);
    fn intersect(&self, ray: &mut Ray) -> u32;
    fn is_occluded(&self, ray: &Ray) -> bool;
    fn refit(&mut self) -> Result<(), Error>;
    fn optimize(&mut self, iterations: u32) -> Result<(), Error>;
    fn sah_cost(&self) -> f32;
    fn save(&self, path: &std::path::Path) -> Result<(), Error>;
    fn load(&mut self, path: &std::path::Path, geometry: Geometry) -> Result<(), Error>;
    fn load_self_contained(path: &std::path::Path) -> Result<Self, Error>;
    fn data(&self) -> &[u8];
    fn is_built(&self) -> bool;
    fn is_refittable(&self) -> bool;
    fn is_rebuildable(&self) -> bool;
    fn has_intermediates(&self) -> bool;
    fn prim_count(&self) -> u32;
    fn used_nodes(&self) -> u32;
    fn used_blocks(&self) -> u32;
    fn layout(&self) -> Layout;
    fn aabb(&self) -> (Vec3, Vec3);
    fn vertices(&self) -> Option<&[Vec4]>;
    fn vertices_mut(&mut self) -> Option<&mut [Vec4]>;
    fn indices(&self) -> Option<&[u32]>;
    fn geometry(&self) -> Option<&Geometry>;
    fn into_geometry(self) -> Option<Geometry>;
    fn clear(&mut self);
    fn set_opacity_micro_maps(&mut self, map: Vec<u32>, n: u32);
    fn clear_opacity_micro_maps(&mut self);
    fn opacity_micro_map_n(&self) -> u32;
    /// Builds a wide tree over `geometry` and converts it.
    fn from_mbvh_over(geometry: Geometry) -> Self;
    fn debug(&self) -> String;
}

macro_rules! impl_cpu_layout_trait {
    ($ty:ident, $mbvh:ident, $layout:expr) => {
        impl CpuLayout for $ty {
            const NAME: &'static str = stringify!($ty);
            const LAYOUT: Layout = $layout;
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
            fn intersect(&self, ray: &mut Ray) -> u32 {
                $ty::intersect(self, ray)
            }
            fn is_occluded(&self, ray: &Ray) -> bool {
                $ty::is_occluded(self, ray)
            }
            fn refit(&mut self) -> Result<(), Error> {
                $ty::refit(self)
            }
            fn optimize(&mut self, iterations: u32) -> Result<(), Error> {
                $ty::optimize(self, iterations)
            }
            fn sah_cost(&self) -> f32 {
                $ty::sah_cost(self)
            }
            fn save(&self, path: &std::path::Path) -> Result<(), Error> {
                $ty::save(self, path)
            }
            fn load(&mut self, path: &std::path::Path, geometry: Geometry) -> Result<(), Error> {
                $ty::load(self, path, geometry)
            }
            fn load_self_contained(path: &std::path::Path) -> Result<Self, Error> {
                $ty::load_self_contained(path)
            }
            fn data(&self) -> &[u8] {
                $ty::data(self)
            }
            fn is_built(&self) -> bool {
                $ty::is_built(self)
            }
            fn is_refittable(&self) -> bool {
                $ty::is_refittable(self)
            }
            fn is_rebuildable(&self) -> bool {
                $ty::is_rebuildable(self)
            }
            fn has_intermediates(&self) -> bool {
                $ty::has_intermediates(self)
            }
            fn prim_count(&self) -> u32 {
                $ty::prim_count(self)
            }
            fn used_nodes(&self) -> u32 {
                $ty::used_nodes(self)
            }
            fn used_blocks(&self) -> u32 {
                $ty::used_blocks(self)
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
            fn set_opacity_micro_maps(&mut self, map: Vec<u32>, n: u32) {
                $ty::set_opacity_micro_maps(self, map, n)
            }
            fn clear_opacity_micro_maps(&mut self) {
                $ty::clear_opacity_micro_maps(self)
            }
            fn opacity_micro_map_n(&self) -> u32 {
                $ty::opacity_micro_map_n(self)
            }
            fn from_mbvh_over(geometry: Geometry) -> Self {
                let mut mbvh = $mbvh::new();
                mbvh.build(geometry);
                $ty::from_mbvh(mbvh)
            }
            fn debug(&self) -> String {
                format!("{self:?}")
            }
        }
    };
}

impl_cpu_layout_trait!(Bvh4Cpu, Mbvh4, Layout::Bvh4Cpu);
impl_cpu_layout_trait!(Bvh8Cpu, Mbvh8, Layout::Bvh8Cpu);

// ---------------------------------------------------------------------------------------------
// Generic test bodies, run once per layout.
// ---------------------------------------------------------------------------------------------

fn build_and_intersect<L: CpuLayout>() {
    let mut bvh = L::new();
    assert!(!bvh.is_built(), "{}", L::NAME);
    assert!(bvh.data().is_empty());
    bvh.build(Geometry::new(two_triangles()));
    assert!(bvh.is_built(), "{}", L::NAME);
    assert_eq!(bvh.layout(), L::LAYOUT);
    assert_eq!(bvh.prim_count(), 2);
    assert!(bvh.has_intermediates());
    assert!(bvh.is_rebuildable());
    // tinybvh keeps the node count only for the intermediate tree, not for this layout.
    assert_eq!(bvh.used_nodes(), 0, "{}: usedNodes is not maintained for the CPU layouts", L::NAME);
    assert!(bvh.used_blocks() > 0);

    let mut ray = Ray::new(Vec3::ZERO, Vec3::Z);
    // The SIMD kernels only count traversal steps in debug builds of tinybvh.
    let _steps = bvh.intersect(&mut ray);
    assert!(ray.hit.is_hit());
    assert_relative_eq!(ray.hit.t, 3.0, epsilon = 1e-5);
    assert_eq!(ray.hit.prim, 1);
    assert_eq!(ray.hit.inst, 0);
    assert_relative_eq!(ray.hit.v, 0.5, epsilon = 1e-5);
    assert_relative_eq!(ray.hit.u, 0.25, epsilon = 1e-5);

    let mut miss = Ray::new(Vec3::ZERO, -Vec3::Z);
    bvh.intersect(&mut miss);
    assert!(!miss.hit.is_hit());

    // t_max limits the search.
    let mut limited = Ray::with_t_max(Vec3::ZERO, Vec3::Z, 2.0);
    bvh.intersect(&mut limited);
    assert_eq!(limited.hit.t, 2.0);

    // A non-normalized direction scales t.
    let mut scaled = Ray::new(Vec3::ZERO, Vec3::Z * 3.0);
    bvh.intersect(&mut scaled);
    assert_relative_eq!(scaled.hit.t, 1.0, epsilon = 1e-5);

    // The root bounds cover both triangles.
    let (mn, mx) = bvh.aabb();
    assert_relative_eq!(mn.z, 3.0, epsilon = 1e-6);
    assert_relative_eq!(mx.z, 5.0, epsilon = 1e-6);
    assert!(bvh.debug().contains(L::NAME));
}

fn occlusion_semantics<L: CpuLayout>() {
    let mut bvh = L::new();
    bvh.build(Geometry::new(two_triangles()));
    assert!(bvh.is_occluded(&Ray::new(Vec3::ZERO, Vec3::Z)));
    assert!(bvh.is_occluded(&Ray::with_t_max(Vec3::ZERO, Vec3::Z, 4.0)));
    assert!(!bvh.is_occluded(&Ray::with_t_max(Vec3::ZERO, Vec3::Z, 2.0)));
    assert!(!bvh.is_occluded(&Ray::new(Vec3::ZERO, -Vec3::Z)));
    assert!(!bvh.is_occluded(&Ray::new(Vec3::new(5.0, 5.0, 0.0), Vec3::Z)));

    // hit.t is a strict upper bound: a ray that already holds the hit at t = 3 reports the
    // triangle at t = 3 as not occluding, while a slightly longer limit finds it.
    let mut ray = Ray::new(Vec3::ZERO, Vec3::Z);
    bvh.intersect(&mut ray);
    assert_relative_eq!(ray.hit.t, 3.0, epsilon = 1e-5);
    assert!(!bvh.is_occluded(&ray), "{}: a hit at t is not occluded before t", L::NAME);
    assert!(bvh.is_occluded(&Ray::with_t_max(Vec3::ZERO, Vec3::Z, 3.0 + 1e-3)));

    // Occlusion from inside the slab between the triangles only sees the far one.
    assert!(bvh.is_occluded(&Ray::new(Vec3::new(0.0, 0.0, 4.0), Vec3::Z)));
    assert!(!bvh.is_occluded(&Ray::with_t_max(Vec3::new(0.0, 0.0, 4.0), Vec3::Z, 0.5)));
}

fn soup_and_indexed_match_brute_force<L: CpuLayout>() {
    let verts = random_triangles(2000, 7);
    let mut soup = L::new();
    soup.build(Geometry::new(verts.clone()));
    assert_eq!(soup.prim_count(), 2000);
    assert!(soup.indices().is_none());
    assert_eq!(soup.vertices().unwrap().len(), 6000);
    check_against_brute_force(&verts, 300, 11, |ray| {
        soup.intersect(ray);
    });

    // The same triangles as indexed geometry: deduplicate nothing, just index 0..n.
    let indices: Vec<u32> = (0..verts.len() as u32).collect();
    let mut indexed = L::new();
    indexed.build(Geometry::indexed(verts.clone(), indices.clone()));
    assert_eq!(indexed.prim_count(), 2000);
    assert_eq!(indexed.indices().unwrap(), &indices[..]);
    assert!(indexed.geometry().unwrap().indices.is_some());
    check_against_brute_force(&verts, 300, 12, |ray| {
        indexed.intersect(ray);
    });

    // Shared vertices on a grid.
    let (gv, gi) = grid_indexed(16, 4.0, 1.0);
    let gsoup = grid_soup(16, 4.0, 1.0);
    let mut a = L::new();
    a.build(Geometry::new(gsoup));
    let mut b = L::new();
    b.build(Geometry::indexed(gv.clone(), gi));
    assert_eq!(a.prim_count(), 512);
    assert_eq!(b.prim_count(), 512);
    for i in 0..40 {
        let o = Vec3::new(0.1 + i as f32 * 0.09, 3.9 - i as f32 * 0.08, -2.0);
        let mut ra = Ray::new(o, Vec3::Z);
        let mut rb = Ray::new(o, Vec3::Z);
        a.intersect(&mut ra);
        b.intersect(&mut rb);
        assert!(ra.hit.is_hit() && rb.hit.is_hit());
        assert_relative_eq!(ra.hit.t, 3.0, epsilon = 1e-4);
        assert_eq!(ra.hit.prim, rb.hit.prim);
        assert_eq!(a.is_occluded(&Ray::new(o, Vec3::Z)), b.is_occluded(&Ray::new(o, Vec3::Z)));
    }
    let geometry = b.into_geometry().unwrap();
    assert_eq!(geometry.vertices, gv);
}

fn hq_build<L: CpuLayout>() {
    let verts = random_triangles(2000, 8);
    let mut hq = L::new();
    hq.build_hq(Geometry::new(verts.clone()));
    assert!(hq.is_built());
    assert_eq!(hq.prim_count(), 2000);
    assert!(!hq.is_refittable(), "{}: spatial splits make the tree not refittable", L::NAME);
    assert!(hq.sah_cost() > 0.0);
    check_against_brute_force(&verts, 300, 13, |ray| {
        hq.intersect(ray);
    });
    let indices: Vec<u32> = (0..verts.len() as u32).collect();
    let mut hq_indexed = L::new();
    hq_indexed.build_hq(Geometry::indexed(verts.clone(), indices));
    check_against_brute_force(&verts, 200, 14, |ray| {
        hq_indexed.intersect(ray);
    });
    // Settings-driven variants behave too.
    let mut settings = L::new().with_settings(&BuildSettings::high_quality());
    settings.build(Geometry::new(verts.clone()));
    check_against_brute_force(&verts, 100, 15, |ray| {
        settings.intersect(ray);
    });
    let mut scalar = L::new().with_settings(&BuildSettings { use_simd_if_available: false, binned_bvh: true, ..Default::default() });
    scalar.build(Geometry::new(verts.clone()));
    check_against_brute_force(&verts, 100, 16, |ray| {
        scalar.intersect(ray);
    });
}

fn large_threaded_build<L: CpuLayout>() {
    // Above tinybvh's threading threshold (50k primitives) the build runs on the pool.
    let verts = random_triangles(80_000, 99);
    let mut bvh = L::new();
    bvh.build(Geometry::new(verts.clone()));
    assert_eq!(bvh.prim_count(), 80_000);
    assert!(bvh.used_blocks() > 1000);
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
    // Refitting after a threaded build works (the intermediate tree is compacted on the way).
    bvh.refit().unwrap();
    check_against_brute_force(&verts, 50, 6, |ray| {
        bvh.intersect(ray);
    });
}

fn refit_follows_moved_vertices<L: CpuLayout>() {
    let mut bvh = L::new().with_settings(&BuildSettings::refittable());
    bvh.build(Geometry::new(two_triangles()));
    assert!(bvh.is_refittable());
    for v in bvh.vertices_mut().unwrap() {
        v.z += 10.0;
    }
    bvh.refit().unwrap();
    let mut ray = Ray::new(Vec3::ZERO, Vec3::Z);
    bvh.intersect(&mut ray);
    assert_relative_eq!(ray.hit.t, 13.0, epsilon = 1e-4);
    assert_eq!(ray.hit.prim, 1);
    let (mn, mx) = bvh.aabb();
    assert_relative_eq!(mn.z, 13.0, epsilon = 1e-5);
    assert_relative_eq!(mx.z, 15.0, epsilon = 1e-5);
    assert!(!bvh.is_occluded(&Ray::with_t_max(Vec3::ZERO, Vec3::Z, 12.0)));
    assert!(bvh.is_occluded(&Ray::with_t_max(Vec3::ZERO, Vec3::Z, 14.0)));

    // A larger scene: move everything, refit, and compare against brute force on moved data.
    let mut verts = random_triangles(1500, 31);
    let mut big = L::new().with_settings(&BuildSettings::refittable());
    big.build(Geometry::new(verts.clone()));
    for v in big.vertices_mut().unwrap() {
        v.x += 0.25;
        v.y -= 0.1;
    }
    for v in &mut verts {
        v.x += 0.25;
        v.y -= 0.1;
    }
    big.refit().unwrap();
    check_against_brute_force(&verts, 200, 32, |ray| {
        big.intersect(ray);
    });
}

fn refit_is_refused_when_not_possible<L: CpuLayout>() {
    let mut empty = L::new();
    assert_eq!(empty.refit(), Err(Error::NotBuilt));
    assert_eq!(empty.optimize(5), Err(Error::NotBuilt));

    let mut hq = L::new();
    hq.build_hq(Geometry::new(random_triangles(500, 1)));
    assert!(!hq.is_refittable());
    assert_eq!(hq.refit(), Err(Error::NotRefittable));
    // Optimizing does not need refittability.
    hq.optimize(2).unwrap();

    let mut converted = L::from_mbvh_over(Geometry::new(random_triangles(500, 2)));
    assert!(converted.is_built());
    assert!(!converted.is_rebuildable(), "{}: converted from an external tree", L::NAME);
    assert_eq!(converted.refit(), Err(Error::NotRebuildable));
    assert_eq!(converted.optimize(5), Err(Error::NotRebuildable));
    // sah_cost still works on the referenced wide tree.
    assert!(converted.sah_cost() > 0.0);

    // discard_intermediates drops the trees that refit and optimize need; tinybvh itself still
    // reports the structure as rebuildable, so the wrapper has to track this on its own.
    let mut discarded = L::new().with_settings(&BuildSettings { discard_intermediates: true, ..Default::default() });
    let verts = random_triangles(2000, 3);
    discarded.build(Geometry::new(verts.clone()));
    assert!(discarded.is_built());
    assert!(!discarded.has_intermediates());
    assert!(discarded.is_rebuildable(), "{}: tinybvh reports rebuildable even with discarded intermediates", L::NAME);
    assert_eq!(discarded.refit(), Err(Error::NotRebuildable));
    assert_eq!(discarded.optimize(5), Err(Error::NotRebuildable));
    check_against_brute_force(&verts, 200, 4, |ray| {
        discarded.intersect(ray);
    });
    // A rebuild without the flag restores the intermediates.
    discarded = discarded.with_settings(&BuildSettings::default());
    discarded.build(Geometry::new(verts));
    assert!(discarded.has_intermediates());
    discarded.refit().unwrap();
}

fn optimize_does_not_hurt<L: CpuLayout>() {
    let verts = random_triangles(3000, 21);
    let mut bvh = L::new().with_settings(&BuildSettings { enable_threading: false, ..Default::default() });
    bvh.build(Geometry::new(verts.clone()));
    let before = bvh.sah_cost();
    assert!(before > 0.0);
    bvh.optimize(10).unwrap();
    let after = bvh.sah_cost();
    assert!(after <= before * 1.001, "{}: optimize made the SAH cost worse: {before} -> {after}", L::NAME);
    assert!(bvh.is_built());
    check_against_brute_force(&verts, 150, 2, |ray| {
        bvh.intersect(ray);
    });
    // Refitting still works afterwards, and the vertices are the owned ones.
    bvh.refit().unwrap();
    check_against_brute_force(&verts, 50, 3, |ray| {
        bvh.intersect(ray);
    });
}

fn from_mbvh_matches_direct_build<L: CpuLayout>() {
    let verts = random_triangles(2000, 41);
    let mut direct = L::new();
    direct.build(Geometry::new(verts.clone()));
    let converted = L::from_mbvh_over(Geometry::new(verts.clone()));
    assert_eq!(converted.prim_count(), 2000);
    assert_eq!(converted.vertices().unwrap(), &verts[..]);
    assert!(converted.geometry().is_some());
    check_against_brute_force(&verts, 300, 42, |ray| {
        converted.intersect(ray);
    });
    let mut rng = common::Rng::new(43);
    for _ in 0..200 {
        let o = Vec3::new(rng.unit() * 2.0 - 0.5, rng.unit() * 2.0 - 0.5, -1.0);
        let d = Vec3::new(rng.unit(), rng.unit(), rng.unit()) - o;
        let mut a = Ray::new(o, d);
        let mut b = Ray::new(o, d);
        direct.intersect(&mut a);
        converted.intersect(&mut b);
        assert_eq!(a.hit.is_hit(), b.hit.is_hit());
        if a.hit.is_hit() {
            assert_eq!(a.hit.prim, b.hit.prim);
            assert_relative_eq!(a.hit.t, b.hit.t, epsilon = 1e-5);
        }
        assert_eq!(direct.is_occluded(&Ray::new(o, d)), converted.is_occluded(&Ray::new(o, d)));
    }
    // A fresh build over the converted structure makes it rebuildable again and releases the
    // wide tree.
    let mut rebuilt = converted;
    rebuilt.build(Geometry::new(random_triangles(300, 44)));
    assert!(rebuilt.is_rebuildable());
    rebuilt.refit().unwrap();
    assert_eq!(rebuilt.prim_count(), 300);
}

fn from_mbvh_repeatedly<L: CpuLayout>() {
    // Conversions of varying sizes, dropped in different orders, with rebuilds in between:
    // exercises the ownership hand-over between the wide tree and the CPU layout.
    let mut kept = Vec::new();
    for i in 0..30u64 {
        let count = 3 + (i * 97 % 1500) as usize;
        let verts = random_triangles(count, 200 + i);
        let mut cpu = L::from_mbvh_over(Geometry::new(verts.clone()));
        assert_eq!(cpu.prim_count(), count as u32);
        let mut ray = Ray::new(Vec3::new(0.5, 0.5, -1.0), Vec3::Z);
        cpu.intersect(&mut ray);
        let reference = common::brute_force_hit(&verts, ray.origin, ray.direction());
        assert_eq!(ray.hit.is_hit(), reference.is_some());
        if i % 3 == 0 {
            cpu.build(Geometry::new(random_triangles(10, i)));
            assert!(cpu.is_rebuildable());
        }
        if i % 2 == 0 {
            kept.push(cpu);
        }
    }
    assert_eq!(kept.len(), 15);
    drop(kept);
}

fn save_and_load<L: CpuLayout>() {
    let dir = temp_dir(&format!("wide-{}", L::NAME.to_lowercase()));
    let path = dir.join("layout.bin");
    let verts = random_triangles(1000, 5);

    let mut bvh = L::new();
    bvh.build(Geometry::new(verts.clone()));
    bvh.save(&path).unwrap();
    let reference_data = bvh.data().to_vec();
    assert_eq!(bvh.used_blocks() as usize * 64, reference_data.len());

    // Bound to caller geometry.
    let mut bound = L::new();
    bound.load(&path, Geometry::new(verts.clone())).unwrap();
    assert!(bound.is_built());
    assert_eq!(bound.prim_count(), 1000);
    assert_eq!(bound.data(), &reference_data[..]);
    assert_eq!(bound.vertices().unwrap(), &verts[..]);
    assert!(!bound.has_intermediates());
    // tinybvh restores the rebuildable flag from the file; the intermediate trees are gone
    // nonetheless, so the wrapper refuses to refit or optimize.
    assert!(bound.is_rebuildable(), "{}: the file carries rebuildable = true", L::NAME);
    assert_eq!(bound.refit(), Err(Error::NotRebuildable));
    assert_eq!(bound.optimize(5), Err(Error::NotRebuildable));
    assert_eq!(bound.opacity_micro_map_n(), 0);
    check_against_brute_force(&verts, 100, 9, |ray| {
        bound.intersect(ray);
    });

    // Self-contained: the triangles come from the leaf blocks.
    let loaded = L::load_self_contained(&path).unwrap();
    assert!(loaded.is_built());
    assert_eq!(loaded.prim_count(), 1000);
    assert_eq!(loaded.data(), &reference_data[..]);
    assert!(loaded.vertices().is_none());
    assert!(!loaded.has_intermediates());
    check_against_brute_force(&verts, 100, 8, |ray| {
        loaded.intersect(ray);
    });
    let mut rng = common::Rng::new(10);
    for _ in 0..100 {
        let o = Vec3::new(rng.unit() * 2.0 - 0.5, rng.unit() * 2.0 - 0.5, -1.0);
        let d = Vec3::new(rng.unit(), rng.unit(), rng.unit()) - o;
        let mut a = Ray::new(o, d);
        let mut b = Ray::new(o, d);
        bvh.intersect(&mut a);
        loaded.intersect(&mut b);
        assert_eq!(a.hit.is_hit(), b.hit.is_hit());
        if a.hit.is_hit() {
            assert_eq!(a.hit.prim, b.hit.prim);
            assert_eq!(a.hit.t, b.hit.t);
        }
    }

    // The wrong triangle count is rejected; the structure stays empty.
    let mut wrong = L::new();
    assert!(matches!(wrong.load(&path, Geometry::new(random_triangles(10, 1))), Err(Error::LoadFailed(_))));
    assert!(!wrong.is_built());
    assert!(matches!(L::load_self_contained(&dir.join("missing.bin")), Err(Error::LoadFailed(_))));
    assert_eq!(L::new().save(&path), Err(Error::NotBuilt));

    // A plain Bvh file with geometry is converted on load.
    let bvh_path = dir.join("plain.bvh");
    let mut plain = Bvh::new();
    plain.build(verts.clone());
    plain.save(&bvh_path, true).unwrap();
    let from_bvh_file = L::load_self_contained(&bvh_path).unwrap();
    assert_eq!(from_bvh_file.prim_count(), 1000);
    check_against_brute_force(&verts, 100, 17, |ray| {
        from_bvh_file.intersect(ray);
    });
    // Without geometry there is nothing to convert from.
    let nodes_only = dir.join("nodes.bvh");
    plain.save(&nodes_only, false).unwrap();
    assert!(matches!(L::load_self_contained(&nodes_only), Err(Error::LoadFailed(_))));

    // Loading into a built structure replaces it; the old geometry is released.
    let mut reused = L::new();
    reused.build(Geometry::new(random_triangles(50, 2)));
    reused.load(&path, Geometry::new(verts.clone())).unwrap();
    assert_eq!(reused.prim_count(), 1000);

    std::fs::remove_dir_all(&dir).ok();
}

fn data_blocks<L: CpuLayout>() {
    let mut bvh = L::new();
    assert!(bvh.data().is_empty());
    assert_eq!(bvh.used_blocks(), 0);
    bvh.build(Geometry::new(random_triangles(500, 3)));
    let data = bvh.data();
    assert!(!data.is_empty());
    assert_eq!(data.len() % 64, 0);
    assert_eq!(data.len(), bvh.used_blocks() as usize * 64);
    // At least one leaf (3 blocks) and one node must exist.
    assert!(bvh.used_blocks() >= 5);
    bvh.clear();
    assert!(bvh.data().is_empty());
    assert!(!bvh.is_built());
}

fn opacity_micro_maps<L: CpuLayout>() {
    let mut bvh = L::new();
    bvh.build(Geometry::new(triangle_at(2.0).to_vec()));
    assert_eq!(bvh.opacity_micro_map_n(), 0);
    bvh.set_opacity_micro_maps(vec![0u32], 1);
    assert_eq!(bvh.opacity_micro_map_n(), 1);
    let mut ray = Ray::new(Vec3::ZERO, Vec3::Z);
    bvh.intersect(&mut ray);
    assert!(!ray.hit.is_hit(), "{}: a fully transparent map hides the triangle", L::NAME);
    assert!(!bvh.is_occluded(&Ray::new(Vec3::ZERO, Vec3::Z)));
    bvh.set_opacity_micro_maps(vec![1u32], 1);
    let mut ray = Ray::new(Vec3::ZERO, Vec3::Z);
    bvh.intersect(&mut ray);
    assert!(ray.hit.is_hit());
    assert!(bvh.is_occluded(&Ray::new(Vec3::ZERO, Vec3::Z)));

    // The map survives a refit (the layout is rebuilt from the intermediate tree).
    bvh.set_opacity_micro_maps(vec![0u32], 1);
    bvh.refit().unwrap();
    assert_eq!(bvh.opacity_micro_map_n(), 1);
    let mut ray = Ray::new(Vec3::ZERO, Vec3::Z);
    bvh.intersect(&mut ray);
    assert!(!ray.hit.is_hit(), "{}: the map is kept across a refit", L::NAME);
    bvh.optimize(2).unwrap();
    let mut ray = Ray::new(Vec3::ZERO, Vec3::Z);
    bvh.intersect(&mut ray);
    assert!(!ray.hit.is_hit(), "{}: the map is kept across an optimize", L::NAME);

    bvh.clear_opacity_micro_maps();
    assert_eq!(bvh.opacity_micro_map_n(), 0);
    let mut ray = Ray::new(Vec3::ZERO, Vec3::Z);
    bvh.intersect(&mut ray);
    assert!(ray.hit.is_hit());

    // A new build detaches the map.
    bvh.set_opacity_micro_maps(vec![0u32], 1);
    bvh.build(Geometry::new(two_triangles()));
    assert_eq!(bvh.opacity_micro_map_n(), 0);
    let mut ray = Ray::new(Vec3::ZERO, Vec3::Z);
    bvh.intersect(&mut ray);
    assert!(ray.hit.is_hit());

    // A 2x2 map on a grid: hide half of each triangle's micro-triangles and check that hits
    // land only on the opaque ones (u + v < 0.5 is micro-triangle 0 for n = 2).
    let mut grid = L::new();
    grid.build(Geometry::new(grid_soup(4, 4.0, 1.0)));
    let map = vec![0b0001u32; grid.prim_count() as usize];
    grid.set_opacity_micro_maps(map, 2);
    let mut rng = common::Rng::new(77);
    let (mut hits, mut misses) = (0, 0);
    for _ in 0..200 {
        let o = Vec3::new(rng.unit() * 4.0, rng.unit() * 4.0, 0.0);
        let mut ray = Ray::new(o, Vec3::Z);
        grid.intersect(&mut ray);
        if ray.hit.is_hit() {
            assert!(ray.hit.u + ray.hit.v < 0.5 + 1e-4, "hit on a transparent micro-triangle: u = {}, v = {}", ray.hit.u, ray.hit.v);
            hits += 1;
        } else {
            misses += 1;
        }
    }
    assert!(hits > 0 && misses > 0);
}

fn as_blas_in_tlas<L: CpuLayout>() {
    let mut blas = L::new();
    blas.build(Geometry::new(two_triangles()));
    let blas: Arc<dyn Blas> = Arc::new(blas);
    let instances = vec![
        BlasInstance::new(0),
        BlasInstance::with_transform(0, Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0))).with_mask(2),
    ];
    let mut tlas = Bvh::new();
    tlas.build_tlas(instances, vec![blas.clone()]);
    assert!(tlas.is_tlas());
    assert_eq!(tlas.prim_count(), 2);
    assert_eq!(Arc::strong_count(&blas), 2);
    let (mn, mx) = tlas.instances()[1].aabb();
    assert_relative_eq!(mn.x, 9.0, epsilon = 1e-5);
    assert_relative_eq!(mx.x, 11.0, epsilon = 1e-5);

    let mut ray = Ray::new(Vec3::ZERO, Vec3::Z);
    tlas.intersect(&mut ray);
    assert_relative_eq!(ray.hit.t, 3.0, epsilon = 1e-5);
    assert_eq!(ray.hit.inst, 0);
    assert_eq!(ray.hit.prim, 1);

    let mut ray = Ray::new(Vec3::new(10.0, 0.0, 0.0), Vec3::Z);
    tlas.intersect(&mut ray);
    assert_eq!(ray.hit.inst, 1);
    assert_eq!(ray.hit.prim, 1);
    assert_relative_eq!(ray.hit.t, 3.0, epsilon = 1e-5);
    assert!(tlas.is_occluded(&Ray::new(Vec3::new(10.0, 0.0, 0.0), Vec3::Z)));

    // Masks: instance 1 only answers rays with bit 1 set.
    let mut masked = Ray::new(Vec3::new(10.0, 0.0, 0.0), Vec3::Z).with_mask(1);
    tlas.intersect(&mut masked);
    assert!(!masked.hit.is_hit());
    assert!(!tlas.is_occluded(&masked));
    assert!(tlas.is_occluded(&Ray::new(Vec3::new(10.0, 0.0, 0.0), Vec3::Z).with_mask(2)));

    // A miss between the instances.
    let mut between = Ray::new(Vec3::new(5.0, 0.0, 0.0), Vec3::Z);
    tlas.intersect(&mut between);
    assert!(!between.hit.is_hit());

    // Many random rays through a translated instance agree with the BLAS in object space.
    let verts = random_triangles(1000, 51);
    let mut big = L::new();
    big.build(Geometry::new(verts.clone()));
    let big: Arc<dyn Blas> = Arc::new(big);
    let mut tlas2 = Bvh::new();
    tlas2.build_tlas(vec![BlasInstance::with_transform(0, Mat4::from_translation(Vec3::new(0.0, 0.0, 5.0)))], vec![big]);
    let mut rng = common::Rng::new(52);
    for _ in 0..200 {
        let o = Vec3::new(rng.unit() * 2.0 - 0.5, rng.unit() * 2.0 - 0.5, -1.0);
        let d = Vec3::new(rng.unit(), rng.unit(), rng.unit()) - o;
        let mut ray = Ray::new(o + Vec3::new(0.0, 0.0, 5.0), d);
        tlas2.intersect(&mut ray);
        let reference = common::brute_force_hit(&verts, o, d);
        assert_eq!(ray.hit.is_hit(), reference.is_some());
        if let Some((t, prim)) = reference {
            assert_relative_eq!(ray.hit.t, t, epsilon = 1e-4 * t.max(1.0));
            assert_eq!(ray.hit.prim as usize, prim);
            assert_eq!(ray.hit.inst, 0);
        }
    }
    drop(tlas);
    assert_eq!(Arc::strong_count(&blas), 1);
}

fn shared_across_threads<L: CpuLayout>() {
    let verts = random_triangles(5000, 77);
    let mut bvh = L::new();
    bvh.build(Geometry::new(verts.clone()));
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

fn clear_and_rebuild<L: CpuLayout>() {
    let mut bvh = L::new();
    bvh.build(Geometry::new(two_triangles()));
    bvh.clear();
    assert!(!bvh.is_built());
    assert!(bvh.vertices().is_none());
    bvh.build(Geometry::new(random_triangles(100, 3)));
    assert!(bvh.is_built());
    assert_eq!(bvh.prim_count(), 100);
    bvh.build(Geometry::new(two_triangles())); // rebuilding replaces the geometry
    assert_eq!(bvh.prim_count(), 2);
    assert_eq!(bvh.vertices().unwrap().len(), 6);
    let mut ray = Ray::new(Vec3::ZERO, Vec3::Z);
    bvh.intersect(&mut ray);
    assert_eq!(ray.hit.prim, 1);
    // Default is the same as new.
    assert!(!L::new().is_built());
}

// ---------------------------------------------------------------------------------------------
// Test entry points.
// ---------------------------------------------------------------------------------------------

#[test]
fn bvh4_build_and_intersect() {
    build_and_intersect::<Bvh4Cpu>();
}
#[test]
fn bvh8_build_and_intersect() {
    build_and_intersect::<Bvh8Cpu>();
}

#[test]
fn bvh4_occlusion_semantics() {
    occlusion_semantics::<Bvh4Cpu>();
}
#[test]
fn bvh8_occlusion_semantics() {
    occlusion_semantics::<Bvh8Cpu>();
}

#[test]
fn bvh4_soup_and_indexed_match_brute_force() {
    soup_and_indexed_match_brute_force::<Bvh4Cpu>();
}
#[test]
fn bvh8_soup_and_indexed_match_brute_force() {
    soup_and_indexed_match_brute_force::<Bvh8Cpu>();
}

#[test]
fn bvh4_hq_build() {
    hq_build::<Bvh4Cpu>();
}
#[test]
fn bvh8_hq_build() {
    hq_build::<Bvh8Cpu>();
}

#[test]
fn bvh4_large_threaded_build() {
    large_threaded_build::<Bvh4Cpu>();
}
#[test]
fn bvh8_large_threaded_build() {
    large_threaded_build::<Bvh8Cpu>();
}

#[test]
fn bvh4_refit_follows_moved_vertices() {
    refit_follows_moved_vertices::<Bvh4Cpu>();
}
#[test]
fn bvh8_refit_follows_moved_vertices() {
    refit_follows_moved_vertices::<Bvh8Cpu>();
}

#[test]
fn bvh4_refit_is_refused_when_not_possible() {
    refit_is_refused_when_not_possible::<Bvh4Cpu>();
}
#[test]
fn bvh8_refit_is_refused_when_not_possible() {
    refit_is_refused_when_not_possible::<Bvh8Cpu>();
}

#[test]
fn bvh4_optimize_does_not_hurt() {
    optimize_does_not_hurt::<Bvh4Cpu>();
}
#[test]
fn bvh8_optimize_does_not_hurt() {
    optimize_does_not_hurt::<Bvh8Cpu>();
}

#[test]
fn bvh4_from_mbvh_matches_direct_build() {
    from_mbvh_matches_direct_build::<Bvh4Cpu>();
}
#[test]
fn bvh8_from_mbvh_matches_direct_build() {
    from_mbvh_matches_direct_build::<Bvh8Cpu>();
}

#[test]
fn bvh4_from_mbvh_repeatedly() {
    from_mbvh_repeatedly::<Bvh4Cpu>();
}
#[test]
fn bvh8_from_mbvh_repeatedly() {
    from_mbvh_repeatedly::<Bvh8Cpu>();
}

#[test]
fn bvh4_save_and_load() {
    save_and_load::<Bvh4Cpu>();
}
#[test]
fn bvh8_save_and_load() {
    save_and_load::<Bvh8Cpu>();
}

#[test]
fn bvh4_data_blocks() {
    data_blocks::<Bvh4Cpu>();
}
#[test]
fn bvh8_data_blocks() {
    data_blocks::<Bvh8Cpu>();
}

#[test]
fn bvh4_opacity_micro_maps() {
    opacity_micro_maps::<Bvh4Cpu>();
}
#[test]
fn bvh8_opacity_micro_maps() {
    opacity_micro_maps::<Bvh8Cpu>();
}

#[test]
fn bvh4_as_blas_in_tlas() {
    as_blas_in_tlas::<Bvh4Cpu>();
}
#[test]
fn bvh8_as_blas_in_tlas() {
    as_blas_in_tlas::<Bvh8Cpu>();
}

#[test]
fn bvh4_shared_across_threads() {
    shared_across_threads::<Bvh4Cpu>();
}
#[test]
fn bvh8_shared_across_threads() {
    shared_across_threads::<Bvh8Cpu>();
}

#[test]
fn bvh4_clear_and_rebuild() {
    clear_and_rebuild::<Bvh4Cpu>();
}
#[test]
fn bvh8_clear_and_rebuild() {
    clear_and_rebuild::<Bvh8Cpu>();
}

#[test]
fn layouts_agree() {
    let verts = random_triangles(3000, 61);
    let mut reference = Bvh::new();
    reference.build(verts.clone());
    let mut b4 = Bvh4Cpu::new();
    b4.build(verts.clone());
    let mut b8 = Bvh8Cpu::new();
    b8.build(verts.clone());
    let mut rng = common::Rng::new(62);
    let mut hits = 0;
    for _ in 0..500 {
        let o = Vec3::new(rng.unit() * 2.0 - 0.5, rng.unit() * 2.0 - 0.5, -1.0);
        let d = Vec3::new(rng.unit(), rng.unit(), rng.unit()) - o;
        let mut r = Ray::new(o, d);
        let mut r4 = Ray::new(o, d);
        let mut r8 = Ray::new(o, d);
        reference.intersect(&mut r);
        b4.intersect(&mut r4);
        b8.intersect(&mut r8);
        assert_eq!(r.hit.is_hit(), r4.hit.is_hit());
        assert_eq!(r.hit.is_hit(), r8.hit.is_hit());
        if r.hit.is_hit() {
            hits += 1;
            assert_eq!(r.hit.prim, r4.hit.prim);
            assert_eq!(r.hit.prim, r8.hit.prim);
            assert!((r.hit.t - r4.hit.t).abs() <= 1e-4, "Bvh {} vs Bvh4Cpu {}", r.hit.t, r4.hit.t);
            assert!((r.hit.t - r8.hit.t).abs() <= 1e-4, "Bvh {} vs Bvh8Cpu {}", r.hit.t, r8.hit.t);
        }
        // Fresh rays: hit.t is the strict search limit for occlusion.
        let blocked = reference.is_occluded(&Ray::new(o, d));
        assert_eq!(blocked, b4.is_occluded(&Ray::new(o, d)));
        assert_eq!(blocked, b8.is_occluded(&Ray::new(o, d)));
    }
    assert!(hits > 100);
}

#[test]
fn bvh4_bundles() {
    let n = tinybvh::bundle_rays();
    let mut bvh = Bvh4Cpu::new();
    bvh.build(grid_soup(8, 4.0, 2.0));

    // Coherent rays through a grid: each one matches the single-ray result. The offsets keep
    // the rays off the quad edges and diagonals, where the two kernels may break ties
    // differently.
    let make_rays = || -> Vec<Ray> {
        (0..n)
            .map(|i| {
                let x = 0.07 + (i % 16) as f32 * 0.2437;
                let y = 0.11 + (i / 16) as f32 * 0.2437;
                Ray::new(Vec3::new(x, y, 0.0), Vec3::Z)
            })
            .collect()
    };
    let mut rays = make_rays();
    bvh.intersect_bundle(&mut rays);
    let mut same_prim = 0;
    for (i, ray) in rays.iter().enumerate() {
        let mut single = make_rays()[i];
        bvh.intersect(&mut single);
        assert_eq!(ray.hit.is_hit(), single.hit.is_hit(), "ray {i}");
        if single.hit.is_hit() {
            assert_relative_eq!(ray.hit.t, single.hit.t, epsilon = 1e-5);
            same_prim += (ray.hit.prim == single.hit.prim) as usize;
        }
    }
    let hits = rays.iter().filter(|r| r.hit.is_hit()).count();
    assert!(hits > n / 2);
    assert!(same_prim * 10 >= hits * 9, "bundle and single-ray primitives disagree on {} of {hits} hits", hits - same_prim);

    let mut rays = make_rays();
    let mut occluded = vec![false; n];
    bvh.is_occluded_bundle(&mut rays, &mut occluded);
    for (i, ray) in make_rays().iter().enumerate() {
        assert_eq!(occluded[i], bvh.is_occluded(ray), "ray {i}");
    }
    assert!(occluded.iter().filter(|&&o| o).count() > n / 2);

    // Rays that all miss.
    let mut rays: Vec<Ray> = (0..n).map(|i| Ray::new(Vec3::new(i as f32 * 0.01, 0.0, 0.0), -Vec3::Z)).collect();
    bvh.intersect_bundle(&mut rays);
    assert!(rays.iter().all(|r| !r.hit.is_hit()));
    let mut occluded = vec![true; n];
    bvh.is_occluded_bundle(&mut rays, &mut occluded);
    assert!(occluded.iter().all(|&o| !o));

    // A limited t_max is honoured per ray: the grid at z = 2 is beyond t = 1.
    let mut rays: Vec<Ray> = (0..n).map(|i| Ray::with_t_max(Vec3::new(1.0 + i as f32 * 0.01, 1.0, 0.0), Vec3::Z, 1.0)).collect();
    bvh.intersect_bundle(&mut rays);
    assert!(rays.iter().all(|r| r.hit.t == 1.0));
    let mut occluded = vec![true; n];
    bvh.is_occluded_bundle(&mut rays, &mut occluded);
    assert!(occluded.iter().all(|&o| !o));
}

#[test]
#[should_panic(expected = "exactly")]
fn bvh4_bundle_wrong_length_panics() {
    let mut bvh = Bvh4Cpu::new();
    bvh.build(two_triangles());
    let mut rays = vec![Ray::new(Vec3::ZERO, Vec3::Z); tinybvh::bundle_rays() + 1];
    bvh.intersect_bundle(&mut rays);
}

#[test]
#[should_panic(expected = "exactly")]
fn bvh4_occluded_bundle_wrong_length_panics() {
    let mut bvh = Bvh4Cpu::new();
    bvh.build(two_triangles());
    let n = tinybvh::bundle_rays();
    let mut rays = vec![Ray::new(Vec3::ZERO, Vec3::Z); n];
    let mut occluded = vec![false; n - 1];
    bvh.is_occluded_bundle(&mut rays, &mut occluded);
}

#[test]
#[should_panic(expected = "no vertices")]
fn bvh4_empty_geometry_panics() {
    Bvh4Cpu::new().build(Vec::<Vec4>::new());
}

#[test]
#[should_panic(expected = "no vertices")]
fn bvh8_empty_geometry_panics() {
    Bvh8Cpu::new().build(Vec::<Vec4>::new());
}

#[test]
#[should_panic(expected = "multiple of three")]
fn bvh4_partial_triangle_panics() {
    Bvh4Cpu::new().build_hq(vec![Vec4::ZERO, Vec4::ONE]);
}

#[test]
#[should_panic(expected = "out of range")]
fn bvh8_bad_index_panics() {
    Bvh8Cpu::new().build(Geometry::indexed(vec![Vec4::ZERO, Vec4::ONE, Vec4::X], vec![0, 1, 3]));
}

#[test]
#[should_panic(expected = "has not been built")]
fn bvh4_intersect_before_build_panics() {
    let bvh = Bvh4Cpu::new();
    let mut ray = Ray::new(Vec3::ZERO, Vec3::Z);
    bvh.intersect(&mut ray);
}

#[test]
#[should_panic(expected = "has not been built")]
fn bvh8_intersect_before_build_panics() {
    let bvh = Bvh8Cpu::new();
    bvh.is_occluded(&Ray::new(Vec3::ZERO, Vec3::Z));
}

#[test]
#[should_panic(expected = "has not been built")]
fn bvh4_from_unbuilt_mbvh_panics() {
    Bvh4Cpu::from_mbvh(Mbvh4::new());
}

#[test]
#[should_panic(expected = "has not been built")]
fn bvh8_sah_cost_before_build_panics() {
    Bvh8Cpu::new().sah_cost();
}

#[test]
#[should_panic(expected = "intermediate tree")]
fn bvh4_sah_cost_after_load_panics() {
    let dir = temp_dir("sah-load");
    let path = dir.join("layout.bin");
    let mut bvh = Bvh4Cpu::new();
    bvh.build(two_triangles());
    bvh.save(&path).unwrap();
    let loaded = Bvh4Cpu::load_self_contained(&path).unwrap();
    std::fs::remove_dir_all(&dir).ok();
    loaded.sah_cost();
}

#[test]
#[should_panic(expected = "NaN")]
fn bvh8_nan_ray_panics() {
    let mut bvh = Bvh8Cpu::new();
    bvh.build(two_triangles());
    let mut ray = Ray::new(Vec3::new(f32::NAN, 0.0, 0.0), Vec3::Z);
    bvh.intersect(&mut ray);
}

#[test]
#[should_panic(expected = "words")]
fn bvh4_short_opacity_map_panics() {
    let mut bvh = Bvh4Cpu::new();
    bvh.build(two_triangles());
    bvh.set_opacity_micro_maps(vec![0u32], 2); // 2x2 bits per triangle, two triangles: 2 words
}
