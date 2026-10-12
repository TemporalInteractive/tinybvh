//! [`Bvh8Cpu`]: tinybvh's `BVH8_CPU`, the 8-wide SIMD CPU traversal layout.

use crate::bvh4_cpu::impl_cpu_layout;
use crate::mbvh::Mbvh8;
use crate::ray::Ray;

impl_cpu_layout!(
    /// The 8-wide SIMD CPU layout (tinybvh's `BVH8_CPU`): eight child boxes tested per step
    /// with AVX2, four triangles per leaf. Usually close to [`Bvh4Cpu`](crate::Bvh4Cpu) in
    /// speed; which one wins depends on the scene and the CPU.
    ///
    /// Build it over triangles like a [`Bvh`](crate::Bvh), or convert a tuned
    /// [`Mbvh8`](crate::Mbvh8) with [`from_mbvh`](Bvh8Cpu::from_mbvh). The layout keeps an
    /// intermediate binary tree so that it can be [refitted](Bvh8Cpu::refit) for animation and
    /// [optimized](Bvh8Cpu::optimize); set `discard_intermediates` in the
    /// [settings](Bvh8Cpu::settings) to drop it and save memory. A `Bvh8Cpu` can serve as a
    /// bottom-level structure in a [`Bvh`](crate::Bvh) TLAS, and honours
    /// [opacity micro maps](Bvh8Cpu::set_opacity_micro_maps).
    ///
    /// The layout lives in 64-byte blocks ([`data`](Bvh8Cpu::data), [`used_blocks`](Bvh8Cpu::used_blocks));
    /// tinybvh does not maintain [`used_nodes`](Bvh8Cpu::used_nodes) for it, which reads 0.
    ///
    /// # Example
    ///
    /// ```
    /// use glam::{Vec3, Vec4};
    /// use tinybvh::{Bvh8Cpu, Ray};
    ///
    /// let tri = |z: f32| [Vec4::new(-1.0, -1.0, z, 0.0), Vec4::new(1.0, -1.0, z, 0.0), Vec4::new(0.0, 1.0, z, 0.0)];
    /// let mut vertices = Vec::new();
    /// vertices.extend(tri(5.0));
    /// vertices.extend(tri(3.0));
    ///
    /// let mut bvh = Bvh8Cpu::new();
    /// bvh.build(vertices);
    ///
    /// let mut ray = Ray::new(Vec3::ZERO, Vec3::Z);
    /// bvh.intersect(&mut ray);
    /// assert_eq!(ray.hit.prim, 1); // the closer triangle
    /// assert!((ray.hit.t - 3.0).abs() < 1e-5);
    /// assert!(bvh.is_occluded(&Ray::with_t_max(Vec3::ZERO, Vec3::Z, 4.0)));
    /// assert!(!bvh.is_occluded(&Ray::with_t_max(Vec3::ZERO, Vec3::Z, 2.0)));
    /// ```
    Bvh8Cpu, Mbvh8, tbvh_bvh8_cpu, "Bvh8Cpu",
    new = tbvh_bvh8_cpu_new, free = tbvh_bvh8_cpu_free, base = tbvh_bvh8_cpu_base, base_const = tbvh_bvh8_cpu_base_const,
    build = tbvh_bvh8_cpu_build, build_indexed = tbvh_bvh8_cpu_build_indexed,
    build_hq = tbvh_bvh8_cpu_build_hq, build_hq_indexed = tbvh_bvh8_cpu_build_hq_indexed,
    convert_from = tbvh_bvh8_cpu_convert_from, optimize = tbvh_bvh8_cpu_optimize, refit = tbvh_bvh8_cpu_refit,
    sah_cost = tbvh_bvh8_cpu_sah_cost, intersect = tbvh_bvh8_cpu_intersect, is_occluded = tbvh_bvh8_cpu_is_occluded,
    save = tbvh_bvh8_cpu_save, load = tbvh_bvh8_cpu_load, load_self_contained = tbvh_bvh8_cpu_load_self_contained,
    data = tbvh_bvh8_cpu_data
);
