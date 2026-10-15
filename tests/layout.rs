//! Verifies that the Rust mirrors of tinybvh's structs match the C++ layouts, and that the
//! library-wide queries return sane values.

use core::mem::size_of;

use glam::Vec3;
use tinybvh::ffi;
use tinybvh::{
    BlasInstance, BlasInstanceEx, BuildSettings, Intersection, IntersectionEx, Ray, RayEx,
};

fn c_sizeof(id: ffi::tbvh_struct_id) -> usize {
    unsafe { ffi::tbvh_sizeof(id) }
}

#[test]
fn struct_sizes_match_cpp() {
    assert_eq!(size_of::<ffi::tbvh_vec3>(), c_sizeof(ffi::TBVH_STRUCT_VEC3));
    assert_eq!(size_of::<ffi::tbvh_vec4>(), c_sizeof(ffi::TBVH_STRUCT_VEC4));
    assert_eq!(size_of::<ffi::tbvh_dbl3>(), c_sizeof(ffi::TBVH_STRUCT_DBL3));
    assert_eq!(size_of::<ffi::tbvh_mat4>(), c_sizeof(ffi::TBVH_STRUCT_MAT4));
    assert_eq!(
        size_of::<ffi::tbvh_dblmat4>(),
        c_sizeof(ffi::TBVH_STRUCT_DBLMAT4)
    );
    assert_eq!(
        size_of::<Intersection>(),
        c_sizeof(ffi::TBVH_STRUCT_INTERSECTION)
    );
    assert_eq!(
        size_of::<IntersectionEx>(),
        c_sizeof(ffi::TBVH_STRUCT_INTERSECTION_EX)
    );
    assert_eq!(size_of::<Ray>(), c_sizeof(ffi::TBVH_STRUCT_RAY));
    assert_eq!(size_of::<RayEx>(), c_sizeof(ffi::TBVH_STRUCT_RAY_EX));
    assert_eq!(
        size_of::<BlasInstance>(),
        c_sizeof(ffi::TBVH_STRUCT_BLAS_INSTANCE)
    );
    assert_eq!(
        size_of::<BlasInstanceEx>(),
        c_sizeof(ffi::TBVH_STRUCT_BLAS_INSTANCE_EX)
    );
    assert_eq!(
        size_of::<ffi::tbvh_build_settings>(),
        c_sizeof(ffi::TBVH_STRUCT_BUILD_SETTINGS)
    );
    assert_eq!(
        size_of::<tinybvh::BvhNode>(),
        c_sizeof(ffi::TBVH_STRUCT_BVH_NODE)
    );
    assert_eq!(
        size_of::<ffi::tbvh_bvh_node_ex>(),
        c_sizeof(ffi::TBVH_STRUCT_BVH_NODE_EX)
    );
    assert_eq!(
        size_of::<ffi::tbvh_bvh_verbose_node>(),
        c_sizeof(ffi::TBVH_STRUCT_BVH_VERBOSE_NODE)
    );
    assert_eq!(
        size_of::<ffi::tbvh_bvh_gpu_node>(),
        c_sizeof(ffi::TBVH_STRUCT_BVH_GPU_NODE)
    );
    assert_eq!(
        size_of::<ffi::tbvh_mbvh4_node>(),
        c_sizeof(ffi::TBVH_STRUCT_MBVH4_NODE)
    );
    assert_eq!(
        size_of::<ffi::tbvh_mbvh8_node>(),
        c_sizeof(ffi::TBVH_STRUCT_MBVH8_NODE)
    );
    assert_eq!(
        c_sizeof(ffi::TBVH_STRUCT_COUNT),
        0,
        "unknown ids report size 0"
    );
}

#[test]
fn ray_matches_cpp_constructor() {
    let origin = Vec3::new(1.0, -2.0, 0.5);
    let dir = Vec3::new(0.0, 0.3, -2.0); // a zero component exercises the safe reciprocal
    let rust = Ray::with_t_max(origin, dir, 12.5).with_mask(0x1_0005);

    let mut c = ffi::tbvh_ray {
        o: ffi::tbvh_vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
        mask: 0,
        d: ffi::tbvh_vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
        inst_idx: 0,
        rd: ffi::tbvh_vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
        hit: ffi::tbvh_intersection {
            inst: 0,
            t: 0.0,
            u: 0.0,
            v: 0.0,
            prim: 0,
            aux_data: [0; ffi::PTR_SIZE],
            user: [0; 14],
        },
    };
    unsafe {
        ffi::tbvh_ray_init(
            &mut c,
            ffi::tbvh_vec3 {
                x: origin.x,
                y: origin.y,
                z: origin.z,
            },
            ffi::tbvh_vec3 {
                x: dir.x,
                y: dir.y,
                z: dir.z,
            },
            12.5,
            0x1_0005,
        );
    }
    assert_eq!([c.o.x, c.o.y, c.o.z], rust.origin.to_array());
    assert_eq!([c.d.x, c.d.y, c.d.z], rust.direction().to_array());
    assert_eq!([c.rd.x, c.rd.y, c.rd.z], rust.rcp_direction().to_array());
    assert_eq!(c.mask, rust.mask);
    assert_eq!(c.mask, 5);
    assert_eq!(c.inst_idx, rust.inst_idx());
    assert_eq!(c.hit.t, rust.hit.t);
    assert_eq!(c.hit.u, rust.hit.u);
    assert_eq!(c.hit.v, rust.hit.v);
    assert_eq!(c.hit.prim, rust.hit.prim);
    assert_eq!(c.hit.inst, rust.hit.inst);

    // set_direction and normalize agree with the C side as well.
    let mut rust2 = rust;
    rust2.set_direction(Vec3::new(3.0, 0.0, 4.0));
    unsafe {
        ffi::tbvh_ray_set_direction(
            &mut c,
            ffi::tbvh_vec3 {
                x: 3.0,
                y: 0.0,
                z: 4.0,
            },
        )
    };
    assert_eq!([c.rd.x, c.rd.y, c.rd.z], rust2.rcp_direction().to_array());
    rust2.normalize();
    unsafe { ffi::tbvh_ray_normalize(&mut c) };
    assert!(
        (c.d.x - rust2.direction().x).abs() < 1e-6 && (c.d.z - rust2.direction().z).abs() < 1e-6
    );
    assert!((c.rd.x - rust2.rcp_direction().x).abs() < 1e-5);
}

#[test]
fn ray_ex_matches_cpp_constructor() {
    let origin = glam::DVec3::new(1.0, -2.0, 0.5);
    let dir = glam::DVec3::new(0.0, 0.3, -2.0);
    let rust = RayEx::with_t_max(origin, dir, 42.0).with_mask(7);
    let mut c: ffi::tbvh_ray_ex = unsafe { core::mem::zeroed() };
    unsafe {
        ffi::tbvh_ray_ex_init(
            &mut c,
            ffi::tbvh_dbl3 {
                x: origin.x,
                y: origin.y,
                z: origin.z,
            },
            ffi::tbvh_dbl3 {
                x: dir.x,
                y: dir.y,
                z: dir.z,
            },
            42.0,
            7,
        );
    }
    assert_eq!([c.o.x, c.o.y, c.o.z], rust.origin.to_array());
    assert_eq!([c.rd.x, c.rd.y, c.rd.z], rust.rcp_direction().to_array());
    assert_eq!(c.mask, 7);
    assert_eq!(c.hit.t, 42.0);
    assert_eq!(c.hit.t, rust.hit.t);
    assert_eq!(c.inst_idx, rust.inst_idx());
}

#[test]
fn blas_instance_matches_cpp_init() {
    let rust = BlasInstance::new(3);
    let mut c: ffi::tbvh_blas_instance = unsafe { core::mem::zeroed() };
    unsafe { ffi::tbvh_blas_instance_init(&mut c, 3) };
    assert_eq!(c.blas_idx, rust.blas_idx);
    assert_eq!(c.mask, rust.mask);
    assert_eq!(
        c.transform.cell,
        rust.transform().transpose().to_cols_array()
    );
    assert_eq!(
        [c.aabb_min.x, c.aabb_min.y, c.aabb_min.z],
        rust.aabb().0.to_array()
    );
    assert_eq!(
        [c.aabb_max.x, c.aabb_max.y, c.aabb_max.z],
        rust.aabb().1.to_array()
    );

    // The transform round-trips through the row-major storage and the inverse is correct.
    let m = glam::Mat4::from_scale_rotation_translation(
        Vec3::new(2.0, 3.0, 4.0),
        glam::Quat::from_rotation_y(0.7),
        Vec3::new(1.0, 2.0, 3.0),
    );
    let mut inst = BlasInstance::with_transform(0, m);
    assert!(inst.transform().abs_diff_eq(m, 1e-6));
    inst.invert_transform();
    assert!(inst.inverse_transform().abs_diff_eq(m.inverse(), 1e-5));
    let p = Vec3::new(0.5, -1.0, 2.0);
    let c_p = unsafe {
        ffi::tbvh_transform_point(
            ffi::tbvh_vec3 {
                x: p.x,
                y: p.y,
                z: p.z,
            },
            &ffi::tbvh_mat4 {
                cell: m.transpose().to_cols_array(),
            },
        )
    };
    assert!(Vec3::new(c_p.x, c_p.y, c_p.z).abs_diff_eq(m.transform_point3(p), 1e-5));
}

#[test]
fn default_settings_match_cpp() {
    let bvh = tinybvh::Bvh::new();
    assert_eq!(bvh.settings(), BuildSettings::default());
    let mut bvh = bvh.with_settings(&BuildSettings::high_quality());
    assert!(bvh.settings().use_spatial_splits);
    bvh.set_settings(&BuildSettings::default());
    assert!(!bvh.settings().use_spatial_splits);
    assert_eq!(bvh.settings().optimize_iterations, 25);
}

#[test]
fn library_queries() {
    let (major, minor, _) = tinybvh::version();
    assert!(
        major >= 1 && (major > 1 || minor >= 9),
        "expected tinybvh 1.9 or later, got {major}.{minor}"
    );
    assert_eq!(tinybvh::bundle_rays(), 64);
    assert!(tinybvh::traversal_stack_size() >= 64);
    assert_eq!(
        unsafe { ffi::tbvh_inst_idx_bits() },
        32,
        "the instance index must live in its own field"
    );
    assert_eq!(unsafe { ffi::tbvh_far() }, tinybvh::FAR);
    assert_eq!(unsafe { ffi::tbvh_dbl_far() }, tinybvh::DBL_FAR);
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        assert!(tinybvh::simd_enabled());
        assert!(tinybvh::simd_builder_available());
    }
    #[cfg(not(feature = "simd"))]
    assert!(!tinybvh::simd_enabled());
    tinybvh::shutdown_thread_pool();
}
