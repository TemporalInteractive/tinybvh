#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(clashing_extern_declarations)]

use glam::Vec3;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

impl From<Vec3> for tinybvh_bvhvec3 {
    fn from(value: Vec3) -> Self {
        tinybvh_bvhvec3 {
            __bindgen_anon_1: tinybvh_bvhvec3__bindgen_ty_1 {
                cell: [value.x, value.y, value.z],
            },
        }
    }
}

impl From<tinybvh_bvhvec3> for Vec3 {
    fn from(value: tinybvh_bvhvec3) -> Self {
        unsafe {
            Vec3::new(
                value.__bindgen_anon_1.cell[0],
                value.__bindgen_anon_1.cell[1],
                value.__bindgen_anon_1.cell[2],
            )
        }
    }
}
