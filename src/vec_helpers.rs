use glam::{Vec2, Vec3, Vec4};

use crate::ffi;

pub trait Vec2Helpers {
    fn x(&self) -> f32;
    fn y(&self) -> f32;

    fn xy(&self) -> Vec2;
}

impl Vec2Helpers for ffi::tinybvh_bvhvec2 {
    #[inline]
    fn x(&self) -> f32 {
        unsafe { self.__bindgen_anon_1.cell[0] }
    }

    #[inline]
    fn y(&self) -> f32 {
        unsafe { self.__bindgen_anon_1.cell[1] }
    }

    #[inline]
    fn xy(&self) -> Vec2 {
        Vec2::new(self.x(), self.y())
    }
}

pub trait Vec3Helpers {
    fn x(&self) -> f32;
    fn y(&self) -> f32;
    fn z(&self) -> f32;

    fn xyz(&self) -> Vec3;
}

impl Vec3Helpers for ffi::tinybvh_bvhvec3 {
    #[inline]
    fn x(&self) -> f32 {
        unsafe { self.__bindgen_anon_1.cell[0] }
    }

    #[inline]
    fn y(&self) -> f32 {
        unsafe { self.__bindgen_anon_1.cell[1] }
    }

    #[inline]
    fn z(&self) -> f32 {
        unsafe { self.__bindgen_anon_1.cell[2] }
    }

    #[inline]
    fn xyz(&self) -> Vec3 {
        Vec3::new(self.x(), self.y(), self.z())
    }
}

pub trait Vec4Helpers {
    fn x(&self) -> f32;
    fn y(&self) -> f32;
    fn z(&self) -> f32;
    fn w(&self) -> f32;

    fn xyzw(&self) -> Vec4;
}

impl Vec4Helpers for ffi::tinybvh_bvhvec4 {
    #[inline]
    fn x(&self) -> f32 {
        unsafe { self.__bindgen_anon_1.cell[0] }
    }

    #[inline]
    fn y(&self) -> f32 {
        unsafe { self.__bindgen_anon_1.cell[1] }
    }

    #[inline]
    fn z(&self) -> f32 {
        unsafe { self.__bindgen_anon_1.cell[2] }
    }

    #[inline]
    fn w(&self) -> f32 {
        unsafe { self.__bindgen_anon_1.cell[3] }
    }

    #[inline]
    fn xyzw(&self) -> Vec4 {
        Vec4::new(self.x(), self.y(), self.z(), self.w())
    }
}
