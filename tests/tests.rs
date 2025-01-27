#[cfg(test)]
mod tests {
    use core::f32;

    use approx::assert_relative_eq;
    use glam::Vec3;
    use tinybvh::*;

    #[test]
    fn layout_wald32() {
        let ray = Ray::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
    }
}
