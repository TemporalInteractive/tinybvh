#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use glam::{Vec3, Vec4};
    use tinybvh::*;

    fn test_triangles_vertices() -> Vec<Vec4> {
        vec![
            Vec4::new(-2.0, 1.0, -1.0, 0.0),
            Vec4::new(-1.0, 1.0, -1.0, 0.0),
            Vec4::new(-2.0, 0.0, -1.0, 0.0),
            Vec4::new(2.0, 1.0, -1.0, 0.0),
            Vec4::new(2.0, 0.0, -1.0, 0.0),
            Vec4::new(1.0, 0.0, -1.0, 0.0),
        ]
    }

    #[test]
    fn layout_wald32() {
        let triangles_vertices = test_triangles_vertices();

        let mut bvh = Bvh::new();
        bvh.build(&triangles_vertices);

        let mut ray = Ray::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0));
        assert_eq!(bvh.intersect(&mut ray), 7);
        assert_relative_eq!(ray.hit.t, 1e30);

        let mut ray = Ray::new(Vec3::new(-1.5, 0.5, 0.0), Vec3::new(0.0, 0.0, -1.0));
        bvh.intersect(&mut ray);
        assert_relative_eq!(ray.hit.t, 1.0);
        assert_eq!(ray.hit.prim, 0);

        let mut ray = Ray::new(Vec3::new(1.5, 0.45, 0.0), Vec3::new(0.0, 0.0, -1.0));
        bvh.intersect(&mut ray);
        assert_relative_eq!(ray.hit.t, 1.0);
        assert_eq!(ray.hit.prim, 1);
    }
}
