# tinybvh for Rust

Safe Rust bindings to [tinybvh](https://github.com/jbikker/tinybvh), Jacco Bikker's header-only
C++ library for building and traversing bounding volume hierarchies.

tinybvh builds high quality BVHs over triangles, custom primitives and instances, traverses them on
the CPU with AVX2 / NEON kernels, and produces node layouts for GPU ray tracing. This crate wraps
every layout of tinybvh 1.9 behind owned Rust types:

* geometry a tree references is owned by the tree and kept alive for as long as the tree exists;
* inputs are validated before they reach C++, so the library never reads out of bounds;
* preconditions tinybvh treats as fatal (refitting a tree with spatial splits, loading a file that
  does not match, ...) are reported as `Result`s;
* every structure is `Send + Sync`: queries take `&self` and can run from many threads;
* math types are [`glam`](https://crates.io/crates/glam)'s `Vec3` / `Vec4` / `Mat4` (and their
  `f64` twins for the double precision layouts).

tinybvh itself is compiled by the `cc` crate from the vendored headers; there is no `bindgen` and
no system library to install, just a C++17 compiler.

## Quick start

```rust
use glam::{Vec3, Vec4};
use tinybvh::{Bvh, Ray};

// Three vertices per triangle; w is padding.
let vertices = vec![
    Vec4::new(-1.0, -1.0, 2.0, 0.0),
    Vec4::new(1.0, -1.0, 2.0, 0.0),
    Vec4::new(0.0, 1.0, 2.0, 0.0),
];
let mut bvh = Bvh::new();
bvh.build(vertices);

let mut ray = Ray::new(Vec3::ZERO, Vec3::Z);
bvh.intersect(&mut ray);
assert!(ray.hit.is_hit());
assert_eq!(ray.hit.prim, 0);
assert!((ray.hit.t - 2.0).abs() < 1e-5);
```

## Layouts

| type | purpose |
| --- | --- |
| `Bvh` | the 32-byte node layout every other layout is derived from; the only layout that can be a TLAS; supports custom geometry, refitting, optimization, tree surgery and serialization |
| `Bvh4Cpu` | 4-wide SIMD traversal with four triangles per leaf: the fastest CPU layout |
| `Bvh8Cpu` | 8-wide SIMD traversal |
| `BvhGpu`, `Bvh4Gpu`, `Bvh8Cwbvh`, `Bvh4AmdHw` | node data for GPU ray tracing kernels (Aila & Laine, quantized 4-wide, compressed wide BVH, AMD hardware format) |
| `Mbvh4`, `Mbvh8` | wide intermediate layouts, convertible into the CPU and GPU layouts |
| `BvhVerbose` | nodes with parent links, for tree surgery and the optimizer |
| `VoxelSet` | a 256³ brick-based voxel object with DDA traversal that can be instanced in a TLAS |
| `BvhDouble`, `Bvh4Double` | double precision with 64-bit indices |

All layouts share `BuildSettings`, which selects the builder: Bonsai by default, Bonsai HQ with
`use_spatial_splits`, binned SAH or SBVH with `binned_bvh`, plus the reference, full-sweep and
H-PLOC builders for experiments.

```rust
use tinybvh::{Bvh4Cpu, BuildSettings};

let mut bvh = Bvh4Cpu::new().with_settings(&BuildSettings::high_quality());
bvh.build(vertices);
```

## Instancing

A top-level acceleration structure is a `Bvh` built over `BlasInstance`s. Each instance pairs a
transform and a 16-bit mask with one of the bottom-level structures, which are shared through
`Arc<dyn Blas>` (`Bvh`, `Bvh4Cpu`, `Bvh8Cpu` and `VoxelSet` implement `Blas`).

```rust
use std::sync::Arc;
use glam::{Mat4, Vec3};
use tinybvh::{Blas, BlasInstance, Bvh, Ray};

let mut mesh = Bvh::new();
mesh.build(vertices);
let blasses: Vec<Arc<dyn Blas>> = vec![Arc::new(mesh)];

let instances = vec![
    BlasInstance::new(0),
    BlasInstance::with_transform(0, Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0))).with_mask(0b10),
];
let mut tlas = Bvh::new();
tlas.build_tlas(instances, blasses);

let mut ray = Ray::new(Vec3::new(10.0, 0.0, 0.0), Vec3::Z);
tlas.intersect(&mut ray);
println!("instance {} primitive {} at t = {}", ray.hit.inst, ray.hit.prim, ray.hit.t);

// Rigid animation: change transforms and rebuild the (small) TLAS.
tlas.instances_mut()[1].set_transform(Mat4::from_translation(Vec3::new(0.0, 5.0, 0.0)));
tlas.rebuild_tlas().unwrap();
```

## Custom primitives

Implement `CustomGeometry` (bounds plus closest-hit and any-hit tests) and build a `Bvh` with
`build_custom`. Custom structures can be instanced in a TLAS like any other BLAS; inside a TLAS
the callbacks see the ray in object space, with `Ray::inst_idx` naming the instance.

## Maintaining a tree

* **Rebuild**: call `build` again.
* **Refit**: change vertices through `vertices_mut` and call `refit`; works for trees built without
  spatial splits (`BuildSettings::refittable`) and is far cheaper than a rebuild.
* **Optimize**: `optimize(iterations)` lowers the SAH cost with insertion-based tree rotations.
* **Inspect**: `nodes`, `prim_indices`, `sah_cost`, `epo_cost`, `node_count`, `leaf_count`.
* **Serialize**: `save` / `load` / `load_self_contained` for `Bvh`, `Bvh4Cpu`, `Bvh8Cpu`,
  `Bvh8Cwbvh` and the double precision layouts; every layout can also load a self-contained `Bvh`
  file and convert it.

## Rays

`Ray::new(origin, direction)` does not normalize the direction, so `hit.t` is measured in units of
the direction length; call `normalize` when you need world distances. `Ray::with_t_max` limits the
search. `hit.t` is also the upper bound of `is_occluded`, so a ray that already records a hit is not
"occluded" by that same hit.

Bundles of 64 rays (`intersect_bundle`, `is_occluded_bundle`) trace coherent rays together on a
TLAS (`Bvh`) or on a `Bvh4Cpu`.

## Features

* `simd` (default): compile tinybvh with AVX2 + FMA on x86-64 and NEON on AArch64. The binary
  then requires such a CPU. Disable it for a portable (slower) build; the API is unchanged.

## Building

A C++17 compiler is required: MSVC on Windows, GCC or Clang elsewhere; the `cc` crate finds it, and
`CXX` overrides the choice. The vendored tinybvh headers are compiled once into a static library.

## Conversions

Every layout builds its own intermediate trees, but a tree tuned once can also be converted:
`Mbvh4::from_bvh(bvh)` collapses a `Bvh`, and `Bvh4Cpu::from_mbvh`, `Bvh4Gpu::from_mbvh`,
`Bvh8Cwbvh::from_mbvh`, `BvhGpu::from_bvh` and `BvhVerbose::from_bvh` consume their source. A
converted structure keeps its source alive and cannot be refitted or optimized in place; build it
from geometry when you need that.

## Notes on tinybvh behaviour

Some tinybvh 1.9.1 behaviours are smoothed over by the wrappers, so they are worth knowing:

* `BvhGpu` traversal crashes in tinybvh for meshes whose root is a single leaf (up to four
  triangles); the wrapper detects this and answers such queries with a scalar triangle test.
* The wide CPU layouts report `used_nodes() == 0`; use `is_built()` and the data block counts.
* The SIMD traversal kernels count traversal steps only in debug builds of tinybvh; in release
  builds `intersect` returns 0 for `Bvh4Cpu`, `Bvh8Cpu`, `Bvh8Cwbvh` and `Bvh4Double`.
* `merge_leafs` leaves unused nodes; `compact` (called automatically by `refit` and `optimize`)
  clears that state through a round trip the C shim performs.
* A tree loaded from a file keeps no per-primitive bounds, so `split_leafs` is refused on it.

## Fatal errors

tinybvh reports unrecoverable conditions through a fatal error handler and then terminates. This
crate prevents them where it can (empty geometry, indices out of range and NaN rays panic instead;
state-dependent preconditions return `Err`). Anything that still reaches tinybvh is printed to
standard error and aborts the process; `set_fatal_error_handler` installs your own reporter.

## License

MIT, like tinybvh itself. See `LICENSE` and `ffi/tinybvh/LICENSE`.
