//! Safe Rust bindings to [tinybvh](https://github.com/jbikker/tinybvh), Jacco Bikker's header-only
//! C++ library for building and traversing bounding volume hierarchies.
//!
//! tinybvh builds high quality BVHs over triangles, custom primitives and instances, and traverses
//! them on the CPU with SIMD kernels; it also produces node layouts for GPU ray tracing. This crate
//! wraps every layout of tinybvh 1.9 behind owned Rust types, keeps the geometry a tree references
//! alive for as long as the tree exists, and validates inputs so that the C++ code never reads out
//! of bounds.
//!
//! # Quick start
//!
//! ```
//! use glam::{Vec3, Vec4};
//! use tinybvh::{Bvh, Ray};
//!
//! // Three vertices per triangle; w is padding.
//! let vertices = vec![
//!     Vec4::new(-1.0, -1.0, 2.0, 0.0),
//!     Vec4::new(1.0, -1.0, 2.0, 0.0),
//!     Vec4::new(0.0, 1.0, 2.0, 0.0),
//! ];
//! let mut bvh = Bvh::new();
//! bvh.build(vertices);
//!
//! let mut ray = Ray::new(Vec3::ZERO, Vec3::Z);
//! bvh.intersect(&mut ray);
//! assert!(ray.hit.is_hit());
//! assert_eq!(ray.hit.prim, 0);
//! assert!((ray.hit.t - 2.0).abs() < 1e-5);
//! ```
//!
//! # Layouts
//!
//! | type | purpose |
//! | --- | --- |
//! | [`Bvh`] | the 32-byte node layout every other layout is derived from; the only layout that can be a TLAS; supports custom geometry, refitting, optimization and serialization |
//! | [`Bvh4Cpu`] | 4-wide SIMD traversal, the fastest CPU layout |
//! | [`Bvh8Cpu`] | 8-wide SIMD traversal |
//! | [`BvhGpu`], [`Bvh4Gpu`], [`Bvh8Cwbvh`], [`Bvh4AmdHw`] | node data for GPU ray tracing kernels |
//! | [`Mbvh4`], [`Mbvh8`] | wide intermediate layouts |
//! | [`BvhVerbose`] | nodes with parent links, for tree surgery |
//! | [`VoxelSet`] | a 256³ voxel object that can be instanced in a TLAS |
//! | [`BvhDouble`], [`Bvh4Double`] | double precision with 64-bit indices |
//!
//! All layouts share the [`BuildSettings`] configuration; the quality / speed trade-off of the
//! builder is selected there.
//!
//! # Instancing
//!
//! A top-level acceleration structure is a [`Bvh`] built with [`Bvh::build_tlas`] over
//! [`BlasInstance`]s, each pairing a transform with one of the bottom-level structures that
//! implement [`Blas`]. Bottom-level structures are shared through `Arc`, so the TLAS keeps them
//! alive; rebuild the TLAS after changing instance transforms.
//!
//! # Threads
//!
//! Builds run on tinybvh's process-wide thread pool unless `enable_threading` is off. Every
//! structure is `Send + Sync`: queries take `&self` and can run from many threads at once.
//!
//! # Features
//!
//! `simd` (default) compiles tinybvh with AVX2 + FMA on x86-64 and NEON on AArch64; the binary then
//! requires such a CPU. Disable it for a portable build. The C++ side is compiled by the `cc`
//! crate with the system C++ compiler.

#![warn(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(unsafe_op_in_unsafe_fn)]

#[doc(hidden)]
pub mod ffi;

#[macro_use]
mod base;
mod bvh;
mod bvh4_cpu;
mod bvh8_cpu;
mod bvh_verbose;
mod double;
mod error;
mod geometry;
mod gpu;
mod instance;
mod mbvh;
mod ray;
mod settings;
mod voxel;

pub use glam;

pub use bvh::{Bvh, BvhNode};
pub use bvh4_cpu::Bvh4Cpu;
pub use bvh8_cpu::Bvh8Cpu;
pub use bvh_verbose::{BvhVerbose, BvhVerboseNode};
pub use double::{Bvh4Double, BvhDouble, BvhNodeEx};
pub use error::{Error, Result};
pub use geometry::{Aabb, CustomGeometry, CustomGeometryEx, Geometry, GeometryEx};
pub use gpu::{Bvh4AmdHw, Bvh4Gpu, Bvh8Cwbvh, BvhGpu, BvhGpuNode};
pub use instance::{Blas, BlasEx, BlasInstance, BlasInstanceEx};
pub use mbvh::{Mbvh4, Mbvh4Node, Mbvh8, Mbvh8Node};
pub use ray::{Intersection, IntersectionEx, Ray, RayEx, DBL_FAR, FAR, MASK_ALL};
pub use settings::{BuildSettings, Layout};
pub use voxel::VoxelSet;

use core::ffi::{c_char, c_void};
use std::sync::Mutex;

/// Version of the tinybvh library compiled into this crate, as `(major, minor, sub)`.
pub fn version() -> (i32, i32, i32) {
    let (mut major, mut minor, mut sub) = (0, 0, 0);
    unsafe { ffi::tbvh_version(&mut major, &mut minor, &mut sub) };
    (major, minor, sub)
}

/// Number of rays in a bundle, for [`Bvh::intersect_bundle`] and [`Bvh4Cpu::intersect_bundle`].
pub fn bundle_rays() -> usize {
    unsafe { ffi::tbvh_bundle_rays() as usize }
}

/// Whether the full SIMD traversal path (AVX2 on x86-64, NEON on AArch64) is compiled in.
pub fn simd_enabled() -> bool {
    unsafe { ffi::tbvh_simd_enabled() }
}

/// Whether a SIMD builder is available, which Bonsai and [`Bvh::build_simd`] require. Without it
/// the scalar reference builders are used.
pub fn simd_builder_available() -> bool {
    unsafe { ffi::tbvh_simd_builder_available() }
}

/// Maximum tree depth supported by the CPU traversal kernels (tinybvh's `TINYBVH_STACK_SIZE`).
pub fn traversal_stack_size() -> u32 {
    unsafe { ffi::tbvh_stack_size() }
}

/// Releases the process-wide build thread pool, joining its worker threads. The pool is created
/// again by the next threaded build; calling this is optional.
pub fn shutdown_thread_pool() {
    unsafe { ffi::tbvh_shutdown_builtin_pool() }
}

type FatalHandler = Box<dyn Fn(&str) + Send + Sync>;
static FATAL_HANDLER: Mutex<Option<FatalHandler>> = Mutex::new(None);

unsafe extern "C" fn fatal_trampoline(message: *const c_char, _user: *mut c_void) {
    let message = geometry::cstr_to_string(message);
    let handler = FATAL_HANDLER.lock().unwrap_or_else(|e| e.into_inner());
    match handler.as_ref() {
        Some(handler) => handler(&message),
        None => eprintln!("tinybvh: {message}"),
    }
    // tinybvh would call exit(1) if we returned; abort instead so a debugger or crash handler
    // sees the real location.
    std::process::abort();
}

/// Installs a reporter for tinybvh's fatal errors, which are raised for invalid input or corrupt
/// trees that this crate could not rule out beforehand. The handler receives the message; when it
/// returns, the process aborts. Panicking inside the handler is not supported and also aborts.
///
/// By default the message is printed to standard error before aborting.
pub fn set_fatal_error_handler<F>(handler: F)
where
    F: Fn(&str) + Send + Sync + 'static,
{
    *FATAL_HANDLER.lock().unwrap_or_else(|e| e.into_inner()) = Some(Box::new(handler));
    install_fatal_trampoline();
}

/// Removes a handler installed with [`set_fatal_error_handler`].
pub fn clear_fatal_error_handler() {
    *FATAL_HANDLER.lock().unwrap_or_else(|e| e.into_inner()) = None;
    install_fatal_trampoline();
}

pub(crate) fn install_fatal_trampoline() {
    use std::sync::Once;
    static INSTALLED: Once = Once::new();
    INSTALLED.call_once(|| unsafe {
        ffi::tbvh_set_fatal_error_handler(Some(fatal_trampoline), core::ptr::null_mut());
    });
}
