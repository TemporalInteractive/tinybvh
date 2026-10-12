//! Compiles the C shim (`ffi/tinybvh_c.cpp`), which is the only translation unit that includes the
//! tinybvh implementation. No bindgen is involved: `src/ffi.rs` mirrors `ffi/tinybvh_c.h` by hand.

use std::env;

fn main() {
    for path in [
        "ffi/tinybvh_c.cpp",
        "ffi/tinybvh_c.h",
        "ffi/tinybvh/tiny_bvh.h",
        "ffi/tinybvh/tiny_bvh_base.h",
        "ffi/tinybvh/tiny_bvh_x86_float.h",
        "ffi/tinybvh/tiny_bvh_x86_double.h",
        "ffi/tinybvh/tiny_bvh_arm_float.h",
        "ffi/tinybvh/tiny_bvh_arm_double.h",
    ] {
        println!("cargo:rerun-if-changed={path}");
    }

    let simd = env::var_os("CARGO_FEATURE_SIMD").is_some();
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .file("ffi/tinybvh_c.cpp")
        .include("ffi")
        .warnings(false);

    // tinybvh at -O0 is extremely slow; a debug profile of the Rust side still wants a usable BVH.
    if env::var("OPT_LEVEL").map(|o| o == "0").unwrap_or(false) {
        build.opt_level(2);
    }

    let msvc = build.get_compiler().is_like_msvc();
    if msvc {
        build.flag("/EHsc");
    }

    if simd {
        match arch.as_str() {
            "x86_64" => {
                if msvc {
                    build.flag("/arch:AVX2");
                } else {
                    build.flag("-mavx2").flag("-mfma");
                }
            }
            // AArch64 has NEON unconditionally; tinybvh enables it on its own.
            "aarch64" => {}
            _ => {
                println!("cargo:warning=tinybvh: no SIMD kernels for target arch `{arch}`, building the scalar path");
                build.define("TINYBVH_NO_SIMD", None);
            }
        }
    } else {
        build.define("TINYBVH_NO_SIMD", None);
    }

    build.compile("tinybvh_c");
}
