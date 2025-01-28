use std::env;
use std::path::PathBuf;

fn main() {
    // Compile and link to the c++ binaries
    #[cfg(feature = "simd")]
    cc::Build::new().file("ffi/wrapper.cpp").compile("tinybvh");
    #[cfg(not(feature = "simd"))]
    cc::Build::new()
        .file("ffi/wrapper.cpp")
        .flag("-D TINYBVH_NO_SIMD")
        .compile("tinybvh");

    println!("cargo:rustc-link-search={}", env::var("OUT_DIR").unwrap());
    println!("cargo:rustc-link-lib=tinybvh");

    #[cfg(feature = "simd")]
    std::env::set_var("BINDGEN_EXTRA_CLANG_ARGS", "-mavx2");
    #[cfg(not(feature = "simd"))]
    std::env::set_var("BINDGEN_EXTRA_CLANG_ARGS", "-D TINYBVH_NO_SIMD");

    // Generate rust bindings
    let bindings = bindgen::Builder::default()
        .header("ffi/wrapper.hpp")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
