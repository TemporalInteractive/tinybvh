use std::env;
use std::path::PathBuf;

fn main() {
    // Compile and link to the c++ binaries
    #[cfg(feature = "simd")]
    cc::Build::new()
        .file("ffi/wrapper.cpp")
        .flag("-mavx")
        .flag("-Ofast")
        //.flag("-O3")
        //.flag("-ffast-math")
        //.flag_if_supported("/arch:AVX")
        //.flag_if_supported("/O2")
        //.flag_if_supported("/Oi")
        //.flag_if_supported("/Ot")
        .compiler("C:\\mingw64\\bin\\gcc.exe")
        //.compiler("C:\\Program Files\\LLVM\\bin\\clang.exe")
        .compile("tinybvh");
    #[cfg(not(feature = "simd"))]
    cc::Build::new()
        .file("ffi/wrapper.cpp")
        .flag("-D TINYBVH_NO_SIMD")
        .compile("tinybvh");

    println!("cargo:rustc-link-search={}", env::var("OUT_DIR").unwrap());
    println!("cargo:rustc-link-lib=tinybvh");

    //#[cfg(feature = "simd")]
    //std::env::set_var("BINDGEN_EXTRA_CLANG_ARGS", "-mavx -O3");
    #[cfg(not(feature = "simd"))]
    std::env::set_var("BINDGEN_EXTRA_CLANG_ARGS", "-D TINYBVH_NO_SIMD");

    // Generate rust bindings
    let bindings = bindgen::Builder::default()
        .header("ffi/wrapper.hpp")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .blocklist_item("__gnu_cxx::__min")
        .blocklist_item("__gnu_cxx::__max")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
