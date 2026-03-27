fn main() {
    if std::env::var("CARGO_FEATURE_TCH_BACKEND").is_ok() {
        build_tch();
    }
    if std::env::var("CARGO_FEATURE_MLX").is_ok() {
        build_mlx();
    }
}

/// Embed the libtorch library path as RPATH in the binary so that it runs
/// without LD_LIBRARY_PATH.  The path is taken from the LIBTORCH env var
/// (same variable used by the `tch` build script) — defaults to /opt/libtorch.
///
/// This mirrors the approach in second-state/qwen3_asr_rs: bake the path at
/// build time so the binary is self-contained with respect to library lookup.
fn build_tch() {
    let libtorch = std::env::var("LIBTORCH").unwrap_or_else(|_| "/opt/libtorch".to_string());

    // For release packages the binary must find libtorch relative to itself
    // ($ORIGIN = directory containing the ELF binary), so the user can unzip
    // and run without any environment variables.
    // For dev/CI builds we embed the absolute LIBTORCH path instead.
    let rpath = if std::env::var("RELEASE_RPATH_ORIGIN").is_ok() {
        "$ORIGIN/libtorch/lib".to_string()
    } else {
        format!("{}/lib", libtorch)
    };

    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", rpath);

    println!("cargo:rerun-if-env-changed=LIBTORCH");
    println!("cargo:rerun-if-env-changed=RELEASE_RPATH_ORIGIN");
    println!("cargo:rerun-if-changed=build.rs");
}

/// Build mlx-c from the git submodule via CMake and link statically.
/// This mirrors the approach in second-state/qwen3_asr_rs: the submodule
/// pulls in mlx-c which fetches and builds the MLX C++ library as a
/// dependency, producing static libraries and a compiled Metal shader
/// library (mlx.metallib).
///
/// No Homebrew, no dylibbundler, no DYLD_LIBRARY_PATH — fully self-contained.
fn build_mlx() {
    let mlx_c_dir = std::path::PathBuf::from("mlx-c");
    if !mlx_c_dir.join("CMakeLists.txt").exists() {
        panic!(
            "mlx-c submodule not found. Please run:\n\
             \n\
             git submodule update --init --recursive\n\
             \n\
             to clone the mlx-c dependency."
        );
    }

    // Ensure CMake and Rust agree on the macOS deployment target.
    // Without this, CMake may compile C++ for macOS 15.x while Rust links
    // for macOS 11.0, causing `___isPlatformVersionAtLeast` linker errors.
    let deployment_target = std::env::var("MACOSX_DEPLOYMENT_TARGET")
        .unwrap_or_else(|_| "14.0".to_string());

    // Build mlx-c via CMake (fetches and builds MLX C++ as a dependency)
    let dst = cmake::Config::new(&mlx_c_dir)
        .define("MLX_BUILD_TESTS", "OFF")
        .define("MLX_BUILD_EXAMPLES", "OFF")
        .define("MLX_BUILD_BENCHMARKS", "OFF")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("CMAKE_OSX_DEPLOYMENT_TARGET", &deployment_target)
        .build();

    // Link paths — CMake may output to lib/ or lib64/
    let lib_dir = dst.join("lib");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    let lib64_dir = dst.join("lib64");
    if lib64_dir.exists() {
        println!("cargo:rustc-link-search=native={}", lib64_dir.display());
    }

    // Link mlx-c and mlx static libraries
    println!("cargo:rustc-link-lib=static=mlxc");
    println!("cargo:rustc-link-lib=static=mlx");

    // Apple system frameworks required by MLX
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Accelerate");
    println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");

    // C++ standard library (MLX is C++)
    println!("cargo:rustc-link-lib=c++");

    // MLX C++ code uses @available() checks that reference ___isPlatformVersionAtLeast
    // from the compiler runtime. Rust passes -nodefaultlibs to the linker, so the
    // compiler runtime is not automatically linked. We must explicitly link it.
    // Find the clang resource directory to locate libclang_rt.osx.a.
    let clang_rt = std::process::Command::new("clang")
        .args(["--print-file-name", "libclang_rt.osx.a"])
        .output()
        .expect("failed to run clang --print-file-name");
    let clang_rt_path = String::from_utf8(clang_rt.stdout)
        .expect("non-utf8 clang output")
        .trim()
        .to_string();
    if std::path::Path::new(&clang_rt_path).exists() {
        println!("cargo:rustc-link-arg={}", clang_rt_path);
    }

    // Rerun if mlx-c sources change
    println!("cargo:rerun-if-changed=mlx-c/CMakeLists.txt");
    println!("cargo:rerun-if-changed=build.rs");
}
