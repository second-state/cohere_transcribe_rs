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

/// Link Apple MLX C library and required macOS frameworks, and embed the
/// library path as RPATH so the binary runs without DYLD_LIBRARY_PATH.
///
/// Expected layout (configurable via MLX_DIR env var, default /opt/mlx):
///   $MLX_DIR/lib/libmlxc.dylib
///   $MLX_DIR/include/mlx/c/
///
/// Build mlx-c from source:
///   brew install mlx
///   git clone --depth 1 https://github.com/ml-explore/mlx-c /tmp/mlx-c
///   cmake -S /tmp/mlx-c -B /tmp/mlx-c/build \
///     -DCMAKE_BUILD_TYPE=Release \
///     -DCMAKE_PREFIX_PATH=$(brew --prefix mlx) \
///     -DCMAKE_INSTALL_PREFIX=/opt/mlx
///   cmake --build /tmp/mlx-c/build --parallel && sudo cmake --install /tmp/mlx-c/build
fn build_mlx() {
    let mlx_dir = std::env::var("MLX_DIR").unwrap_or_else(|_| "/opt/mlx".to_string());
    let lib_dir = format!("{}/lib", mlx_dir);

    // Library search path (link time)
    println!("cargo:rustc-link-search=native={}", lib_dir);

    // mlx-c: C wrapper around the MLX C++ library
    println!("cargo:rustc-link-lib=dylib=mlxc");

    // Embed RPATH so the binary finds libmlxc.dylib at runtime.
    // For release packages we skip this — dylibbundler rewrites all dylib
    // load commands to @loader_path/lib after the build, so no rpath is needed.
    if std::env::var("RELEASE_RPATH_ORIGIN").is_err() {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir);
    }

    // Apple system frameworks required by MLX
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
    println!("cargo:rustc-link-lib=framework=MetalPerformanceShadersGraph");
    println!("cargo:rustc-link-lib=framework=Accelerate");
    println!("cargo:rustc-link-lib=framework=Foundation");

    // C++ standard library (MLX is C++)
    println!("cargo:rustc-link-lib=c++");

    println!("cargo:rerun-if-env-changed=MLX_DIR");
    println!("cargo:rerun-if-env-changed=RELEASE_RPATH_ORIGIN");
    println!("cargo:rerun-if-changed=build.rs");
}
