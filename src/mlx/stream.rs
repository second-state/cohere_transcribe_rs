//! MLX stream and device initialisation.
//!
//! MLX uses an execution stream model.  All array operations are enqueued on
//! a stream (CPU or GPU).  We keep a single global GPU stream for the lifetime
//! of the process, initialised once via `init_mlx`.

use std::sync::OnceLock;

use super::ffi::{self, mlx_device, mlx_device_type, mlx_stream};

struct MlxContext {
    stream: mlx_stream,
    #[allow(dead_code)]
    device: mlx_device,
}

// Safety: the stream/device pointers are reference-counted C objects owned by
// the MLX runtime; they are safe to use from any thread after init.
unsafe impl Send for MlxContext {}
unsafe impl Sync for MlxContext {}

static CONTEXT: OnceLock<MlxContext> = OnceLock::new();

/// Initialise the MLX runtime.  Must be called once before any array ops.
///
/// `use_gpu = true` selects the Metal GPU (default).
/// `use_gpu = false` forces CPU execution (useful for debugging).
pub fn init_mlx(use_gpu: bool) {
    CONTEXT.get_or_init(|| {
        let device_type = if use_gpu {
            mlx_device_type::MLX_GPU
        } else {
            mlx_device_type::MLX_CPU
        };

        let device = unsafe { ffi::mlx_device_new_type(device_type, 0) };
        unsafe { ffi::mlx_set_default_device(device) };

        let mut stream: mlx_stream = std::ptr::null_mut();
        unsafe { ffi::mlx_get_default_stream(&mut stream, device) };

        MlxContext { stream, device }
    });
}

/// Return the global default stream.  Panics if `init_mlx` has not been called.
pub fn default_stream() -> mlx_stream {
    CONTEXT
        .get()
        .expect("MLX not initialised — call init_mlx() before using arrays")
        .stream
}

/// Block until all pending operations on the default stream have completed.
pub fn synchronize() {
    if let Some(ctx) = CONTEXT.get() {
        unsafe { ffi::mlx_synchronize(ctx.stream) };
    }
}
