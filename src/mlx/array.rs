//! Safe RAII wrapper around `mlx_array`.
//!
//! `Array` owns an MLX array handle and frees it on drop.
//! All heavy computation is lazy — MLX only materialises results when
//! `eval()` is called (or data is read).

use std::ffi::c_void;

use super::ffi::{self, mlx_array, mlx_dtype};

// ---------------------------------------------------------------------------
// Array
// ---------------------------------------------------------------------------

/// A reference-counted MLX array.  Cheap to clone (increments ref count).
pub struct Array {
    pub(crate) ptr: mlx_array,
}

// Safety: MLX arrays are ref-counted C objects; they are safe to send between
// threads after the stream they were created on has been synchronized.
unsafe impl Send for Array {}
unsafe impl Sync for Array {}

impl Array {
    /// Wrap an existing raw pointer (takes ownership).
    pub(crate) unsafe fn from_ptr(ptr: mlx_array) -> Self {
        assert!(!ptr.is_null(), "mlx_array pointer is null");
        Self { ptr }
    }

    /// Create an uninitialised (empty) placeholder — used as the output slot
    /// for FFI calls before they write a real value.
    pub(crate) fn empty() -> Self {
        let ptr = unsafe { ffi::mlx_array_new() };
        Self { ptr }
    }

    // -----------------------------------------------------------------------
    // Creation helpers
    // -----------------------------------------------------------------------

    /// Create a 1-D float32 array from a Rust slice.
    pub fn from_slice_f32(data: &[f32]) -> Self {
        let shape = [data.len() as i32];
        let ptr = unsafe {
            ffi::mlx_array_new_data(
                data.as_ptr() as *const c_void,
                shape.as_ptr(),
                1,
                mlx_dtype::MLX_FLOAT32,
            )
        };
        unsafe { Self::from_ptr(ptr) }
    }

    /// Create a float32 array from a flat buffer with an explicit shape.
    pub fn from_data_f32(data: &[f32], shape: &[i32]) -> Self {
        assert_eq!(
            data.len(),
            shape.iter().product::<i32>() as usize,
            "data length does not match shape product"
        );
        let ptr = unsafe {
            ffi::mlx_array_new_data(
                data.as_ptr() as *const c_void,
                shape.as_ptr(),
                shape.len() as i32,
                mlx_dtype::MLX_FLOAT32,
            )
        };
        unsafe { Self::from_ptr(ptr) }
    }

    /// Create a 1-D int32 index array from a Rust slice.
    pub fn from_slice_i32(data: &[i32]) -> Self {
        let shape = [data.len() as i32];
        let ptr = unsafe {
            ffi::mlx_array_new_data(
                data.as_ptr() as *const c_void,
                shape.as_ptr(),
                1,
                mlx_dtype::MLX_INT32,
            )
        };
        unsafe { Self::from_ptr(ptr) }
    }

    // -----------------------------------------------------------------------
    // Shape / metadata
    // -----------------------------------------------------------------------

    pub fn ndim(&self) -> usize {
        unsafe { ffi::mlx_array_ndim(self.ptr) }
    }

    pub fn dim(&self, axis: i32) -> i32 {
        let n = self.ndim();
        let shape_ptr = unsafe { ffi::mlx_array_shape(self.ptr) };
        assert!(!shape_ptr.is_null(), "mlx_array_shape returned null");
        let idx = if axis >= 0 {
            axis as usize
        } else {
            (n as i32 + axis) as usize
        };
        assert!(idx < n, "axis {} out of range for ndim {}", axis, n);
        unsafe { *shape_ptr.add(idx) }
    }

    pub fn size(&self) -> usize {
        unsafe { ffi::mlx_array_size(self.ptr) }
    }

    pub fn shape(&self) -> Vec<i32> {
        let n = self.ndim();
        let shape_ptr = unsafe { ffi::mlx_array_shape(self.ptr) };
        if shape_ptr.is_null() || n == 0 {
            return vec![];
        }
        unsafe { std::slice::from_raw_parts(shape_ptr, n).to_vec() }
    }

    // -----------------------------------------------------------------------
    // Evaluation and data access
    // -----------------------------------------------------------------------

    /// Force materialisation of any pending lazy computation.
    pub fn eval(&self) {
        unsafe { ffi::mlx_array_eval(self.ptr) };
    }

    /// Copy all values to a Rust `Vec<f32>`.  Calls `eval()` first.
    pub fn to_vec_f32(&self) -> Vec<f32> {
        self.eval();
        let n = self.size();
        let ptr = unsafe { ffi::mlx_array_data_float32(self.ptr) };
        assert!(!ptr.is_null(), "mlx_array_data_float32 returned null");
        unsafe { std::slice::from_raw_parts(ptr, n).to_vec() }
    }

    /// Read back a single f32 scalar (0-d or 1-element array).
    pub fn item_f32(&self) -> f32 {
        self.eval();
        let ptr = unsafe { ffi::mlx_array_data_float32(self.ptr) };
        unsafe { *ptr }
    }

    /// Read the index of the maximum element (argmax over flattened array).
    /// Convenience for greedy decoding.
    pub fn argmax_flat(&self) -> i64 {
        let am = super::ops::argmax(self, -1, false);
        am.eval();
        let ptr = unsafe { ffi::mlx_array_data_int32(am.ptr) };
        unsafe { *ptr as i64 }
    }
}

impl Drop for Array {
    fn drop(&mut self) {
        unsafe { ffi::mlx_array_free(self.ptr) };
    }
}

// Arrays cannot be cheaply cloned without incrementing the ref count via the
// C API.  Provide an explicit method instead of implementing Clone to avoid
// accidental copies.
impl Array {
    /// Shallow copy — wraps the same storage.
    /// The caller is responsible for ensuring the original outlives the copy
    /// (or that eval has been called, materialising the data).
    ///
    /// TODO: use mlx_array_retain if/when available in mlx-c.
    pub fn shallow_clone(&self) -> Self {
        // Re-create from data to guarantee independent ownership.
        // This is the safe fallback — the eval round-trip is acceptable for
        // weight tensors that are only cloned once at load time.
        let data = self.to_vec_f32();
        let shape = self.shape();
        Self::from_data_f32(&data, &shape)
    }
}
