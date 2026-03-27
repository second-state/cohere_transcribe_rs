//! Safe wrappers around MLX C FFI operations.
//!
//! Every function follows the same pattern:
//!   1. Create an empty output array with `Array::empty()`
//!   2. Call the FFI function with `&mut res.ptr`
//!   3. Return the result
//!
//! Shapes use `i32`, not `i64` — MLX's C API uses 32-bit dimension sizes.

use std::ffi::CString;

use super::array::Array;
use super::ffi::{self, mlx_dtype};
use super::stream::default_stream;

// ---------------------------------------------------------------------------
// Linear algebra
// ---------------------------------------------------------------------------

/// Matrix multiplication: (…, M, K) × (…, K, N) → (…, M, N).
pub fn matmul(a: &Array, b: &Array) -> Array {
    let mut res = Array::empty();
    unsafe { ffi::mlx_matmul(&mut res.ptr, a.ptr, b.ptr, default_stream()) };
    res
}

/// Linear layer: x @ w.T + b.
/// x: (B, T, in), w: (out, in), b: (out,) → (B, T, out)
pub fn linear(x: &Array, w: &Array, b: &Array) -> Array {
    let wt = transpose(w, &[1, 0]);
    let y = matmul(x, &wt);
    add(&y, b)
}

// ---------------------------------------------------------------------------
// Shape manipulation
// ---------------------------------------------------------------------------

pub fn reshape(a: &Array, shape: &[i32]) -> Array {
    let mut res = Array::empty();
    unsafe {
        ffi::mlx_reshape(
            &mut res.ptr,
            a.ptr,
            shape.as_ptr(),
            shape.len(),
            default_stream(),
        )
    };
    res
}

pub fn transpose(a: &Array, axes: &[i32]) -> Array {
    let mut res = Array::empty();
    unsafe {
        ffi::mlx_transpose_axes(
            &mut res.ptr,
            a.ptr,
            axes.as_ptr(),
            axes.len(),
            default_stream(),
        )
    };
    res
}

/// Swap the last two dimensions — shorthand used everywhere in attention.
pub fn transpose_last2(a: &Array) -> Array {
    let ndim = a.ndim() as i32;
    let axes: Vec<i32> = (0..ndim - 2).chain([ndim - 1, ndim - 2]).collect();
    transpose(a, &axes)
}

pub fn expand_dims(a: &Array, axes: &[i32]) -> Array {
    // The mlx-c API has mlx_expand_dims for a single axis and
    // mlx_expand_dims_axes for multiple axes.
    if axes.len() == 1 {
        let mut res = Array::empty();
        unsafe { ffi::mlx_expand_dims(&mut res.ptr, a.ptr, axes[0], default_stream()) };
        res
    } else {
        let mut res = Array::empty();
        unsafe {
            ffi::mlx_expand_dims_axes(
                &mut res.ptr,
                a.ptr,
                axes.as_ptr(),
                axes.len(),
                default_stream(),
            )
        };
        res
    }
}

pub fn squeeze(a: &Array, axes: &[i32]) -> Array {
    let mut res = Array::empty();
    unsafe {
        ffi::mlx_squeeze_axes(
            &mut res.ptr,
            a.ptr,
            axes.as_ptr(),
            axes.len(),
            default_stream(),
        )
    };
    res
}

pub fn flatten(a: &Array, start: i32, end: i32) -> Array {
    let mut res = Array::empty();
    unsafe { ffi::mlx_flatten(&mut res.ptr, a.ptr, start, end, default_stream()) };
    res
}

// ---------------------------------------------------------------------------
// Concatenation
// ---------------------------------------------------------------------------

pub fn cat(arrays: &[&Array], axis: i32) -> Array {
    let vec = unsafe { ffi::mlx_vector_array_new() };
    for a in arrays {
        unsafe { ffi::mlx_vector_array_append_value(vec, a.ptr) };
    }
    let mut res = Array::empty();
    unsafe { ffi::mlx_concatenate_axis(&mut res.ptr, vec, axis, default_stream()) };
    unsafe { ffi::mlx_vector_array_free(vec) };
    res
}

// ---------------------------------------------------------------------------
// Arithmetic
// ---------------------------------------------------------------------------

pub fn add(a: &Array, b: &Array) -> Array {
    let mut res = Array::empty();
    unsafe { ffi::mlx_add(&mut res.ptr, a.ptr, b.ptr, default_stream()) };
    res
}

pub fn sub(a: &Array, b: &Array) -> Array {
    let mut res = Array::empty();
    unsafe { ffi::mlx_subtract(&mut res.ptr, a.ptr, b.ptr, default_stream()) };
    res
}

pub fn mul(a: &Array, b: &Array) -> Array {
    let mut res = Array::empty();
    unsafe { ffi::mlx_multiply(&mut res.ptr, a.ptr, b.ptr, default_stream()) };
    res
}

/// Scale by a scalar — creates a temporary scalar Array.
pub fn scale(a: &Array, s: f32) -> Array {
    let scalar = unsafe { Array::from_ptr(ffi::mlx_array_new_float(s)) };
    mul(a, &scalar)
}

pub fn neg(a: &Array) -> Array {
    let mut res = Array::empty();
    unsafe { ffi::mlx_negative(&mut res.ptr, a.ptr, default_stream()) };
    res
}

// ---------------------------------------------------------------------------
// Activations
// ---------------------------------------------------------------------------

/// ReLU activation: max(0, x).
pub fn relu(a: &Array) -> Array {
    let zero = unsafe { Array::from_ptr(ffi::mlx_array_new_float(0.0)) };
    let mut res = Array::empty();
    unsafe { ffi::mlx_maximum(&mut res.ptr, a.ptr, zero.ptr, default_stream()) };
    res
}

/// SiLU (Swish) activation: x * sigmoid(x).
pub fn silu(a: &Array) -> Array {
    let sig = sigmoid(a);
    mul(a, &sig)
}

/// Sigmoid activation via mlx_sigmoid.
pub fn sigmoid(a: &Array) -> Array {
    let mut res = Array::empty();
    unsafe { ffi::mlx_sigmoid(&mut res.ptr, a.ptr, default_stream()) };
    res
}

pub fn softmax(a: &Array, axis: i32) -> Array {
    let axes = [axis];
    let mut res = Array::empty();
    unsafe {
        ffi::mlx_softmax_axes(
            &mut res.ptr,
            a.ptr,
            axes.as_ptr(),
            1,
            true, // precise = true for numerical stability
            default_stream(),
        )
    };
    res
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

pub fn argmax(a: &Array, axis: i32, keepdims: bool) -> Array {
    let mut res = Array::empty();
    unsafe { ffi::mlx_argmax_axis(&mut res.ptr, a.ptr, axis, keepdims, default_stream()) };
    res
}

pub fn mean(a: &Array, axes: &[i32], keepdims: bool) -> Array {
    let mut res = Array::empty();
    unsafe {
        ffi::mlx_mean_axes(
            &mut res.ptr,
            a.ptr,
            axes.as_ptr(),
            axes.len(),
            keepdims,
            default_stream(),
        )
    };
    res
}

pub fn sum(a: &Array, axes: &[i32], keepdims: bool) -> Array {
    let mut res = Array::empty();
    unsafe {
        ffi::mlx_sum_axes(
            &mut res.ptr,
            a.ptr,
            axes.as_ptr(),
            axes.len(),
            keepdims,
            default_stream(),
        )
    };
    res
}

// ---------------------------------------------------------------------------
// Normalisation
// ---------------------------------------------------------------------------

/// Fused layer norm using the fast kernel in mlx.core.fast.
pub fn layer_norm(x: &Array, weight: &Array, bias: &Array, eps: f32) -> Array {
    let mut res = Array::empty();
    unsafe {
        ffi::mlx_fast_layer_norm(
            &mut res.ptr,
            x.ptr,
            weight.ptr,
            bias.ptr,
            eps,
            default_stream(),
        )
    };
    res
}

/// Scaled dot-product attention using MLX's fused fast kernel.
/// q/k/v: (B, H, T, d_k).  mask: optional additive mask.
pub fn scaled_dot_product_attention(
    q: &Array,
    k: &Array,
    v: &Array,
    scale: f32,
    mask: Option<&Array>,
) -> Array {
    let mask_mode = if mask.is_some() {
        CString::new("additive").unwrap()
    } else {
        CString::new("").unwrap()
    };
    let mask_ptr = mask.map_or(std::ptr::null_mut(), |m| m.ptr);
    let sinks_ptr: ffi::mlx_array = std::ptr::null_mut();
    let mut res = Array::empty();
    unsafe {
        ffi::mlx_fast_scaled_dot_product_attention(
            &mut res.ptr,
            q.ptr,
            k.ptr,
            v.ptr,
            scale,
            mask_mode.as_ptr(),
            mask_ptr,
            sinks_ptr,
            default_stream(),
        )
    };
    res
}

// ---------------------------------------------------------------------------
// Indexing
// ---------------------------------------------------------------------------

/// Gather rows from `arr` at integer `indices` along the given axis.
/// Equivalent to arr[indices] — used for token/positional embeddings.
pub fn take(arr: &Array, indices: &Array, axis: i32) -> Array {
    let mut res = Array::empty();
    unsafe { ffi::mlx_take_axis(&mut res.ptr, arr.ptr, indices.ptr, axis, default_stream()) };
    res
}

// ---------------------------------------------------------------------------
// Creation
// ---------------------------------------------------------------------------

pub fn zeros(shape: &[i32]) -> Array {
    let mut res = Array::empty();
    unsafe {
        ffi::mlx_zeros(
            &mut res.ptr,
            shape.as_ptr(),
            shape.len(),
            mlx_dtype::MLX_FLOAT32,
            default_stream(),
        )
    };
    res
}

pub fn arange_f32(start: f32, stop: f32, step: f32) -> Array {
    let mut res = Array::empty();
    unsafe {
        ffi::mlx_arange(
            &mut res.ptr,
            start as f64,
            stop as f64,
            step as f64,
            mlx_dtype::MLX_FLOAT32,
            default_stream(),
        )
    };
    res
}

// ---------------------------------------------------------------------------
// Slicing (used directly by encoder.rs)
// ---------------------------------------------------------------------------

/// Slice an array along all dimensions with given start/stop indices and
/// unit strides.
pub fn slice(x: &Array, starts: &[i32], stops: &[i32]) -> Array {
    let n = starts.len();
    let strides = vec![1i32; n];
    let mut res = Array::empty();
    unsafe {
        ffi::mlx_slice(
            &mut res.ptr,
            x.ptr,
            starts.as_ptr(),
            n,
            stops.as_ptr(),
            n,
            strides.as_ptr(),
            n,
            default_stream(),
        )
    };
    res
}

// ---------------------------------------------------------------------------
// Convolutions
// ---------------------------------------------------------------------------

/// 2-D convolution.
/// input: (N, H, W, C_in) — MLX channels-last
/// weight: (C_out, kH, kW, C_in/groups) — transposed from PyTorch OIHW
/// Returns (N, H', W', C_out)
pub fn conv2d(
    input: &Array,
    weight: &Array,
    stride_h: i32,
    stride_w: i32,
    pad_h: i32,
    pad_w: i32,
    groups: i32,
) -> Array {
    let mut res = Array::empty();
    unsafe {
        ffi::mlx_conv2d(
            &mut res.ptr,
            input.ptr,
            weight.ptr,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            1, // dilation_h
            1, // dilation_w
            groups,
            default_stream(),
        )
    };
    res
}

/// 1-D convolution.
/// input: (N, L, C_in) — MLX channels-last
/// weight: (C_out, kW, C_in/groups) — transposed from PyTorch OIW
/// Returns (N, L', C_out)
pub fn conv1d(input: &Array, weight: &Array, stride: i32, padding: i32, groups: i32) -> Array {
    let mut res = Array::empty();
    unsafe {
        ffi::mlx_conv1d(
            &mut res.ptr,
            input.ptr,
            weight.ptr,
            stride,
            padding,
            1, // dilation
            groups,
            default_stream(),
        )
    };
    res
}
