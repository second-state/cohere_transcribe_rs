//! MLX Conformer encoder — mirrors src/encoder.rs using Array + ops::*.
//!
//! Key layout differences from the tch backend:
//!   - MLX convolutions are channels-last: input (N,H,W,C), weight (O,kH,kW,I).
//!     PyTorch/safetensors weights are OIHW → we transpose them at load time.
//!   - Conv1d: input (N,L,C), weight (O,kW,I).
//!
//! All other ops (matmul, layer_norm, silu, softmax, …) match the tch version.

use anyhow::{Context, Result};

use super::array::Array;
use super::ops;
use super::weights::MlxWeights;
use crate::config::ModelConfig;

// ---------------------------------------------------------------------------
// Weight transposition helpers (PyTorch layout → MLX layout)
// ---------------------------------------------------------------------------

/// PyTorch Conv2d weight: (O, I/g, kH, kW) → MLX: (O, kH, kW, I/g)
fn pt_conv2d_to_mlx(w: Array) -> Array {
    ops::transpose(&w, &[0, 2, 3, 1])
}

/// PyTorch Conv1d weight: (O, I/g, kW) → MLX: (O, kW, I/g)
fn pt_conv1d_to_mlx(w: Array) -> Array {
    ops::transpose(&w, &[0, 2, 1])
}

// ---------------------------------------------------------------------------
// ConvSubsampling
//   Three stride-2 Conv2d passes with ReLU, then a linear projection.
//   PyTorch input convention:  (N, C, H, W) = (1, 1, T, n_mels)
//   MLX input convention:      (N, H, W, C) = (1, T, n_mels, 1)
// ---------------------------------------------------------------------------
struct ConvSubsampling {
    c0_w: Array, // (256, 3, 3, 1)  — MLX layout after transpose
    c0_b: Array,
    c2_w: Array, // (256, 3, 3, 1)  — depthwise groups=256
    c2_b: Array,
    c3_w: Array, // (256, 1, 1, 256) — pointwise
    c3_b: Array,
    c5_w: Array,
    c5_b: Array,
    c6_w: Array,
    c6_b: Array,
    out_w: Array, // linear: (d_model, feat)
    out_b: Array,
    conv_channels: i32,
}

impl ConvSubsampling {
    fn load(weights: &MlxWeights, prefix: &str) -> Result<Self> {
        let get = |n: &str| -> Result<Array> {
            Ok(weights.get(&format!("{}{}", prefix, n))?.shallow_clone())
        };
        Ok(Self {
            c0_w: pt_conv2d_to_mlx(get("conv.0.weight")?),
            c0_b: get("conv.0.bias")?,
            c2_w: pt_conv2d_to_mlx(get("conv.2.weight")?),
            c2_b: get("conv.2.bias")?,
            c3_w: pt_conv2d_to_mlx(get("conv.3.weight")?),
            c3_b: get("conv.3.bias")?,
            c5_w: pt_conv2d_to_mlx(get("conv.5.weight")?),
            c5_b: get("conv.5.bias")?,
            c6_w: pt_conv2d_to_mlx(get("conv.6.weight")?),
            c6_b: get("conv.6.bias")?,
            out_w: get("out.weight")?,
            out_b: get("out.bias")?,
            conv_channels: 256,
        })
    }

    /// x: (1, n_mels, T) → (1, T', d_model), returns T'
    fn forward(&self, x: &Array) -> (Array, i32) {
        // (1, n_mels, T) → (1, T, n_mels) → (1, T, n_mels, 1)
        let x = ops::transpose(x, &[0, 2, 1]);
        let x = ops::expand_dims(&x, &[3]);

        // Conv0: (1, T, n_mels, 1) → (1, T/2, n_mels/2, 256)
        let x = ops::conv2d(&x, &self.c0_w, 2, 2, 1, 1, 1);
        let x = add_bias_nhwc(&x, &self.c0_b);
        let x = ops::relu(&x);

        // Conv2 (depthwise) + Conv3 (pointwise)
        let x = ops::conv2d(&x, &self.c2_w, 2, 2, 1, 1, self.conv_channels);
        let x = add_bias_nhwc(&x, &self.c2_b);
        let x = ops::conv2d(&x, &self.c3_w, 1, 1, 0, 0, 1);
        let x = add_bias_nhwc(&x, &self.c3_b);
        let x = ops::relu(&x);

        // Conv5 (depthwise) + Conv6 (pointwise)
        let x = ops::conv2d(&x, &self.c5_w, 2, 2, 1, 1, self.conv_channels);
        let x = add_bias_nhwc(&x, &self.c5_b);
        let x = ops::conv2d(&x, &self.c6_w, 1, 1, 0, 0, 1);
        let x = add_bias_nhwc(&x, &self.c6_b);
        let x = ops::relu(&x);

        // x: (1, T', n_mels/8, 256) → (1, T', 256 * n_mels/8)
        let t_prime = x.dim(1);
        let feat = x.dim(2) * x.dim(3);
        let x = ops::reshape(&x, &[1, t_prime, feat]);

        // Linear projection: (1, T', feat) → (1, T', d_model)
        let out = ops::linear(&x, &self.out_w, &self.out_b);
        (out, t_prime)
    }
}

/// Add a (C,) bias to an NHWC array by broadcasting over N, H, W.
fn add_bias_nhwc(x: &Array, bias: &Array) -> Array {
    // bias: (C,) → (1, 1, 1, C) via reshape
    let c = bias.dim(0);
    let b = ops::reshape(bias, &[1, 1, 1, c]);
    ops::add(x, &b)
}

// ---------------------------------------------------------------------------
// Relative Positional Encoding (same formula as tch backend)
// ---------------------------------------------------------------------------
fn rel_positional_encoding(length: usize, d_model: usize) -> Array {
    let n_pos = 2 * length - 1;
    let mut pe = vec![0.0f32; n_pos * d_model];
    for i in 0..n_pos {
        let pos = (length as i32 - 1 - i as i32) as f32;
        for k in (0..d_model).step_by(2) {
            let div = ((k as f32) * -(10000.0f32.ln()) / d_model as f32).exp();
            pe[i * d_model + k] = (pos * div).sin();
            if k + 1 < d_model {
                pe[i * d_model + k + 1] = (pos * div).cos();
            }
        }
    }
    Array::from_data_f32(&pe, &[1, n_pos as i32, d_model as i32])
}

// ---------------------------------------------------------------------------
// ConformerFeedForward  (Linear → SiLU → Linear)
// ---------------------------------------------------------------------------
struct FeedForward {
    l1_w: Array,
    l1_b: Array,
    l2_w: Array,
    l2_b: Array,
}

impl FeedForward {
    fn load(weights: &MlxWeights, prefix: &str) -> Result<Self> {
        let get = |n: &str| -> Result<Array> {
            Ok(weights.get(&format!("{}{}", prefix, n))?.shallow_clone())
        };
        Ok(Self {
            l1_w: get("linear1.weight")?,
            l1_b: get("linear1.bias")?,
            l2_w: get("linear2.weight")?,
            l2_b: get("linear2.bias")?,
        })
    }

    fn forward(&self, x: &Array) -> Array {
        let h = ops::silu(&ops::linear(x, &self.l1_w, &self.l1_b));
        ops::linear(&h, &self.l2_w, &self.l2_b)
    }
}

// ---------------------------------------------------------------------------
// ConformerConvolution
//   pointwise_conv1 → GLU → depthwise_conv → BatchNorm → SiLU → pointwise_conv2
//
//   MLX Conv1d: input (N,L,C), weight (O,kW,I) — NHWC analog for 1D.
//   PyTorch Conv1d weights (O,I/g,kW) must be transposed → (O,kW,I/g).
// ---------------------------------------------------------------------------
struct ConformerConv {
    pw1_w: Array, // (2*d_model, 1, d_model) after MLX transpose
    pw1_b: Array,
    dw_w: Array, // (d_model, kW, 1) after MLX transpose
    dw_b: Array,
    // BatchNorm parameters (eval mode: normalize by running stats then affine)
    bn_w: Array,
    bn_b: Array,
    bn_rm: Array,
    bn_rv: Array,
    pw2_w: Array, // (d_model, 1, d_model) after MLX transpose
    pw2_b: Array,
    d_model: i32,
}

impl ConformerConv {
    fn load(weights: &MlxWeights, prefix: &str, d_model: i32) -> Result<Self> {
        let get = |n: &str| -> Result<Array> {
            Ok(weights.get(&format!("{}{}", prefix, n))?.shallow_clone())
        };
        Ok(Self {
            pw1_w: pt_conv1d_to_mlx(get("pointwise_conv1.weight")?),
            pw1_b: get("pointwise_conv1.bias")?,
            dw_w: pt_conv1d_to_mlx(get("depthwise_conv.weight")?),
            dw_b: get("depthwise_conv.bias")?,
            bn_w: get("batch_norm.weight")?,
            bn_b: get("batch_norm.bias")?,
            bn_rm: get("batch_norm.running_mean")?,
            bn_rv: get("batch_norm.running_var")?,
            pw2_w: pt_conv1d_to_mlx(get("pointwise_conv2.weight")?),
            pw2_b: get("pointwise_conv2.bias")?,
            d_model,
        })
    }

    fn forward(&self, x: &Array) -> Array {
        // x: (B, T, d_model) — already NLC (channels-last) for MLX conv1d
        // Pointwise conv1: (B, T, 2*d_model)
        let x = ops::conv1d(&x, &self.pw1_w, 1, 0, 1);
        let b1 = ops::reshape(&self.pw1_b, &[1, 1, self.pw1_b.dim(0)]);
        let x = ops::add(&x, &b1);

        // GLU: split along last dim, gate with sigmoid
        let (a, gate) = split_last(&x, self.d_model);
        let x = ops::mul(&a, &ops_sigmoid(&gate));

        // Depthwise conv1d: kernel size inferred from weight shape (O, kW, 1)
        let kw = self.dw_w.dim(1);
        let pad = (kw - 1) / 2;
        let x = ops::conv1d(&x, &self.dw_w, 1, pad, self.d_model);
        let b2 = ops::reshape(&self.dw_b, &[1, 1, self.dw_b.dim(0)]);
        let x = ops::add(&x, &b2);

        // BatchNorm eval mode: (x - running_mean) / sqrt(running_var + eps) * w + b
        // x: (B, T, C) — apply stats along last dim
        let x = batch_norm_nlc(&x, &self.bn_w, &self.bn_b, &self.bn_rm, &self.bn_rv);

        // SiLU activation
        let x = ops::silu(&x);

        // Pointwise conv2: (B, T, d_model)
        let x = ops::conv1d(&x, &self.pw2_w, 1, 0, 1);
        let b3 = ops::reshape(&self.pw2_b, &[1, 1, self.pw2_b.dim(0)]);
        ops::add(&x, &b3)
    }
}

fn ops_sigmoid(x: &Array) -> Array {
    ops::sigmoid(x)
}

/// Split last dimension at `split_at`, returning (left, right).
fn split_last(x: &Array, split_at: i32) -> (Array, Array) {
    let ndim = x.ndim() as i32;
    let last = ndim - 1;
    let total = x.dim(last);

    // Build slice bounds for left half: all dims full, last dim [0, split_at)
    let n = ndim as usize;
    let mut starts = vec![0i32; n];
    let mut stops: Vec<i32> = (0..ndim).map(|d| x.dim(d)).collect();

    stops[last as usize] = split_at;
    let left = mlx_slice(x, &starts, &stops);

    starts[last as usize] = split_at;
    stops[last as usize] = total;
    let right = mlx_slice(x, &starts, &stops);

    (left, right)
}

fn mlx_slice(x: &Array, starts: &[i32], stops: &[i32]) -> Array {
    ops::slice(x, starts, stops)
}

/// BatchNorm in eval mode for NLC layout (x: B,T,C).
fn batch_norm_nlc(
    x: &Array,
    weight: &Array,
    bias: &Array,
    running_mean: &Array,
    running_var: &Array,
) -> Array {
    let eps = 1e-5f32;
    let c = weight.dim(0);

    // Broadcast 1-D stats to (1, 1, C)
    let rm = ops::reshape(running_mean, &[1, 1, c]);
    let rv = ops::reshape(running_var, &[1, 1, c]);
    let w = ops::reshape(weight, &[1, 1, c]);
    let b = ops::reshape(bias, &[1, 1, c]);

    // (x - mean) / sqrt(var + eps)
    let x_centered = ops::sub(x, &rm);
    // sqrt(var + eps): var + eps then rsqrt
    let var_eps = ops::add(&rv, &Array::from_data_f32(&[eps], &[1, 1, 1]));
    let std = ops_sqrt(&var_eps);
    let x_norm = ops::mul(&x_centered, &ops_recip(&std));

    ops::add(&ops::mul(&x_norm, &w), &b)
}

fn ops_sqrt(x: &Array) -> Array {
    // sqrt via pow(0.5) — not in ops yet, use element-wise via take trick.
    // Simplest: create scalar 0.5 and use mlx_power if available, else approximate.
    // Using: sqrt(x) = exp(0.5 * log(x)) — need log and exp.
    // TODO: expose mlx_sqrt directly in ops.rs.
    // For now: use a Rust-side computation for this single-use stats scalar.
    let data = x.to_vec_f32();
    let sqrt_data: Vec<f32> = data.iter().map(|v| v.sqrt()).collect();
    let shape = x.shape();
    Array::from_data_f32(&sqrt_data, &shape)
}

fn ops_recip(x: &Array) -> Array {
    let data = x.to_vec_f32();
    let recip_data: Vec<f32> = data.iter().map(|v| 1.0 / v).collect();
    let shape = x.shape();
    Array::from_data_f32(&recip_data, &shape)
}

// ---------------------------------------------------------------------------
// RelPositionMultiHeadAttention
// ---------------------------------------------------------------------------
struct RelPosAttn {
    q_w: Array,
    q_b: Array,
    k_w: Array,
    k_b: Array,
    v_w: Array,
    v_b: Array,
    pos_w: Array,
    out_w: Array,
    out_b: Array,
    pos_bias_u: Array, // (n_heads, d_k)
    pos_bias_v: Array,
    n_heads: i32,
    d_k: i32,
    scale: f32,
}

impl RelPosAttn {
    fn load(weights: &MlxWeights, prefix: &str, n_heads: i32, d_model: i32) -> Result<Self> {
        let d_k = d_model / n_heads;
        let get = |n: &str| -> Result<Array> {
            Ok(weights.get(&format!("{}{}", prefix, n))?.shallow_clone())
        };
        Ok(Self {
            q_w: get("linear_q.weight")?,
            q_b: get("linear_q.bias")?,
            k_w: get("linear_k.weight")?,
            k_b: get("linear_k.bias")?,
            v_w: get("linear_v.weight")?,
            v_b: get("linear_v.bias")?,
            pos_w: get("linear_pos.weight")?,
            out_w: get("linear_out.weight")?,
            out_b: get("linear_out.bias")?,
            pos_bias_u: get("pos_bias_u")?,
            pos_bias_v: get("pos_bias_v")?,
            n_heads,
            d_k,
            scale: (d_k as f32).powf(-0.5),
        })
    }

    /// Relative shift: x (B, H, T, 2T-1) → (B, H, T, T)
    fn rel_shift(&self, x: &Array, t: i32) -> Array {
        let b = x.dim(0);
        let h = x.dim(1);
        // Pad one zero column on the left along axis=-1:
        //   (B, H, T, 2T-1) → (B, H, T, 2T)
        let x = pad_left_zero(x);
        // Reshape: (B, H, T, 2T) → (B, H, 2T, T)
        let x = ops::reshape(&x, &[b, h, -1, t]);
        // Take rows [1, T+1) along axis 2: (B, H, T, T)
        // Equivalent to tch narrow(2, 1, t).
        let n = x.ndim() as i32;
        let full_stops: Vec<i32> = (0..n).map(|d| x.dim(d)).collect();
        let mut starts = vec![0i32; n as usize];
        let mut stops = full_stops;
        starts[2] = 1;
        stops[2] = t + 1;
        mlx_slice(&x, &starts, &stops)
    }

    fn forward(&self, x: &Array, pos_emb: &Array) -> Array {
        let b = x.dim(0);
        let t = x.dim(1);

        // Project Q, K, V: (B, T, d_model) → (B, H, T, d_k)
        let reshape_qkv = |z: &Array| -> Array {
            let r = ops::reshape(z, &[b, t, self.n_heads, self.d_k]);
            ops::transpose(&r, &[0, 2, 1, 3])
        };

        let q = reshape_qkv(&ops::linear(x, &self.q_w, &self.q_b));
        let k = reshape_qkv(&ops::linear(x, &self.k_w, &self.k_b));
        let v = reshape_qkv(&ops::linear(x, &self.v_w, &self.v_b));

        // Positional projection: (1, 2T-1, d_model) → (1, H, 2T-1, d_k)
        let n_pos = pos_emb.dim(1);
        // pos_w has no bias — zero bias
        let pos_bias_zero = ops::zeros(&[1]);
        let p = ops::linear(pos_emb, &self.pos_w, &pos_bias_zero);
        let p = ops::reshape(&p, &[1, n_pos, self.n_heads, self.d_k]);
        let p = ops::transpose(&p, &[0, 2, 1, 3]);

        // Add content/position biases
        let u = ops::reshape(&self.pos_bias_u, &[1, self.n_heads, 1, self.d_k]);
        let v_bias = ops::reshape(&self.pos_bias_v, &[1, self.n_heads, 1, self.d_k]);
        let q_u = ops::add(&q, &u);
        let q_v = ops::add(&q, &v_bias);

        // Attention scores
        let p_t = ops::transpose_last2(&p);
        let k_t = ops::transpose_last2(&k);
        let matrix_ac = ops::matmul(&q_u, &k_t);
        let matrix_bd = ops::matmul(&q_v, &p_t);
        let matrix_bd = self.rel_shift(&matrix_bd, t);

        let scores = ops::scale(&ops::add(&matrix_ac, &matrix_bd), self.scale);
        let attn = ops::softmax(&scores, -1);
        let out = ops::matmul(&attn, &v);

        // (B, H, T, d_k) → (B, T, d_model)
        let out = ops::transpose(&out, &[0, 2, 1, 3]);
        let out = ops::reshape(&out, &[b, t, self.n_heads * self.d_k]);
        ops::linear(&out, &self.out_w, &self.out_b)
    }
}

/// Pad one zero column on the left of the last dimension.
fn pad_left_zero(x: &Array) -> Array {
    let ndim = x.ndim() as i32;
    let mut zero_shape: Vec<i32> = x.shape();
    *zero_shape.last_mut().unwrap() = 1;
    let zeros = ops::zeros(&zero_shape);
    ops::cat(&[&zeros, x], ndim - 1)
}

// ---------------------------------------------------------------------------
// ConformerLayer
// ---------------------------------------------------------------------------
struct ConformerLayer {
    norm_ff1: (Array, Array),
    ff1: FeedForward,
    norm_self_att: (Array, Array),
    self_attn: RelPosAttn,
    norm_conv: (Array, Array),
    conv: ConformerConv,
    norm_ff2: (Array, Array),
    ff2: FeedForward,
    norm_out: (Array, Array),
}

impl ConformerLayer {
    fn load(weights: &MlxWeights, prefix: &str, n_heads: i32, d_model: i32) -> Result<Self> {
        let norm = |n: &str| -> Result<(Array, Array)> {
            let key = format!("{}{}", prefix, n);
            let w = weights.get(&format!("{}.weight", key))?.shallow_clone();
            let b = weights.get(&format!("{}.bias", key))?.shallow_clone();
            Ok((w, b))
        };
        Ok(Self {
            norm_ff1: norm("norm_feed_forward1")?,
            ff1: FeedForward::load(weights, &format!("{}feed_forward1.", prefix))?,
            norm_self_att: norm("norm_self_att")?,
            self_attn: RelPosAttn::load(
                weights,
                &format!("{}self_attn.", prefix),
                n_heads,
                d_model,
            )?,
            norm_conv: norm("norm_conv")?,
            conv: ConformerConv::load(weights, &format!("{}conv.", prefix), d_model)?,
            norm_ff2: norm("norm_feed_forward2")?,
            ff2: FeedForward::load(weights, &format!("{}feed_forward2.", prefix))?,
            norm_out: norm("norm_out")?,
        })
    }

    fn forward(&self, x: &Array, pos_emb: &Array) -> Array {
        let (nw1, nb1) = &self.norm_ff1;
        let ff1_out = self.ff1.forward(&ops::layer_norm(x, nw1, nb1, 1e-5));
        let x = ops::add(x, &ops::scale(&ff1_out, 0.5));

        let (nw2, nb2) = &self.norm_self_att;
        let attn_out = self
            .self_attn
            .forward(&ops::layer_norm(&x, nw2, nb2, 1e-5), pos_emb);
        let x = ops::add(&x, &attn_out);

        let (nw3, nb3) = &self.norm_conv;
        let conv_out = self.conv.forward(&ops::layer_norm(&x, nw3, nb3, 1e-5));
        let x = ops::add(&x, &conv_out);

        let (nw4, nb4) = &self.norm_ff2;
        let ff2_out = self.ff2.forward(&ops::layer_norm(&x, nw4, nb4, 1e-5));
        let x = ops::add(&x, &ops::scale(&ff2_out, 0.5));

        let (nw5, nb5) = &self.norm_out;
        ops::layer_norm(&x, nw5, nb5, 1e-5)
    }
}

// ---------------------------------------------------------------------------
// ConformerEncoder (public)
// ---------------------------------------------------------------------------
pub struct ConformerEncoder {
    pre_encode: ConvSubsampling,
    layers: Vec<ConformerLayer>,
    enc_dec_proj_w: Option<Array>,
    enc_dec_proj_b: Option<Array>,
    d_model: i32,
}

impl ConformerEncoder {
    pub fn load(weights: &MlxWeights, cfg: &ModelConfig) -> Result<Self> {
        let enc = &cfg.encoder;
        let d_model = enc.d_model as i32;
        let n_heads = enc.n_heads as i32;

        let pre_encode = ConvSubsampling::load(weights, "encoder.pre_encode.")?;

        let mut layers = Vec::with_capacity(enc.n_layers);
        for i in 0..enc.n_layers {
            let prefix = format!("encoder.layers.{}.", i);
            let layer = ConformerLayer::load(weights, &prefix, n_heads, d_model)
                .with_context(|| format!("Loading ConformerLayer {}", i))?;
            layers.push(layer);
        }

        let enc_dec_proj_w = weights
            .get("encoder_decoder_proj.weight")
            .ok()
            .map(|a| a.shallow_clone());
        let enc_dec_proj_b = weights
            .get("encoder_decoder_proj.bias")
            .ok()
            .map(|a| a.shallow_clone());

        Ok(Self {
            pre_encode,
            layers,
            enc_dec_proj_w,
            enc_dec_proj_b,
            d_model,
        })
    }

    /// x: (1, n_mels, T) → (1, T', dec_hidden)
    pub fn forward(&self, x: &Array) -> Array {
        let (x, t_prime) = self.pre_encode.forward(x);
        let pos_emb = rel_positional_encoding(t_prime as usize, self.d_model as usize);

        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(&x, &pos_emb);
        }

        if let (Some(w), Some(b)) = (&self.enc_dec_proj_w, &self.enc_dec_proj_b) {
            x = ops::linear(&x, w, b);
        }

        x
    }
}
