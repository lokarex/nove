//! Elementwise, shape and reduction operation forwarding.

use super::super::Backend;
use super::BackendStorage;
use super::dispatch::{backend_binary_methods, backend_unary_methods, dispatch};
#[cfg(all(feature = "candle", feature = "native"))]
use crate::BackendKind;
use crate::{BackendError, Shape};

backend_unary_methods!(
    zeros_like,
    ones_like,
    relu,
    silu,
    gelu,
    tanh,
    exp,
    log,
    sqrt,
    recip,
    abs,
    neg,
    sum_all,
    max_all,
    min_all,
    mean_all,
    flatten_all,
);

backend_binary_methods!(
    broadcast_add,
    broadcast_mul,
    broadcast_div,
    broadcast_sub,
    broadcast_eq,
    broadcast_ne,
    broadcast_gt,
    broadcast_lt,
    broadcast_ge,
    broadcast_le,
    broadcast_matmul,
    broadcast_pow,
    embedding,
);

impl BackendStorage {
    pub fn reshape(&self, shape: &Shape) -> Result<Self, BackendError> {
        dispatch!(self, reshape, shape)
    }

    pub fn broadcast_as(&self, shape: &Shape) -> Result<Self, BackendError> {
        dispatch!(self, broadcast_as, shape)
    }

    pub fn flatten_from(&self, dim: usize) -> Result<Self, BackendError> {
        dispatch!(self, flatten_from, dim)
    }

    pub fn flatten_to(&self, dim: usize) -> Result<Self, BackendError> {
        dispatch!(self, flatten_to, dim)
    }

    pub fn flatten(&self, start: usize, end: usize) -> Result<Self, BackendError> {
        dispatch!(self, flatten, start, end)
    }

    pub fn squeeze(&self, dim: usize) -> Result<Self, BackendError> {
        dispatch!(self, squeeze, dim)
    }

    pub fn unsqueeze(&self, dim: usize) -> Result<Self, BackendError> {
        dispatch!(self, unsqueeze, dim)
    }

    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Self, BackendError> {
        dispatch!(self, transpose, dim0, dim1)
    }

    pub fn contiguous(&self) -> Result<Self, BackendError> {
        dispatch!(self, contiguous)
    }

    pub fn permute(&self, dims: &[usize]) -> Result<Self, BackendError> {
        dispatch!(self, permute, dims)
    }

    pub fn narrow(&self, dim: usize, start: usize, length: usize) -> Result<Self, BackendError> {
        dispatch!(self, narrow, dim, start, length)
    }

    pub fn affine(&self, weight: f64, bias: f64) -> Result<Self, BackendError> {
        dispatch!(self, affine, weight, bias)
    }

    pub fn clamp(&self, min: f64, max: f64) -> Result<Self, BackendError> {
        dispatch!(self, clamp, min, max)
    }

    pub fn powf(&self, exponent: f64) -> Result<Self, BackendError> {
        dispatch!(self, powf, exponent)
    }

    pub fn sum(&self, dim: usize) -> Result<Self, BackendError> {
        dispatch!(self, sum, dim)
    }

    pub fn sum_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        dispatch!(self, sum_keepdim, dim)
    }

    pub fn max(&self, dim: usize) -> Result<Self, BackendError> {
        dispatch!(self, max, dim)
    }

    pub fn max_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        dispatch!(self, max_keepdim, dim)
    }

    pub fn min(&self, dim: usize) -> Result<Self, BackendError> {
        dispatch!(self, min, dim)
    }

    pub fn min_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        dispatch!(self, min_keepdim, dim)
    }

    pub fn mean(&self, dim: usize) -> Result<Self, BackendError> {
        dispatch!(self, mean, dim)
    }

    pub fn mean_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        dispatch!(self, mean_keepdim, dim)
    }

    pub fn var(&self, dim: usize) -> Result<Self, BackendError> {
        dispatch!(self, var, dim)
    }

    pub fn var_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        dispatch!(self, var_keepdim, dim)
    }

    pub fn argmax(&self, dim: usize) -> Result<Self, BackendError> {
        dispatch!(self, argmax, dim)
    }

    pub fn argmax_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        dispatch!(self, argmax_keepdim, dim)
    }

    pub fn argmin(&self, dim: usize) -> Result<Self, BackendError> {
        dispatch!(self, argmin, dim)
    }

    pub fn argmin_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        dispatch!(self, argmin_keepdim, dim)
    }
}
