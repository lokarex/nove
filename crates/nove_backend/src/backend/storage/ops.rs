//! Elementwise, shape and reduction operation forwarding.

use super::super::Backend;
use super::super::validation;
use super::BackendStorage;
use super::dispatch::{backend_binary_methods, backend_unary_methods, dispatch, dispatch_binary};
use super::ensure_compatible_devices;
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
    broadcast_pow,
);

impl BackendStorage {
    pub fn broadcast_matmul(&self, rhs: &Self) -> Result<Self, BackendError> {
        ensure_compatible_devices("broadcast_matmul", &[self, rhs])?;
        validation::validate_matmul(&self.shape()?, &rhs.shape()?)?;
        dispatch_binary!(self, rhs, broadcast_matmul)
    }

    pub fn embedding(&self, indexes: &Self) -> Result<Self, BackendError> {
        ensure_compatible_devices("embedding", &[self, indexes])?;
        validation::validate_embedding(&self.shape()?, &indexes.shape()?, indexes.dtype()?)?;
        dispatch_binary!(self, indexes, embedding)
    }

    pub fn reshape(&self, shape: &Shape) -> Result<Self, BackendError> {
        validation::validate_reshape(&self.shape()?, shape)?;
        dispatch!(self, reshape, shape)
    }

    pub fn broadcast_as(&self, shape: &Shape) -> Result<Self, BackendError> {
        validation::validate_broadcast(&self.shape()?, shape)?;
        dispatch!(self, broadcast_as, shape)
    }

    pub fn flatten_from(&self, dim: usize) -> Result<Self, BackendError> {
        validation::validate_flatten(&self.shape()?, Some(dim), None, "flatten_from")?;
        dispatch!(self, flatten_from, dim)
    }

    pub fn flatten_to(&self, dim: usize) -> Result<Self, BackendError> {
        validation::validate_flatten(&self.shape()?, None, Some(dim), "flatten_to")?;
        dispatch!(self, flatten_to, dim)
    }

    pub fn flatten(&self, start: usize, end: usize) -> Result<Self, BackendError> {
        validation::validate_flatten(&self.shape()?, Some(start), Some(end), "flatten")?;
        dispatch!(self, flatten, start, end)
    }

    pub fn squeeze(&self, dim: usize) -> Result<Self, BackendError> {
        validation::validate_axis(&self.shape()?, dim, "squeeze")?;
        dispatch!(self, squeeze, dim)
    }

    pub fn unsqueeze(&self, dim: usize) -> Result<Self, BackendError> {
        validation::validate_unsqueeze(&self.shape()?, dim)?;
        dispatch!(self, unsqueeze, dim)
    }

    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Self, BackendError> {
        validation::validate_axis(&self.shape()?, dim0, "transpose")?;
        validation::validate_axis(&self.shape()?, dim1, "transpose")?;
        dispatch!(self, transpose, dim0, dim1)
    }

    pub fn contiguous(&self) -> Result<Self, BackendError> {
        dispatch!(self, contiguous)
    }

    pub fn permute(&self, dims: &[usize]) -> Result<Self, BackendError> {
        validation::validate_permutation(&self.shape()?, dims)?;
        dispatch!(self, permute, dims)
    }

    pub fn narrow(&self, dim: usize, start: usize, length: usize) -> Result<Self, BackendError> {
        validation::validate_narrow(&self.shape()?, dim, start, length)?;
        dispatch!(self, narrow, dim, start, length)
    }

    pub fn affine(&self, weight: f64, bias: f64) -> Result<Self, BackendError> {
        dispatch!(self, affine, weight, bias)
    }

    pub fn clamp(&self, min: f64, max: f64) -> Result<Self, BackendError> {
        validation::validate_clamp(min, max)?;
        dispatch!(self, clamp, min, max)
    }

    pub fn powf(&self, exponent: f64) -> Result<Self, BackendError> {
        dispatch!(self, powf, exponent)
    }

    pub fn sum(&self, dim: usize) -> Result<Self, BackendError> {
        validation::validate_reduction(&self.shape()?, dim, false, "sum")?;
        dispatch!(self, sum, dim)
    }

    pub fn sum_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        validation::validate_reduction(&self.shape()?, dim, true, "sum_keepdim")?;
        dispatch!(self, sum_keepdim, dim)
    }

    pub fn max(&self, dim: usize) -> Result<Self, BackendError> {
        validation::validate_reduction(&self.shape()?, dim, false, "max")?;
        dispatch!(self, max, dim)
    }

    pub fn max_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        validation::validate_reduction(&self.shape()?, dim, true, "max_keepdim")?;
        dispatch!(self, max_keepdim, dim)
    }

    pub fn min(&self, dim: usize) -> Result<Self, BackendError> {
        validation::validate_reduction(&self.shape()?, dim, false, "min")?;
        dispatch!(self, min, dim)
    }

    pub fn min_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        validation::validate_reduction(&self.shape()?, dim, true, "min_keepdim")?;
        dispatch!(self, min_keepdim, dim)
    }

    pub fn mean(&self, dim: usize) -> Result<Self, BackendError> {
        validation::validate_reduction(&self.shape()?, dim, false, "mean")?;
        dispatch!(self, mean, dim)
    }

    pub fn mean_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        validation::validate_reduction(&self.shape()?, dim, true, "mean_keepdim")?;
        dispatch!(self, mean_keepdim, dim)
    }

    pub fn var(&self, dim: usize) -> Result<Self, BackendError> {
        validation::validate_reduction(&self.shape()?, dim, false, "var")?;
        dispatch!(self, var, dim)
    }

    pub fn var_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        validation::validate_reduction(&self.shape()?, dim, true, "var_keepdim")?;
        dispatch!(self, var_keepdim, dim)
    }

    pub fn argmax(&self, dim: usize) -> Result<Self, BackendError> {
        validation::validate_reduction(&self.shape()?, dim, false, "argmax")?;
        dispatch!(self, argmax, dim)
    }

    pub fn argmax_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        validation::validate_reduction(&self.shape()?, dim, true, "argmax_keepdim")?;
        dispatch!(self, argmax_keepdim, dim)
    }

    pub fn argmin(&self, dim: usize) -> Result<Self, BackendError> {
        validation::validate_reduction(&self.shape()?, dim, false, "argmin")?;
        dispatch!(self, argmin, dim)
    }

    pub fn argmin_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        validation::validate_reduction(&self.shape()?, dim, true, "argmin_keepdim")?;
        dispatch!(self, argmin_keepdim, dim)
    }
}
