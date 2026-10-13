//! Convolution, transposed convolution and pooling forwarding.

use super::super::{Backend, validation};
use super::dispatch::{dispatch, dispatch_binary};
use super::{BackendStorage, ensure_compatible_devices};
use crate::BackendError;

impl BackendStorage {
    pub fn conv1d(
        &self,
        kernel: &Self,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self, BackendError> {
        ensure_compatible_devices("conv1d", &[self, kernel])?;
        validation::validate_convolution_basics(
            &self.shape()?,
            &kernel.shape()?,
            1,
            stride,
            dilation,
            groups,
            false,
        )?;
        dispatch_binary!(self, kernel, conv1d, padding, stride, dilation, groups)
    }

    pub fn conv2d(
        &self,
        kernel: &Self,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self, BackendError> {
        ensure_compatible_devices("conv2d", &[self, kernel])?;
        validation::validate_convolution_basics(
            &self.shape()?,
            &kernel.shape()?,
            2,
            stride,
            dilation,
            groups,
            false,
        )?;
        dispatch_binary!(self, kernel, conv2d, padding, stride, dilation, groups)
    }

    pub fn conv_transpose1d(
        &self,
        kernel: &Self,
        padding: usize,
        output_padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self, BackendError> {
        ensure_compatible_devices("conv_transpose1d", &[self, kernel])?;
        validation::validate_convolution_basics(
            &self.shape()?,
            &kernel.shape()?,
            1,
            stride,
            dilation,
            groups,
            true,
        )?;
        dispatch_binary!(
            self,
            kernel,
            conv_transpose1d,
            padding,
            output_padding,
            stride,
            dilation,
            groups
        )
    }

    pub fn conv_transpose2d(
        &self,
        kernel: &Self,
        padding: usize,
        output_padding: (usize, usize),
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self, BackendError> {
        ensure_compatible_devices("conv_transpose2d", &[self, kernel])?;
        validation::validate_convolution_basics(
            &self.shape()?,
            &kernel.shape()?,
            2,
            stride,
            dilation,
            groups,
            true,
        )?;
        dispatch_binary!(
            self,
            kernel,
            conv_transpose2d,
            padding,
            output_padding,
            stride,
            dilation,
            groups
        )
    }

    pub fn max_pool2d_with_stride(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self, BackendError> {
        validation::validate_pool2d(
            &self.shape()?,
            kernel_size,
            stride,
            "max_pool2d_with_stride",
        )?;
        dispatch!(self, max_pool2d_with_stride, kernel_size, stride)
    }

    pub fn avg_pool2d_with_stride(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self, BackendError> {
        validation::validate_pool2d(
            &self.shape()?,
            kernel_size,
            stride,
            "avg_pool2d_with_stride",
        )?;
        dispatch!(self, avg_pool2d_with_stride, kernel_size, stride)
    }
}
