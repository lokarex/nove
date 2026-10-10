//! Convolution, transposed convolution and pooling forwarding.

use super::super::Backend;
use super::BackendStorage;
use crate::BackendError;
#[cfg(all(feature = "candle", feature = "native"))]
use crate::BackendKind;

impl BackendStorage {
    pub fn conv1d(
        &self,
        kernel: &Self,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self, BackendError> {
        match (self, kernel) {
            #[cfg(feature = "candle")]
            (Self::Candle(lhs), Self::Candle(rhs)) => Ok(Self::Candle(
                lhs.conv1d(rhs, padding, stride, dilation, groups)?,
            )),
            #[cfg(feature = "native")]
            (Self::Native(lhs), Self::Native(rhs)) => Ok(Self::Native(
                lhs.conv1d(rhs, padding, stride, dilation, groups)?,
            )),
            #[cfg(all(feature = "candle", feature = "native"))]
            (Self::Candle(_), other) => Err(BackendError::BackendMismatch {
                expected: BackendKind::Candle,
                found: other.backend_kind().unwrap_or(BackendKind::Candle),
            }),
            #[cfg(all(feature = "candle", feature = "native"))]
            (Self::Native(_), other) => Err(BackendError::BackendMismatch {
                expected: BackendKind::Native,
                found: other.backend_kind().unwrap_or(BackendKind::Native),
            }),
        }
    }

    pub fn conv2d(
        &self,
        kernel: &Self,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self, BackendError> {
        match (self, kernel) {
            #[cfg(feature = "candle")]
            (Self::Candle(lhs), Self::Candle(rhs)) => Ok(Self::Candle(
                lhs.conv2d(rhs, padding, stride, dilation, groups)?,
            )),
            #[cfg(feature = "native")]
            (Self::Native(lhs), Self::Native(rhs)) => Ok(Self::Native(
                lhs.conv2d(rhs, padding, stride, dilation, groups)?,
            )),
            #[cfg(all(feature = "candle", feature = "native"))]
            (Self::Candle(_), other) => Err(BackendError::BackendMismatch {
                expected: BackendKind::Candle,
                found: other.backend_kind().unwrap_or(BackendKind::Candle),
            }),
            #[cfg(all(feature = "candle", feature = "native"))]
            (Self::Native(_), other) => Err(BackendError::BackendMismatch {
                expected: BackendKind::Native,
                found: other.backend_kind().unwrap_or(BackendKind::Native),
            }),
        }
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
        match (self, kernel) {
            #[cfg(feature = "candle")]
            (Self::Candle(lhs), Self::Candle(rhs)) => Ok(Self::Candle(lhs.conv_transpose1d(
                rhs,
                padding,
                output_padding,
                stride,
                dilation,
                groups,
            )?)),
            #[cfg(feature = "native")]
            (Self::Native(lhs), Self::Native(rhs)) => Ok(Self::Native(lhs.conv_transpose1d(
                rhs,
                padding,
                output_padding,
                stride,
                dilation,
                groups,
            )?)),
            #[cfg(all(feature = "candle", feature = "native"))]
            (Self::Candle(_), other) => Err(BackendError::BackendMismatch {
                expected: BackendKind::Candle,
                found: other.backend_kind().unwrap_or(BackendKind::Candle),
            }),
            #[cfg(all(feature = "candle", feature = "native"))]
            (Self::Native(_), other) => Err(BackendError::BackendMismatch {
                expected: BackendKind::Native,
                found: other.backend_kind().unwrap_or(BackendKind::Native),
            }),
        }
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
        match (self, kernel) {
            #[cfg(feature = "candle")]
            (Self::Candle(lhs), Self::Candle(rhs)) => Ok(Self::Candle(lhs.conv_transpose2d(
                rhs,
                padding,
                output_padding,
                stride,
                dilation,
                groups,
            )?)),
            #[cfg(feature = "native")]
            (Self::Native(lhs), Self::Native(rhs)) => Ok(Self::Native(lhs.conv_transpose2d(
                rhs,
                padding,
                output_padding,
                stride,
                dilation,
                groups,
            )?)),
            #[cfg(all(feature = "candle", feature = "native"))]
            (Self::Candle(_), other) => Err(BackendError::BackendMismatch {
                expected: BackendKind::Candle,
                found: other.backend_kind().unwrap_or(BackendKind::Candle),
            }),
            #[cfg(all(feature = "candle", feature = "native"))]
            (Self::Native(_), other) => Err(BackendError::BackendMismatch {
                expected: BackendKind::Native,
                found: other.backend_kind().unwrap_or(BackendKind::Native),
            }),
        }
    }

    pub fn max_pool2d_with_stride(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self, BackendError> {
        match self {
            #[cfg(feature = "candle")]
            Self::Candle(storage) => Ok(Self::Candle(
                storage.max_pool2d_with_stride(kernel_size, stride)?,
            )),
            #[cfg(feature = "native")]
            Self::Native(storage) => Ok(Self::Native(
                storage.max_pool2d_with_stride(kernel_size, stride)?,
            )),
        }
    }

    pub fn avg_pool2d_with_stride(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self, BackendError> {
        match self {
            #[cfg(feature = "candle")]
            Self::Candle(storage) => Ok(Self::Candle(
                storage.avg_pool2d_with_stride(kernel_size, stride)?,
            )),
            #[cfg(feature = "native")]
            Self::Native(storage) => Ok(Self::Native(
                storage.avg_pool2d_with_stride(kernel_size, stride)?,
            )),
        }
    }
}
