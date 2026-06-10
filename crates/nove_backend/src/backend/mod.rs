//! Backend facade and backend-independent tensor payload types.
//!
//! This module keeps tensor creation and tensor operations independent from any
//! single numerical backend. User input is first normalized into a
//! [`TensorPayload`], then the selected backend turns that payload into
//! backend-owned storage.
//!
//! # Notes
//! [`BackendStorage`] is hidden from public documentation because it is the
//! internal dispatch boundary used by `nove_tensor`.
//!
//! # Examples
//! ```
//! use nove_backend::{
//!     BackendKind, DType, Shape, TensorBuffer, TensorPayload,
//! };
//!
//! let payload = TensorPayload::new(
//!     TensorBuffer::F32(vec![1.0, 2.0, 3.0, 4.0]),
//!     Shape::from_dims(&[2, 2]),
//! )
//! .unwrap();
//!
//! assert_eq!(payload.buffer().dtype(), DType::F32);
//! assert_eq!(payload.shape().dims(), &[2, 2]);
//! ```

#![allow(private_interfaces, dead_code, unused_variables)]

#[cfg(feature = "candle")]
pub mod candle;

#[cfg(feature = "native")]
pub mod native;

use crate::{DType, Device, Shape, device::DeviceKind};
use std::{collections::HashMap, fmt::Display};
use thiserror::Error;

/// Identifies a tensor backend implementation.
///
/// # Examples
/// ```no_run
/// use nove_backend::BackendKind;
///
/// assert_eq!(BackendKind::Candle, BackendKind::Candle);
/// assert_ne!(BackendKind::Candle, BackendKind::Native);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKind {
    /// Candle-backed storage and numerical operations.
    Candle,
    /// Native reference storage and numerical operations.
    Native,
}

/// Errors returned by backend implementations and dispatch helpers.
///
/// # Notes
/// Backend adapters should preserve stable high-level error categories, such as
/// unsupported dtype, unsupported operation, and backend mismatch, instead of
/// exposing backend-specific error types through public tensor APIs.
#[derive(Error, Debug)]
pub enum BackendError {
    /// A backend implementation returned an error message.
    #[error("{backend:?} backend error: {message}")]
    BackendImplementation {
        /// Backend that produced the error.
        backend: BackendKind,
        /// Backend-provided message.
        message: String,
    },

    /// The selected backend is not available in the compiled feature set.
    #[error("Unsupported backend: {0:?}")]
    UnsupportedBackend(BackendKind),

    /// The selected device kind is not supported by the backend.
    #[error("Unsupported device: {0:?}")]
    UnsupportedDevice(DeviceKind),

    /// The selected dtype is not supported by the backend.
    #[error("Unsupported dtype for {backend:?}: {dtype:?}")]
    UnsupportedDType { backend: BackendKind, dtype: DType },

    /// The selected operation is not supported by the backend.
    #[error("Unsupported operation for {backend:?}: {operation}")]
    UnsupportedOperation {
        /// Backend that rejected the operation.
        backend: BackendKind,
        /// Operation name.
        operation: String,
    },

    /// Two tensors or storages belong to different backends.
    #[error("Backend mismatch: expected {expected:?}, found {found:?}")]
    BackendMismatch {
        /// Backend required by the current operation.
        expected: BackendKind,
        /// Backend found on the input storage.
        found: BackendKind,
    },

    /// The operation request is invalid before reaching a backend kernel.
    #[error("Invalid backend operation: {0}")]
    InvalidOperation(String),
}

/// A flattened typed tensor buffer independent of any backend implementation.
///
/// # Notes
/// `TensorBuffer` stores materialized element values in row-major order. It is
/// used by [`TensorPayload`] before backend-specific storage is created.
///
/// # Examples
/// ```
/// use nove_backend::{DType, TensorBuffer};
///
/// let buffer = TensorBuffer::F32(vec![1.0, 2.0, 3.0]);
///
/// assert_eq!(buffer.dtype(), DType::F32);
/// assert_eq!(buffer.len(), 3);
/// assert!(!buffer.is_empty());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum TensorBuffer {
    /// Unsigned 8-bit integer elements.
    U8(Vec<u8>),
    /// Unsigned 32-bit integer elements.
    U32(Vec<u32>),
    /// Signed 64-bit integer elements.
    I64(Vec<i64>),
    /// Single precision floating point elements.
    F32(Vec<f32>),
    /// Double precision floating point elements.
    F64(Vec<f64>),
}

impl TensorBuffer {
    /// Returns the dtype represented by this buffer.
    ///
    /// # Returns
    /// * [`DType`] - The dtype matching this buffer variant.
    ///
    /// # Examples
    /// ```
    /// use nove_backend::{DType, TensorBuffer};
    ///
    /// assert_eq!(TensorBuffer::I64(vec![1, 2]).dtype(), DType::I64);
    /// ```
    pub fn dtype(&self) -> DType {
        match self {
            Self::U8(_) => DType::U8,
            Self::U32(_) => DType::U32,
            Self::I64(_) => DType::I64,
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
        }
    }

    /// Returns the number of elements in this buffer.
    ///
    /// # Returns
    /// * `usize` - The number of stored tensor elements.
    ///
    /// # Examples
    /// ```
    /// use nove_backend::TensorBuffer;
    ///
    /// assert_eq!(TensorBuffer::U8(vec![1, 2, 3]).len(), 3);
    /// ```
    pub fn len(&self) -> usize {
        match self {
            Self::U8(data) => data.len(),
            Self::U32(data) => data.len(),
            Self::I64(data) => data.len(),
            Self::F32(data) => data.len(),
            Self::F64(data) => data.len(),
        }
    }

    /// Returns true when this buffer has no elements.
    ///
    /// # Returns
    /// * `bool` - `true` when the buffer stores no elements.
    ///
    /// # Examples
    /// ```
    /// use nove_backend::TensorBuffer;
    ///
    /// assert!(TensorBuffer::F64(vec![]).is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Backend-independent, materialized tensor payload.
///
/// A tensor payload stores typed element data and shape metadata before a
/// backend turns it into backend-specific storage.
///
/// # Examples
/// ```
/// use nove_backend::{DType, Shape, TensorBuffer, TensorPayload};
///
/// let payload = TensorPayload::new(
///     TensorBuffer::F32(vec![1.0, 2.0, 3.0, 4.0]),
///     Shape::from_dims(&[2, 2]),
/// )
/// .unwrap();
///
/// assert_eq!(payload.buffer().dtype(), DType::F32);
/// assert_eq!(payload.shape().elem_count(), 4);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct TensorPayload {
    buffer: TensorBuffer,
    shape: Shape,
}

impl TensorPayload {
    /// Creates a backend-independent tensor payload.
    ///
    /// # Arguments
    /// * `buffer` - The typed element data in row-major order.
    /// * `shape` - The shape metadata for the tensor.
    ///
    /// # Returns
    /// * `Ok(TensorPayload)` - The payload when the buffer length matches the shape.
    /// * `Err(BackendError)` - The error when the payload is inconsistent.
    ///
    /// # Examples
    /// ```
    /// use nove_backend::{Shape, TensorBuffer, TensorPayload};
    ///
    /// let payload = TensorPayload::new(
    ///     TensorBuffer::F32(vec![1.0, 2.0, 3.0, 4.0]),
    ///     Shape::from_dims(&[2, 2]),
    /// )
    /// .unwrap();
    ///
    /// assert_eq!(payload.shape().dims(), &[2, 2]);
    /// assert!(TensorPayload::new(TensorBuffer::F32(vec![1.0]), Shape::from_dims(&[2])).is_err());
    /// ```
    pub fn new(buffer: TensorBuffer, shape: Shape) -> Result<Self, BackendError> {
        if buffer.len() != shape.elem_count() {
            return Err(BackendError::InvalidOperation(format!(
                "data length {} does not match shape {:?}",
                buffer.len(),
                shape
            )));
        }
        Ok(Self { buffer, shape })
    }

    /// Returns the typed data buffer.
    ///
    /// # Returns
    /// * [`TensorBuffer`] - The materialized element values.
    ///
    /// # Examples
    /// ```
    /// use nove_backend::{DType, Shape, TensorBuffer, TensorPayload};
    ///
    /// let payload = TensorPayload::new(TensorBuffer::U8(vec![1, 2]), Shape::from_dims(&[2])).unwrap();
    ///
    /// assert_eq!(payload.buffer().dtype(), DType::U8);
    /// ```
    pub fn buffer(&self) -> &TensorBuffer {
        &self.buffer
    }

    /// Returns the tensor shape.
    ///
    /// # Returns
    /// * [`Shape`] - The tensor shape carried with the payload.
    ///
    /// # Examples
    /// ```
    /// use nove_backend::{Shape, TensorBuffer, TensorPayload};
    ///
    /// let payload = TensorPayload::new(TensorBuffer::F64(vec![1.0]), Shape::from(())).unwrap();
    ///
    /// assert_eq!(payload.shape().rank(), 0);
    /// ```
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
}

/// Trait implemented by values that can be converted into a tensor payload.
///
/// # Examples
/// ```
/// use nove_backend::{DType, IntoTensorPayload};
///
/// let payload = vec![1.0f32, 2.0, 3.0].into_tensor_payload().unwrap();
///
/// assert_eq!(payload.buffer().dtype(), DType::F32);
/// assert_eq!(payload.shape().dims(), &[3]);
/// ```
pub trait IntoTensorPayload: Sized {
    /// Converts the value into a backend-independent tensor payload.
    fn into_tensor_payload(self) -> Result<TensorPayload, BackendError>;
}

/// Trait implemented by scalar element types supported by tensor conversion APIs.
///
/// # Examples
/// ```
/// use nove_backend::{DType, TensorBuffer, TensorElement};
///
/// assert_eq!(<f32 as TensorElement>::dtype(), DType::F32);
///
/// let buffer = <f32 as TensorElement>::into_buffer(vec![1.0, 2.0]);
/// assert_eq!(buffer, TensorBuffer::F32(vec![1.0, 2.0]));
/// ```
pub trait TensorElement: Sized + Copy + Send + Sync + 'static {
    /// Returns the dtype represented by this scalar element type.
    fn dtype() -> DType;
    /// Converts a vector of scalar values into a typed tensor buffer.
    fn into_buffer(data: Vec<Self>) -> TensorBuffer;
    /// Reads values of this scalar type from a typed tensor buffer.
    fn from_buffer(buffer: &TensorBuffer) -> Result<Vec<Self>, BackendError>;
}

/// Trait implemented by floating point element types supported by random factories.
///
/// # Examples
/// ```
/// use nove_backend::FloatTensorElement;
///
/// assert_eq!(1.5f32.to_f64(), 1.5);
/// assert_eq!(2.0f64.to_f64(), 2.0);
/// ```
pub trait FloatTensorElement: TensorElement {
    /// Converts the value to `f64` for backend-independent random factories.
    fn to_f64(self) -> f64;
}

macro_rules! impl_tensor_element {
    ($ty:ty, $dtype:expr, $variant:ident) => {
        impl TensorElement for $ty {
            fn dtype() -> DType {
                $dtype
            }

            fn into_buffer(data: Vec<Self>) -> TensorBuffer {
                TensorBuffer::$variant(data)
            }

            fn from_buffer(buffer: &TensorBuffer) -> Result<Vec<Self>, BackendError> {
                match buffer {
                    TensorBuffer::$variant(data) => Ok(data.clone()),
                    _ => Err(BackendError::InvalidOperation(format!(
                        "cannot convert {:?} buffer into {}",
                        buffer.dtype(),
                        stringify!($ty)
                    ))),
                }
            }
        }

        impl IntoTensorPayload for $ty {
            fn into_tensor_payload(self) -> Result<TensorPayload, BackendError> {
                TensorPayload::new(
                    <$ty as TensorElement>::into_buffer(vec![self]),
                    Shape::from(()),
                )
            }
        }
    };
}

impl_tensor_element!(u8, DType::U8, U8);
impl_tensor_element!(u32, DType::U32, U32);
impl_tensor_element!(i64, DType::I64, I64);
impl_tensor_element!(f32, DType::F32, F32);
impl_tensor_element!(f64, DType::F64, F64);

impl FloatTensorElement for f32 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl FloatTensorElement for f64 {
    fn to_f64(self) -> f64 {
        self
    }
}

impl<T> IntoTensorPayload for Vec<T>
where
    T: TensorElement,
{
    fn into_tensor_payload(self) -> Result<TensorPayload, BackendError> {
        let shape = Shape::from([self.len()].as_slice());
        TensorPayload::new(T::into_buffer(self), shape)
    }
}

impl<T> IntoTensorPayload for &[T]
where
    T: TensorElement,
{
    fn into_tensor_payload(self) -> Result<TensorPayload, BackendError> {
        let shape = Shape::from([self.len()].as_slice());
        TensorPayload::new(T::into_buffer(self.to_vec()), shape)
    }
}

impl<T, const N: usize> IntoTensorPayload for &[T; N]
where
    T: TensorElement,
{
    fn into_tensor_payload(self) -> Result<TensorPayload, BackendError> {
        self.as_slice().into_tensor_payload()
    }
}

impl<T> IntoTensorPayload for Vec<&[T]>
where
    T: TensorElement,
{
    fn into_tensor_payload(self) -> Result<TensorPayload, BackendError> {
        let rows = self.len();
        let cols = self.first().map(|row| row.len()).unwrap_or(0);
        if self.iter().any(|row| row.len() != cols) {
            return Err(BackendError::InvalidOperation(
                "ragged Vec<&[T]> cannot create a tensor payload".to_string(),
            ));
        }
        let data = self
            .into_iter()
            .flat_map(|row| row.iter().copied())
            .collect::<Vec<_>>();
        TensorPayload::new(T::into_buffer(data), Shape::from_dims(&[rows, cols]))
    }
}

impl<T> IntoTensorPayload for Vec<Vec<T>>
where
    T: TensorElement,
{
    fn into_tensor_payload(self) -> Result<TensorPayload, BackendError> {
        let rows = self.len();
        let cols = self.first().map(|row| row.len()).unwrap_or(0);
        if self.iter().any(|row| row.len() != cols) {
            return Err(BackendError::InvalidOperation(
                "ragged Vec<Vec<T>> cannot create a tensor payload".to_string(),
            ));
        }
        let data = self.into_iter().flatten().collect::<Vec<_>>();
        TensorPayload::new(T::into_buffer(data), Shape::from_dims(&[rows, cols]))
    }
}

impl<T> IntoTensorPayload for Vec<Vec<Vec<T>>>
where
    T: TensorElement,
{
    fn into_tensor_payload(self) -> Result<TensorPayload, BackendError> {
        let d0 = self.len();
        let d1 = self.first().map(|value| value.len()).unwrap_or(0);
        let d2 = self
            .first()
            .and_then(|value| value.first())
            .map(|value| value.len())
            .unwrap_or(0);
        if self
            .iter()
            .any(|value| value.len() != d1 || value.iter().any(|inner| inner.len() != d2))
        {
            return Err(BackendError::InvalidOperation(
                "ragged Vec<Vec<Vec<T>>> cannot create a tensor payload".to_string(),
            ));
        }
        let data = self.into_iter().flatten().flatten().collect::<Vec<_>>();
        TensorPayload::new(T::into_buffer(data), Shape::from_dims(&[d0, d1, d2]))
    }
}

impl<T> IntoTensorPayload for Vec<Vec<Vec<Vec<T>>>>
where
    T: TensorElement,
{
    fn into_tensor_payload(self) -> Result<TensorPayload, BackendError> {
        let d0 = self.len();
        let d1 = self.first().map(|value| value.len()).unwrap_or(0);
        let d2 = self
            .first()
            .and_then(|value| value.first())
            .map(|value| value.len())
            .unwrap_or(0);
        let d3 = self
            .first()
            .and_then(|value| value.first())
            .and_then(|value| value.first())
            .map(|value| value.len())
            .unwrap_or(0);
        if self.iter().any(|value| {
            value.len() != d1
                || value
                    .iter()
                    .any(|inner| inner.len() != d2 || inner.iter().any(|leaf| leaf.len() != d3))
        }) {
            return Err(BackendError::InvalidOperation(
                "ragged Vec<Vec<Vec<Vec<T>>>> cannot create a tensor payload".to_string(),
            ));
        }
        let data = self
            .into_iter()
            .flatten()
            .flatten()
            .flatten()
            .collect::<Vec<_>>();
        TensorPayload::new(T::into_buffer(data), Shape::from_dims(&[d0, d1, d2, d3]))
    }
}

impl<T, const M: usize, const N: usize> IntoTensorPayload for &[[T; N]; M]
where
    T: TensorElement,
{
    fn into_tensor_payload(self) -> Result<TensorPayload, BackendError> {
        let data = self.iter().flatten().copied().collect::<Vec<_>>();
        TensorPayload::new(T::into_buffer(data), Shape::from_dims(&[M, N]))
    }
}

impl<T, const A: usize, const B: usize, const C: usize> IntoTensorPayload for &[[[T; C]; B]; A]
where
    T: TensorElement,
{
    fn into_tensor_payload(self) -> Result<TensorPayload, BackendError> {
        let data = self.iter().flatten().flatten().copied().collect::<Vec<_>>();
        TensorPayload::new(T::into_buffer(data), Shape::from_dims(&[A, B, C]))
    }
}

impl<T, const A: usize, const B: usize, const C: usize, const D: usize> IntoTensorPayload
    for &[[[[T; D]; C]; B]; A]
where
    T: TensorElement,
{
    fn into_tensor_payload(self) -> Result<TensorPayload, BackendError> {
        let data = self
            .iter()
            .flatten()
            .flatten()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        TensorPayload::new(T::into_buffer(data), Shape::from_dims(&[A, B, C, D]))
    }
}

/// Internal trait defining the contract every backend must implement.
///
/// Methods are dispatched through this trait so the compiler guarantees every
/// backend exposes the same set of operations. Adding a new operation requires
/// implementations for both `CandleStorage` and `NativeStorage`.
#[doc(hidden)]
pub(crate) trait Backend: Clone + std::fmt::Debug + Sized {
    // -- properties --
    fn dtype(&self) -> DType;
    fn shape(&self) -> Shape;
    fn to_tensor_buffer(&self) -> Result<TensorBuffer, BackendError>;
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;

    // -- lifecycle --
    fn zero_set(&mut self) -> Result<(), BackendError>;

    // -- factories --
    fn from_payload(payload: &TensorPayload, device: &Device) -> Result<Self, BackendError>;
    fn zeros(shape: &Shape, dtype: DType, device: &Device) -> Result<Self, BackendError>;
    fn ones(shape: &Shape, dtype: DType, device: &Device) -> Result<Self, BackendError>;
    fn rand(
        dtype: DType,
        low: f64,
        high: f64,
        shape: &Shape,
        device: &Device,
    ) -> Result<Self, BackendError>;
    fn randn(
        dtype: DType,
        mean: f64,
        std: f64,
        shape: &Shape,
        device: &Device,
    ) -> Result<Self, BackendError>;

    // -- collection --
    fn stack(tensors: &[Self], dim: usize) -> Result<Self, BackendError>;
    fn cat(tensors: &[Self], dim: usize) -> Result<Self, BackendError>;

    // -- conversion --
    fn to_dtype(&self, dtype: DType) -> Result<Self, BackendError>;
    fn to_device(&self, _device: &Device) -> Result<Self, BackendError> {
        Ok(self.clone())
    }

    // -- unary (17) --
    fn zeros_like(&self) -> Result<Self, BackendError>;
    fn ones_like(&self) -> Result<Self, BackendError>;
    fn relu(&self) -> Result<Self, BackendError>;
    fn silu(&self) -> Result<Self, BackendError>;
    fn gelu(&self) -> Result<Self, BackendError>;
    fn tanh(&self) -> Result<Self, BackendError>;
    fn exp(&self) -> Result<Self, BackendError>;
    fn log(&self) -> Result<Self, BackendError>;
    fn sqrt(&self) -> Result<Self, BackendError>;
    fn recip(&self) -> Result<Self, BackendError>;
    fn abs(&self) -> Result<Self, BackendError>;
    fn neg(&self) -> Result<Self, BackendError>;
    fn sum_all(&self) -> Result<Self, BackendError>;
    fn max_all(&self) -> Result<Self, BackendError>;
    fn min_all(&self) -> Result<Self, BackendError>;
    fn mean_all(&self) -> Result<Self, BackendError>;
    fn flatten_all(&self) -> Result<Self, BackendError>;

    // -- binary (13) --
    fn broadcast_add(&self, rhs: &Self) -> Result<Self, BackendError>;
    fn broadcast_mul(&self, rhs: &Self) -> Result<Self, BackendError>;
    fn broadcast_div(&self, rhs: &Self) -> Result<Self, BackendError>;
    fn broadcast_sub(&self, rhs: &Self) -> Result<Self, BackendError>;
    fn broadcast_eq(&self, rhs: &Self) -> Result<Self, BackendError>;
    fn broadcast_ne(&self, rhs: &Self) -> Result<Self, BackendError>;
    fn broadcast_gt(&self, rhs: &Self) -> Result<Self, BackendError>;
    fn broadcast_lt(&self, rhs: &Self) -> Result<Self, BackendError>;
    fn broadcast_ge(&self, rhs: &Self) -> Result<Self, BackendError>;
    fn broadcast_le(&self, rhs: &Self) -> Result<Self, BackendError>;
    fn broadcast_matmul(&self, rhs: &Self) -> Result<Self, BackendError>;
    fn broadcast_pow(&self, rhs: &Self) -> Result<Self, BackendError>;
    fn embedding(&self, rhs: &Self) -> Result<Self, BackendError>;

    // -- shape ops (11) --
    fn reshape(&self, shape: &Shape) -> Result<Self, BackendError>;
    fn broadcast_as(&self, shape: &Shape) -> Result<Self, BackendError>;
    fn flatten_from(&self, dim: usize) -> Result<Self, BackendError>;
    fn flatten_to(&self, dim: usize) -> Result<Self, BackendError>;
    fn flatten(&self, start: usize, end: usize) -> Result<Self, BackendError>;
    fn squeeze(&self, dim: usize) -> Result<Self, BackendError>;
    fn unsqueeze(&self, dim: usize) -> Result<Self, BackendError>;
    fn transpose(&self, dim0: usize, dim1: usize) -> Result<Self, BackendError>;
    fn contiguous(&self) -> Result<Self, BackendError> {
        Ok(self.clone())
    }
    fn permute(&self, dims: &[usize]) -> Result<Self, BackendError>;
    fn narrow(&self, dim: usize, start: usize, length: usize) -> Result<Self, BackendError>;

    // -- math (3) --
    fn affine(&self, weight: f64, bias: f64) -> Result<Self, BackendError>;
    fn clamp(&self, min: f64, max: f64) -> Result<Self, BackendError>;
    fn powf(&self, exponent: f64) -> Result<Self, BackendError>;

    // -- reduction (10) --
    fn sum(&self, dim: usize) -> Result<Self, BackendError>;
    fn sum_keepdim(&self, dim: usize) -> Result<Self, BackendError>;
    fn max(&self, dim: usize) -> Result<Self, BackendError>;
    fn max_keepdim(&self, dim: usize) -> Result<Self, BackendError>;
    fn min(&self, dim: usize) -> Result<Self, BackendError>;
    fn min_keepdim(&self, dim: usize) -> Result<Self, BackendError>;
    fn mean(&self, dim: usize) -> Result<Self, BackendError>;
    fn mean_keepdim(&self, dim: usize) -> Result<Self, BackendError>;
    fn var(&self, dim: usize) -> Result<Self, BackendError>;
    fn var_keepdim(&self, dim: usize) -> Result<Self, BackendError>;

    // -- arg ops (4) --
    fn argmax(&self, dim: usize) -> Result<Self, BackendError>;
    fn argmax_keepdim(&self, dim: usize) -> Result<Self, BackendError>;
    fn argmin(&self, dim: usize) -> Result<Self, BackendError>;
    fn argmin_keepdim(&self, dim: usize) -> Result<Self, BackendError>;

    // -- convolution (2) --
    fn conv1d(
        &self,
        kernel: &Self,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self, BackendError>;
    fn conv2d(
        &self,
        kernel: &Self,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self, BackendError>;

    // -- pooling (2) --
    fn max_pool2d_with_stride(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self, BackendError>;
    fn avg_pool2d_with_stride(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self, BackendError>;

    // -- indexing (3) --
    fn gather(&self, indexes: &Self, dim: usize) -> Result<Self, BackendError>;
    fn index_select(&self, indexes: &Self, dim: usize) -> Result<Self, BackendError>;
    fn where_cond(
        condition: &Self,
        true_value: &Self,
        false_value: &Self,
    ) -> Result<Self, BackendError>;
}

/// Backend-owned tensor storage.
#[derive(Clone, Debug)]
#[doc(hidden)]
pub enum BackendStorage {
    #[cfg(feature = "candle")]
    Candle(candle::CandleStorage),
    #[cfg(feature = "native")]
    Native(native::NativeStorage),
}

impl BackendStorage {
    pub fn from_data<A>(data: A, device: &Device) -> Result<Self, BackendError>
    where
        A: IntoTensorPayload,
    {
        Self::from_payload(data.into_tensor_payload()?, device)
    }

    pub fn from_payload(payload: TensorPayload, device: &Device) -> Result<Self, BackendError> {
        match device.backend() {
            #[cfg(feature = "candle")]
            BackendKind::Candle => Ok(Self::Candle(candle::CandleStorage::from_payload(
                &payload, device,
            )?)),
            #[cfg(feature = "native")]
            BackendKind::Native => {
                if device.kind() != DeviceKind::Cpu {
                    return Err(BackendError::UnsupportedDevice(device.kind()));
                }
                Ok(Self::Native(native::NativeStorage::from_payload(
                    &payload, device,
                )?))
            }
            #[cfg(not(all(feature = "candle", feature = "native")))]
            backend => Err(BackendError::UnsupportedBackend(backend)),
        }
    }

    pub fn from_vec<D>(data: Vec<D>, shape: &Shape, device: &Device) -> Result<Self, BackendError>
    where
        D: TensorElement,
    {
        Self::from_payload(
            TensorPayload::new(D::into_buffer(data), shape.clone())?,
            device,
        )
    }

    pub fn from_slice<D>(data: &[D], shape: &Shape, device: &Device) -> Result<Self, BackendError>
    where
        D: TensorElement,
    {
        Self::from_vec(data.to_vec(), shape, device)
    }

    pub fn rand<T>(low: T, high: T, shape: &Shape, device: &Device) -> Result<Self, BackendError>
    where
        T: FloatTensorElement,
    {
        match device.backend() {
            #[cfg(feature = "candle")]
            BackendKind::Candle => Ok(Self::Candle(candle::CandleStorage::rand(
                T::dtype(),
                low.to_f64(),
                high.to_f64(),
                shape,
                device,
            )?)),
            #[cfg(feature = "native")]
            BackendKind::Native => Ok(Self::Native(native::NativeStorage::rand(
                T::dtype(),
                low.to_f64(),
                high.to_f64(),
                shape,
                device,
            )?)),
            #[cfg(not(all(feature = "candle", feature = "native")))]
            backend => Err(BackendError::UnsupportedBackend(backend)),
        }
    }

    pub fn randn<T>(mean: T, std: T, shape: &Shape, device: &Device) -> Result<Self, BackendError>
    where
        T: FloatTensorElement,
    {
        match device.backend() {
            #[cfg(feature = "candle")]
            BackendKind::Candle => Ok(Self::Candle(candle::CandleStorage::randn(
                T::dtype(),
                mean.to_f64(),
                std.to_f64(),
                shape,
                device,
            )?)),
            #[cfg(feature = "native")]
            BackendKind::Native => Ok(Self::Native(native::NativeStorage::randn(
                T::dtype(),
                mean.to_f64(),
                std.to_f64(),
                shape,
                device,
            )?)),
            #[cfg(not(all(feature = "candle", feature = "native")))]
            backend => Err(BackendError::UnsupportedBackend(backend)),
        }
    }

    pub fn zeros(shape: &Shape, dtype: DType, device: &Device) -> Result<Self, BackendError> {
        match device.backend() {
            #[cfg(feature = "candle")]
            BackendKind::Candle => Ok(Self::Candle(candle::CandleStorage::zeros(
                shape, dtype, device,
            )?)),
            #[cfg(feature = "native")]
            BackendKind::Native => Ok(Self::Native(native::NativeStorage::zeros(
                shape, dtype, device,
            )?)),
            #[cfg(not(all(feature = "candle", feature = "native")))]
            backend => Err(BackendError::UnsupportedBackend(backend)),
        }
    }

    pub fn ones(shape: &Shape, dtype: DType, device: &Device) -> Result<Self, BackendError> {
        match device.backend() {
            #[cfg(feature = "candle")]
            BackendKind::Candle => Ok(Self::Candle(candle::CandleStorage::ones(
                shape, dtype, device,
            )?)),
            #[cfg(feature = "native")]
            BackendKind::Native => Ok(Self::Native(native::NativeStorage::ones(
                shape, dtype, device,
            )?)),
            #[cfg(not(all(feature = "candle", feature = "native")))]
            backend => Err(BackendError::UnsupportedBackend(backend)),
        }
    }

    pub fn to_vec<S>(&self) -> Result<Vec<S>, BackendError>
    where
        S: TensorElement,
    {
        S::from_buffer(&self.to_tensor_buffer()?)
    }

    pub fn to_scalar<S>(&self) -> Result<S, BackendError>
    where
        S: TensorElement,
    {
        let buffer = self.to_tensor_buffer()?;
        if buffer.len() != 1 {
            return Err(BackendError::InvalidOperation(format!(
                "cannot convert tensor with {} elements to scalar: expected 1 element",
                buffer.len()
            )));
        }
        let values = S::from_buffer(&buffer)?;
        Ok(values[0])
    }

    pub fn to_tensor_buffer(&self) -> Result<TensorBuffer, BackendError> {
        match self {
            #[cfg(feature = "candle")]
            Self::Candle(storage) => storage.to_tensor_buffer(),
            #[cfg(feature = "native")]
            Self::Native(storage) => storage.to_tensor_buffer(),
        }
    }

    pub fn stack(tensors: &[Self], dim: usize) -> Result<Self, BackendError> {
        #[cfg(feature = "candle")]
        {
            if tensors
                .iter()
                .all(|storage| matches!(storage, Self::Candle(_)))
            {
                let tensors = tensors
                    .iter()
                    .map(|storage| match storage {
                        Self::Candle(storage) => Ok(storage.clone()),
                        #[cfg(all(feature = "candle", feature = "native"))]
                        _ => Err(BackendError::BackendMismatch {
                            expected: BackendKind::Candle,
                            found: storage.backend_kind().unwrap_or(BackendKind::Candle),
                        }),
                    })
                    .collect::<Result<Vec<_>, BackendError>>()?;
                return Ok(Self::Candle(candle::CandleStorage::stack(&tensors, dim)?));
            }
        }

        #[cfg(feature = "native")]
        {
            if tensors
                .iter()
                .all(|storage| matches!(storage, Self::Native(_)))
            {
                let tensors = tensors
                    .iter()
                    .map(|storage| match storage {
                        Self::Native(storage) => Ok(storage.clone()),
                        #[cfg(all(feature = "candle", feature = "native"))]
                        _ => Err(BackendError::BackendMismatch {
                            expected: BackendKind::Native,
                            found: storage.backend_kind().unwrap_or(BackendKind::Native),
                        }),
                    })
                    .collect::<Result<Vec<_>, BackendError>>()?;
                return Ok(Self::Native(native::NativeStorage::stack(&tensors, dim)?));
            }
        }

        let _ = (tensors, dim);
        Err(BackendError::InvalidOperation(
            "stack requires all tensors to use the same backend".to_string(),
        ))
    }

    pub fn cat(tensors: &[Self], dim: usize) -> Result<Self, BackendError> {
        #[cfg(feature = "candle")]
        {
            if tensors
                .iter()
                .all(|storage| matches!(storage, Self::Candle(_)))
            {
                let tensors = tensors
                    .iter()
                    .map(|storage| match storage {
                        Self::Candle(storage) => Ok(storage.clone()),
                        #[cfg(all(feature = "candle", feature = "native"))]
                        _ => Err(BackendError::BackendMismatch {
                            expected: BackendKind::Candle,
                            found: storage.backend_kind().unwrap_or(BackendKind::Candle),
                        }),
                    })
                    .collect::<Result<Vec<_>, BackendError>>()?;
                return Ok(Self::Candle(candle::CandleStorage::cat(&tensors, dim)?));
            }
        }

        #[cfg(feature = "native")]
        {
            if tensors
                .iter()
                .all(|storage| matches!(storage, Self::Native(_)))
            {
                let tensors = tensors
                    .iter()
                    .map(|storage| match storage {
                        Self::Native(storage) => Ok(storage.clone()),
                        #[cfg(all(feature = "candle", feature = "native"))]
                        _ => Err(BackendError::BackendMismatch {
                            expected: BackendKind::Native,
                            found: storage.backend_kind().unwrap_or(BackendKind::Native),
                        }),
                    })
                    .collect::<Result<Vec<_>, BackendError>>()?;
                return Ok(Self::Native(native::NativeStorage::cat(&tensors, dim)?));
            }
        }

        let _ = (tensors, dim);
        Err(BackendError::InvalidOperation(
            "cat requires all tensors to use the same backend".to_string(),
        ))
    }

    pub fn backend_kind(&self) -> Result<BackendKind, BackendError> {
        match self {
            #[cfg(feature = "candle")]
            Self::Candle(_) => Ok(BackendKind::Candle),
            #[cfg(feature = "native")]
            Self::Native(_) => Ok(BackendKind::Native),
        }
    }

    pub fn detach(&self) -> Result<Self, BackendError> {
        Ok(self.clone())
    }

    pub fn assign_from(&mut self, other: &Self) -> Result<(), BackendError> {
        *self = other.clone();
        Ok(())
    }

    pub fn dtype(&self) -> Result<DType, BackendError> {
        match self {
            #[cfg(feature = "candle")]
            Self::Candle(storage) => Ok(storage.dtype()),
            #[cfg(feature = "native")]
            Self::Native(storage) => Ok(storage.dtype()),
        }
    }

    pub fn shape(&self) -> Result<Shape, BackendError> {
        match self {
            #[cfg(feature = "candle")]
            Self::Candle(storage) => Ok(storage.shape()),
            #[cfg(feature = "native")]
            Self::Native(storage) => Ok(storage.shape()),
        }
    }

    pub fn zero_set(&mut self) -> Result<(), BackendError> {
        match self {
            #[cfg(feature = "candle")]
            Self::Candle(storage) => storage.zero_set(),
            #[cfg(feature = "native")]
            Self::Native(storage) => storage.zero_set(),
        }
    }

    pub fn fmt_backend(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(feature = "candle")]
            Self::Candle(storage) => storage.fmt(f),
            #[cfg(feature = "native")]
            Self::Native(storage) => storage.fmt(f),
        }
    }
}

macro_rules! dispatch {
    ($self:ident, $method:ident $(,)?) => {
        match $self {
            #[cfg(feature = "candle")]
            Self::Candle(storage) => Ok(Self::Candle(storage.$method()?)),
            #[cfg(feature = "native")]
            Self::Native(storage) => Ok(Self::Native(storage.$method()?)),
        }
    };
    ($self:ident, $method:ident, $($arg:expr),+ $(,)?) => {
        match $self {
            #[cfg(feature = "candle")]
            Self::Candle(storage) => Ok(Self::Candle(storage.$method($($arg),+)?)),
            #[cfg(feature = "native")]
            Self::Native(storage) => Ok(Self::Native(storage.$method($($arg),+)?)),
        }
    };
}

macro_rules! backend_unary_methods {
    ($($method:ident),+ $(,)?) => {
        impl BackendStorage {
            $(
                pub fn $method(&self) -> Result<Self, BackendError> {
                    match self {
                        #[cfg(feature = "candle")]
                        Self::Candle(storage) => Ok(Self::Candle(storage.$method()?)),
                        #[cfg(feature = "native")]
                        Self::Native(storage) => Ok(Self::Native(storage.$method()?)),

                    }
                }
            )+
        }
    };
}

macro_rules! backend_binary_methods {
    ($($method:ident),+ $(,)?) => {
        impl BackendStorage {
            $(
                pub fn $method(&self, rhs: &Self) -> Result<Self, BackendError> {
                    match (self, rhs) {
                        #[cfg(feature = "candle")]
                        (Self::Candle(lhs), Self::Candle(rhs)) => Ok(Self::Candle(lhs.$method(rhs)?)),
                        #[cfg(feature = "native")]
                        (Self::Native(lhs), Self::Native(rhs)) => Ok(Self::Native(lhs.$method(rhs)?)),
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
            )+
        }
    };
}

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
    pub fn to_device(&self, device: &Device) -> Result<Self, BackendError> {
        match (self, device.backend()) {
            #[cfg(feature = "candle")]
            (Self::Candle(storage), BackendKind::Candle) => {
                Ok(Self::Candle(storage.to_device(device)?))
            }
            #[cfg(feature = "native")]
            (Self::Native(storage), BackendKind::Native) if device.kind() == DeviceKind::Cpu => {
                Ok(Self::Native(storage.clone()))
            }
            #[cfg(all(feature = "candle", feature = "native"))]
            (Self::Native(storage), BackendKind::Candle) => {
                let payload = TensorPayload::new(storage.to_tensor_buffer()?, storage.shape())?;
                Ok(Self::Candle(candle::CandleStorage::from_payload(
                    &payload, device,
                )?))
            }
            #[cfg(all(feature = "candle", feature = "native"))]
            (Self::Candle(storage), BackendKind::Native) if device.kind() == DeviceKind::Cpu => {
                let payload = TensorPayload::new(storage.to_tensor_buffer()?, storage.shape())?;
                Ok(Self::Native(native::NativeStorage::from_payload(
                    &payload, device,
                )?))
            }
            (_, backend) => Err(BackendError::UnsupportedBackend(backend)),
        }
    }

    pub fn to_dtype(&self, dtype: DType) -> Result<Self, BackendError> {
        dispatch!(self, to_dtype, dtype)
    }

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

    pub fn gather(&self, indexes: &Self, dim: usize) -> Result<Self, BackendError> {
        match (self, indexes) {
            #[cfg(feature = "candle")]
            (Self::Candle(lhs), Self::Candle(rhs)) => Ok(Self::Candle(lhs.gather(rhs, dim)?)),
            #[cfg(feature = "native")]
            (Self::Native(lhs), Self::Native(rhs)) => Ok(Self::Native(lhs.gather(rhs, dim)?)),
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

    pub fn index_select(&self, indexes: &Self, dim: usize) -> Result<Self, BackendError> {
        match (self, indexes) {
            #[cfg(feature = "candle")]
            (Self::Candle(lhs), Self::Candle(rhs)) => Ok(Self::Candle(lhs.index_select(rhs, dim)?)),
            #[cfg(feature = "native")]
            (Self::Native(lhs), Self::Native(rhs)) => Ok(Self::Native(lhs.index_select(rhs, dim)?)),
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

    pub fn where_cond(
        condition: &Self,
        true_value: &Self,
        false_value: &Self,
    ) -> Result<Self, BackendError> {
        match (condition, true_value, false_value) {
            #[cfg(feature = "candle")]
            (Self::Candle(condition), Self::Candle(true_value), Self::Candle(false_value)) => {
                Ok(Self::Candle(candle::CandleStorage::where_cond(
                    condition,
                    true_value,
                    false_value,
                )?))
            }
            #[cfg(feature = "native")]
            (Self::Native(condition), Self::Native(true_value), Self::Native(false_value)) => {
                Ok(Self::Native(native::NativeStorage::where_cond(
                    condition,
                    true_value,
                    false_value,
                )?))
            }
            #[cfg(all(feature = "candle", feature = "native"))]
            (Self::Candle(_), _, _) => Err(BackendError::BackendMismatch {
                expected: BackendKind::Candle,
                found: BackendKind::Native,
            }),
            #[cfg(all(feature = "candle", feature = "native"))]
            (Self::Native(_), _, _) => Err(BackendError::BackendMismatch {
                expected: BackendKind::Native,
                found: BackendKind::Candle,
            }),
        }
    }
}

impl Display for BackendStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.fmt_backend(f)
    }
}

/// Saves backend storage values to a safetensors file.
///
/// # Notes
/// This helper is part of the backend dispatch boundary used by
/// `nove_tensor::safetensor`. Native CPU safetensors support is intentionally
/// reported as unsupported until that backend implements serialization.
///
/// # Arguments
/// * `file_path` - Destination path for the safetensors file.
/// * `tensors` - Named backend storage values to serialize.
///
/// # Returns
/// * `Ok(())` - The file was written successfully.
/// * `Err(BackendError)` - The backend does not support the operation or failed.
///
/// # Examples
/// ```no_run
/// use std::collections::HashMap;
/// use nove_backend::backend;
///
/// backend::save_safetensors("model.safetensors", HashMap::new()).unwrap();
/// ```
pub fn save_safetensors(
    file_path: &str,
    tensors: HashMap<String, BackendStorage>,
) -> Result<(), BackendError> {
    #[cfg(feature = "candle")]
    {
        candle::save_safetensors(file_path, tensors)
    }
    #[cfg(not(feature = "candle"))]
    {
        Err(BackendError::UnsupportedOperation {
            backend: BackendKind::Candle,
            operation: "save_safetensors".to_string(),
        })
    }
}

/// Loads backend storage values from a safetensors file.
///
/// # Notes
/// Loaded storage is created on the backend selected by `device`.
///
/// # Arguments
/// * `file_path` - Source safetensors file path.
/// * `device` - The target Nove device descriptor.
///
/// # Returns
/// * `Ok(HashMap<String, BackendStorage>)` - Named backend storage values.
/// * `Err(BackendError)` - The backend does not support the operation or failed.
///
/// # Examples
/// ```no_run
/// use nove_backend::{backend, device};
///
/// let device = device::candle::cpu().unwrap();
/// let tensors = backend::load_safetensors("model.safetensors", &device).unwrap();
/// assert!(tensors.is_empty() || !tensors.is_empty());
/// ```
pub fn load_safetensors(
    file_path: &str,
    device: &Device,
) -> Result<HashMap<String, BackendStorage>, BackendError> {
    match device.backend() {
        #[cfg(feature = "candle")]
        BackendKind::Candle => candle::load_safetensors(file_path, device),
        #[cfg(not(feature = "candle"))]
        BackendKind::Candle => Err(BackendError::UnsupportedBackend(BackendKind::Candle)),
        BackendKind::Native => Err(BackendError::UnsupportedOperation {
            backend: BackendKind::Native,
            operation: "load_safetensors".to_string(),
        }),
    }
}
