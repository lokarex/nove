//! Native CPU reference backend for Nove.
//!
//! This crate owns a small, typed, contiguous CPU storage implementation used
//! by Nove as the first non-Candle backend. It is intentionally simple and
//! deterministic so backend behavior can be compared against Candle-backed
//! storage through `nove_tensor` tests.
//!
//! # Notes
//! Half precision dtypes are not implemented by the native CPU backend v1.
//! Unsupported operations return stable backend errors instead of falling back
//! to another backend.
//!
//! # Examples
//! ```
//! use nove_cpu::{CpuBuffer, CpuDType, CpuStorage};
//!
//! let storage = CpuStorage::ones(&[2, 2], CpuDType::F32).unwrap();
//!
//! assert_eq!(storage.shape(), &[2, 2]);
//! assert_eq!(storage.dtype(), CpuDType::F32);
//! assert_eq!(storage.buffer(), &CpuBuffer::F32(vec![1.0, 1.0, 1.0, 1.0]));
//! ```

use std::fmt::Display;

/// Dtypes supported by the native CPU backend.
///
/// # Examples
/// ```
/// use nove_cpu::CpuDType;
///
/// assert_eq!(CpuDType::F32, CpuDType::F32);
/// assert_ne!(CpuDType::F32, CpuDType::F64);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CpuDType {
    /// Unsigned 8-bit integer.
    U8,
    /// Unsigned 32-bit integer.
    U32,
    /// Signed 64-bit integer.
    I64,
    /// IEEE 754 single precision floating point value.
    F32,
    /// IEEE 754 double precision floating point value.
    F64,
}

/// Errors returned by the native CPU backend.
#[derive(Debug, thiserror::Error)]
pub enum CpuBackendError {
    /// The dtype is not supported by the native CPU backend.
    #[error("unsupported CPU dtype: {0:?}")]
    UnsupportedDType(CpuDType),

    /// The operation is not implemented by the native CPU backend.
    #[error("unsupported CPU backend operation: {0}")]
    UnsupportedOperation(String),

    /// The requested shape is incompatible with the storage values.
    #[error("shape mismatch: expected {expected:?}, found {found:?}")]
    ShapeMismatch {
        /// Expected shape or element count.
        expected: Vec<usize>,
        /// Found shape or element count.
        found: Vec<usize>,
    },

    /// Two buffers have incompatible dtypes.
    #[error("dtype mismatch: expected {expected:?}, found {found:?}")]
    DTypeMismatch { expected: CpuDType, found: CpuDType },

    /// The requested operation is invalid for the provided inputs.
    #[error("invalid CPU backend operation: {0}")]
    InvalidOperation(String),
}

/// Result type returned by the native CPU backend.
pub type CpuResult<T> = Result<T, CpuBackendError>;

/// Typed contiguous buffer owned by the native CPU backend.
///
/// # Examples
/// ```
/// use nove_cpu::{CpuBuffer, CpuDType};
///
/// let buffer = CpuBuffer::I64(vec![1, 2, 3]);
///
/// assert_eq!(buffer.dtype(), CpuDType::I64);
/// assert_eq!(buffer.len(), 3);
/// assert!(!buffer.is_empty());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum CpuBuffer {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PoolKind {
    Max,
    Average,
}

impl CpuBuffer {
    /// Returns the dtype represented by this buffer.
    ///
    /// # Returns
    /// * [`CpuDType`] - The dtype matching this buffer variant.
    ///
    /// # Examples
    /// ```
    /// use nove_cpu::{CpuBuffer, CpuDType};
    ///
    /// assert_eq!(CpuBuffer::U8(vec![1, 2]).dtype(), CpuDType::U8);
    /// ```
    pub fn dtype(&self) -> CpuDType {
        match self {
            Self::U8(_) => CpuDType::U8,
            Self::U32(_) => CpuDType::U32,
            Self::I64(_) => CpuDType::I64,
            Self::F32(_) => CpuDType::F32,
            Self::F64(_) => CpuDType::F64,
        }
    }

    /// Returns the number of elements stored in this buffer.
    ///
    /// # Returns
    /// * `usize` - The number of stored elements.
    ///
    /// # Examples
    /// ```
    /// use nove_cpu::CpuBuffer;
    ///
    /// assert_eq!(CpuBuffer::F64(vec![1.0, 2.0]).len(), 2);
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

    /// Returns true when this buffer contains no elements.
    ///
    /// # Returns
    /// * `bool` - `true` when no values are stored.
    ///
    /// # Examples
    /// ```
    /// use nove_cpu::CpuBuffer;
    ///
    /// assert!(CpuBuffer::F32(vec![]).is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Native CPU tensor storage.
///
/// # Examples
/// ```
/// use nove_cpu::{CpuBuffer, CpuDType, CpuStorage};
///
/// let storage = CpuStorage::from_buffer(
///     CpuBuffer::F32(vec![1.0, 2.0, 3.0, 4.0]),
///     &[2, 2],
/// )
/// .unwrap();
///
/// assert_eq!(storage.shape(), &[2, 2]);
/// assert_eq!(storage.dtype(), CpuDType::F32);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct CpuStorage {
    shape: Vec<usize>,
    buffer: CpuBuffer,
}

impl CpuStorage {
    /// Creates storage from a typed buffer and shape.
    ///
    /// # Arguments
    /// * `buffer` - Typed row-major values.
    /// * `shape` - Shape metadata for the storage.
    ///
    /// # Returns
    /// * `Ok(CpuStorage)` - Storage when element count and shape agree.
    /// * `Err(CpuBackendError)` - The error when the buffer cannot match the shape.
    ///
    /// # Examples
    /// ```
    /// use nove_cpu::{CpuBuffer, CpuStorage};
    ///
    /// let storage = CpuStorage::from_buffer(CpuBuffer::U8(vec![1, 2, 3, 4]), &[2, 2]).unwrap();
    ///
    /// assert_eq!(storage.shape(), &[2, 2]);
    /// assert!(CpuStorage::from_buffer(CpuBuffer::U8(vec![1]), &[2]).is_err());
    /// ```
    pub fn from_buffer(buffer: CpuBuffer, shape: &[usize]) -> CpuResult<Self> {
        let expected = elem_count(shape);
        let found = buffer.len();
        if expected != found {
            return Err(CpuBackendError::ShapeMismatch {
                expected: vec![expected],
                found: vec![found],
            });
        }
        Ok(Self {
            shape: shape.to_vec(),
            buffer,
        })
    }

    /// Creates zero-filled storage.
    ///
    /// # Arguments
    /// * `shape` - Shape metadata for the storage.
    /// * `dtype` - Element dtype to allocate.
    ///
    /// # Returns
    /// * `Ok(CpuStorage)` - Zero-filled storage.
    /// * `Err(CpuBackendError)` - The error when allocation or validation fails.
    ///
    /// # Examples
    /// ```
    /// use nove_cpu::{CpuBuffer, CpuDType, CpuStorage};
    ///
    /// let storage = CpuStorage::zeros(&[3], CpuDType::I64).unwrap();
    ///
    /// assert_eq!(storage.buffer(), &CpuBuffer::I64(vec![0, 0, 0]));
    /// ```
    pub fn zeros(shape: &[usize], dtype: CpuDType) -> CpuResult<Self> {
        let count = elem_count(shape);
        let buffer = match dtype {
            CpuDType::U8 => CpuBuffer::U8(vec![0; count]),
            CpuDType::U32 => CpuBuffer::U32(vec![0; count]),
            CpuDType::I64 => CpuBuffer::I64(vec![0; count]),
            CpuDType::F32 => CpuBuffer::F32(vec![0.0; count]),
            CpuDType::F64 => CpuBuffer::F64(vec![0.0; count]),
        };
        Self::from_buffer(buffer, shape)
    }

    /// Creates one-filled storage.
    ///
    /// # Arguments
    /// * `shape` - Shape metadata for the storage.
    /// * `dtype` - Element dtype to allocate.
    ///
    /// # Returns
    /// * `Ok(CpuStorage)` - One-filled storage.
    /// * `Err(CpuBackendError)` - The error when allocation or validation fails.
    ///
    /// # Examples
    /// ```
    /// use nove_cpu::{CpuBuffer, CpuDType, CpuStorage};
    ///
    /// let storage = CpuStorage::ones(&[2], CpuDType::F32).unwrap();
    ///
    /// assert_eq!(storage.buffer(), &CpuBuffer::F32(vec![1.0, 1.0]));
    /// ```
    pub fn ones(shape: &[usize], dtype: CpuDType) -> CpuResult<Self> {
        let count = elem_count(shape);
        let buffer = match dtype {
            CpuDType::U8 => CpuBuffer::U8(vec![1; count]),
            CpuDType::U32 => CpuBuffer::U32(vec![1; count]),
            CpuDType::I64 => CpuBuffer::I64(vec![1; count]),
            CpuDType::F32 => CpuBuffer::F32(vec![1.0; count]),
            CpuDType::F64 => CpuBuffer::F64(vec![1.0; count]),
        };
        Self::from_buffer(buffer, shape)
    }

    /// Returns the dtype of this storage.
    ///
    /// # Returns
    /// * [`CpuDType`] - The storage dtype.
    ///
    /// # Examples
    /// ```
    /// use nove_cpu::{CpuDType, CpuStorage};
    ///
    /// let storage = CpuStorage::zeros(&[1], CpuDType::U32).unwrap();
    ///
    /// assert_eq!(storage.dtype(), CpuDType::U32);
    /// ```
    pub fn dtype(&self) -> CpuDType {
        self.buffer.dtype()
    }

    /// Returns the shape of this storage.
    ///
    /// # Returns
    /// * `&[usize]` - The storage shape.
    ///
    /// # Examples
    /// ```
    /// use nove_cpu::{CpuDType, CpuStorage};
    ///
    /// let storage = CpuStorage::zeros(&[2, 3], CpuDType::F64).unwrap();
    ///
    /// assert_eq!(storage.shape(), &[2, 3]);
    /// ```
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the underlying typed buffer.
    ///
    /// # Returns
    /// * [`CpuBuffer`] - The typed row-major values.
    ///
    /// # Examples
    /// ```
    /// use nove_cpu::{CpuBuffer, CpuDType, CpuStorage};
    ///
    /// let storage = CpuStorage::ones(&[2], CpuDType::U8).unwrap();
    ///
    /// assert_eq!(storage.buffer(), &CpuBuffer::U8(vec![1, 1]));
    /// ```
    pub fn buffer(&self) -> &CpuBuffer {
        &self.buffer
    }

    /// Sets all values to zero in place.
    pub fn zero_set(&mut self) -> CpuResult<()> {
        self.buffer = match self.dtype() {
            CpuDType::U8 => CpuBuffer::U8(vec![0; self.elem_count()]),
            CpuDType::U32 => CpuBuffer::U32(vec![0; self.elem_count()]),
            CpuDType::I64 => CpuBuffer::I64(vec![0; self.elem_count()]),
            CpuDType::F32 => CpuBuffer::F32(vec![0.0; self.elem_count()]),
            CpuDType::F64 => CpuBuffer::F64(vec![0.0; self.elem_count()]),
        };
        Ok(())
    }

    /// Converts this storage to a new dtype.
    ///
    /// # Arguments
    /// * `dtype` - Target dtype.
    ///
    /// # Returns
    /// * `Ok(CpuStorage)` - A new storage value with the target dtype.
    /// * `Err(CpuBackendError)` - The error when conversion is unsupported.
    ///
    /// # Examples
    /// ```
    /// use nove_cpu::{CpuBuffer, CpuDType, CpuStorage};
    ///
    /// let storage = CpuStorage::ones(&[2], CpuDType::U8).unwrap();
    /// let converted = storage.to_dtype(CpuDType::F32).unwrap();
    ///
    /// assert_eq!(converted.buffer(), &CpuBuffer::F32(vec![1.0, 1.0]));
    /// ```
    pub fn to_dtype(&self, dtype: CpuDType) -> CpuResult<Self> {
        if self.dtype() == dtype {
            return Ok(self.clone());
        }
        let values = self.to_f64_vec()?;
        let buffer = match dtype {
            CpuDType::U8 => CpuBuffer::U8(values.iter().map(|value| *value as u8).collect()),
            CpuDType::U32 => CpuBuffer::U32(values.iter().map(|value| *value as u32).collect()),
            CpuDType::I64 => CpuBuffer::I64(values.iter().map(|value| *value as i64).collect()),
            CpuDType::F32 => CpuBuffer::F32(values.iter().map(|value| *value as f32).collect()),
            CpuDType::F64 => CpuBuffer::F64(values),
        };
        Self::from_buffer(buffer, &self.shape)
    }

    /// Reshapes this storage without changing element order.
    ///
    /// # Arguments
    /// * `shape` - Target shape with the same element count.
    ///
    /// # Returns
    /// * `Ok(CpuStorage)` - A reshaped storage value.
    /// * `Err(CpuBackendError)` - The error when element counts differ.
    ///
    /// # Examples
    /// ```
    /// use nove_cpu::{CpuDType, CpuStorage};
    ///
    /// let storage = CpuStorage::ones(&[4], CpuDType::F32).unwrap();
    /// let reshaped = storage.reshape(&[2, 2]).unwrap();
    ///
    /// assert_eq!(reshaped.shape(), &[2, 2]);
    /// assert_eq!(reshaped.dtype(), CpuDType::F32);
    /// ```
    pub fn reshape(&self, shape: &[usize]) -> CpuResult<Self> {
        if elem_count(shape) != self.elem_count() {
            return Err(CpuBackendError::ShapeMismatch {
                expected: vec![self.elem_count()],
                found: vec![elem_count(shape)],
            });
        }
        Ok(Self {
            shape: shape.to_vec(),
            buffer: self.buffer.clone(),
        })
    }

    /// Broadcasts this storage to the target shape.
    pub fn broadcast_as(&self, shape: &[usize]) -> CpuResult<Self> {
        let src_shape = align_shape(&self.shape, shape.len());
        for (source, target) in src_shape.iter().zip(shape.iter()) {
            if *source != 1 && source != target {
                return Err(CpuBackendError::ShapeMismatch {
                    expected: shape.to_vec(),
                    found: self.shape.clone(),
                });
            }
        }

        let source_strides = strides(&src_shape);
        let source_values = self.to_f64_vec()?;
        let output_count = elem_count(shape);
        let mut values = Vec::with_capacity(output_count);

        for output_index in 0..output_count {
            let coords = unravel_index(output_index, shape);
            let mut source_index = 0;
            for axis in 0..shape.len() {
                let source_coord = if src_shape[axis] == 1 {
                    0
                } else {
                    coords[axis]
                };
                source_index += source_coord * source_strides[axis];
            }
            values.push(source_values[source_index]);
        }

        Self::from_f64_values(values, shape, self.dtype())
    }

    /// Flattens all dimensions.
    pub fn flatten_all(&self) -> CpuResult<Self> {
        self.reshape(&[self.elem_count()])
    }

    /// Flattens from a dimension to the end.
    pub fn flatten_from(&self, dim: usize) -> CpuResult<Self> {
        if dim >= self.shape.len() {
            return Ok(self.clone());
        }
        let mut shape = self.shape[..dim].to_vec();
        shape.push(self.shape[dim..].iter().product());
        self.reshape(&shape)
    }

    /// Flattens from the start through a dimension.
    pub fn flatten_to(&self, dim: usize) -> CpuResult<Self> {
        if dim >= self.shape.len() {
            return self.flatten_all();
        }
        let mut shape = vec![self.shape[..=dim].iter().product()];
        shape.extend_from_slice(&self.shape[dim + 1..]);
        self.reshape(&shape)
    }

    /// Flattens an inclusive dimension range.
    pub fn flatten(&self, start: usize, end: usize) -> CpuResult<Self> {
        if start > end || end >= self.shape.len() {
            return Err(CpuBackendError::InvalidOperation(format!(
                "invalid flatten range {start}..={end} for shape {:?}",
                self.shape
            )));
        }
        let mut shape = self.shape[..start].to_vec();
        shape.push(self.shape[start..=end].iter().product());
        shape.extend_from_slice(&self.shape[end + 1..]);
        self.reshape(&shape)
    }

    /// Removes a dimension of size one.
    pub fn squeeze(&self, dim: usize) -> CpuResult<Self> {
        if self.shape.get(dim) != Some(&1) {
            return Err(CpuBackendError::InvalidOperation(format!(
                "cannot squeeze dimension {dim} from shape {:?}",
                self.shape
            )));
        }
        let mut shape = self.shape.clone();
        shape.remove(dim);
        self.reshape(&shape)
    }

    /// Inserts a dimension of size one.
    pub fn unsqueeze(&self, dim: usize) -> CpuResult<Self> {
        if dim > self.shape.len() {
            return Err(CpuBackendError::InvalidOperation(format!(
                "cannot unsqueeze dimension {dim} for shape {:?}",
                self.shape
            )));
        }
        let mut shape = self.shape.clone();
        shape.insert(dim, 1);
        self.reshape(&shape)
    }

    /// Transposes two dimensions.
    pub fn transpose(&self, dim0: usize, dim1: usize) -> CpuResult<Self> {
        let mut dims = (0..self.shape.len()).collect::<Vec<_>>();
        dims.swap(dim0, dim1);
        self.permute(&dims)
    }

    /// Permutes dimensions.
    pub fn permute(&self, dims: &[usize]) -> CpuResult<Self> {
        if dims.len() != self.shape.len() {
            return Err(CpuBackendError::InvalidOperation(format!(
                "permutation {:?} does not match shape {:?}",
                dims, self.shape
            )));
        }
        let output_shape = dims.iter().map(|dim| self.shape[*dim]).collect::<Vec<_>>();
        let input_values = self.to_f64_vec()?;
        let input_strides = strides(&self.shape);
        let output_count = elem_count(&output_shape);
        let mut output = Vec::with_capacity(output_count);
        for output_index in 0..output_count {
            let output_coords = unravel_index(output_index, &output_shape);
            let mut input_coords = vec![0; self.shape.len()];
            for (output_axis, input_axis) in dims.iter().enumerate() {
                input_coords[*input_axis] = output_coords[output_axis];
            }
            let input_index = input_coords
                .iter()
                .zip(input_strides.iter())
                .map(|(coord, stride)| coord * stride)
                .sum::<usize>();
            output.push(input_values[input_index]);
        }
        Self::from_f64_values(output, &output_shape, self.dtype())
    }

    /// Narrows a dimension.
    pub fn narrow(&self, dim: usize, start: usize, length: usize) -> CpuResult<Self> {
        if dim >= self.shape.len() || start + length > self.shape[dim] {
            return Err(CpuBackendError::InvalidOperation(format!(
                "invalid narrow dim={dim}, start={start}, length={length} for shape {:?}",
                self.shape
            )));
        }
        let mut output_shape = self.shape.clone();
        output_shape[dim] = length;
        let input_values = self.to_f64_vec()?;
        let input_strides = strides(&self.shape);
        let output_count = elem_count(&output_shape);
        let mut output = Vec::with_capacity(output_count);
        for output_index in 0..output_count {
            let mut coords = unravel_index(output_index, &output_shape);
            coords[dim] += start;
            let input_index = coords
                .iter()
                .zip(input_strides.iter())
                .map(|(coord, stride)| coord * stride)
                .sum::<usize>();
            output.push(input_values[input_index]);
        }
        Self::from_f64_values(output, &output_shape, self.dtype())
    }

    /// Returns a zero-filled tensor with the same shape and dtype.
    pub fn zeros_like(&self) -> CpuResult<Self> {
        Self::zeros(&self.shape, self.dtype())
    }

    /// Returns a one-filled tensor with the same shape and dtype.
    pub fn ones_like(&self) -> CpuResult<Self> {
        Self::ones(&self.shape, self.dtype())
    }

    /// Adds two tensors using broadcasting.
    pub fn broadcast_add(&self, rhs: &Self) -> CpuResult<Self> {
        self.binary_op(rhs, "broadcast_add", |lhs, rhs| lhs + rhs)
    }

    /// Subtracts two tensors using broadcasting.
    pub fn broadcast_sub(&self, rhs: &Self) -> CpuResult<Self> {
        self.binary_op(rhs, "broadcast_sub", |lhs, rhs| lhs - rhs)
    }

    /// Multiplies two tensors using broadcasting.
    pub fn broadcast_mul(&self, rhs: &Self) -> CpuResult<Self> {
        self.binary_op(rhs, "broadcast_mul", |lhs, rhs| lhs * rhs)
    }

    /// Divides two tensors using broadcasting.
    pub fn broadcast_div(&self, rhs: &Self) -> CpuResult<Self> {
        self.binary_op(rhs, "broadcast_div", |lhs, rhs| lhs / rhs)
    }

    /// Computes elementwise power using broadcasting.
    pub fn broadcast_pow(&self, rhs: &Self) -> CpuResult<Self> {
        self.binary_op(rhs, "broadcast_pow", |lhs, rhs| lhs.powf(rhs))
    }

    /// Matrix multiplication for rank-2 tensors.
    pub fn broadcast_matmul(&self, rhs: &Self) -> CpuResult<Self> {
        if self.shape.len() != 2 || rhs.shape.len() != 2 {
            return unsupported("broadcast_matmul only supports rank-2 CPU tensors");
        }
        let (m, k) = (self.shape[0], self.shape[1]);
        if rhs.shape[0] != k {
            return Err(CpuBackendError::ShapeMismatch {
                expected: vec![k],
                found: vec![rhs.shape[0]],
            });
        }
        let n = rhs.shape[1];
        let lhs = self.to_f64_vec()?;
        let rhs = rhs.to_f64_vec()?;
        let mut output = vec![0.0; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut value = 0.0;
                for inner in 0..k {
                    value += lhs[row * k + inner] * rhs[inner * n + col];
                }
                output[row * n + col] = value;
            }
        }
        Self::from_f64_values(
            output,
            &[m, n],
            promote_float_dtype(self.dtype(), self.dtype()),
        )
    }

    /// Sums all tensor elements.
    pub fn sum_all(&self) -> CpuResult<Self> {
        let sum = self.to_f64_vec()?.iter().sum::<f64>();
        Self::from_f64_values(vec![sum], &[], self.float_result_dtype())
    }

    /// Sums along one dimension.
    pub fn sum(&self, dim: usize) -> CpuResult<Self> {
        self.reduce(dim, false, |values| values.iter().sum())
    }

    /// Sums along one dimension and keeps the reduced dimension.
    pub fn sum_keepdim(&self, dim: usize) -> CpuResult<Self> {
        self.reduce(dim, true, |values| values.iter().sum())
    }

    /// Computes the mean over all elements.
    pub fn mean_all(&self) -> CpuResult<Self> {
        let values = self.to_f64_vec()?;
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        Self::from_f64_values(vec![mean], &[], self.float_result_dtype())
    }

    /// Computes the mean along one dimension.
    pub fn mean(&self, dim: usize) -> CpuResult<Self> {
        self.reduce(dim, false, |values| {
            values.iter().sum::<f64>() / values.len() as f64
        })
    }

    /// Computes the mean along one dimension and keeps the reduced dimension.
    pub fn mean_keepdim(&self, dim: usize) -> CpuResult<Self> {
        self.reduce(dim, true, |values| {
            values.iter().sum::<f64>() / values.len() as f64
        })
    }

    /// Applies an affine transform.
    pub fn affine(&self, weight: f64, bias: f64) -> CpuResult<Self> {
        self.unary_op("affine", |value| value * weight + bias)
    }

    /// Clamps values.
    pub fn clamp(&self, min: f64, max: f64) -> CpuResult<Self> {
        self.unary_op("clamp", |value| value.clamp(min, max))
    }

    /// Raises values to a scalar exponent.
    pub fn powf(&self, exponent: f64) -> CpuResult<Self> {
        self.unary_op("powf", |value| value.powf(exponent))
    }

    /// Applies ReLU.
    pub fn relu(&self) -> CpuResult<Self> {
        self.unary_op("relu", |value| value.max(0.0))
    }

    /// Applies SiLU.
    pub fn silu(&self) -> CpuResult<Self> {
        self.unary_op("silu", |value| value / (1.0 + (-value).exp()))
    }

    /// Applies GELU using the tanh approximation.
    pub fn gelu(&self) -> CpuResult<Self> {
        self.unary_op("gelu", |value| {
            0.5 * value
                * (1.0 + (0.797_884_560_802_865_4 * (value + 0.044_715 * value.powi(3))).tanh())
        })
    }

    /// Applies tanh.
    pub fn tanh(&self) -> CpuResult<Self> {
        self.unary_op("tanh", f64::tanh)
    }

    /// Applies exp.
    pub fn exp(&self) -> CpuResult<Self> {
        self.unary_op("exp", f64::exp)
    }

    /// Applies natural log.
    pub fn log(&self) -> CpuResult<Self> {
        self.unary_op("log", f64::ln)
    }

    /// Applies square root.
    pub fn sqrt(&self) -> CpuResult<Self> {
        self.unary_op("sqrt", f64::sqrt)
    }

    /// Applies reciprocal.
    pub fn recip(&self) -> CpuResult<Self> {
        self.unary_op("recip", |value| 1.0 / value)
    }

    /// Applies absolute value.
    pub fn abs(&self) -> CpuResult<Self> {
        self.unary_op("abs", f64::abs)
    }

    /// Applies negation.
    pub fn neg(&self) -> CpuResult<Self> {
        self.unary_op("neg", |value| -value)
    }

    /// Compares two tensors for equality using broadcasting.
    pub fn broadcast_eq(&self, rhs: &Self) -> CpuResult<Self> {
        self.binary_compare(rhs, |lhs, rhs| lhs == rhs)
    }

    /// Compares two tensors for inequality using broadcasting.
    pub fn broadcast_ne(&self, rhs: &Self) -> CpuResult<Self> {
        self.binary_compare(rhs, |lhs, rhs| lhs != rhs)
    }

    /// Compares two tensors using greater-than.
    pub fn broadcast_gt(&self, rhs: &Self) -> CpuResult<Self> {
        self.binary_compare(rhs, |lhs, rhs| lhs > rhs)
    }

    /// Compares two tensors using less-than.
    pub fn broadcast_lt(&self, rhs: &Self) -> CpuResult<Self> {
        self.binary_compare(rhs, |lhs, rhs| lhs < rhs)
    }

    /// Compares two tensors using greater-or-equal.
    pub fn broadcast_ge(&self, rhs: &Self) -> CpuResult<Self> {
        self.binary_compare(rhs, |lhs, rhs| lhs >= rhs)
    }

    /// Compares two tensors using less-or-equal.
    pub fn broadcast_le(&self, rhs: &Self) -> CpuResult<Self> {
        self.binary_compare(rhs, |lhs, rhs| lhs <= rhs)
    }

    /// Selects values by condition.
    pub fn where_cond(&self, true_value: &Self, false_value: &Self) -> CpuResult<Self> {
        let shape = broadcast_shape(true_value.shape(), false_value.shape())?;
        let shape = broadcast_shape(&shape, self.shape())?;
        let cond = self.broadcast_as(&shape)?.to_f64_vec()?;
        let true_values = true_value.broadcast_as(&shape)?.to_f64_vec()?;
        let false_values = false_value.broadcast_as(&shape)?.to_f64_vec()?;
        let values = cond
            .iter()
            .zip(true_values.iter())
            .zip(false_values.iter())
            .map(|((condition, true_value), false_value)| {
                if *condition != 0.0 {
                    *true_value
                } else {
                    *false_value
                }
            })
            .collect::<Vec<_>>();
        Self::from_f64_values(values, &shape, true_value.dtype())
    }

    pub fn max_all(&self) -> CpuResult<Self> {
        let value = self
            .to_f64_vec()?
            .into_iter()
            .reduce(f64::max)
            .ok_or_else(|| CpuBackendError::InvalidOperation("empty max".to_string()))?;
        Self::from_f64_values(vec![value], &[], self.float_result_dtype())
    }

    pub fn min_all(&self) -> CpuResult<Self> {
        let value = self
            .to_f64_vec()?
            .into_iter()
            .reduce(f64::min)
            .ok_or_else(|| CpuBackendError::InvalidOperation("empty min".to_string()))?;
        Self::from_f64_values(vec![value], &[], self.float_result_dtype())
    }

    pub fn max(&self, dim: usize) -> CpuResult<Self> {
        self.reduce(dim, false, |values| {
            values.iter().copied().reduce(f64::max).unwrap_or(0.0)
        })
    }

    pub fn max_keepdim(&self, dim: usize) -> CpuResult<Self> {
        self.reduce(dim, true, |values| {
            values.iter().copied().reduce(f64::max).unwrap_or(0.0)
        })
    }

    pub fn min(&self, dim: usize) -> CpuResult<Self> {
        self.reduce(dim, false, |values| {
            values.iter().copied().reduce(f64::min).unwrap_or(0.0)
        })
    }

    pub fn min_keepdim(&self, dim: usize) -> CpuResult<Self> {
        self.reduce(dim, true, |values| {
            values.iter().copied().reduce(f64::min).unwrap_or(0.0)
        })
    }

    pub fn var(&self, dim: usize) -> CpuResult<Self> {
        self.reduce(dim, false, variance)
    }

    pub fn var_keepdim(&self, dim: usize) -> CpuResult<Self> {
        self.reduce(dim, true, variance)
    }

    pub fn argmax(&self, dim: usize) -> CpuResult<Self> {
        self.arg_reduce(dim, false, |lhs, rhs| lhs > rhs)
    }

    pub fn argmax_keepdim(&self, dim: usize) -> CpuResult<Self> {
        self.arg_reduce(dim, true, |lhs, rhs| lhs > rhs)
    }

    pub fn argmin(&self, dim: usize) -> CpuResult<Self> {
        self.arg_reduce(dim, false, |lhs, rhs| lhs < rhs)
    }

    pub fn argmin_keepdim(&self, dim: usize) -> CpuResult<Self> {
        self.arg_reduce(dim, true, |lhs, rhs| lhs < rhs)
    }

    pub fn embedding(&self, indexes: &Self) -> CpuResult<Self> {
        if self.shape.len() < 2 {
            return Err(CpuBackendError::InvalidOperation(format!(
                "embedding table must have rank >= 2, got {:?}",
                self.shape
            )));
        }

        let index_values = indexes.to_i64_vec("embedding")?;
        let mut output_shape = indexes.shape.clone();
        output_shape.extend_from_slice(&self.shape[1..]);

        let row_size = self.shape[1..].iter().product::<usize>();
        let mut source_positions = Vec::with_capacity(index_values.len() * row_size);
        for index in index_values {
            let row = checked_index(index, self.shape[0], "embedding")?;
            let row_start = row * row_size;
            for offset in 0..row_size {
                source_positions.push(row_start + offset);
            }
        }

        self.take_positions(&source_positions, &output_shape)
    }

    pub fn conv1d(
        &self,
        kernel: &Self,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> CpuResult<Self> {
        if !matches!(
            (self.dtype(), kernel.dtype()),
            (CpuDType::F32 | CpuDType::F64, CpuDType::F32 | CpuDType::F64)
        ) {
            return unsupported("conv1d");
        }
        if self.shape.len() != 3 || kernel.shape.len() != 3 {
            return Err(CpuBackendError::InvalidOperation(format!(
                "conv1d expects rank-3 input and kernel, got {:?} and {:?}",
                self.shape, kernel.shape
            )));
        }
        if stride == 0 || dilation == 0 || groups == 0 {
            return Err(CpuBackendError::InvalidOperation(
                "conv1d stride, dilation, and groups must be non-zero".to_string(),
            ));
        }

        let [batch, input_channels, input_length] = [self.shape[0], self.shape[1], self.shape[2]];
        let [output_channels, kernel_channels, kernel_size] =
            [kernel.shape[0], kernel.shape[1], kernel.shape[2]];
        if input_channels % groups != 0 || output_channels % groups != 0 {
            return Err(CpuBackendError::InvalidOperation(format!(
                "conv1d channels must be divisible by groups={groups}, got input_channels={input_channels}, output_channels={output_channels}"
            )));
        }
        let input_channels_per_group = input_channels / groups;
        let output_channels_per_group = output_channels / groups;
        if kernel_channels != input_channels_per_group {
            return Err(CpuBackendError::ShapeMismatch {
                expected: vec![output_channels, input_channels_per_group, kernel_size],
                found: kernel.shape.clone(),
            });
        }

        let output_length = conv_output_size(
            input_length,
            kernel_size,
            padding,
            stride,
            dilation,
            "conv1d",
        )?;
        let input = self.to_f64_vec()?;
        let weights = kernel.to_f64_vec()?;
        let input_strides = strides(&self.shape);
        let kernel_strides = strides(&kernel.shape);
        let mut output = vec![0.0; batch * output_channels * output_length];

        for n in 0..batch {
            for oc in 0..output_channels {
                let group = oc / output_channels_per_group;
                let input_channel_start = group * input_channels_per_group;
                for out_pos in 0..output_length {
                    let mut value = 0.0;
                    for ic_local in 0..kernel_channels {
                        let ic = input_channel_start + ic_local;
                        for k in 0..kernel_size {
                            let padded_pos = out_pos * stride + k * dilation;
                            if padded_pos < padding {
                                continue;
                            }
                            let input_pos = padded_pos - padding;
                            if input_pos >= input_length {
                                continue;
                            }
                            let input_index =
                                n * input_strides[0] + ic * input_strides[1] + input_pos;
                            let kernel_index =
                                oc * kernel_strides[0] + ic_local * kernel_strides[1] + k;
                            value += input[input_index] * weights[kernel_index];
                        }
                    }
                    output[(n * output_channels + oc) * output_length + out_pos] = value;
                }
            }
        }

        Self::from_f64_values(
            output,
            &[batch, output_channels, output_length],
            promote_float_dtype(self.dtype(), kernel.dtype()),
        )
    }

    pub fn conv2d(
        &self,
        kernel: &Self,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> CpuResult<Self> {
        if !matches!(
            (self.dtype(), kernel.dtype()),
            (CpuDType::F32 | CpuDType::F64, CpuDType::F32 | CpuDType::F64)
        ) {
            return unsupported("conv2d");
        }
        if self.shape.len() != 4 || kernel.shape.len() != 4 {
            return Err(CpuBackendError::InvalidOperation(format!(
                "conv2d expects rank-4 input and kernel, got {:?} and {:?}",
                self.shape, kernel.shape
            )));
        }
        if stride == 0 || dilation == 0 || groups == 0 {
            return Err(CpuBackendError::InvalidOperation(
                "conv2d stride, dilation, and groups must be non-zero".to_string(),
            ));
        }

        let [batch, input_channels, input_h, input_w] =
            [self.shape[0], self.shape[1], self.shape[2], self.shape[3]];
        let [output_channels, kernel_channels, kernel_h, kernel_w] = [
            kernel.shape[0],
            kernel.shape[1],
            kernel.shape[2],
            kernel.shape[3],
        ];
        if input_channels % groups != 0 || output_channels % groups != 0 {
            return Err(CpuBackendError::InvalidOperation(format!(
                "conv2d channels must be divisible by groups={groups}, got input_channels={input_channels}, output_channels={output_channels}"
            )));
        }
        let input_channels_per_group = input_channels / groups;
        let output_channels_per_group = output_channels / groups;
        if kernel_channels != input_channels_per_group {
            return Err(CpuBackendError::ShapeMismatch {
                expected: vec![
                    output_channels,
                    input_channels_per_group,
                    kernel_h,
                    kernel_w,
                ],
                found: kernel.shape.clone(),
            });
        }

        let output_h = conv_output_size(input_h, kernel_h, padding, stride, dilation, "conv2d")?;
        let output_w = conv_output_size(input_w, kernel_w, padding, stride, dilation, "conv2d")?;
        let input = self.to_f64_vec()?;
        let weights = kernel.to_f64_vec()?;
        let input_strides = strides(&self.shape);
        let kernel_strides = strides(&kernel.shape);
        let mut output = vec![0.0; batch * output_channels * output_h * output_w];

        for n in 0..batch {
            for oc in 0..output_channels {
                let group = oc / output_channels_per_group;
                let input_channel_start = group * input_channels_per_group;
                for oh in 0..output_h {
                    for ow in 0..output_w {
                        let mut value = 0.0;
                        for ic_local in 0..kernel_channels {
                            let ic = input_channel_start + ic_local;
                            for kh in 0..kernel_h {
                                let padded_h = oh * stride + kh * dilation;
                                if padded_h < padding {
                                    continue;
                                }
                                let ih = padded_h - padding;
                                if ih >= input_h {
                                    continue;
                                }
                                for kw in 0..kernel_w {
                                    let padded_w = ow * stride + kw * dilation;
                                    if padded_w < padding {
                                        continue;
                                    }
                                    let iw = padded_w - padding;
                                    if iw >= input_w {
                                        continue;
                                    }
                                    let input_index = n * input_strides[0]
                                        + ic * input_strides[1]
                                        + ih * input_strides[2]
                                        + iw;
                                    let kernel_index = oc * kernel_strides[0]
                                        + ic_local * kernel_strides[1]
                                        + kh * kernel_strides[2]
                                        + kw;
                                    value += input[input_index] * weights[kernel_index];
                                }
                            }
                        }
                        output[((n * output_channels + oc) * output_h + oh) * output_w + ow] =
                            value;
                    }
                }
            }
        }

        Self::from_f64_values(
            output,
            &[batch, output_channels, output_h, output_w],
            promote_float_dtype(self.dtype(), kernel.dtype()),
        )
    }

    pub fn conv_transpose1d(
        &self,
        kernel: &Self,
        padding: usize,
        output_padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> CpuResult<Self> {
        if self.shape.len() != 3 || kernel.shape.len() != 3 || groups == 0 {
            return Err(CpuBackendError::InvalidOperation(format!(
                "conv_transpose1d expects rank-3 tensors and non-zero groups, got {:?}, {:?}, groups={groups}",
                self.shape, kernel.shape
            )));
        }
        let [batch, input_channels, input_length] = [self.shape[0], self.shape[1], self.shape[2]];
        let [
            kernel_input_channels,
            output_channels_per_group,
            kernel_size,
        ] = [kernel.shape[0], kernel.shape[1], kernel.shape[2]];
        if kernel_input_channels != input_channels || input_channels % groups != 0 {
            return Err(CpuBackendError::InvalidOperation(format!(
                "conv_transpose1d expected matching input/kernel channels divisible by groups={groups}, got {input_channels} and {kernel_input_channels}"
            )));
        }
        let output_length = conv_transpose_output_size(
            input_length,
            kernel_size,
            padding,
            output_padding,
            stride,
            dilation,
            "conv_transpose1d",
        )?;
        let input_values = self.to_f64_vec()?;
        let kernel_values = kernel.to_f64_vec()?;
        let output_channels = output_channels_per_group * groups;
        let input_channels_per_group = input_channels / groups;
        let mut output = vec![0.0; batch * output_channels * output_length];

        for n in 0..batch {
            for ic in 0..input_channels {
                let group = ic / input_channels_per_group;
                for il in 0..input_length {
                    let input_value = input_values[(n * input_channels + ic) * input_length + il];
                    for oc_local in 0..output_channels_per_group {
                        let oc = group * output_channels_per_group + oc_local;
                        for k in 0..kernel_size {
                            let padded = il * stride + k * dilation;
                            if padded < padding {
                                continue;
                            }
                            let ol = padded - padding;
                            if ol < output_length {
                                let kernel_index =
                                    (ic * output_channels_per_group + oc_local) * kernel_size + k;
                                let output_index = (n * output_channels + oc) * output_length + ol;
                                output[output_index] += input_value * kernel_values[kernel_index];
                            }
                        }
                    }
                }
            }
        }
        Self::from_f64_values(
            output,
            &[batch, output_channels, output_length],
            promote_float_dtype(self.dtype(), kernel.dtype()),
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
    ) -> CpuResult<Self> {
        if self.shape.len() != 4 || kernel.shape.len() != 4 || groups == 0 {
            return Err(CpuBackendError::InvalidOperation(format!(
                "conv_transpose2d expects rank-4 tensors and non-zero groups, got {:?}, {:?}, groups={groups}",
                self.shape, kernel.shape
            )));
        }
        let [batch, input_channels, input_h, input_w] =
            [self.shape[0], self.shape[1], self.shape[2], self.shape[3]];
        let [
            kernel_input_channels,
            output_channels_per_group,
            kernel_h,
            kernel_w,
        ] = [
            kernel.shape[0],
            kernel.shape[1],
            kernel.shape[2],
            kernel.shape[3],
        ];
        if kernel_input_channels != input_channels || input_channels % groups != 0 {
            return Err(CpuBackendError::InvalidOperation(format!(
                "conv_transpose2d expected matching input/kernel channels divisible by groups={groups}, got {input_channels} and {kernel_input_channels}"
            )));
        }
        let output_h = conv_transpose_output_size(
            input_h,
            kernel_h,
            padding,
            output_padding.0,
            stride,
            dilation,
            "conv_transpose2d",
        )?;
        let output_w = conv_transpose_output_size(
            input_w,
            kernel_w,
            padding,
            output_padding.1,
            stride,
            dilation,
            "conv_transpose2d",
        )?;
        let input_values = self.to_f64_vec()?;
        let kernel_values = kernel.to_f64_vec()?;
        let output_channels = output_channels_per_group * groups;
        let input_channels_per_group = input_channels / groups;
        let mut output = vec![0.0; batch * output_channels * output_h * output_w];

        for n in 0..batch {
            for ic in 0..input_channels {
                let group = ic / input_channels_per_group;
                for ih in 0..input_h {
                    for iw in 0..input_w {
                        let input_index = ((n * input_channels + ic) * input_h + ih) * input_w + iw;
                        let input_value = input_values[input_index];
                        for oc_local in 0..output_channels_per_group {
                            let oc = group * output_channels_per_group + oc_local;
                            for kh in 0..kernel_h {
                                let padded_h = ih * stride + kh * dilation;
                                if padded_h < padding {
                                    continue;
                                }
                                let oh = padded_h - padding;
                                if oh >= output_h {
                                    continue;
                                }
                                for kw in 0..kernel_w {
                                    let padded_w = iw * stride + kw * dilation;
                                    if padded_w < padding {
                                        continue;
                                    }
                                    let ow = padded_w - padding;
                                    if ow >= output_w {
                                        continue;
                                    }
                                    let kernel_index =
                                        ((ic * output_channels_per_group + oc_local) * kernel_h
                                            + kh)
                                            * kernel_w
                                            + kw;
                                    let output_index = ((n * output_channels + oc) * output_h + oh)
                                        * output_w
                                        + ow;
                                    output[output_index] +=
                                        input_value * kernel_values[kernel_index];
                                }
                            }
                        }
                    }
                }
            }
        }
        Self::from_f64_values(
            output,
            &[batch, output_channels, output_h, output_w],
            promote_float_dtype(self.dtype(), kernel.dtype()),
        )
    }

    pub fn max_pool2d_with_stride(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> CpuResult<Self> {
        self.pool2d_with_stride(kernel_size, stride, PoolKind::Max)
    }

    pub fn avg_pool2d_with_stride(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> CpuResult<Self> {
        self.pool2d_with_stride(kernel_size, stride, PoolKind::Average)
    }

    pub fn gather(&self, indexes: &Self, dim: usize) -> CpuResult<Self> {
        if indexes.dtype() != CpuDType::I64 {
            return Err(CpuBackendError::DTypeMismatch {
                expected: CpuDType::I64,
                found: indexes.dtype(),
            });
        }
        if dim >= self.shape.len() || indexes.shape.len() != self.shape.len() {
            return Err(CpuBackendError::InvalidOperation(format!(
                "gather dim={dim} requires input and index tensors with matching rank, got {:?} and {:?}",
                self.shape, indexes.shape
            )));
        }
        for axis in 0..self.shape.len() {
            if axis != dim && indexes.shape[axis] > self.shape[axis] {
                return Err(CpuBackendError::ShapeMismatch {
                    expected: self.shape.clone(),
                    found: indexes.shape.clone(),
                });
            }
        }

        let index_values = indexes.to_i64_vec("gather")?;
        let output_shape = indexes.shape.clone();
        let input_strides = strides(&self.shape);
        let output_count = elem_count(&output_shape);
        let mut source_positions = Vec::with_capacity(output_count);
        for (output_index, &index_value) in index_values.iter().enumerate().take(output_count) {
            let mut coords = unravel_index(output_index, &output_shape);
            let selected = checked_index(index_value, self.shape[dim], "gather")?;
            coords[dim] = selected;
            source_positions.push(ravel_index_with_strides(&coords, &input_strides));
        }
        self.take_positions(&source_positions, &output_shape)
    }

    pub fn scatter_add(&self, indexes: &Self, source: &Self, dim: usize) -> CpuResult<Self> {
        if indexes.dtype() != CpuDType::I64 {
            return Err(CpuBackendError::DTypeMismatch {
                expected: CpuDType::I64,
                found: indexes.dtype(),
            });
        }
        if self.dtype() != source.dtype() {
            return Err(CpuBackendError::DTypeMismatch {
                expected: self.dtype(),
                found: source.dtype(),
            });
        }
        if dim >= self.shape.len()
            || indexes.shape != source.shape
            || source.shape.len() != self.shape.len()
            || source
                .shape
                .iter()
                .enumerate()
                .any(|(axis, size)| axis != dim && *size > self.shape[axis])
        {
            return Err(CpuBackendError::InvalidOperation(format!(
                "scatter_add dim={dim} requires indexes/source with matching shapes compatible with {:?}, got {:?} and {:?}",
                self.shape, indexes.shape, source.shape
            )));
        }

        let index_values = indexes.to_i64_vec("scatter_add")?;
        let source_values = source.to_f64_vec()?;
        let mut output = self.to_f64_vec()?;
        let output_strides = strides(&self.shape);
        for source_index in 0..source_values.len() {
            let mut coords = unravel_index(source_index, &source.shape);
            coords[dim] =
                checked_index(index_values[source_index], self.shape[dim], "scatter_add")?;
            let output_index = ravel_index_with_strides(&coords, &output_strides);
            output[output_index] += source_values[source_index];
        }
        Self::from_f64_values(output, &self.shape, self.dtype())
    }

    pub fn index_add(&self, indexes: &Self, source: &Self, dim: usize) -> CpuResult<Self> {
        if indexes.dtype() != CpuDType::I64 {
            return Err(CpuBackendError::DTypeMismatch {
                expected: CpuDType::I64,
                found: indexes.dtype(),
            });
        }
        if self.dtype() != source.dtype() {
            return Err(CpuBackendError::DTypeMismatch {
                expected: self.dtype(),
                found: source.dtype(),
            });
        }
        if dim >= self.shape.len()
            || indexes.shape.len() != 1
            || source.shape.len() != self.shape.len()
            || source.shape[dim] != indexes.shape[0]
            || source
                .shape
                .iter()
                .enumerate()
                .any(|(axis, size)| axis != dim && *size != self.shape[axis])
        {
            return Err(CpuBackendError::InvalidOperation(format!(
                "index_add dim={dim} requires rank-1 indexes and source compatible with {:?}, got {:?} and {:?}",
                self.shape, indexes.shape, source.shape
            )));
        }

        let index_values = indexes.to_i64_vec("index_add")?;
        let source_values = source.to_f64_vec()?;
        let mut output = self.to_f64_vec()?;
        let output_strides = strides(&self.shape);
        for (source_index, source_value) in source_values.iter().enumerate() {
            let mut coords = unravel_index(source_index, &source.shape);
            coords[dim] = checked_index(index_values[coords[dim]], self.shape[dim], "index_add")?;
            let output_index = ravel_index_with_strides(&coords, &output_strides);
            output[output_index] += source_value;
        }
        Self::from_f64_values(output, &self.shape, self.dtype())
    }

    pub fn index_select(&self, indexes: &Self, dim: usize) -> CpuResult<Self> {
        if indexes.dtype() != CpuDType::I64 {
            return Err(CpuBackendError::DTypeMismatch {
                expected: CpuDType::I64,
                found: indexes.dtype(),
            });
        }
        if dim >= self.shape.len() || indexes.shape.len() != 1 {
            return Err(CpuBackendError::InvalidOperation(format!(
                "index_select dim={dim} requires rank-1 indexes for input shape {:?}, got {:?}",
                self.shape, indexes.shape
            )));
        }

        let index_values = indexes.to_i64_vec("index_select")?;
        let mut output_shape = self.shape.clone();
        output_shape[dim] = index_values.len();
        let input_strides = strides(&self.shape);
        let output_count = elem_count(&output_shape);
        let mut source_positions = Vec::with_capacity(output_count);
        for output_index in 0..output_count {
            let mut coords = unravel_index(output_index, &output_shape);
            let selected =
                checked_index(index_values[coords[dim]], self.shape[dim], "index_select")?;
            coords[dim] = selected;
            source_positions.push(ravel_index_with_strides(&coords, &input_strides));
        }
        self.take_positions(&source_positions, &output_shape)
    }

    fn elem_count(&self) -> usize {
        elem_count(&self.shape)
    }

    fn to_f64_vec(&self) -> CpuResult<Vec<f64>> {
        Ok(match &self.buffer {
            CpuBuffer::U8(data) => data.iter().map(|value| *value as f64).collect(),
            CpuBuffer::U32(data) => data.iter().map(|value| *value as f64).collect(),
            CpuBuffer::I64(data) => data.iter().map(|value| *value as f64).collect(),
            CpuBuffer::F32(data) => data.iter().map(|value| *value as f64).collect(),
            CpuBuffer::F64(data) => data.clone(),
        })
    }

    fn to_i64_vec(&self, operation: &str) -> CpuResult<Vec<i64>> {
        match &self.buffer {
            CpuBuffer::I64(data) => Ok(data.clone()),
            _ => Err(CpuBackendError::InvalidOperation(format!(
                "{operation} indexes must have dtype I64, got {:?}",
                self.dtype()
            ))),
        }
    }

    fn take_positions(&self, positions: &[usize], shape: &[usize]) -> CpuResult<Self> {
        if positions.len() != elem_count(shape) {
            return Err(CpuBackendError::ShapeMismatch {
                expected: vec![elem_count(shape)],
                found: vec![positions.len()],
            });
        }

        let buffer = match &self.buffer {
            CpuBuffer::U8(data) => {
                CpuBuffer::U8(positions.iter().map(|index| data[*index]).collect())
            }
            CpuBuffer::U32(data) => {
                CpuBuffer::U32(positions.iter().map(|index| data[*index]).collect())
            }
            CpuBuffer::I64(data) => {
                CpuBuffer::I64(positions.iter().map(|index| data[*index]).collect())
            }
            CpuBuffer::F32(data) => {
                CpuBuffer::F32(positions.iter().map(|index| data[*index]).collect())
            }
            CpuBuffer::F64(data) => {
                CpuBuffer::F64(positions.iter().map(|index| data[*index]).collect())
            }
        };
        Self::from_buffer(buffer, shape)
    }

    fn pool2d_with_stride(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        kind: PoolKind,
    ) -> CpuResult<Self> {
        if self.shape.len() != 4 {
            return Err(CpuBackendError::InvalidOperation(format!(
                "pool2d expects rank-4 input, got {:?}",
                self.shape
            )));
        }
        let (kernel_h, kernel_w) = kernel_size;
        let (stride_h, stride_w) = stride;
        if kernel_h == 0 || kernel_w == 0 || stride_h == 0 || stride_w == 0 {
            return Err(CpuBackendError::InvalidOperation(
                "pool2d kernel and stride values must be non-zero".to_string(),
            ));
        }
        let [batch, channels, height, width] =
            [self.shape[0], self.shape[1], self.shape[2], self.shape[3]];
        if kernel_h > height || kernel_w > width {
            return Err(CpuBackendError::InvalidOperation(format!(
                "pool2d kernel {:?} is larger than input spatial shape {:?}",
                kernel_size,
                &self.shape[2..]
            )));
        }

        let output_h = (height - kernel_h) / stride_h + 1;
        let output_w = (width - kernel_w) / stride_w + 1;
        let input = self.to_f64_vec()?;
        let input_strides = strides(&self.shape);
        let mut output = Vec::with_capacity(batch * channels * output_h * output_w);

        for n in 0..batch {
            for c in 0..channels {
                for oh in 0..output_h {
                    for ow in 0..output_w {
                        let h_start = oh * stride_h;
                        let w_start = ow * stride_w;
                        let mut value = match kind {
                            PoolKind::Max => f64::NEG_INFINITY,
                            PoolKind::Average => 0.0,
                        };
                        for kh in 0..kernel_h {
                            for kw in 0..kernel_w {
                                let input_index = n * input_strides[0]
                                    + c * input_strides[1]
                                    + (h_start + kh) * input_strides[2]
                                    + (w_start + kw) * input_strides[3];
                                match kind {
                                    PoolKind::Max => value = value.max(input[input_index]),
                                    PoolKind::Average => value += input[input_index],
                                }
                            }
                        }
                        if matches!(kind, PoolKind::Average) {
                            value /= (kernel_h * kernel_w) as f64;
                        }
                        output.push(value);
                    }
                }
            }
        }

        Self::from_f64_values(output, &[batch, channels, output_h, output_w], self.dtype())
    }

    /// Creates storage from f64 values cast into the requested dtype.
    pub fn from_f64_values(values: Vec<f64>, shape: &[usize], dtype: CpuDType) -> CpuResult<Self> {
        let buffer = match dtype {
            CpuDType::U8 => CpuBuffer::U8(values.iter().map(|value| *value as u8).collect()),
            CpuDType::U32 => CpuBuffer::U32(values.iter().map(|value| *value as u32).collect()),
            CpuDType::I64 => CpuBuffer::I64(values.iter().map(|value| *value as i64).collect()),
            CpuDType::F32 => CpuBuffer::F32(values.iter().map(|value| *value as f32).collect()),
            CpuDType::F64 => CpuBuffer::F64(values),
        };
        Self::from_buffer(buffer, shape)
    }

    /// Stacks tensors along a new dimension.
    pub fn stack(tensors: &[Self], dim: usize) -> CpuResult<Self> {
        let first = tensors.first().ok_or_else(|| {
            CpuBackendError::InvalidOperation("cannot stack empty input".to_string())
        })?;
        if tensors
            .iter()
            .any(|tensor| tensor.shape != first.shape || tensor.dtype() != first.dtype())
        {
            return Err(CpuBackendError::InvalidOperation(
                "all tensors passed to stack must have the same shape and dtype".to_string(),
            ));
        }
        if dim > first.shape.len() {
            return Err(CpuBackendError::InvalidOperation(format!(
                "stack dimension {dim} is invalid for shape {:?}",
                first.shape
            )));
        }
        let mut output_shape = first.shape.clone();
        output_shape.insert(dim, tensors.len());
        let output_count = elem_count(&output_shape);
        let mut output = vec![0.0; output_count];
        let output_strides = strides(&output_shape);
        for (tensor_index, tensor) in tensors.iter().enumerate() {
            let values = tensor.to_f64_vec()?;
            for (input_index, value) in values.iter().enumerate() {
                let input_coords = unravel_index(input_index, &first.shape);
                let mut output_coords = input_coords;
                output_coords.insert(dim, tensor_index);
                let output_index = output_coords
                    .iter()
                    .zip(output_strides.iter())
                    .map(|(coord, stride)| coord * stride)
                    .sum::<usize>();
                output[output_index] = *value;
            }
        }
        Self::from_f64_values(output, &output_shape, first.dtype())
    }

    /// Concatenates tensors along an existing dimension.
    pub fn cat(tensors: &[Self], dim: usize) -> CpuResult<Self> {
        let first = tensors.first().ok_or_else(|| {
            CpuBackendError::InvalidOperation("cannot concatenate empty input".to_string())
        })?;
        if dim >= first.shape.len() {
            return Err(CpuBackendError::InvalidOperation(format!(
                "concat dimension {dim} is invalid for shape {:?}",
                first.shape
            )));
        }
        if tensors.iter().any(|tensor| {
            tensor.dtype() != first.dtype()
                || tensor.shape.len() != first.shape.len()
                || tensor
                    .shape
                    .iter()
                    .enumerate()
                    .any(|(axis, size)| axis != dim && *size != first.shape[axis])
        }) {
            return Err(CpuBackendError::InvalidOperation(
                "all tensors passed to cat must have compatible shapes and dtypes".to_string(),
            ));
        }
        let mut output_shape = first.shape.clone();
        output_shape[dim] = tensors.iter().map(|tensor| tensor.shape[dim]).sum();
        let output_count = elem_count(&output_shape);
        let output_strides = strides(&output_shape);
        let mut output = vec![0.0; output_count];
        let mut dim_offset = 0;
        for tensor in tensors {
            let values = tensor.to_f64_vec()?;
            for (input_index, value) in values.iter().enumerate() {
                let mut output_coords = unravel_index(input_index, &tensor.shape);
                output_coords[dim] += dim_offset;
                let output_index = output_coords
                    .iter()
                    .zip(output_strides.iter())
                    .map(|(coord, stride)| coord * stride)
                    .sum::<usize>();
                output[output_index] = *value;
            }
            dim_offset += tensor.shape[dim];
        }
        Self::from_f64_values(output, &output_shape, first.dtype())
    }

    fn float_result_dtype(&self) -> CpuDType {
        match self.dtype() {
            CpuDType::F64 => CpuDType::F64,
            _ => CpuDType::F32,
        }
    }

    fn unary_op(&self, name: &'static str, op: impl Fn(f64) -> f64) -> CpuResult<Self> {
        if !matches!(self.dtype(), CpuDType::F32 | CpuDType::F64) {
            return unsupported(name);
        }
        let values = self.to_f64_vec()?.into_iter().map(op).collect::<Vec<_>>();
        Self::from_f64_values(values, &self.shape, self.dtype())
    }

    fn binary_op(
        &self,
        rhs: &Self,
        name: &'static str,
        op: impl Fn(f64, f64) -> f64,
    ) -> CpuResult<Self> {
        if !matches!(
            (self.dtype(), rhs.dtype()),
            (CpuDType::F32 | CpuDType::F64, CpuDType::F32 | CpuDType::F64)
        ) {
            return unsupported(name);
        }
        let rhs_dtype = rhs.dtype();
        let shape = broadcast_shape(&self.shape, &rhs.shape)?;
        let lhs = self.broadcast_as(&shape)?.to_f64_vec()?;
        let rhs = rhs.broadcast_as(&shape)?.to_f64_vec()?;
        let values = lhs
            .iter()
            .zip(rhs.iter())
            .map(|(lhs, rhs)| op(*lhs, *rhs))
            .collect::<Vec<_>>();
        Self::from_f64_values(values, &shape, promote_float_dtype(self.dtype(), rhs_dtype))
    }

    fn binary_compare(&self, rhs: &Self, op: impl Fn(f64, f64) -> bool) -> CpuResult<Self> {
        let shape = broadcast_shape(&self.shape, &rhs.shape)?;
        let lhs = self.broadcast_as(&shape)?.to_f64_vec()?;
        let rhs = rhs.broadcast_as(&shape)?.to_f64_vec()?;
        let values = lhs
            .iter()
            .zip(rhs.iter())
            .map(|(lhs, rhs)| u8::from(op(*lhs, *rhs)))
            .collect::<Vec<_>>();
        Self::from_buffer(CpuBuffer::U8(values), &shape)
    }

    fn reduce(&self, dim: usize, keepdim: bool, op: impl Fn(&[f64]) -> f64) -> CpuResult<Self> {
        if dim >= self.shape.len() {
            return Err(CpuBackendError::InvalidOperation(format!(
                "invalid reduction dim {dim} for shape {:?}",
                self.shape
            )));
        }
        let input = self.to_f64_vec()?;
        let mut output_shape = self.shape.clone();
        let reduced = output_shape.remove(dim);
        if keepdim {
            output_shape.insert(dim, 1);
        }
        let output_count = elem_count(&output_shape);
        let mut buckets = vec![Vec::with_capacity(reduced); output_count];
        for (input_index, &value) in input.iter().enumerate().take(self.elem_count()) {
            let mut coords = unravel_index(input_index, &self.shape);
            coords.remove(dim);
            if keepdim {
                coords.insert(dim, 0);
            }
            let output_index = ravel_index(&coords, &output_shape);
            buckets[output_index].push(value);
        }
        let output = buckets.iter().map(|values| op(values)).collect::<Vec<_>>();
        Self::from_f64_values(output, &output_shape, self.float_result_dtype())
    }

    fn arg_reduce(
        &self,
        dim: usize,
        keepdim: bool,
        better: impl Fn(f64, f64) -> bool,
    ) -> CpuResult<Self> {
        if dim >= self.shape.len() {
            return Err(CpuBackendError::InvalidOperation(format!(
                "invalid arg reduction dim {dim} for shape {:?}",
                self.shape
            )));
        }
        let input = self.to_f64_vec()?;
        let mut output_shape = self.shape.clone();
        let reduced = output_shape.remove(dim);
        if keepdim {
            output_shape.insert(dim, 1);
        }
        let output_count = elem_count(&output_shape);
        let mut best = vec![(0usize, 0.0f64, false); output_count];
        for (input_index, &value) in input.iter().enumerate().take(self.elem_count()) {
            let mut coords = unravel_index(input_index, &self.shape);
            let reduced_index = coords[dim];
            coords.remove(dim);
            if keepdim {
                coords.insert(dim, 0);
            }
            let output_index = ravel_index(&coords, &output_shape);
            let entry = &mut best[output_index];
            if !entry.2 || better(value, entry.1) {
                *entry = (reduced_index, value, true);
            }
        }
        let values = best
            .into_iter()
            .map(|(index, _, _)| index as u32)
            .collect::<Vec<_>>();
        let _ = reduced;
        Self::from_buffer(CpuBuffer::U32(values), &output_shape)
    }
}

impl Display for CpuStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CpuStorage({:?}, {:?})", self.shape, self.buffer)
    }
}

fn elem_count(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn align_shape(shape: &[usize], rank: usize) -> Vec<usize> {
    let mut aligned = vec![1; rank.saturating_sub(shape.len())];
    aligned.extend_from_slice(shape);
    aligned
}

fn strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for index in (1..shape.len()).rev() {
        strides[index - 1] = strides[index] * shape[index];
    }
    strides
}

fn unravel_index(mut index: usize, shape: &[usize]) -> Vec<usize> {
    let mut coords = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        coords[axis] = index % shape[axis];
        index /= shape[axis];
    }
    coords
}

fn ravel_index(coords: &[usize], shape: &[usize]) -> usize {
    coords
        .iter()
        .zip(strides(shape).iter())
        .map(|(coord, stride)| coord * stride)
        .sum()
}

fn ravel_index_with_strides(coords: &[usize], strides: &[usize]) -> usize {
    coords
        .iter()
        .zip(strides.iter())
        .map(|(coord, stride)| coord * stride)
        .sum()
}

fn checked_index(index: i64, upper_bound: usize, operation: &str) -> CpuResult<usize> {
    if index < 0 || index as usize >= upper_bound {
        return Err(CpuBackendError::InvalidOperation(format!(
            "{operation} index {index} is out of bounds for dimension size {upper_bound}"
        )));
    }
    Ok(index as usize)
}

fn broadcast_shape(lhs: &[usize], rhs: &[usize]) -> CpuResult<Vec<usize>> {
    let rank = lhs.len().max(rhs.len());
    let lhs_aligned = align_shape(lhs, rank);
    let rhs_aligned = align_shape(rhs, rank);
    let mut shape = Vec::with_capacity(rank);
    for (lhs_dim, rhs_dim) in lhs_aligned.iter().zip(rhs_aligned.iter()) {
        if lhs_dim == rhs_dim {
            shape.push(*lhs_dim);
        } else if *lhs_dim == 1 {
            shape.push(*rhs_dim);
        } else if *rhs_dim == 1 {
            shape.push(*lhs_dim);
        } else {
            return Err(CpuBackendError::ShapeMismatch {
                expected: lhs.to_vec(),
                found: rhs.to_vec(),
            });
        }
    }
    Ok(shape)
}

fn promote_float_dtype(lhs: CpuDType, rhs: CpuDType) -> CpuDType {
    if lhs == CpuDType::F64 || rhs == CpuDType::F64 {
        CpuDType::F64
    } else {
        CpuDType::F32
    }
}

fn conv_output_size(
    input_size: usize,
    kernel_size: usize,
    padding: usize,
    stride: usize,
    dilation: usize,
    operation: &str,
) -> CpuResult<usize> {
    if kernel_size == 0 {
        return Err(CpuBackendError::InvalidOperation(format!(
            "{operation} kernel dimensions must be non-zero"
        )));
    }
    let effective_kernel = dilation
        .checked_mul(kernel_size - 1)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| {
            CpuBackendError::InvalidOperation(format!(
                "{operation} effective kernel size overflowed"
            ))
        })?;
    let padded_input = padding
        .checked_mul(2)
        .and_then(|value| input_size.checked_add(value))
        .ok_or_else(|| {
            CpuBackendError::InvalidOperation(format!("{operation} padded input size overflowed"))
        })?;
    if padded_input < effective_kernel {
        return Err(CpuBackendError::InvalidOperation(format!(
            "{operation} effective kernel size {effective_kernel} is larger than padded input size {padded_input}"
        )));
    }
    Ok((padded_input - effective_kernel) / stride + 1)
}

fn conv_transpose_output_size(
    input: usize,
    kernel: usize,
    padding: usize,
    output_padding: usize,
    stride: usize,
    dilation: usize,
    operation: &str,
) -> CpuResult<usize> {
    if input == 0 || kernel == 0 || stride == 0 || dilation == 0 || output_padding >= stride {
        return Err(CpuBackendError::InvalidOperation(format!(
            "{operation} requires non-zero input/kernel/stride/dilation and output_padding < stride"
        )));
    }
    let padded = (input - 1)
        .checked_mul(stride)
        .and_then(|value| value.checked_add(dilation * (kernel - 1)))
        .and_then(|value| value.checked_add(output_padding + 1))
        .ok_or_else(|| CpuBackendError::InvalidOperation(format!("{operation} size overflow")))?;
    padded.checked_sub(2 * padding).ok_or_else(|| {
        CpuBackendError::InvalidOperation(format!("{operation} padding is too large"))
    })
}

fn variance(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values
        .iter()
        .map(|value| {
            let delta = *value - mean;
            delta * delta
        })
        .sum::<f64>()
        / (values.len() - 1) as f64
}

fn unsupported<T>(operation: &str) -> CpuResult<T> {
    Err(CpuBackendError::UnsupportedOperation(operation.to_string()))
}
