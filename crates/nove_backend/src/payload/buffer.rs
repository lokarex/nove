use crate::DType;

/// A flattened typed tensor buffer independent of any backend implementation.
///
/// # Notes
/// `TensorBuffer` stores materialized element values in row-major order. It is
/// used by [`super::TensorPayload`] before backend-specific storage is created.
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
