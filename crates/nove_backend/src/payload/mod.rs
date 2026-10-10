//! Backend-independent tensor payloads and input conversion.

mod buffer;
mod conversion;
mod element;

pub use buffer::TensorBuffer;
pub use element::{FloatTensorElement, TensorElement};

use crate::{BackendError, Shape};

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
