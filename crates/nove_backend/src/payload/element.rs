use super::{IntoTensorPayload, TensorBuffer, TensorPayload};
use crate::{BackendError, DType, Shape};

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
