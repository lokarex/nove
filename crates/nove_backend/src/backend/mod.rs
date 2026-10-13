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

mod contract;
mod serialization;
mod storage;
mod validation;

pub use crate::{
    BackendError, BackendKind, FloatTensorElement, IntoTensorPayload, TensorBuffer, TensorElement,
    TensorPayload,
};
pub(crate) use contract::Backend;
pub use serialization::{load_safetensors, save_safetensors};
pub use storage::BackendStorage;
