//! Backend abstractions for Nove tensors.
//!
//! This crate defines the backend-independent pieces that `nove_tensor` uses
//! to create tensor storage, select devices, move data between backend
//! implementations, and describe tensor payloads before backend storage is
//! materialized.
//!
//! # Notes
//! The default feature enables the Candle-backed CPU implementation. The
//! `native` feature enables the native backend family facade, and
//! `native-cpu` enables the native CPU device capability.
//!
//! # Examples
//! ```
//! use nove_backend::{BackendKind, Device, DType, Shape, TensorBuffer, TensorPayload, device};
//!
//! let device = Device::default();
//! assert!(matches!(device.backend(), BackendKind::Candle | BackendKind::Native));
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

pub mod device;
pub use device::{Device, DeviceError, DeviceKind};

mod dtype;
pub use dtype::DType;

mod shape;
pub use shape::Shape;

pub mod backend;
pub use backend::{
    BackendError, BackendKind, FloatTensorElement, TensorBuffer,
    IntoTensorPayload, TensorElement, TensorPayload,
};
