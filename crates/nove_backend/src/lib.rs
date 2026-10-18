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
//! Native family storage is available with `native`; its `Cpu` variant requires
//! `native-cpu`. Library compilation requires `candle` or `native`; device
//! construction additionally requires the corresponding device capability feature.
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

#[cfg(not(any(feature = "candle", feature = "native")))]
compile_error!("At least one compiled backend must be enabled; enable candle or native");

#[cfg(any(feature = "candle", feature = "native"))]
pub mod device;
#[cfg(any(feature = "candle", feature = "native"))]
pub use device::{Device, DeviceError, DeviceKind, DeviceLocation};

#[cfg(any(feature = "candle", feature = "native"))]
mod dtype;
#[cfg(any(feature = "candle", feature = "native"))]
pub use dtype::DType;

#[cfg(any(feature = "candle", feature = "native"))]
mod shape;
#[cfg(any(feature = "candle", feature = "native"))]
pub use shape::Shape;

#[cfg(any(feature = "candle", feature = "native"))]
mod kind;
#[cfg(any(feature = "candle", feature = "native"))]
pub use kind::BackendKind;

#[cfg(any(feature = "candle", feature = "native"))]
mod error;
#[cfg(any(feature = "candle", feature = "native"))]
pub use error::BackendError;

#[cfg(any(feature = "candle", feature = "native"))]
mod payload;
#[cfg(any(feature = "candle", feature = "native"))]
pub use payload::{
    FloatTensorElement, IntoTensorPayload, TensorBuffer, TensorElement, TensorPayload,
};

#[cfg(any(feature = "candle", feature = "native"))]
pub mod backend;
