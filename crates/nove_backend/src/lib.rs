#![doc = include_str!("../README.md")]

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
