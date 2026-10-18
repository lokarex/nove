//! Storage materialization, dtype conversion and device transfer.

use super::super::Backend;
#[cfg(all(feature = "candle", feature = "native"))]
use super::super::{candle, native};
use super::BackendStorage;
use super::dispatch::dispatch;
#[cfg(all(feature = "candle", feature = "native"))]
use crate::TensorPayload;
use crate::{BackendError, BackendKind, DType, Device, TensorBuffer, TensorElement};

impl BackendStorage {
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

    pub fn to_device(&self, device: &Device) -> Result<Self, BackendError> {
        match (self, device.backend()) {
            #[cfg(feature = "candle")]
            (Self::Candle(storage), BackendKind::Candle) => {
                Ok(Self::Candle(storage.to_device(device)?))
            }
            #[cfg(feature = "native")]
            (Self::Native(storage), BackendKind::Native) => {
                Ok(Self::Native(storage.to_device(device)?))
            }
            #[cfg(all(feature = "candle", feature = "native"))]
            (Self::Native(storage), BackendKind::Candle) => {
                let payload = TensorPayload::new(storage.to_tensor_buffer()?, storage.shape())?;
                Ok(Self::Candle(candle::CandleStorage::from_payload(
                    payload, device,
                )?))
            }
            #[cfg(all(feature = "candle", feature = "native"))]
            (Self::Candle(storage), BackendKind::Native) => {
                let payload = TensorPayload::new(storage.to_tensor_buffer()?, storage.shape())?;
                Ok(Self::Native(native::NativeStorage::from_payload(
                    payload, device,
                )?))
            }
            #[cfg(not(all(feature = "candle", feature = "native")))]
            (_, backend) => Err(BackendError::UnsupportedBackend(backend)),
        }
    }

    pub fn to_dtype(&self, dtype: DType) -> Result<Self, BackendError> {
        dispatch!(self, to_dtype, dtype)
    }
}
