//! Storage creation from payloads, typed data and factories.

use super::super::Backend;
#[cfg(feature = "candle")]
use super::super::candle;
#[cfg(feature = "native")]
use super::super::native;
use super::super::validation;
use super::BackendStorage;
use crate::{
    BackendError, BackendKind, DType, Device, FloatTensorElement, IntoTensorPayload, Shape,
    TensorElement, TensorPayload,
};

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
                payload, device,
            )?)),
            #[cfg(feature = "native")]
            BackendKind::Native => Ok(Self::Native(native::NativeStorage::from_payload(
                payload, device,
            )?)),
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
        shape.checked_elem_count()?;
        Self::from_vec(data.to_vec(), shape, device)
    }

    pub fn rand<T>(low: T, high: T, shape: &Shape, device: &Device) -> Result<Self, BackendError>
    where
        T: FloatTensorElement,
    {
        shape.checked_elem_count()?;
        validation::validate_uniform(low.to_f64(), high.to_f64())?;
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
        shape.checked_elem_count()?;
        validation::validate_normal(mean.to_f64(), std.to_f64())?;
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
        shape.checked_elem_count()?;
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
        shape.checked_elem_count()?;
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
}
