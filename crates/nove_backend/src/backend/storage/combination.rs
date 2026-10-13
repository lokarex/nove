//! Stacking and concatenation of backend storage.

use super::super::Backend;
#[cfg(feature = "candle")]
use super::super::candle;
#[cfg(feature = "native")]
use super::super::native;
use super::super::validation;
use super::BackendStorage;
use super::ensure_compatible_devices;
use crate::BackendError;
#[cfg(all(feature = "candle", feature = "native"))]
use crate::BackendKind;

impl BackendStorage {
    pub fn stack(tensors: &[Self], dim: usize) -> Result<Self, BackendError> {
        ensure_compatible_devices("stack", &tensors.iter().collect::<Vec<_>>())?;
        let metadata = tensors
            .iter()
            .map(|tensor| Ok((tensor.shape()?, tensor.dtype()?)))
            .collect::<Result<Vec<_>, BackendError>>()?;
        validation::validate_stack(
            &metadata
                .iter()
                .map(|(shape, dtype)| (shape, *dtype))
                .collect::<Vec<_>>(),
            dim,
        )?;
        match tensors
            .first()
            .ok_or(BackendError::EmptyInputs { operation: "stack" })?
        {
            #[cfg(feature = "candle")]
            Self::Candle(_) => {
                let tensors = tensors
                    .iter()
                    .map(|storage| match storage {
                        Self::Candle(storage) => Ok(storage.clone()),
                        #[cfg(all(feature = "candle", feature = "native"))]
                        _ => Err(BackendError::BackendMismatch {
                            expected: BackendKind::Candle,
                            found: storage.backend_kind()?,
                        }),
                    })
                    .collect::<Result<Vec<_>, BackendError>>()?;
                Ok(Self::Candle(candle::CandleStorage::stack(&tensors, dim)?))
            }
            #[cfg(feature = "native")]
            Self::Native(_) => {
                let tensors = tensors
                    .iter()
                    .map(|storage| match storage {
                        Self::Native(storage) => Ok(storage.clone()),
                        #[cfg(all(feature = "candle", feature = "native"))]
                        _ => Err(BackendError::BackendMismatch {
                            expected: BackendKind::Native,
                            found: storage.backend_kind()?,
                        }),
                    })
                    .collect::<Result<Vec<_>, BackendError>>()?;
                Ok(Self::Native(native::NativeStorage::stack(&tensors, dim)?))
            }
        }
    }

    pub fn cat(tensors: &[Self], dim: usize) -> Result<Self, BackendError> {
        ensure_compatible_devices("cat", &tensors.iter().collect::<Vec<_>>())?;
        let metadata = tensors
            .iter()
            .map(|tensor| Ok((tensor.shape()?, tensor.dtype()?)))
            .collect::<Result<Vec<_>, BackendError>>()?;
        validation::validate_cat(
            &metadata
                .iter()
                .map(|(shape, dtype)| (shape, *dtype))
                .collect::<Vec<_>>(),
            dim,
        )?;
        match tensors
            .first()
            .ok_or(BackendError::EmptyInputs { operation: "cat" })?
        {
            #[cfg(feature = "candle")]
            Self::Candle(_) => {
                let tensors = tensors
                    .iter()
                    .map(|storage| match storage {
                        Self::Candle(storage) => Ok(storage.clone()),
                        #[cfg(all(feature = "candle", feature = "native"))]
                        _ => Err(BackendError::BackendMismatch {
                            expected: BackendKind::Candle,
                            found: storage.backend_kind()?,
                        }),
                    })
                    .collect::<Result<Vec<_>, BackendError>>()?;
                Ok(Self::Candle(candle::CandleStorage::cat(&tensors, dim)?))
            }
            #[cfg(feature = "native")]
            Self::Native(_) => {
                let tensors = tensors
                    .iter()
                    .map(|storage| match storage {
                        Self::Native(storage) => Ok(storage.clone()),
                        #[cfg(all(feature = "candle", feature = "native"))]
                        _ => Err(BackendError::BackendMismatch {
                            expected: BackendKind::Native,
                            found: storage.backend_kind()?,
                        }),
                    })
                    .collect::<Result<Vec<_>, BackendError>>()?;
                Ok(Self::Native(native::NativeStorage::cat(&tensors, dim)?))
            }
        }
    }
}
