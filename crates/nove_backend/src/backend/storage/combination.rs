//! Stacking and concatenation of backend storage.

use super::super::Backend;
#[cfg(feature = "candle")]
use super::super::candle;
#[cfg(feature = "native")]
use super::super::native;
use super::BackendStorage;
use crate::BackendError;
#[cfg(all(feature = "candle", feature = "native"))]
use crate::BackendKind;

impl BackendStorage {
    pub fn stack(tensors: &[Self], dim: usize) -> Result<Self, BackendError> {
        #[cfg(feature = "candle")]
        {
            if tensors
                .iter()
                .all(|storage| matches!(storage, Self::Candle(_)))
            {
                let tensors = tensors
                    .iter()
                    .map(|storage| match storage {
                        Self::Candle(storage) => Ok(storage.clone()),
                        #[cfg(all(feature = "candle", feature = "native"))]
                        _ => Err(BackendError::BackendMismatch {
                            expected: BackendKind::Candle,
                            found: storage.backend_kind().unwrap_or(BackendKind::Candle),
                        }),
                    })
                    .collect::<Result<Vec<_>, BackendError>>()?;
                return Ok(Self::Candle(candle::CandleStorage::stack(&tensors, dim)?));
            }
        }

        #[cfg(feature = "native")]
        {
            if tensors
                .iter()
                .all(|storage| matches!(storage, Self::Native(_)))
            {
                let tensors = tensors
                    .iter()
                    .map(|storage| match storage {
                        Self::Native(storage) => Ok(storage.clone()),
                        #[cfg(all(feature = "candle", feature = "native"))]
                        _ => Err(BackendError::BackendMismatch {
                            expected: BackendKind::Native,
                            found: storage.backend_kind().unwrap_or(BackendKind::Native),
                        }),
                    })
                    .collect::<Result<Vec<_>, BackendError>>()?;
                return Ok(Self::Native(native::NativeStorage::stack(&tensors, dim)?));
            }
        }

        let _ = (tensors, dim);
        Err(BackendError::InvalidOperation(
            "stack requires all tensors to use the same backend".to_string(),
        ))
    }

    pub fn cat(tensors: &[Self], dim: usize) -> Result<Self, BackendError> {
        #[cfg(feature = "candle")]
        {
            if tensors
                .iter()
                .all(|storage| matches!(storage, Self::Candle(_)))
            {
                let tensors = tensors
                    .iter()
                    .map(|storage| match storage {
                        Self::Candle(storage) => Ok(storage.clone()),
                        #[cfg(all(feature = "candle", feature = "native"))]
                        _ => Err(BackendError::BackendMismatch {
                            expected: BackendKind::Candle,
                            found: storage.backend_kind().unwrap_or(BackendKind::Candle),
                        }),
                    })
                    .collect::<Result<Vec<_>, BackendError>>()?;
                return Ok(Self::Candle(candle::CandleStorage::cat(&tensors, dim)?));
            }
        }

        #[cfg(feature = "native")]
        {
            if tensors
                .iter()
                .all(|storage| matches!(storage, Self::Native(_)))
            {
                let tensors = tensors
                    .iter()
                    .map(|storage| match storage {
                        Self::Native(storage) => Ok(storage.clone()),
                        #[cfg(all(feature = "candle", feature = "native"))]
                        _ => Err(BackendError::BackendMismatch {
                            expected: BackendKind::Native,
                            found: storage.backend_kind().unwrap_or(BackendKind::Native),
                        }),
                    })
                    .collect::<Result<Vec<_>, BackendError>>()?;
                return Ok(Self::Native(native::NativeStorage::cat(&tensors, dim)?));
            }
        }

        let _ = (tensors, dim);
        Err(BackendError::InvalidOperation(
            "cat requires all tensors to use the same backend".to_string(),
        ))
    }
}
