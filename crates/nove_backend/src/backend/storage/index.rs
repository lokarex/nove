//! Indexing, indexed accumulation and conditional selection.

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
    pub fn gather(&self, indexes: &Self, dim: usize) -> Result<Self, BackendError> {
        match (self, indexes) {
            #[cfg(feature = "candle")]
            (Self::Candle(lhs), Self::Candle(rhs)) => Ok(Self::Candle(lhs.gather(rhs, dim)?)),
            #[cfg(feature = "native")]
            (Self::Native(lhs), Self::Native(rhs)) => Ok(Self::Native(lhs.gather(rhs, dim)?)),
            #[cfg(all(feature = "candle", feature = "native"))]
            (Self::Candle(_), other) => Err(BackendError::BackendMismatch {
                expected: BackendKind::Candle,
                found: other.backend_kind().unwrap_or(BackendKind::Candle),
            }),
            #[cfg(all(feature = "candle", feature = "native"))]
            (Self::Native(_), other) => Err(BackendError::BackendMismatch {
                expected: BackendKind::Native,
                found: other.backend_kind().unwrap_or(BackendKind::Native),
            }),
        }
    }

    pub fn scatter_add(
        &self,
        indexes: &Self,
        source: &Self,
        dim: usize,
    ) -> Result<Self, BackendError> {
        match (self, indexes, source) {
            #[cfg(feature = "candle")]
            (Self::Candle(init), Self::Candle(indexes), Self::Candle(source)) => {
                Ok(Self::Candle(init.scatter_add(indexes, source, dim)?))
            }
            #[cfg(feature = "native")]
            (Self::Native(init), Self::Native(indexes), Self::Native(source)) => {
                Ok(Self::Native(init.scatter_add(indexes, source, dim)?))
            }
            #[cfg(all(feature = "candle", feature = "native"))]
            (Self::Candle(_), _, _) => Err(BackendError::BackendMismatch {
                expected: BackendKind::Candle,
                found: BackendKind::Native,
            }),
            #[cfg(all(feature = "candle", feature = "native"))]
            (Self::Native(_), _, _) => Err(BackendError::BackendMismatch {
                expected: BackendKind::Native,
                found: BackendKind::Candle,
            }),
        }
    }

    pub fn index_select(&self, indexes: &Self, dim: usize) -> Result<Self, BackendError> {
        match (self, indexes) {
            #[cfg(feature = "candle")]
            (Self::Candle(lhs), Self::Candle(rhs)) => Ok(Self::Candle(lhs.index_select(rhs, dim)?)),
            #[cfg(feature = "native")]
            (Self::Native(lhs), Self::Native(rhs)) => Ok(Self::Native(lhs.index_select(rhs, dim)?)),
            #[cfg(all(feature = "candle", feature = "native"))]
            (Self::Candle(_), other) => Err(BackendError::BackendMismatch {
                expected: BackendKind::Candle,
                found: other.backend_kind().unwrap_or(BackendKind::Candle),
            }),
            #[cfg(all(feature = "candle", feature = "native"))]
            (Self::Native(_), other) => Err(BackendError::BackendMismatch {
                expected: BackendKind::Native,
                found: other.backend_kind().unwrap_or(BackendKind::Native),
            }),
        }
    }

    pub fn index_add(
        &self,
        indexes: &Self,
        source: &Self,
        dim: usize,
    ) -> Result<Self, BackendError> {
        match (self, indexes, source) {
            #[cfg(feature = "candle")]
            (Self::Candle(init), Self::Candle(indexes), Self::Candle(source)) => {
                Ok(Self::Candle(init.index_add(indexes, source, dim)?))
            }
            #[cfg(feature = "native")]
            (Self::Native(init), Self::Native(indexes), Self::Native(source)) => {
                Ok(Self::Native(init.index_add(indexes, source, dim)?))
            }
            #[cfg(all(feature = "candle", feature = "native"))]
            (Self::Candle(_), _, _) => Err(BackendError::BackendMismatch {
                expected: BackendKind::Candle,
                found: BackendKind::Native,
            }),
            #[cfg(all(feature = "candle", feature = "native"))]
            (Self::Native(_), _, _) => Err(BackendError::BackendMismatch {
                expected: BackendKind::Native,
                found: BackendKind::Candle,
            }),
        }
    }

    pub fn where_cond(
        condition: &Self,
        true_value: &Self,
        false_value: &Self,
    ) -> Result<Self, BackendError> {
        match (condition, true_value, false_value) {
            #[cfg(feature = "candle")]
            (Self::Candle(condition), Self::Candle(true_value), Self::Candle(false_value)) => {
                Ok(Self::Candle(candle::CandleStorage::where_cond(
                    condition,
                    true_value,
                    false_value,
                )?))
            }
            #[cfg(feature = "native")]
            (Self::Native(condition), Self::Native(true_value), Self::Native(false_value)) => {
                Ok(Self::Native(native::NativeStorage::where_cond(
                    condition,
                    true_value,
                    false_value,
                )?))
            }
            #[cfg(all(feature = "candle", feature = "native"))]
            (Self::Candle(_), _, _) => Err(BackendError::BackendMismatch {
                expected: BackendKind::Candle,
                found: BackendKind::Native,
            }),
            #[cfg(all(feature = "candle", feature = "native"))]
            (Self::Native(_), _, _) => Err(BackendError::BackendMismatch {
                expected: BackendKind::Native,
                found: BackendKind::Candle,
            }),
        }
    }
}
