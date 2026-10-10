//! Backend-owned storage, metadata and formatting.

mod combination;
mod conversion;
mod convolution;
mod dispatch;
mod factory;
mod index;
mod lifecycle;
mod ops;

use super::Backend;
#[cfg(feature = "candle")]
use super::candle;
#[cfg(feature = "native")]
use super::native;
use crate::{BackendError, BackendKind, DType, Shape};
use std::fmt::Display;

/// Backend-owned tensor storage.
#[derive(Clone, Debug)]
#[doc(hidden)]
pub enum BackendStorage {
    #[cfg(feature = "candle")]
    Candle(candle::CandleStorage),
    #[cfg(feature = "native")]
    Native(native::NativeStorage),
}

impl BackendStorage {
    pub fn backend_kind(&self) -> Result<BackendKind, BackendError> {
        match self {
            #[cfg(feature = "candle")]
            Self::Candle(_) => Ok(BackendKind::Candle),
            #[cfg(feature = "native")]
            Self::Native(_) => Ok(BackendKind::Native),
        }
    }

    pub fn dtype(&self) -> Result<DType, BackendError> {
        match self {
            #[cfg(feature = "candle")]
            Self::Candle(storage) => Ok(storage.dtype()),
            #[cfg(feature = "native")]
            Self::Native(storage) => Ok(storage.dtype()),
        }
    }

    pub fn shape(&self) -> Result<Shape, BackendError> {
        match self {
            #[cfg(feature = "candle")]
            Self::Candle(storage) => Ok(storage.shape()),
            #[cfg(feature = "native")]
            Self::Native(storage) => Ok(storage.shape()),
        }
    }

    pub fn fmt_backend(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(feature = "candle")]
            Self::Candle(storage) => storage.fmt(f),
            #[cfg(feature = "native")]
            Self::Native(storage) => storage.fmt(f),
        }
    }
}

impl Display for BackendStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.fmt_backend(f)
    }
}
