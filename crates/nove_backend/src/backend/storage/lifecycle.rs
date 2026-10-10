//! Storage detachment, assignment and zeroing.

use super::super::Backend;
use super::BackendStorage;
use crate::BackendError;

impl BackendStorage {
    pub fn detach(&self) -> Result<Self, BackendError> {
        Ok(self.clone())
    }

    pub fn assign_from(&mut self, other: &Self) -> Result<(), BackendError> {
        *self = other.clone();
        Ok(())
    }

    pub fn zero_set(&mut self) -> Result<(), BackendError> {
        match self {
            #[cfg(feature = "candle")]
            Self::Candle(storage) => storage.zero_set(),
            #[cfg(feature = "native")]
            Self::Native(storage) => storage.zero_set(),
        }
    }
}
