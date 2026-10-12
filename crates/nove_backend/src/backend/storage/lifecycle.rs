//! Storage cloning, device comparison and zeroing.

use super::super::Backend;
use super::BackendStorage;
use crate::BackendError;

impl BackendStorage {
    /// Try to clone the storage.
    ///
    /// # Notes
    /// * The cloned storage has an independent data buffer.
    /// * The backend, device, dtype and shape are preserved.
    /// * Unlike [`Clone::clone`], this method guarantees independent data for every backend.
    ///
    /// # Returns
    /// * `Ok(Self)` - A new storage with cloned data if successful.
    /// * `Err(BackendError)` - The error when cloning the storage.
    ///
    /// # Examples
    /// ```
    /// use nove_backend::{Device, backend::BackendStorage};
    ///
    /// let mut source = BackendStorage::from_data(vec![1.0f32, 2.0], &Device::default()).unwrap();
    /// let cloned = source.try_clone().unwrap();
    ///
    /// source.zero_set().unwrap();
    /// assert_eq!(cloned.to_vec::<f32>().unwrap(), vec![1.0, 2.0]);
    /// ```
    pub fn try_clone(&self) -> Result<Self, BackendError> {
        match self {
            #[cfg(feature = "candle")]
            Self::Candle(storage) => storage.try_clone().map(Self::Candle),
            #[cfg(feature = "native")]
            Self::Native(storage) => storage.try_clone().map(Self::Native),
        }
    }

    /// Check whether the storages use the same backend and actual device.
    ///
    /// # Notes
    /// * Equal device indexes do not guarantee the same Candle device handle.
    /// * Native storage currently supports only the CPU device.
    ///
    /// # Arguments
    /// * `other` - The storage to compare devices with.
    ///
    /// # Returns
    /// * `true` - The storages use the same backend and actual device.
    /// * `false` - The backend or actual device differs.
    pub fn same_device(&self, other: &Self) -> bool {
        match (self, other) {
            #[cfg(feature = "candle")]
            (Self::Candle(a), Self::Candle(b)) => a.same_device(b),
            #[cfg(feature = "native")]
            (Self::Native(a), Self::Native(b)) => a.same_device(b),
            #[cfg(all(feature = "candle", feature = "native"))]
            (Self::Candle(_), Self::Native(_)) | (Self::Native(_), Self::Candle(_)) => false,
        }
    }

    /// Set the storage values to zero.
    ///
    /// # Notes
    /// * Storages sharing the buffer may also observe the change.
    /// * Use [`Self::try_clone`] to create an independent buffer.
    ///
    /// # Returns
    /// * `Ok(())` - The storage values are set to zero.
    /// * `Err(BackendError)` - The error when setting the values to zero.
    pub fn zero_set(&mut self) -> Result<(), BackendError> {
        match self {
            #[cfg(feature = "candle")]
            Self::Candle(storage) => storage.zero_set(),
            #[cfg(feature = "native")]
            Self::Native(storage) => storage.zero_set(),
        }
    }
}
