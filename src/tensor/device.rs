use thiserror::Error;

/// Error type from device operations.
#[derive(Error, Debug)]
pub enum DeviceError {
    /// Error from Candle library.
    #[error("Error from Candle: {0}")]
    CandleError(#[from] candle_core::Error),
}

/// The `Device` struct to describe the device.
///
/// # Fields
/// * `inner` - The inner device from candle.
pub struct Device {
    pub inner: candle_core::Device,
}

impl Device {
    /// Get the CPU device.
    ///
    /// # Returns
    /// * `Self` - The CPU device.
    pub fn get_cpu() -> Self {
        Self {
            inner: candle_core::Device::Cpu,
        }
    }

    /// Get the CUDA device with the given index.
    ///
    /// # Arguments
    /// * `index` - The index of the CUDA device.
    ///
    /// # Returns
    /// * `Ok(Self)` - The CUDA device with the given index.
    /// * `Err(DeviceError)` - The error when creating the CUDA device.
    pub fn get_cuda(index: usize) -> Result<Self, DeviceError> {
        Ok(Self {
            inner: candle_core::Device::new_cuda(index)?,
        })
    }

    /// Get the CUDA device with the given index if available.
    ///
    /// # Note
    /// If the CUDA device with the given index is not available, the CPU device will be returned.
    ///
    /// # Arguments
    /// * `index` - The index of the CUDA device.
    ///
    /// # Returns
    /// * `Self` - The CUDA device with the given index if available, otherwise the CPU device.
    pub fn get_cuda_if_available(index: usize) -> Self {
        let device =
            candle_core::Device::new_cuda(index).unwrap_or_else(|_| candle_core::Device::Cpu);

        Self { inner: device }
    }

    /// Get the Metal device with the given index.
    ///
    /// # Arguments
    /// * `index` - The index of the Metal device.
    ///
    /// # Returns
    /// * `Ok(Self)` - The Metal device with the given index.
    /// * `Err(DeviceError)` - The error when creating the Metal device.
    pub fn get_metal(index: usize) -> Result<Self, DeviceError> {
        Ok(Self {
            inner: candle_core::Device::new_metal(index)?,
        })
    }

    /// Get the Metal device with the given index if available.
    ///
    /// # Note
    /// If the Metal device with the given index is not available, the CPU device will be returned.
    ///
    /// # Arguments
    /// * `index` - The index of the Metal device.
    ///
    /// # Returns
    /// * `Self` - The Metal device with the given index if available, otherwise the CPU device.
    pub fn get_metal_if_available(index: usize) -> Self {
        Self {
            inner: candle_core::Device::new_metal(index)
                .unwrap_or_else(|_| candle_core::Device::Cpu),
        }
    }
}

impl std::ops::Deref for Device {
    type Target = candle_core::Device;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
