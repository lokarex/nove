/// Alias BurnDevice for burn::backend::wgpu::WgpuDevice
type BurnDevice = burn::backend::wgpu::WgpuDevice;

/// The `Device` struct to describe the device.
///
/// # Fields
/// * `inner` - The inner device from burn.
pub struct Device {
    pub inner: BurnDevice,
}

impl Device {
    /// Get the discrete gpu device by index.
    ///
    /// # Arguments
    /// * `index` - The index of the discrete gpu.
    ///
    /// # Returns
    /// A new `Device` instance with the discrete gpu device.
    pub fn discrete_gpu(index: usize) -> Self {
        Self {
            inner: BurnDevice::DiscreteGpu(index),
        }
    }

    /// Get the integrated gpu device by index.
    ///
    /// # Arguments
    /// * `index` - The index of the integrated gpu.
    ///
    /// # Returns
    /// A new `Device` instance with the integrated gpu device.
    pub fn integrated_gpu(index: usize) -> Self {
        Self {
            inner: BurnDevice::IntegratedGpu(index),
        }
    }

    /// Get the default device.
    ///
    /// # Returns
    /// A new `Device` instance with the default device.
    pub fn default_device() -> Self {
        Self {
            inner: BurnDevice::DefaultDevice,
        }
    }
}

impl std::ops::Deref for Device {
    type Target = BurnDevice;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
