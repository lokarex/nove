#[cfg(any(feature = "cuda", feature = "metal"))]
use thiserror::Error;

/// Error type from device operations.
#[cfg(any(feature = "cuda", feature = "metal"))]
#[derive(Error, Debug)]
pub enum DeviceError {
    /// Error from Candle library.
    #[error("Error from Candle: {0}")]
    CandleError(#[from] candle_core::Error),
}

/// The type of the device.
#[derive(Debug, PartialEq)]
enum DeviceTpye {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda,
    #[cfg(feature = "metal")]
    Metal,
}

/// The `Device` struct to describe the device.
///
/// # Fields
/// * `inner` - The inner device from candle.
/// * `device_type` - The type of the device.
/// * `index` - The index of the device.
#[derive(Debug)]
pub struct Device {
    inner: candle_core::Device,
    device_type: DeviceTpye,
    index: usize,
}

impl Device {
    /// Get the CPU device.
    ///
    /// # Returns
    /// * `Self` - The CPU device.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::Device;
    ///
    /// let cpu = Device::get_cpu();
    /// println!("{:?}", cpu);
    /// ```
    pub fn get_cpu() -> Self {
        Self {
            inner: candle_core::Device::Cpu,
            device_type: DeviceTpye::Cpu,
            index: 0,
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
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::Device;
    ///
    /// let cuda = Device::get_cuda(0);
    /// match cuda {
    ///     Ok(cuda) => println!("{:?}", cuda),
    ///     Err(err) => println!("{:?}", err),
    /// }
    /// ```
    #[cfg(feature = "cuda")]
    pub fn get_cuda(index: usize) -> Result<Self, DeviceError> {
        Ok(Self {
            inner: candle_core::Device::new_cuda(index)?,
            device_type: DeviceTpye::Cuda,
            index,
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
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::Device;
    ///
    /// let device = Device::get_cuda_if_available(0);
    /// println!("{:?}", device);
    /// ```
    #[cfg(feature = "cuda")]
    pub fn get_cuda_if_available(index: usize) -> Self {
        let (device, device_type, index) = match candle_core::Device::new_cuda(index) {
            Ok(device) => (device, DeviceTpye::Cuda, index),
            Err(_) => (candle_core::Device::Cpu, DeviceTpye::Cpu, 0),
        };

        Self {
            inner: device,
            device_type,
            index,
        }
    }

    /// Get the Metal device with the given index.
    ///
    /// # Arguments
    /// * `index` - The index of the Metal device.
    ///
    /// # Returns
    /// * `Ok(Self)` - The Metal device with the given index.
    /// * `Err(DeviceError)` - The error when creating the Metal device.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::Device;
    ///
    /// let metal = Device::get_metal(0);
    /// match metal {
    ///     Ok(metal) => println!("{:?}", metal),
    ///     Err(err) => println!("{:?}", err),
    /// }
    /// ```
    #[cfg(feature = "metal")]
    pub fn get_metal(index: usize) -> Result<Self, DeviceError> {
        Ok(Self {
            inner: candle_core::Device::new_metal(index)?,
            device_type: DeviceTpye::Metal,
            index,
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
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::Device;
    ///
    /// let device = Device::get_metal_if_available(0);
    /// println!("{:?}", device);
    /// ```
    #[cfg(feature = "metal")]
    pub fn get_metal_if_available(index: usize) -> Self {
        let (device, device_type, index) = match candle_core::Device::new_metal(index) {
            Ok(device) => (device, DeviceTpye::Metal, index),
            Err(_) => (candle_core::Device::Cpu, DeviceTpye::Cpu, 0),
        };

        Self {
            inner: device,
            device_type,
            index,
        }
    }
}

impl std::ops::Deref for Device {
    type Target = candle_core::Device;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl PartialEq for Device {
    fn eq(&self, other: &Self) -> bool {
        self.device_type == other.device_type && self.index == other.index
    }
}

impl Eq for Device {}
