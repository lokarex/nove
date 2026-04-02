use std::fmt::Display;
use thiserror::Error;

/// Error type from device operations.
#[derive(Error, Debug)]
pub enum DeviceError {
    /// Error from Candle library.
    #[error("Error from Candle: {0}")]
    CandleError(#[from] candle_core::Error),

    /// Backend feature is not enabled.
    #[error("Backend feature is not enabled: {0}")]
    BackendFeatureNotEnabled(String),

    /// Other errors.
    #[error("Other error: {0}")]
    OtherError(String),
}

/// The type of the device.
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum DeviceKind {
    Cpu,
    Cuda,
    Metal,
}

/// The `Device` struct to describe the device.
///
/// # Fields
/// * `inner` - The inner device from candle.
/// * `kind` - The kind of compute device.
/// * `index` - The index of the device.
#[derive(Debug, Clone)]
pub struct Device {
    inner: candle_core::Device,
    kind: DeviceKind,
    index: usize,
}

impl Device {
    /// Get the CPU device.
    ///
    /// # Returns
    /// * `Self` - The CPU device.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::tensor::Device;
    ///
    /// let cpu = Device::cpu();
    /// println!("{:?}", cpu);
    /// ```
    pub fn cpu() -> Self {
        Self {
            inner: candle_core::Device::Cpu,
            kind: DeviceKind::Cpu,
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
    /// ```no_run
    /// use nove::tensor::Device;
    ///
    /// let cuda = Device::cuda(0);
    /// match cuda {
    ///     Ok(cuda) => println!("{:?}", cuda),
    ///     Err(err) => println!("{:?}", err),
    /// }
    /// ```
    pub fn cuda(index: usize) -> Result<Self, DeviceError> {
        #[cfg(feature = "cuda")]
        return Ok(Self {
            inner: candle_core::Device::new_cuda(index)?,
            kind: DeviceKind::Cuda,
            index,
        });
        #[cfg(not(feature = "cuda"))]
        {
            let _ = index;
            Err(DeviceError::BackendFeatureNotEnabled("cuda".to_string()))
        }
    }

    /// Get the CUDA device with the given index if available.
    ///
    /// # Notes
    /// If the CUDA device with the given index is not available, the CPU device will be returned.
    ///
    /// # Arguments
    /// * `index` - The index of the CUDA device.
    ///
    /// # Returns
    /// * `Self` - The CUDA device with the given index if available, otherwise the CPU device.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::tensor::Device;
    ///
    /// let device = Device::cuda_if_available(0);
    /// println!("{:?}", device);
    /// ```
    pub fn cuda_if_available(index: usize) -> Self {
        let (device, kind, index) = match candle_core::Device::new_cuda(index) {
            Ok(device) => (device, DeviceKind::Cuda, index),
            Err(_) => (candle_core::Device::Cpu, DeviceKind::Cpu, 0),
        };

        Self {
            inner: device,
            kind,
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
    /// ```no_run
    /// use nove::tensor::Device;
    ///
    /// let metal = Device::metal(0);
    /// match metal {
    ///     Ok(metal) => println!("{:?}", metal),
    ///     Err(err) => println!("{:?}", err),
    /// }
    /// ```
    pub fn metal(index: usize) -> Result<Self, DeviceError> {
        #[cfg(feature = "metal")]
        return Ok(Self {
            inner: candle_core::Device::new_metal(index)?,
            kind: DeviceKind::Metal,
            index,
        });
        #[cfg(not(feature = "metal"))]
        {
            let _ = index;
            Err(DeviceError::BackendFeatureNotEnabled("metal".to_string()))
        }
    }

    /// Get the Metal device with the given index if available.
    ///
    /// # Notes
    /// If the Metal device with the given index is not available, the CPU device will be returned.
    ///
    /// # Arguments
    /// * `index` - The index of the Metal device.
    ///
    /// # Returns
    /// * `Self` - The Metal device with the given index if available, otherwise the CPU device.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::tensor::Device;
    ///
    /// let device = Device::metal_if_available(0);
    /// println!("{:?}", device);
    /// ```
    pub fn metal_if_available(index: usize) -> Self {
        let (device, kind, index) = match candle_core::Device::new_metal(index) {
            Ok(device) => (device, DeviceKind::Metal, index),
            Err(_) => (candle_core::Device::Cpu, DeviceKind::Cpu, 0),
        };

        Self {
            inner: device,
            kind,
            index,
        }
    }

    /// Get the kind of the device.
    ///
    /// # Returns
    /// * `DeviceKind` - The kind of the device.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::tensor::Device;
    ///
    /// let device = Device::cpu();
    /// let kind = device.kind();
    /// println!("{:?}", kind);
    /// ```
    pub fn kind(&self) -> DeviceKind {
        self.kind
    }

    /// Get the index of the device.
    ///
    /// # Returns
    /// * `usize` - The index of the device.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::tensor::Device;
    ///
    /// let device = Device::cpu();
    /// let index = device.index();
    /// println!("{}", index);
    /// ```
    pub fn index(&self) -> usize {
        self.index
    }

    /// Check if the device is a CPU device.
    ///
    /// # Returns
    /// * `bool` - `true` if the device is a CPU device, otherwise `false`.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::tensor::Device;
    ///
    /// let device = Device::cpu();
    /// let is_cpu = device.is_cpu();
    /// println!("{}", is_cpu);
    /// ```
    pub fn is_cpu(&self) -> bool {
        self.kind == DeviceKind::Cpu
    }

    /// Check if the device is a CUDA device.
    ///
    /// # Returns
    /// * `bool` - `true` if the device is a CUDA device, otherwise `false`.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::tensor::Device;
    ///
    /// let device = Device::cuda_if_available(0);
    /// let is_cuda = device.is_cuda();
    /// println!("{}", is_cuda);
    /// ```
    pub fn is_cuda(&self) -> bool {
        self.kind == DeviceKind::Cuda
    }

    /// Check if the device is a Metal device.
    ///
    /// # Returns
    /// * `bool` - `true` if the device is a Metal device, otherwise `false`.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::tensor::Device;
    ///
    /// let device = Device::metal_if_available(0);
    /// let is_metal = device.is_metal();
    /// println!("{}", is_metal);
    /// ```
    pub fn is_metal(&self) -> bool {
        self.kind == DeviceKind::Metal
    }

    /// Synchronize the device.
    ///
    /// This method blocks until all previously submitted operations on the device
    /// have completed. This is useful for ensuring that GPU operations are finished
    /// before proceeding, which can help with memory management.
    ///
    /// # Returns
    /// * `Ok(())` - The device is synchronized successfully.
    /// * `Err(DeviceError)` - The error when synchronizing the device.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::tensor::Device;
    ///
    /// let device = Device::cpu();
    /// device.synchronize().unwrap();
    /// ```
    pub fn synchronize(&self) -> Result<(), DeviceError> {
        self.inner.synchronize()?;
        Ok(())
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
        self.kind == other.kind && self.index == other.index
    }
}

impl Eq for Device {}

impl Display for DeviceKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceKind::Cpu => write!(f, "cpu"),
            DeviceKind::Cuda => write!(f, "cuda"),
            DeviceKind::Metal => write!(f, "metal"),
        }
    }
}

impl Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.kind {
            DeviceKind::Cpu => write!(f, "{}", self.kind),
            DeviceKind::Cuda => write!(f, "{}({})", self.kind, self.index),
            DeviceKind::Metal => write!(f, "{}({})", self.kind, self.index),
        }
    }
}
