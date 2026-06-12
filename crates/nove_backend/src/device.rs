//! Device descriptors and explicit backend-specific constructors.
//!
//! Devices in Nove always carry both a backend and a device kind. Use the
//! constructors in [`candle`] or [`native`] so call sites make the backend
//! choice visible.

use crate::backend::{BackendError, BackendKind};
use std::fmt::Display;
use thiserror::Error;

/// Error type for device construction and synchronization operations.
///
/// # Notes
/// Feature-gated constructors return [`DeviceError::BackendFeatureNotEnabled`]
/// when the requested backend/device capability is not compiled into the
/// crate.
#[derive(Error, Debug)]
pub enum DeviceError {
    /// Error propagated from a backend implementation.
    #[error("Backend error: {0}")]
    BackendError(#[from] BackendError),

    /// The requested backend/device feature is not enabled.
    #[error("Backend feature is not enabled: {0}")]
    BackendFeatureNotEnabled(String),

    /// Any other device-level error.
    #[error("Other error: {0}")]
    OtherError(String),
}

/// The hardware class used by a Nove device.
///
/// # Examples
/// ```
/// use nove_backend::DeviceKind;
///
/// assert_eq!(DeviceKind::Cpu.to_string(), "cpu");
/// assert_eq!(DeviceKind::Cuda.to_string(), "cuda");
/// assert_eq!(DeviceKind::Metal.to_string(), "metal");
/// ```
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum DeviceKind {
    /// Host CPU memory and execution.
    Cpu,
    /// CUDA GPU memory and execution.
    Cuda,
    /// Apple Metal GPU memory and execution.
    Metal,
}

/// Backend-qualified device descriptor used by tensors and backend storage.
///
/// # Notes
/// A device is constructed through explicit backend-specific helpers in [`candle`] or [`native`].
/// For tests and common cases use [`Device::default()`] which selects the
/// highest-priority enabled backend.
#[derive(Debug, Clone)]
pub struct Device {
    backend: BackendKind,
    kind: DeviceKind,
    index: usize,
    #[allow(dead_code)]
    handle: BackendDeviceHandle,
}

#[derive(Debug, Clone)]
pub(crate) enum BackendDeviceHandle {
    #[cfg(feature = "candle")]
    Candle(nove_candle::CandleDevice),
    #[cfg(feature = "native")]
    Native,
}

impl BackendDeviceHandle {
    fn backend(&self) -> BackendKind {
        match self {
            #[cfg(feature = "candle")]
            Self::Candle(_) => BackendKind::Candle,
            #[cfg(feature = "native")]
            Self::Native => BackendKind::Native,
        }
    }
}

impl Device {
    pub(crate) fn from_handle(handle: BackendDeviceHandle, kind: DeviceKind, index: usize) -> Self {
        Self {
            backend: handle.backend(),
            kind,
            index,
            handle,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn handle(&self) -> &BackendDeviceHandle {
        &self.handle
    }

    /// Returns the backend selected for this device.
    ///
    /// # Returns
    /// * [`BackendKind`] - The backend implementation that owns tensors on this device.
    ///
    /// # Examples
    /// ```
    /// use nove_backend::{BackendKind, device, Device};
    ///
    /// let device = Device::default();
    /// assert!(matches!(device.backend(), BackendKind::Candle | BackendKind::Native));
    /// ```
    pub fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Returns the hardware kind of the device.
    ///
    /// # Returns
    /// * [`DeviceKind`] - The hardware class used by the device.
    ///
    /// # Examples
    /// ```
    /// use nove_backend::{DeviceKind, device};
    ///
    /// #[cfg(feature = "candle-cpu")]
    /// {
    ///     let device = device::candle::cpu().unwrap();
    ///     assert_eq!(device.kind(), DeviceKind::Cpu);
    /// }
    /// #[cfg(feature = "native-cpu")]
    /// {
    ///     let device = device::native::cpu().unwrap();
    ///     assert_eq!(device.kind(), DeviceKind::Cpu);
    /// }
    /// ```
    pub fn kind(&self) -> DeviceKind {
        self.kind
    }

    /// Returns the device index.
    ///
    /// CPU devices use index `0`. CUDA and Metal devices use the backend
    /// device index passed to their constructors.
    ///
    /// # Returns
    /// * `usize` - The device index.
    ///
    /// # Examples
    /// ```
    /// use nove_backend::device;
    ///
    /// #[cfg(feature = "candle-cpu")]
    /// {
    ///     let device = device::candle::cpu().unwrap();
    ///     assert_eq!(device.index(), 0);
    /// }
    /// #[cfg(feature = "native-cpu")]
    /// {
    ///     let device = device::native::cpu().unwrap();
    ///     assert_eq!(device.index(), 0);
    /// }
    /// ```
    pub fn index(&self) -> usize {
        self.index
    }

    /// Returns true when this device uses CPU execution.
    ///
    /// # Returns
    /// * `bool` - `true` for [`DeviceKind::Cpu`], otherwise `false`.
    ///
    /// # Examples
    /// ```
    /// use nove_backend::device;
    ///
    /// #[cfg(feature = "candle-cpu")]
    /// {
    ///     let device = device::candle::cpu().unwrap();
    ///     assert!(device.is_cpu());
    /// }
    /// #[cfg(feature = "native-cpu")]
    /// {
    ///     let device = device::native::cpu().unwrap();
    ///     assert!(device.is_cpu());
    /// }
    /// ```
    pub fn is_cpu(&self) -> bool {
        self.kind == DeviceKind::Cpu
    }

    /// Returns true when this device uses CUDA execution.
    ///
    /// # Returns
    /// * `bool` - `true` for [`DeviceKind::Cuda`], otherwise `false`.
    ///
    /// # Examples
    /// ```
    /// use nove_backend::device;
    ///
    /// #[cfg(feature = "candle-cuda")]
    /// {
    ///     let device = device::candle::cuda(0).unwrap();
    ///     assert!(device.is_cuda());
    /// }
    /// ```
    pub fn is_cuda(&self) -> bool {
        self.kind == DeviceKind::Cuda
    }

    /// Returns true when this device uses Metal execution.
    ///
    /// # Returns
    /// * `bool` - `true` for [`DeviceKind::Metal`], otherwise `false`.
    ///
    /// # Examples
    /// ```
    /// use nove_backend::device;
    ///
    /// #[cfg(feature = "candle-metal")]
    /// {
    ///     let device = device::candle::metal(0).unwrap();
    ///     assert!(device.is_cuda());
    /// }
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
    /// * `Ok(())` - The device was synchronized successfully.
    /// * `Err(DeviceError)` - The error returned by the selected backend.
    ///
    /// # Examples
    /// ```
    /// use nove_backend::{device, Device};
    ///
    /// let device = Device::default();
    /// device.synchronize().unwrap();
    /// ```
    pub fn synchronize(&self) -> Result<(), DeviceError> {
        match self.backend {
            #[cfg(feature = "candle")]
            BackendKind::Candle => {
                crate::backend::candle::synchronize_device(self)?;
            }
            #[cfg(not(feature = "candle"))]
            BackendKind::Candle => {}
            #[cfg(feature = "native")]
            BackendKind::Native => {
                crate::backend::native::synchronize_device(self)?;
            }
            #[cfg(not(feature = "native"))]
            BackendKind::Native => {}
        }
        Ok(())
    }
}

/// Candle-backed device constructors.
///
/// # Notes
/// `*_if_available` constructors fall back only to Candle CPU. They never
/// silently switch to the native Nove CPU backend.
pub mod candle {
    #[cfg(feature = "candle")]
    use super::{BackendDeviceHandle, DeviceKind};
    use super::{Device, DeviceError};

    /// Creates a Candle-backed CPU device.
    ///
    /// # Returns
    /// * `Ok(Device)` - A Candle CPU device when the `candle-cpu` feature is enabled.
    /// * `Err(DeviceError)` - A feature error when `candle-cpu` is disabled.
    ///
    /// # Examples
    /// ```
    /// use nove_backend::{BackendKind, Device, DeviceKind, device};
    ///
    /// # #[cfg(feature = "candle-cpu")]
    /// # {
    /// let device = device::candle::cpu().unwrap();
    ///
    /// assert_eq!(device.backend(), BackendKind::Candle);
    /// assert_eq!(device.kind(), DeviceKind::Cpu);
    /// # }
    /// ```
    pub fn cpu() -> Result<Device, DeviceError> {
        #[cfg(feature = "candle-cpu")]
        {
            let handle = crate::backend::candle::create_candle_device(DeviceKind::Cpu, 0)?;
            Ok(Device::from_handle(
                BackendDeviceHandle::Candle(handle),
                DeviceKind::Cpu,
                0,
            ))
        }

        #[cfg(not(feature = "candle-cpu"))]
        {
            Err(DeviceError::BackendFeatureNotEnabled(
                "candle-cpu".to_string(),
            ))
        }
    }

    /// Creates a Candle-backed CUDA device with the given index.
    ///
    /// # Arguments
    /// * `index` - The CUDA device index.
    ///
    /// # Returns
    /// * `Ok(Device)` - A Candle CUDA device when available.
    /// * `Err(DeviceError)` - A feature or backend validation error.
    ///
    /// # Examples
    /// ```
    /// use nove_backend::{device, BackendKind};
    ///
    /// #[cfg(feature = "candle-cuda")]
    /// {
    ///     let device = device::candle::cuda(0).unwrap();
    ///     assert_eq!(device.backend(), BackendKind::Candle);
    ///     assert!(device.is_cuda());
    /// }
    /// ```
    pub fn cuda(index: usize) -> Result<Device, DeviceError> {
        #[cfg(feature = "candle-cuda")]
        {
            let handle = crate::backend::candle::create_candle_device(DeviceKind::Cuda, index)?;
            Ok(Device::from_handle(
                BackendDeviceHandle::Candle(handle),
                DeviceKind::Cuda,
                index,
            ))
        }

        #[cfg(not(feature = "candle-cuda"))]
        {
            let _ = index;
            Err(DeviceError::BackendFeatureNotEnabled(
                "candle-cuda".to_string(),
            ))
        }
    }

    /// Creates a Candle-backed CUDA device when available, otherwise Candle CPU.
    ///
    /// # Arguments
    /// * `index` - The CUDA device index to try first.
    ///
    /// # Returns
    /// * `Ok(Device)` - A Candle CUDA device, or Candle CPU if CUDA is unavailable.
    /// * `Err(DeviceError)` - A feature error when `candle-cuda` is disabled.
    ///
    /// # Examples
    /// ```
    /// use nove_backend::{BackendKind, device};
    ///
    /// #[cfg(feature = "candle-cuda")]
    /// {
    ///     let device = device::candle::cuda_if_available(0).unwrap();
    ///     assert_eq!(device.backend(), BackendKind::Candle);
    /// }
    /// ```
    pub fn cuda_if_available(index: usize) -> Result<Device, DeviceError> {
        #[cfg(feature = "candle-cuda")]
        {
            cuda(index).or_else(|_| cpu())
        }

        #[cfg(not(feature = "candle-cuda"))]
        {
            let _ = index;
            Err(DeviceError::BackendFeatureNotEnabled(
                "candle-cuda".to_string(),
            ))
        }
    }

    /// Creates a Candle-backed Metal device with the given index.
    ///
    /// # Arguments
    /// * `index` - The Metal device index.
    ///
    /// # Returns
    /// * `Ok(Device)` - A Candle Metal device when available.
    /// * `Err(DeviceError)` - A feature or backend validation error.
    ///
    /// # Examples
    /// ```
    /// use nove_backend::{device, BackendKind};
    ///
    /// #[cfg(feature = "candle-metal")]
    /// {
    ///     let device = device::candle::metal(0).unwrap();
    ///     assert_eq!(device.backend(), BackendKind::Candle);
    ///     assert!(device.is_metal());
    /// }
    /// ```
    pub fn metal(index: usize) -> Result<Device, DeviceError> {
        #[cfg(feature = "candle-metal")]
        {
            let handle = crate::backend::candle::create_candle_device(DeviceKind::Metal, index)?;
            Ok(Device::from_handle(
                BackendDeviceHandle::Candle(handle),
                DeviceKind::Metal,
                index,
            ))
        }

        #[cfg(not(feature = "candle-metal"))]
        {
            let _ = index;
            Err(DeviceError::BackendFeatureNotEnabled(
                "candle-metal".to_string(),
            ))
        }
    }

    /// Creates a Candle-backed Metal device when available, otherwise Candle CPU.
    ///
    /// # Arguments
    /// * `index` - The Metal device index to try first.
    ///
    /// # Returns
    /// * `Ok(Device)` - A Candle Metal device, or Candle CPU if Metal is unavailable.
    /// * `Err(DeviceError)` - A feature error when `candle-metal` is disabled.
    ///
    /// # Examples
    /// ```
    /// use nove_backend::{BackendKind, device};
    ///
    /// #[cfg(feature = "candle-metal")]
    /// {
    ///     let device = candle::metal_if_available(0).unwrap();
    ///     assert_eq!(device.backend(), BackendKind::Candle);
    /// }
    /// ```
    pub fn metal_if_available(index: usize) -> Result<Device, DeviceError> {
        #[cfg(feature = "candle-metal")]
        {
            metal(index).or_else(|_| cpu())
        }

        #[cfg(not(feature = "candle-metal"))]
        {
            let _ = index;
            Err(DeviceError::BackendFeatureNotEnabled(
                "candle-metal".to_string(),
            ))
        }
    }
}

impl Default for Device {
    /// Creates the default CPU device using the highest-priority enabled backend.
    ///
    /// # Notes
    /// Priority: `candle-cuda` > `candle-metal` > `candle-cpu` > `native-cpu`.
    ///
    /// # Panics
    /// * Panics if the selected GPU device is unavailable (e.g. `candle-cuda`
    ///   enabled but no CUDA device present).
    /// * Panics if no backend device feature is enabled.
    #[allow(unreachable_code)]
    fn default() -> Self {
        #[cfg(feature = "candle-cuda")]
        {
            return candle::cuda(0).unwrap();
        }
        #[cfg(feature = "candle-metal")]
        {
            return candle::metal(0).unwrap();
        }
        #[cfg(feature = "candle-cpu")]
        {
            return candle::cpu().unwrap();
        }
        #[cfg(feature = "native-cpu")]
        {
            return native::cpu().unwrap();
        }
        panic!("Device::default() requires at least one backend feature to be enabled");
    }
}

/// Nove native device constructors.
pub mod native {
    #[cfg(feature = "native-cpu")]
    use super::{BackendDeviceHandle, DeviceKind};
    use super::{Device, DeviceError};

    /// Creates a native Nove CPU device.
    ///
    /// # Returns
    /// * `Ok(Device)` - A native Nove CPU device when the `native-cpu` feature is enabled.
    /// * `Err(DeviceError)` - A feature error when `native-cpu` is disabled.
    ///
    /// # Examples
    /// ```
    /// use nove_backend::{BackendKind, Device, DeviceKind, device};
    ///
    /// #[cfg(feature = "native-cpu")]
    /// {
    ///     let device = device::native::cpu().unwrap();
    ///
    ///     assert_eq!(device.backend(), BackendKind::Native);
    ///     assert_eq!(device.kind(), DeviceKind::Cpu);
    /// }
    /// ```
    pub fn cpu() -> Result<Device, DeviceError> {
        #[cfg(feature = "native-cpu")]
        {
            Ok(Device::from_handle(
                BackendDeviceHandle::Native,
                DeviceKind::Cpu,
                0,
            ))
        }

        #[cfg(not(feature = "native-cpu"))]
        {
            Err(DeviceError::BackendFeatureNotEnabled(
                "native-cpu".to_string(),
            ))
        }
    }
}

impl PartialEq for Device {
    fn eq(&self, other: &Self) -> bool {
        self.backend == other.backend && self.kind == other.kind && self.index == other.index
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
            DeviceKind::Cuda | DeviceKind::Metal => write!(f, "{}({})", self.kind, self.index),
        }
    }
}
