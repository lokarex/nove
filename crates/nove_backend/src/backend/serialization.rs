use super::BackendStorage;
#[cfg(feature = "candle")]
use super::candle;
use crate::{BackendError, BackendKind, Device};
use std::collections::HashMap;

/// Saves backend storage values to a safetensors file.
///
/// # Notes
/// This helper is part of the backend dispatch boundary used by
/// `nove_tensor::safetensor`. When Candle is enabled, Native storage values
/// are saved through temporary Candle CPU storage. Without Candle, saving
/// is reported as unsupported for Native.
///
/// # Arguments
/// * `file_path` - Destination path for the safetensors file.
/// * `tensors` - Named backend storage values to serialize.
///
/// # Returns
/// * `Ok(())` - The file was written successfully.
/// * `Err(BackendError)` - The backend does not support the operation or failed.
///
/// # Examples
/// ```no_run
/// use std::collections::HashMap;
/// use nove_backend::backend;
///
/// backend::save_safetensors("model.safetensors", HashMap::new()).unwrap();
/// ```
pub fn save_safetensors(
    file_path: &str,
    tensors: HashMap<String, BackendStorage>,
) -> Result<(), BackendError> {
    #[cfg(feature = "candle")]
    {
        candle::save_safetensors(file_path, tensors)
    }
    #[cfg(not(feature = "candle"))]
    {
        Err(BackendError::UnsupportedOperation {
            backend: BackendKind::Native,
            operation: "save_safetensors".to_string(),
        })
    }
}

/// Loads backend storage values from a safetensors file.
///
/// # Notes
/// Loaded storage is created on the backend selected by `device`.
/// Loading into Native is reported as unsupported before reading the file.
///
/// # Arguments
/// * `file_path` - Source safetensors file path.
/// * `device` - The target Nove device descriptor.
///
/// # Returns
/// * `Ok(HashMap<String, BackendStorage>)` - Named backend storage values.
/// * `Err(BackendError)` - The backend does not support the operation or failed.
///
/// # Examples
/// ```no_run
/// use nove_backend::{backend, device};
///
/// let device = device::candle::cpu().unwrap();
/// let tensors = backend::load_safetensors("model.safetensors", &device).unwrap();
/// assert!(tensors.is_empty() || !tensors.is_empty());
/// ```
pub fn load_safetensors(
    file_path: &str,
    device: &Device,
) -> Result<HashMap<String, BackendStorage>, BackendError> {
    match device.backend() {
        #[cfg(feature = "candle")]
        BackendKind::Candle => candle::load_safetensors(file_path, device),
        #[cfg(not(feature = "candle"))]
        BackendKind::Candle => Err(BackendError::UnsupportedBackend(BackendKind::Candle)),
        BackendKind::Native => Err(BackendError::UnsupportedOperation {
            backend: BackendKind::Native,
            operation: "load_safetensors".to_string(),
        }),
    }
}
