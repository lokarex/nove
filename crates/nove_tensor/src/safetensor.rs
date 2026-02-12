use std::collections::HashMap;

use crate::{Device, Tensor, TensorError};

/// Save the tensors to a safetensors file.
///
/// # Arguments
/// * `file_path` - The path to the safetensors file.
/// * `tensors` - The tensors to be saved.
///
/// # Returns
/// * `Ok(())` - If the tensors are saved successfully.
/// * `Err(TensorError)` - If there is an error while saving the tensors.
pub fn save(file_path: &str, tensors: HashMap<String, Tensor>) -> Result<(), TensorError> {
    let candle_tensor = tensors
        .iter()
        .map(|(name, t)| {
            let name = name;
            let tensor = t.to_candle_tensor()?;
            Ok((name, tensor))
        })
        .collect::<Result<HashMap<_, _>, TensorError>>()?;
    candle_core::safetensors::save(&candle_tensor, file_path)?;
    Ok(())
}

/// Load the tensors from a safetensors file.
///
/// # Arguments
/// * `file_path` - The path to the safetensors file.
/// * `device` - The device to load the tensors on.
///
/// # Returns
/// * `Ok(HashMap<String, Tensor>)` - If the tensors are loaded successfully.
/// * `Err(TensorError)` - If there is an error while loading the tensors.
pub fn load(file_path: &str, device: &Device) -> Result<HashMap<String, Tensor>, TensorError> {
    let candle_tensor = candle_core::safetensors::load(file_path, device)?;
    let tensors = candle_tensor
        .iter()
        .map(|(name, tensor)| {
            let name = name.to_string();
            let tensor = Tensor::from_candle_tensor(tensor.clone(), device, false)?;
            Ok((name, tensor))
        })
        .collect::<Result<HashMap<_, _>, TensorError>>()?;
    Ok(tensors)
}
