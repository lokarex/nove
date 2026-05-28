use std::collections::HashMap;

use crate::{Device, Tensor, TensorError, backend};

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
    let storages = tensors
        .into_iter()
        .map(|(name, tensor)| Ok((name, tensor.backend_storage()?)))
        .collect::<Result<HashMap<_, _>, TensorError>>()?;
    backend::save_safetensors(file_path, storages)?;
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
    let storages = backend::load_safetensors(file_path, device)?;
    Ok(storages
        .into_iter()
        .map(|(name, storage)| {
            (
                name,
                Tensor::from_backend_storage(
                    storage,
                    device.clone(),
                    false,
                    vec![],
                    crate::backpropagation::graph::OpKind::Leaf,
                ),
            )
        })
        .collect())
}
