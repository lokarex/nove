use nove_tensor::{DType, Device, Tensor, TensorError};
use std::{collections::HashMap, fmt::Display, path::Path};
use thiserror::Error;

pub mod layer;

#[derive(Error, Debug)]
pub enum ModelError {
    /// I/O errors from the standard library.
    #[error(transparent)]
    IoError(#[from] std::io::Error),

    /// Tensor errors from the `nove_tensor` crate.
    #[error(transparent)]
    TensorError(#[from] TensorError),

    /// Missing parameter.
    #[error("Missing parameter: {0}")]
    MissingParameter(String),

    /// Unexpected parameter.
    #[error("Unexpected parameter: {0}")]
    UnexpectedParameter(String),

    /// Parameter missing name.
    #[error("Parameter missing name")]
    ParameterMissingName(),

    /// Missing field.
    #[error("Missing field: {0}")]
    MissingField(String),

    /// Parameter count mismatch.
    #[error("Parameter count mismatch: expected {expected}, got {actual}")]
    ParameterCountMismatch { expected: usize, actual: usize },

    /// Other errors.
    #[error("{0}")]
    OtherError(String),
}

pub trait Model: Display {
    type Input;
    type Output;

    /// Run the model forward pass.
    ///
    /// # Arguments
    /// * `input` - The input tensor.
    ///
    /// # Returns
    /// * `Ok(Self::Output)` - The output tensor if successful.
    /// * `Err(ModelError)` - The error when running the model forward pass.
    fn forward(&mut self, input: Self::Input) -> Result<Self::Output, ModelError>;

    /// Set whether to enable gradient tracking for the model.
    ///
    /// # Arguments
    /// * `grad_enabled` - Whether to enable gradient tracking for the model.
    ///
    /// # Returns
    /// * `Ok(())` - If successful.
    /// * `Err(ModelError)` - The error when setting the gradient tracking.
    fn require_grad(&mut self, grad_enabled: bool) -> Result<(), ModelError>;

    /// Move the model to the specified device.
    ///
    /// # Arguments
    /// * `device` - The device to move the model to.
    ///
    /// # Returns
    /// * `Ok(())` - If successful.
    /// * `Err(ModelError)` - The error when moving the model to the device.
    fn to_device(&mut self, device: &Device) -> Result<(), ModelError>;

    /// Convert the model to the specified data type.
    ///
    /// # Arguments
    /// * `dtype` - The data type to convert the model to.
    ///
    /// # Returns
    /// * `Ok(())` - If successful.
    /// * `Err(ModelError)` - The error when converting the model to the data type.
    fn to_dtype(&mut self, dtype: &DType) -> Result<(), ModelError>;

    /// Get the parameters of the model.
    ///
    /// # Returns
    /// * `Ok(Vec<Tensor>)` - The parameters of the model.
    /// * `Err(ModelError)` - The error when getting the parameters of the model.
    fn parameters(&self) -> Result<Vec<Tensor>, ModelError>;

    /// Get the named parameters of the model.
    ///
    /// # Returns
    /// * `Ok(HashMap<String, Tensor>)` - The named parameters of the model.
    /// * `Err(ModelError)` - The error when getting the named parameters of the model.
    fn named_parameters(&self) -> Result<HashMap<String, Tensor>, ModelError> {
        Ok(self
            .parameters()?
            .into_iter()
            .map(|t| match t.name()? {
                Some(name) => Ok((name, t)),
                None => Err(ModelError::ParameterMissingName()),
            })
            .collect::<Result<HashMap<_, _>, ModelError>>()?)
    }

    /// Save the model parameters to a file.
    ///
    /// # Arguments
    /// * `file_path` - The path to the file.
    ///
    /// # Returns
    /// * `Ok(())` - If successful.
    /// * `Err(ModelError)` - The error when saving the model parameters to the file.
    fn save(&self, file_path: &str) -> Result<(), ModelError> {
        let file = Path::new(file_path);
        match file.extension().and_then(|ext| ext.to_str()) {
            Some("safetensors") => {
                let params = self.named_parameters()?;
                nove_tensor::safetensor::save(file_path, params)?;
            }
            Some(ext) => {
                return Err(ModelError::OtherError(format!(
                    "Unsupported file extension: {}",
                    ext
                )));
            }
            None => {
                return Err(ModelError::OtherError(
                    "File extension not found".to_string(),
                ));
            }
        }
        Ok(())
    }

    /// Load the model parameters from a file.
    ///
    /// # Arguments
    /// * `file_path` - The path to the file.
    /// * `device` - The device to load the model parameters to.
    ///
    /// # Returns
    /// * `Ok(())` - If successful.
    /// * `Err(ModelError)` - The error when loading the model parameters from the file.
    fn load(&mut self, file_path: &str, device: &Device) -> Result<(), ModelError> {
        let file = Path::new(file_path);
        match file.extension().and_then(|ext| ext.to_str()) {
            Some("safetensors") => {
                let new_params = nove_tensor::safetensor::load(file_path, device)?;
                let old_params = self.named_parameters()?;

                if new_params.len() != old_params.len() {
                    return Err(ModelError::ParameterCountMismatch {
                        expected: old_params.len(),
                        actual: new_params.len(),
                    });
                }

                for (name, new_param) in new_params {
                    let old_param = old_params
                        .get(&name)
                        .ok_or(ModelError::UnexpectedParameter(name))?;

                    old_param.update_from_tensor(&new_param)?;
                }
            }
            Some(ext) => {
                return Err(ModelError::OtherError(format!(
                    "Unsupported file extension: {}",
                    ext
                )));
            }
            None => {
                return Err(ModelError::OtherError(
                    "File extension not found".to_string(),
                ));
            }
        }
        Ok(())
    }
}
