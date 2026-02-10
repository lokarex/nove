use nove_tensor::{DType, Device, Tensor, TensorError};
use std::{collections::HashMap, fmt::Display};
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
    #[error("Parameter missing name: {0}")]
    ParameterMissingName(String),

    /// Missing field.
    #[error("Missing field: {0}")]
    MissingField(String),

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

    /// Convert the model to safetensors format data.
    ///
    /// # Returns
    /// * `Ok(HashMap<String, Tensor>)` - The safetensors format data if successful.
    /// * `Err(ModelError)` - The error when converting the model to safetensors format data.
    fn to_safetensors(&self) -> Result<HashMap<String, Tensor>, ModelError>;

    /// Load the model from safetensors format data.
    ///
    /// # Arguments
    /// * `tensors` - The safetensors format data to load the model from.
    ///
    /// # Returns
    /// * `Ok(())` - If successful.
    /// * `Err(ModelError)` - The error when loading the model from safetensors format data.
    fn load_from_safetensors(&mut self, tensors: HashMap<String, Tensor>)
    -> Result<(), ModelError>;
}
