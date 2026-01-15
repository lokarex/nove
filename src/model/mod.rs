use crate::{
    model::paramstore::ParamStore,
    tensor::{Device, Tensor},
};
use thiserror::Error;

pub mod paramstore;

#[derive(Error, Debug)]
pub enum ModelError {
    #[error("IO error: {0}")]
    IoError(std::io::Error),

    #[error("Other error: {0}")]
    OtherError(String),
}

pub trait Model {
    type ParamStore: ParamStore;
    type Input;
    type Output;

    /// Get the parameter store of the model.
    ///
    /// # Returns
    /// * `Ok(&Self::ParamStore)` - The parameter store of the model.
    /// * `Err(ModelError)` - The error when getting the parameter store.
    fn param_store(&self) -> Result<&Self::ParamStore, ModelError>;

    /// Get the mutable parameter store of the model.
    ///
    /// # Returns
    /// * `Ok(&mut Self::ParamStore)` - The mutable parameter store of the model.
    /// * `Err(ModelError)` - The error when getting the mutable parameter store.
    fn param_store_mut(&mut self) -> Result<&mut Self::ParamStore, ModelError>;

    /// Perform a forward pass of the model.
    ///
    /// # Arguments
    /// * `input` - The input data for the forward pass.
    ///
    /// # Returns
    /// * `Ok(Self::Output)` - The output data of the forward pass.
    /// * `Err(ModelError)` - The error when performing the forward pass.
    fn forward(&self, input: Self::Input) -> Result<Self::Output, ModelError>;

    /// Move the model to the specified devices.
    ///
    /// # Arguments
    /// * `devices` - The devices to move the model to.
    /// * `processor` - The function to process each parameter.
    ///
    /// # Returns
    /// * `Ok(())` - If the model is successfully moved to the devices.
    /// * `Err(ModelError)` - The error when moving the model to the devices.
    fn to_devices(
        &mut self,
        devices: &[Device],
        processor: dyn FnMut((&str, &Tensor), &Device),
    ) -> Result<(), ModelError>;

    /// Save the model to the specified file.
    ///
    /// # Arguments
    /// * `file_path` - The path of the file to save the model to.
    ///
    /// # Returns
    /// * `Ok(())` - If the model is successfully saved to the file.
    /// * `Err(ModelError)` - The error when saving the model to the file.
    fn save(&self, file_path: &str) -> Result<(), ModelError>;

    /// Load the model from the specified file.
    ///
    /// # Arguments
    /// * `file_path` - The path of the file to load the model from.
    /// * `devices` - The devices to map the loaded parameters to.
    /// * `processor` - The function to process each loaded parameter.
    ///
    /// # Returns
    /// * `Ok(())` - If the model is successfully loaded from the file.
    /// * `Err(ModelError)` - The error when loading the model from the file.
    fn load(
        &mut self,
        file_path: &str,
        devices: &[Device],
        processor: dyn FnMut((&str, &Tensor), &Device),
    ) -> Result<(), ModelError>;

    /// Set the model to training mode.
    ///
    /// # Returns
    /// * `Ok(())` - If the model is successfully set to training mode.
    /// * `Err(ModelError)` - The error when setting the model to training mode.
    fn to_train(&mut self) -> Result<(), ModelError>;

    /// Set the model to evaluation mode.
    ///
    /// # Returns
    /// * `Ok(())` - If the model is successfully set to evaluation mode.
    /// * `Err(ModelError)` - The error when setting the model to evaluation mode.
    fn to_eval(&mut self) -> Result<(), ModelError>;
}
