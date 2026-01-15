use crate::tensor::{Device, Tensor, TensorError};
use std::collections::HashMap;
use thiserror::Error;

pub mod safetensors;

#[derive(Error, Debug)]
pub enum ParamStoreError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Tensor error: {0}")]
    TensorError(#[from] TensorError),

    #[error("Other error: {0}")]
    OtherError(String),
}

pub trait ParamStore {
    /// Save the parameters in the store to the specified file.
    ///
    /// # Arguments
    /// * `file_path` - The path of the file to save the parameters to.
    ///
    /// # Returns
    /// * `Ok(())` - If the parameters are successfully saved to the file.
    /// * `Err(ParamStoreError)` - The error when saving the parameters to the file.
    fn save(&self, file_path: &str) -> Result<(), ParamStoreError>;

    /// Load the parameters from the specified file to the store.
    ///
    /// # Arguments
    /// * `file_path` - The path of the file to load the parameters from.
    /// * `devices` - The devices to map the loaded parameters to.
    /// * `processor` - The function to process each loaded parameter.
    ///
    /// # Returns
    /// * `Ok(())` - If the parameters are successfully loaded from the file.
    /// * `Err(ParamStoreError)` - The error when loading the parameters from the file.
    fn load(
        &mut self,
        file_path: &str,
        devices: &[Device],
        processor: impl FnMut((&str, &Tensor), &[Device]) -> Result<(), ParamStoreError>,
    ) -> Result<(), ParamStoreError>;

    /// Add a parameter to the store.
    ///
    /// # Arguments
    /// * `name` - The name of the parameter.
    /// * `param` - The parameter to add.
    ///
    /// # Returns
    /// * `Ok(())` - If the parameter is successfully added to the store.
    /// * `Err(ParamStoreError)` - The error when adding the parameter to the store.
    fn add_param(&mut self, name: &str, param: Tensor) -> Result<(), ParamStoreError>;

    /// Update the parameter with the specified name in the store.
    ///
    /// # Arguments
    /// * `name` - The name of the parameter.
    /// * `param` - The parameter after update.
    ///
    /// # Returns
    /// * `Ok(())` - If the parameter is successfully updated in the store.
    /// * `Err(ParamStoreError)` - The error when updating the parameter in the store.
    fn update_param(&mut self, name: &str, param: Tensor) -> Result<(), ParamStoreError>;

    /// Move the parameters in the store to the specified devices.
    ///
    /// # Arguments
    /// * `devices` - The devices to move the parameters to.
    /// * `processor` - The function to process each parameter.
    ///
    /// # Returns
    /// * `Ok(())` - If the parameters are successfully moved to the devices.
    /// * `Err(ParamStoreError)` - The error when moving the parameters to the devices.
    fn to_device(
        &mut self,
        devices: &[Device],
        processor: impl FnMut((&str, &Tensor), &[Device]) -> Result<(), ParamStoreError>,
    ) -> Result<(), ParamStoreError>;

    /// Get the parameters in the store.
    ///
    /// # Returns
    /// * `Ok(Vec<&Tensor>)` - The parameters in the store.
    /// * `Err(ParamStoreError)` - The error when getting the parameters.
    fn parameters(&self) -> Result<Vec<&Tensor>, ParamStoreError>;

    /// Get the named parameters in the store.
    ///
    /// # Returns
    /// * `Ok(HashMap<&str, &Tensor>)` - The named parameters in the store.
    /// * `Err(ParamStoreError)` - The error when getting the named parameters.
    fn named_parameters(&self) -> Result<HashMap<&str, &Tensor>, ParamStoreError>;
}
