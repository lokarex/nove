use std::collections::HashMap;

use thiserror::Error;

use crate::tensor::{Device, Tensor};

#[derive(Error, Debug)]
pub enum ParamStoreError {
    #[error("IO error: {0}")]
    IoError(std::io::Error),

    #[error("Other error: {0}")]
    OtherError(String),
}

pub trait ParamStore {
    /// Save the parameters in the store to the specified file.
    ///
    /// # Arguments
    /// * `path` - The path of the file to save the parameters to.
    ///
    /// # Returns
    /// * `Ok(())` - If the parameters are successfully saved to the file.
    /// * `Err(ParamStoreError)` - The error when saving the parameters to the file.
    fn save(&self, path: &str) -> Result<(), ParamStoreError>;

    /// Load the parameters from the specified file to the store.
    ///
    /// # Arguments
    /// * `path` - The path of the file to load the parameters from.
    /// * `devices` - The devices to map the loaded parameters to.
    /// * `processor` - The function to process each loaded parameter.
    ///
    /// # Returns
    /// * `Ok(())` - If the parameters are successfully loaded from the file.
    /// * `Err(ParamStoreError)` - The error when loading the parameters from the file.
    fn load(
        &mut self,
        path: &str,
        devices: &[Device],
        processor: dyn FnMut((&str, Tensor), &Device),
    ) -> Result<(), ParamStoreError>;

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
        processor: dyn FnMut((&str, Tensor), &Device),
    ) -> Result<(), ParamStoreError>;

    /// Get the parameters in the store.
    ///
    /// # Returns
    /// * `Ok(Vec<Tensor>)` - The parameters in the store.
    /// * `Err(ParamStoreError)` - The error when getting the parameters.
    fn parameters(&self) -> Result<Vec<Tensor>, ParamStoreError>;

    /// Get the named parameters in the store.
    ///
    /// # Returns
    /// * `Ok(HashMap<&str, Tensor>)` - The named parameters in the store.
    /// * `Err(ParamStoreError)` - The error when getting the named parameters.
    fn named_parameters(&self) -> Result<HashMap<&str, Tensor>, ParamStoreError>;
}

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
    fn to_device(
        &mut self,
        devices: &[Device],
        processor: dyn FnMut((&str, Tensor), &Device),
    ) -> Result<(), ModelError>;

    /// Save the model to the specified file.
    ///
    /// # Arguments
    /// * `path` - The path of the file to save the model to.
    ///
    /// # Returns
    /// * `Ok(())` - If the model is successfully saved to the file.
    /// * `Err(ModelError)` - The error when saving the model to the file.
    fn save(&self, path: &str) -> Result<(), ModelError>;

    /// Load the model from the specified file.
    ///
    /// # Arguments
    /// * `path` - The path of the file to load the model from.
    /// * `devices` - The devices to map the loaded parameters to.
    /// * `processor` - The function to process each loaded parameter.
    ///
    /// # Returns
    /// * `Ok(())` - If the model is successfully loaded from the file.
    /// * `Err(ModelError)` - The error when loading the model from the file.
    fn load(
        &mut self,
        path: &str,
        devices: &[Device],
        processor: dyn FnMut((&str, Tensor), &Device),
    ) -> Result<(), ModelError>;
}
