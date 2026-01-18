use std::fmt::Display;

use crate::{
    model::Parameter,
    tensor::{Device, TensorError},
};
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

pub trait ParamStore: Display {
    /// Set the name of the parameter store.
    ///
    /// # Arguments
    /// * `name` - The name of the parameter store.
    fn set_name(&mut self, name: &str);

    /// Get the name of the parameter store.
    ///
    /// # Returns
    /// * `&str` - The name of the parameter store.
    fn name(&self) -> &str;

    /// Set the type ID of the parameter store.
    ///
    /// # Notes
    /// * The type ID is used to identify the type of the parameter set.
    ///
    /// # Arguments
    /// * `type_id` - The type ID of the parameter store.
    fn set_type_id(&mut self, type_id: usize);

    /// Get the type ID of the parameter store.
    ///
    /// # Returns
    /// * `usize` - The type ID of the parameter store.
    fn type_id(&self) -> usize;

    /// Save the all parameters to the specified folder.
    ///
    /// # Arguments
    /// * `folder_path` - The path of the folder to save the parameters to.
    ///
    /// # Returns
    /// * `Ok(())` - If the parameters are successfully saved to the folder.
    /// * `Err(ParamStoreError)` - The error when saving the parameters to the folder.
    fn save(&self, folder_path: &str) -> Result<(), ParamStoreError>;

    /// Load the parameters from the specified folder to the store.
    ///
    /// # Arguments
    /// * `folder_path` - The path of the folder to load the parameters from.
    /// * `process_fn` - The function to process each parameter store(including submodules).
    ///
    /// # Returns
    /// * `Ok(())` - If the parameters are successfully loaded from the folder.
    /// * `Err(ParamStoreError)` - The error when loading the parameters from the folder.
    fn load(
        &mut self,
        folder_path: &str,
        device: &Device,
        process_fn: impl FnMut(&str, &mut Self) -> Result<(), ParamStoreError>,
    ) -> Result<(), ParamStoreError>;

    /// Set the direct submodule in the parameter store.
    ///
    /// # Arguments
    /// * `module` - The submodule to set.
    ///
    /// # Returns
    /// * `Ok(())` - If the submodule is successfully set.
    /// * `Err(ParamStoreError)` - The error when setting the submodule.
    fn set_module(&mut self, module: Self) -> Result<(), ParamStoreError>;

    /// Get the direct submodules in the parameter store.
    ///
    /// # Returns
    /// * `Ok(Vec<&Self>)` - The submodules in the parameter store.
    /// * `Err(ParamStoreError)` - The error when getting the submodules.
    fn modules(&self) -> Result<Vec<&Self>, ParamStoreError>;

    /// Set the direct parameter in the store.
    ///
    /// # Arguments
    /// * `param` - The parameter to set.
    ///
    /// # Returns
    /// * `Ok(())` - If the parameter is successfully set.
    /// * `Err(ParamStoreError)` - The error when setting the parameter.
    fn set_paramter(&mut self, param: Parameter) -> Result<(), ParamStoreError>;

    /// Get the direct parameters in the store.
    ///
    /// # Returns
    /// * `Ok(Vec<&Parameter>)` - The parameters in the store.
    /// * `Err(ParamStoreError)` - The error when getting the parameters.
    fn parameters(&self) -> Result<Vec<&Parameter>, ParamStoreError>;
}
