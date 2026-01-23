use std::fmt::Display;

use crate::Parameter;
use nove_tensor::{Device, TensorError};
use thiserror::Error;

pub mod safetensors;

#[derive(Error, Debug)]
pub enum ParamStoreError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Tensor error: {0}")]
    TensorError(#[from] TensorError),

    #[error("RwLock poisoned: {0}")]
    RwLockPoisoned(String),

    #[error("Other error: {0}")]
    OtherError(String),
}

impl<T> From<std::sync::PoisonError<T>> for ParamStoreError {
    fn from(error: std::sync::PoisonError<T>) -> Self {
        ParamStoreError::RwLockPoisoned(error.to_string())
    }
}

pub trait ParamStore: Display + Clone {
    /// Create a new parameter store with the specified name.
    ///
    /// # Arguments
    /// * `name` - The name of the parameter store.
    ///
    /// # Returns
    /// * `Ok(Self)` - The new parameter store.
    /// * `Err(ParamStoreError)` - The error when creating the parameter store.
    fn new(name: &str) -> Result<Self, ParamStoreError>;

    /// Set the name of the parameter store.
    ///
    /// # Arguments
    /// * `name` - The name of the parameter store.
    ///
    /// # Returns
    /// * `Ok(())` - If the name is successfully set.
    /// * `Err(ParamStoreError)` - The error when setting the name.
    fn set_name(&self, name: &str) -> Result<(), ParamStoreError>;

    /// Get the name of the parameter store.
    ///
    /// # Returns
    /// * `Ok(String)` - The name of the parameter store.
    /// * `Err(ParamStoreError)` - The error when getting the name.
    fn name(&self) -> Result<String, ParamStoreError>;

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
    fn load<F>(
        &self,
        folder_path: &str,
        device: &Device,
        process_fn: F,
    ) -> Result<(), ParamStoreError>
    where
        F: FnMut(&str, &Self) -> Result<(), ParamStoreError>;

    /// Set the direct submodule in the parameter store.
    ///
    /// # Arguments
    /// * `module` - The submodule to set.
    ///
    /// # Returns
    /// * `Ok(())` - If the submodule is successfully set.
    /// * `Err(ParamStoreError)` - The error when setting the submodule.
    fn set_module(&self, module: Self) -> Result<(), ParamStoreError>;

    /// Get the direct submodules in the parameter store.
    ///
    /// # Returns
    /// * `Ok(Vec<Self>)` - The submodules in the parameter store.
    /// * `Err(ParamStoreError)` - The error when getting the submodules.
    fn modules(&self) -> Result<Vec<Self>, ParamStoreError>;

    /// Set the direct parameter in the store.
    ///
    /// # Arguments
    /// * `param` - The parameter to set.
    ///
    /// # Returns
    /// * `Ok(())` - If the parameter is successfully set.
    /// * `Err(ParamStoreError)` - The error when setting the parameter.
    fn set_parameter(&self, param: Parameter) -> Result<(), ParamStoreError>;

    /// Get the direct parameters in the store.
    ///
    /// # Returns
    /// * `Ok(Vec<Parameter>)` - The parameters in the store.
    /// * `Err(ParamStoreError)` - The error when getting the parameters.
    fn parameters(&self) -> Result<Vec<Parameter>, ParamStoreError>;
}
