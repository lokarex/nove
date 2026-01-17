use crate::{
    model::paramstore::{ParamStore, ParamStoreError},
    tensor::{Device, Tensor},
};
use thiserror::Error;

pub mod paramstore;

#[derive(Error, Debug)]
pub enum ModelError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("ParamStore error: {0}")]
    ParamStoreError(#[from] ParamStoreError),

    #[error("Other error: {0}")]
    OtherError(String),
}

pub struct Parameter(pub String, pub Tensor);

pub trait Model {
    type ParamStore: ParamStore;
    type Input;
    type Output;

    /// Get the parameter stores of the model.
    ///
    /// # Returns
    /// * `Ok(Vec<&Self::ParamStore>)` - The parameter stores of the model.
    /// * `Err(ModelError)` - The error when getting the parameter stores.
    fn param_stores(&self) -> Result<Vec<&Self::ParamStore>, ModelError>;

    /// Get the mutable parameter stores of the model.
    ///
    /// # Returns
    /// * `Ok(Vec<&mut Self::ParamStore>)` - The mutable parameter stores of the model.
    /// * `Err(ModelError)` - The error when getting the mutable parameter stores.
    fn param_store_mut(&mut self) -> Result<Vec<&mut Self::ParamStore>, ModelError>;

    /// Perform a forward pass of the model.
    ///
    /// # Arguments
    /// * `input` - The input data for the forward pass.
    ///
    /// # Returns
    /// * `Ok(Self::Output)` - The output data of the forward pass.
    /// * `Err(ModelError)` - The error when performing the forward pass.
    fn forward(&self, input: Self::Input) -> Result<Self::Output, ModelError>;

    /// Save the model to the specified folder.
    ///
    /// # Notes
    /// * This method would call the `save` method of each parameter store. So the specific storage
    ///   method depends on the ParamStore chosen for the model.
    ///
    /// # Arguments
    /// * `folder_path` - The path of the folder to save the model to.
    ///
    /// # Returns
    /// * `Ok(())` - If the model is successfully saved to the folder.
    /// * `Err(ModelError)` - The error when saving the model to the folder.
    fn save(&self, folder_path: &str) -> Result<(), ModelError> {
        let param_stores = self.param_stores()?;
        for param_store in param_stores {
            param_store.save(folder_path)?;
        }
        Ok(())
    }

    /// Load the model from the specified folder.
    ///
    /// # Notes
    /// * This method would call the `load` method of each parameter store.
    /// * The ParamStore type must be same as the one used for saving.
    ///
    /// # Arguments
    /// * `folder_path` - The path of the folder to load the model from.
    /// * `devices` - The devices to load the model to.
    ///   * If the number of devices is 1, the model would be loaded to all parameter stores to this device.
    ///   * If the number of devices is equal to the number of parameter stores, each parameter store
    ///     would be loaded to the corresponding device.
    ///   * If the number of devices is not equal to 1 or the number of parameter stores, the method
    ///     would return an error.
    /// * `process_fn` - The function to process each parameter store(including submodules).
    ///
    /// # Returns
    /// * `Ok(())` - If the model is successfully loaded from the folder.
    /// * `Err(ModelError)` - The error when loading the model from the folder.
    fn load(
        &mut self,
        folder_path: &str,
        devices: &[Device],
        mut process_fn: impl FnMut(&str, &mut Self::ParamStore) -> Result<(), ParamStoreError>,
    ) -> Result<(), ModelError> {
        if devices.is_empty() {
            return Err(ModelError::OtherError(
                "The devices list is empty.".to_string(),
            ));
        }

        let mut param_stores = self.param_store_mut()?;
        // Get the number of parameter stores.
        let num_param_stores = param_stores.len();
        if num_param_stores == 0 {
            return Err(ModelError::OtherError(
                "The model has no parameter store.".to_string(),
            ));
        }

        if devices.len() != 1 && devices.len() != num_param_stores {
            return Err(ModelError::OtherError(format!(
                "The number of devices({}) is not equal to the number of parameter stores({}) or 1.",
                devices.len(),
                self.param_stores()?.len()
            )));
        }

        match devices.len() {
            1 => {
                for param_store in param_stores.iter_mut() {
                    param_store.load(
                        (folder_path.to_string() + param_store.name()).as_str(),
                        &devices[0],
                        &mut process_fn,
                    )?;
                }
            }
            _ => {
                for (i, param_store) in param_stores.iter_mut().enumerate() {
                    param_store.load(
                        (folder_path.to_string() + param_store.name()).as_str(),
                        &devices[i],
                        &mut process_fn,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// Set the model to training mode.
    ///
    /// # Returns
    /// * `Ok(())` - If the model is successfully set to training mode.
    /// * `Err(ModelError)` - The error when setting the model to training mode.
    fn to_train(&mut self) -> Result<(), ModelError> {
        todo!()
    }

    /// Set the model to evaluation mode.
    ///
    /// # Returns
    /// * `Ok(())` - If the model is successfully set to evaluation mode.
    /// * `Err(ModelError)` - The error when setting the model to evaluation mode.
    fn to_eval(&mut self) -> Result<(), ModelError> {
        todo!()
    }
}
