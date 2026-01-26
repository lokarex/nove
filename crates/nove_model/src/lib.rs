use crate::paramstore::{ParamStore, ParamStoreError};
use nove_tensor::{Device, Tensor, TensorError};
use thiserror::Error;

pub mod layer;
pub mod paramstore;

#[derive(Error, Debug)]
pub enum ModelError {
    /// I/O errors from the standard library.
    #[error(transparent)]
    IoError(#[from] std::io::Error),

    /// Tensor errors from the `nove_tensor` crate.
    #[error(transparent)]
    TensorError(#[from] TensorError),

    /// Parameter store errors from the `nove_model` crate.
    #[error(transparent)]
    ParamStoreError(#[from] ParamStoreError),

    /// Other errors.
    #[error("{0}")]
    OtherError(String),
}

#[derive(Debug, Clone)]
pub struct Parameter(pub String, pub Tensor);

pub trait Model {
    type ParamStore: ParamStore;
    type Input;
    type Output;

    /// Get the direct parameter stores of the model.
    ///
    /// # Returns
    /// * `Ok(Vec<Self::ParamStore>)` - The parameter stores of the model.
    /// * `Err(ModelError)` - The error when getting the parameter stores.
    fn param_stores(&self) -> Result<Vec<Self::ParamStore>, ModelError>;

    /// Get all parameters of the model, including those in the submodules.
    ///
    /// # Returns
    /// * `Ok(Vec<Parameter>)` - All parameters of the model.
    /// * `Err(ModelError)` - The error when getting the parameters.
    fn parameters(&self) -> Result<Vec<Parameter>, ModelError> {
        let mut all_params = Vec::new();

        fn collect_params<S: ParamStore>(
            store: &S,
            all_params: &mut Vec<Parameter>,
        ) -> Result<(), ParamStoreError> {
            // Collect parameters from the current store.
            all_params.extend(store.parameters()?);

            // Recursively collect parameters from submodules.
            for submodule in store.modules()? {
                collect_params(&submodule, all_params)?;
            }

            Ok(())
        }

        for store in self.param_stores()? {
            collect_params(&store, &mut all_params)?;
        }

        Ok(all_params)
    }

    /// Perform a forward pass of the model.
    ///
    /// # Arguments
    /// * `input` - The input data for the forward pass.
    ///
    /// # Returns
    /// * `Ok(Self::Output)` - The output data of the forward pass.
    /// * `Err(ModelError)` - The error when performing the forward pass.
    fn forward(&mut self, input: Self::Input) -> Result<Self::Output, ModelError>;

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
        // Create the folder if it does not exist.
        std::fs::create_dir_all(folder_path)?;

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
    fn load<F>(
        &mut self,
        folder_path: &str,
        devices: &[Device],
        mut process_fn: F,
    ) -> Result<(), ModelError>
    where
        F: FnMut(&str, &Self::ParamStore) -> Result<(), ParamStoreError>,
    {
        if devices.is_empty() {
            return Err(ModelError::OtherError(
                "The devices list is empty.".to_string(),
            ));
        }

        let param_stores = self.param_stores()?;
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
                num_param_stores
            )));
        }

        match devices.len() {
            1 => {
                for param_store in param_stores.iter() {
                    param_store.load(folder_path, &devices[0], &mut process_fn)?;
                }
            }
            _ => {
                for (i, param_store) in param_stores.iter().enumerate() {
                    param_store.load(folder_path, &devices[i], &mut process_fn)?;
                }
            }
        }
        Ok(())
    }

    /// Get the summary of the model.
    ///
    /// # Returns
    /// * `Ok(String)` - The summary of the model.
    /// * `Err(ModelError)` - The error when getting the model summary.
    fn summary(&self) -> Result<String, ModelError> {
        let mut summary = format!("{}(\n", std::any::type_name::<Self>());
        let param_stores = self.param_stores()?;
        for param_store in param_stores {
            let param_store_summary = format!("{}", param_store);
            for line in param_store_summary.lines() {
                summary.push_str(&format!("  {}\n", line));
            }
        }
        summary.push_str(")\n");
        Ok(summary)
    }
}
