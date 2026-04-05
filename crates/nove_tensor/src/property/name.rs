use crate::{Tensor, TensorError};

impl Tensor {
    /// Get the name of the tensor.
    ///
    /// # Returns
    /// * `Ok(Some(name))` - The name of the tensor if it has been set.
    /// * `Ok(None)` - The tensor does not have a name.
    /// * `Err(TensorError)` - The error when getting the name of the tensor.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let cpu = Device::cpu();
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32], &cpu, false).unwrap();
    ///
    /// let name = tensor.name().unwrap();
    /// assert_eq!(name, None);
    /// ```
    ///
    /// # See Also
    /// * [`Tensor::require_name`] - Create a new tensor like the current tensor with the specified name.
    pub fn name(&self) -> Result<Option<String>, TensorError> {
        let data = self.data.read()?;
        Ok(data.name.clone())
    }

    /// Create a new tensor like the current tensor with the specified name.
    ///
    /// # Arguments
    /// * `name` - The name to set for the tensor.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The new tensor with the specified name.
    /// * `Err(TensorError)` - The error when setting the name of the tensor.
    ///
    /// # Examples
    /// * Set name on tensor
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let cpu = Device::cpu();
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32], &cpu, false).unwrap();
    ///
    /// let result = tensor.require_name("my_tensor").unwrap();
    /// assert_eq!(result.name().unwrap(), Some("my_tensor".to_string()));
    /// ```
    ///
    /// * Verify name, data and shape preserved after naming
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let cpu = Device::cpu();
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32, 3.0f32], &cpu, false).unwrap();
    ///
    /// let result = tensor.require_name("test").unwrap();
    /// assert_eq!(result.name().unwrap(), Some("test".to_string()));
    /// assert_eq!(result.shape().unwrap(), (&[3]).into());
    /// assert_eq!(result.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
    /// ```
    pub fn require_name(&self, name: &str) -> Result<Tensor, TensorError> {
        let new_tensor = self.try_clone()?;
        new_tensor.data.write()?.name = Some(name.to_string());
        Ok(new_tensor)
    }
}
