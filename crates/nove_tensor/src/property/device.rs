use crate::{Device, Tensor, TensorError, backpropagation::graph::OpKind};

impl Tensor {
    /// Create a new tensor like the current tensor, but on the specified device.
    ///
    /// # Arguments
    /// * `device` - The device to move the tensor to.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The new tensor on the specified device.
    /// * `Err(TensorError)` - The error when moving the tensor to the device.
    ///
    /// # Examples
    /// * Move tensor to same device
    /// ```
    /// use nove::tensor::{Device, Tensor};
    ///
    /// // CPU device (when CPU backend available)
    /// #[cfg(feature = "candle-cpu")]
    /// {
    ///     let cpu = nove::device::candle::cpu().unwrap();
    ///     let tensor = Tensor::from_data(&[1.0f32, 2.0f32, 3.0f32, 4.0f32], &cpu, false).unwrap();
    ///     let result = tensor.to_device(&cpu).unwrap();
    ///     assert_eq!(result.device().unwrap(), cpu);
    /// }
    /// // CUDA device (if feature enabled and hardware available)
    /// #[cfg(feature = "candle-cuda")]
    /// if let Ok(cuda) = nove::device::candle::cuda(0) {
    ///     let tensor = Tensor::from_data(&[1.0f32, 2.0f32, 3.0f32, 4.0f32], &cuda, false).unwrap();
    ///     let result = tensor.to_device(&cuda).unwrap();
    ///     assert_eq!(result.device().unwrap(), cuda);
    /// }
    /// // Metal device (if feature enabled and hardware available)
    /// #[cfg(feature = "candle-metal")]
    /// if let Ok(metal) = nove::device::candle::metal(0) {
    ///     let tensor = Tensor::from_data(&[1.0f32, 2.0f32, 3.0f32, 4.0f32], &metal, false).unwrap();
    ///     let result = tensor.to_device(&metal).unwrap();
    ///     assert_eq!(result.device().unwrap(), metal);
    /// }
    /// ```
    ///
    /// * Move tensor to different device
    /// ```
    /// use nove::tensor::{Device, Tensor};
    ///
    /// #[cfg(feature = "candle-cpu")]
    /// {
    ///     let cpu = nove::device::candle::cpu().unwrap();
    ///     let tensor = Tensor::from_data(&[1.0f32, 2.0f32, 3.0f32, 4.0f32], &cpu, false).unwrap();
    ///
    ///     // CUDA device (if feature enabled and hardware available)
    ///     #[cfg(feature = "candle-cuda")]
    ///     if let Ok(cuda) = nove::device::candle::cuda(0) {
    ///         let result = tensor.to_device(&cuda).unwrap();
    ///         assert_eq!(result.device().unwrap(), cuda);
    ///         let result_cpu = result.to_device(&cpu).unwrap();
    ///         assert_eq!(result_cpu.device().unwrap(), cpu);
    ///     }
    ///
    ///     // Metal device (if feature enabled and hardware available)
    ///     #[cfg(feature = "candle-metal")]
    ///     if let Ok(metal) = nove::device::candle::metal(0) {
    ///         let result = tensor.to_device(&metal).unwrap();
    ///         assert_eq!(result.device().unwrap(), metal);
    ///         let result_cpu = result.to_device(&cpu).unwrap();
    ///         assert_eq!(result_cpu.device().unwrap(), cpu);
    ///     }
    /// }
    /// ```
    pub fn to_device(&self, device: &Device) -> Result<Tensor, TensorError> {
        if self.device()? == *device {
            return Ok(self.copy());
        }

        let data = self.data.read()?;
        let storage = data.storage.to_device(device)?;
        let grad = match &data.grad {
            Some(grad) => Some(grad.to_device(device)?),
            None => None,
        };
        let requires_grad = data.requires_grad;
        let name = data.name.clone();
        drop(data);

        Ok(Tensor::from_backend_parts(
            storage,
            device.clone(),
            requires_grad,
            vec![self.copy()],
            OpKind::ToDevice {
                from: self.device()?,
                to: device.clone(),
            },
            grad,
            name,
        ))
    }

    /// Get the device of the tensor.
    ///
    /// # Returns
    /// * `Ok(device)` - The device of the tensor.
    /// * `Err(TensorError)` - The error when getting the device of the tensor.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    ///
    /// #[cfg(feature = "candle-cpu")]
    /// {
    ///     let cpu = nove::device::candle::cpu().unwrap();
    ///     let tensor = Tensor::from_data(&[1.0f32, 2.0f32], &cpu, false).unwrap();
    ///     assert_eq!(tensor.device().unwrap(), cpu);
    /// }
    /// ```
    pub fn device(&self) -> Result<Device, TensorError> {
        Ok(self.data.read()?.device.clone())
    }
}
