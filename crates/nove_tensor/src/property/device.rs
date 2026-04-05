use std::sync::{Arc, RwLock};

use crate::{
    Device, Tensor, TensorError,
    tensor::{TensorData, TensorInner},
};

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
    /// // CPU device (always available)
    /// let cpu = Device::cpu();
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32, 3.0f32, 4.0f32], &cpu, false).unwrap();
    /// assert_eq!(tensor.device().unwrap(), cpu);
    /// let result = tensor.to_device(&cpu).unwrap();
    /// assert_eq!(result.device().unwrap(), cpu);
    ///
    /// // CUDA device (if feature enabled)
    /// #[cfg(feature = "cuda")]
    /// if let Ok(cuda) = Device::cuda(0) {
    ///     let tensor = Tensor::from_data(&[1.0f32, 2.0f32, 3.0f32, 4.0f32], &cuda, false).unwrap();
    ///     assert_eq!(tensor.device().unwrap(), cuda);
    ///     let result = tensor.to_device(&cuda).unwrap();
    ///     assert_eq!(result.device().unwrap(), cuda);
    /// }
    ///
    /// // Metal device (if feature enabled)
    /// #[cfg(feature = "metal")]
    /// if let Ok(metal) = Device::metal(0) {
    ///     let tensor = Tensor::from_data(&[1.0f32, 2.0f32, 3.0f32, 4.0f32], &metal, false).unwrap();
    ///     assert_eq!(result.device().unwrap(), metal);
    ///     let result = result.to_device(&metal).unwrap();
    ///     assert_eq!(result.device().unwrap(), metal);
    /// }
    /// ```
    ///
    /// * Move tensor to different device
    /// ```
    /// use nove::tensor::{Device, Tensor};
    ///
    /// let cpu = Device::cpu();
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32, 3.0f32, 4.0f32], &cpu, false).unwrap();
    ///
    /// // CUDA device (if feature enabled)
    /// #[cfg(feature = "cuda")]
    /// if let Ok(cuda) = Device::cuda(0) {
    ///     let result = tensor.to_device(&cuda).unwrap();
    ///     assert_eq!(result.device().unwrap(), cuda);
    ///     let result_cpu = result.to_device(&cpu).unwrap();
    ///     assert_eq!(result_cpu.device().unwrap(), cpu);
    /// }
    ///
    /// // Metal device (if feature enabled)
    /// #[cfg(feature = "metal")]
    /// if let Ok(metal) = Device::metal(0) {
    ///     let result = tensor.to_device(&metal).unwrap();
    ///     assert_eq!(result.device().unwrap(), metal);
    ///     let result_cpu = result.to_device(&cpu).unwrap();
    ///     assert_eq!(result_cpu.device().unwrap(), cpu);
    /// }
    /// ```
    pub fn to_device(&self, device: &Device) -> Result<Tensor, TensorError> {
        // Check the device, if the device is the same, return the original tensor
        if self.device()? == *device {
            return Ok(self.copy());
        }

        let new_inner = match &self.data.read()?.inner {
            TensorInner::Tensor(tensor) => TensorInner::Tensor(tensor.to_device(device)?),
            TensorInner::Var(var) => {
                TensorInner::Var(candle_core::Var::from_tensor(&var.to_device(device)?)?)
            }
        };

        let new_grad = match &self.data.read()?.grad {
            Some(grad) => Some(grad.to_device(device)?),
            None => None,
        };

        Ok(Tensor {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                grad: new_grad,
                device: device.clone(),
                parents: vec![self.copy()],
                name: self.data.read()?.name.clone(),
            })),
        })
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
    /// let cpu = Device::cpu();
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32], &cpu, false).unwrap();
    ///
    /// let device = tensor.device().unwrap();
    /// assert_eq!(device, cpu);
    /// ```
    pub fn device(&self) -> Result<Device, TensorError> {
        let data = self.data.read()?;
        Ok(data.device.clone())
    }
}
