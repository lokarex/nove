use std::sync::{Arc, RwLock};

use crate::{
    DType, Device, Shape, Tensor, TensorError,
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

    /// Create a new tensor like the current tensor but with the specified dtype.
    ///
    /// # Arguments
    /// * `dtype` - The dtype to convert the tensor to.
    ///
    /// # Returns
    /// * `Ok(tensor)` - The new tensor if successful.
    /// * `Err(TensorError)` - The error when converting the tensor to the dtype.
    ///
    /// # Examples
    ///
    /// * Convert to same dtype returns copy
    /// ```
    /// use nove::tensor::{Device, DType, Tensor};
    ///
    /// let cpu = Device::cpu();
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32, 3.0f32], &cpu, false).unwrap();
    /// assert_eq!(tensor.dtype().unwrap(), DType::F32);
    ///
    /// let result = tensor.to_dtype(&DType::F32).unwrap();
    /// assert_eq!(result.shape().unwrap(), (&[3]).into());
    /// assert_eq!(result.dtype().unwrap(), DType::F32);
    /// assert_eq!(result.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
    /// ```
    ///
    /// * Convert tensor to different dtype
    /// ```
    /// use nove::tensor::{Device, DType, Tensor};
    ///
    /// let cpu = Device::cpu();
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32, 3.0f32, 4.0f32], &cpu, false).unwrap();
    /// assert_eq!(tensor.dtype().unwrap(), DType::F32);
    ///
    /// let result = tensor.to_dtype(&DType::F64).unwrap();
    /// assert_eq!(result.shape().unwrap(), (&[4]).into());
    /// assert_eq!(result.dtype().unwrap(), DType::F64);
    /// let data = result.to_vec::<f64>().unwrap();
    /// assert!((data[0] - 1.0).abs() < 1e-6);
    /// assert!((data[1] - 2.0).abs() < 1e-6);
    /// assert!((data[2] - 3.0).abs() < 1e-6);
    /// assert!((data[3] - 4.0).abs() < 1e-6);
    /// ```
    pub fn to_dtype(&self, dtype: &DType) -> Result<Tensor, TensorError> {
        // Check current dtype first to avoid unnecessary conversion
        let current_dtype = {
            let data = self.data.read()?;
            match &data.inner {
                TensorInner::Tensor(tensor) => tensor.dtype(),
                TensorInner::Var(var) => var.dtype(),
            }
        };

        // If already the target dtype, return the tensor itself
        if current_dtype == *dtype {
            return Ok(self.copy());
        }

        let new_inner = match &self.data.read()?.inner {
            TensorInner::Tensor(tensor) => TensorInner::Tensor(tensor.to_dtype(*dtype)?),
            TensorInner::Var(var) => {
                TensorInner::Var(candle_core::Var::from_tensor(&var.to_dtype(*dtype)?)?)
            }
        };

        let new_grad = match &self.data.read()?.grad {
            Some(grad) => Some(grad.to_dtype(dtype)?),
            None => None,
        };

        Ok(Tensor {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                device: self.data.read()?.device.clone(),
                grad: new_grad,
                parents: vec![self.copy()],
                name: self.data.read()?.name.clone(),
            })),
        })
    }

    /// Get the dtype of the tensor.
    ///
    /// # Returns
    /// * `Ok(Dtype)` - The dtype of the tensor.
    /// * `Err(TensorError)` - The error when getting the dtype of the tensor.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, DType, Tensor};
    /// let cpu = Device::cpu();
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32], &cpu, false).unwrap();
    ///
    /// let dtype = tensor.dtype().unwrap();
    /// assert_eq!(dtype, DType::F32);
    /// ```
    pub fn dtype(&self) -> Result<DType, TensorError> {
        let data = self.data.read()?;
        let dtype = match &data.inner {
            TensorInner::Tensor(tensor) => tensor.dtype(),
            TensorInner::Var(var) => var.dtype(),
        };
        Ok(dtype)
    }

    /// Create a new tensor like the current tensor with the specified shape.
    ///
    /// # Notes
    /// * In the nove framework, gradients of intermediate Tensors are not stored.
    ///   When `reshape` is called on a Tensor with `requires_grad=true`, it returns an intermediate Tensor whose `grad` is always `None`.
    ///
    /// # Arguments
    /// * `shape` - The shape to reshape the tensor to.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The new tensor with the specified shape.
    /// * `Err(TensorError)` - The error when reshaping the tensor.
    ///
    /// # Examples
    /// * Reshape 1D tensor to 2D
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let cpu = Device::cpu();
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32, 3.0f32, 4.0f32], &cpu, false).unwrap();
    ///
    /// let result = tensor.reshape(&Shape::from(&[2, 2])).unwrap();
    /// assert_eq!(result.shape().unwrap(), (&[2, 2]).into());
    /// assert_eq!(result.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    /// ```
    ///
    /// * Reshape 1D tensor to column vector
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let cpu = Device::cpu();
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32, 3.0f32], &cpu, false).unwrap();
    ///
    /// let result = tensor.reshape(&Shape::from(&[3, 1])).unwrap();
    /// assert_eq!(result.shape().unwrap(), (&[3, 1]).into());
    /// assert_eq!(result.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
    /// ```
    ///
    /// * Backpropagate through reshape with gradient
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let cpu = Device::cpu();
    ///
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32, 3.0f32, 4.0f32], &cpu, true).unwrap();
    /// let result = tensor.reshape(&Shape::from(&[2, 2])).unwrap();
    /// result.backward().unwrap();
    /// let grad = tensor.grad().unwrap().unwrap();
    /// assert_eq!(grad.shape().unwrap(), (&[4]).into());
    /// assert_eq!(grad.to_vec::<f32>().unwrap(), vec![1.0, 1.0, 1.0, 1.0]);
    ///
    /// // The following Tensor is actually an intermediate Tensor, its grad is always None
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32, 3.0f32, 4.0f32], &cpu, true).unwrap().reshape(&Shape::from(&[2, 2])).unwrap();
    /// tensor.backward().unwrap();
    /// assert!(tensor.grad().unwrap().is_none());
    ///
    /// // If you need to reshape the tensor immediately using chained calls while preserving gradients, you can do so as follows
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32, 3.0f32, 4.0f32], &cpu, true).unwrap().reshape(&Shape::from(&[2, 2])).unwrap().require_grad(true).unwrap();
    /// tensor.backward().unwrap();
    /// let grad = tensor.grad().unwrap().unwrap();
    /// assert_eq!(grad.shape().unwrap(), (&[2, 2]).into());
    /// assert_eq!(grad.to_vec::<f32>().unwrap(), vec![1.0, 1.0, 1.0, 1.0]);
    /// ```
    pub fn reshape(&self, shape: &Shape) -> Result<Tensor, TensorError> {
        let new_inner = match &self.data.read()?.inner {
            TensorInner::Tensor(tensor) => TensorInner::Tensor(tensor.reshape(shape)?),
            TensorInner::Var(var) => TensorInner::Tensor(var.reshape(shape)?),
        };

        let new_grad = match &self.data.read()?.grad {
            Some(grad) => Some(grad.reshape(shape)?),
            None => None,
        };

        Ok(Tensor {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                grad: new_grad,
                device: self.data.read()?.device.clone(),
                parents: vec![self.copy()],
                name: self.data.read()?.name.clone(),
            })),
        })
    }

    /// Get the shape of the tensor.
    ///
    /// # Returns
    /// * `Ok(shape)` - The shape of the tensor.
    /// * `Err(TensorError)` - The error when getting the shape of the tensor.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let cpu = Device::cpu();
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32, 3.0f32], &cpu, false).unwrap();
    ///
    /// let shape = tensor.shape().unwrap();
    /// assert_eq!(shape, (&[3]).into());
    /// ```
    pub fn shape(&self) -> Result<Shape, TensorError> {
        let data = self.data.read()?;
        let shape = match &data.inner {
            TensorInner::Tensor(tensor) => tensor.shape(),
            TensorInner::Var(var) => var.shape(),
        };
        Ok(Shape::from(shape))
    }

    /// Get the number of dimensions of the tensor.
    ///
    /// # Returns
    /// * `Ok(num_dim)` - The number of dimensions of the tensor.
    /// * `Err(TensorError)` - The error when getting the number of dimensions of the tensor.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let cpu = Device::cpu();
    ///
    /// // 1-dimensional tensor (vector)
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32, 3.0f32], &cpu, false).unwrap();
    /// let num_dim = tensor.num_dim().unwrap();
    /// assert_eq!(num_dim, 1);
    ///
    /// // 2-dimensional tensor (matrix)
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32, 3.0f32, 4.0f32], &cpu, false).unwrap();
    /// let result = tensor.reshape(&(&[2, 2]).into()).unwrap();
    /// let num_dim = result.num_dim().unwrap();
    /// assert_eq!(num_dim, 2);
    ///
    /// // 3-dimensional tensor
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32], &cpu, false).unwrap();
    /// let result = tensor.reshape(&(&[2, 2, 2]).into()).unwrap();
    /// let num_dim = result.num_dim().unwrap();
    /// assert_eq!(num_dim, 3);
    ///
    /// // 4-dimensional tensor
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32], &cpu, false).unwrap();
    /// let result = tensor.reshape(&(&[2, 2, 1, 2]).into()).unwrap();
    /// let num_dim = result.num_dim().unwrap();
    /// assert_eq!(num_dim, 4);
    /// ```
    pub fn num_dim(&self) -> Result<usize, TensorError> {
        let shape = self.shape()?;
        Ok(shape.rank())
    }

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
