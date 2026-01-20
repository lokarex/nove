use crate::tensor::{DType, Device, Shape, Tensor, TensorError, tensor::TensorInner};

impl Tensor {
    /// Move the tensor to the specified device.
    ///
    /// # Arguments
    /// * `device` - The device to move the tensor to.
    ///
    /// # Returns
    /// * `Ok(())` - If the tensor is successfully moved to the device.
    /// * `Err(TensorError)` - The error when moving the tensor to the device.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor, TensorError};
    /// let cpu = Device::cpu();
    /// let mut tensor = Tensor::from_data(&[1.0f32, 2.0f32], &cpu, false).unwrap();
    ///
    /// // Move the tensor to the CPU(It already on the CPU, so it will return an error)
    /// match tensor.to_device(&cpu) {
    ///     Ok(()) => println!("Tensor has been moved to CPU"),
    ///     Err(err) => println!("Error moving tensor to CPU: {:?}", err),
    /// }
    /// ```
    pub fn to_device(&self, device: &Device) -> Result<(), TensorError> {
        // Check the device
        if self.device()? == *device {
            return Ok(());
        }

        // Update the device
        *self.data.device.write()? = device.clone();

        // Move the inner to the device
        let mut inner = self.data.inner.write()?;
        match &mut *inner {
            TensorInner::Tensor(tensor) => {
                *tensor = tensor.to_device(device)?;
            }
            TensorInner::Var(var) => {
                *var = candle_core::Var::from_tensor(&var.to_device(device)?)?;
            }
        }

        // Move the gradient to the device
        let mut grad = self.data.grad.write()?;
        if let Some(grad) = grad.as_mut() {
            *grad = grad.to_device(device)?;
        }

        // Clear the parents
        let mut parents = self.data.parents.write()?;
        parents.clear();
        Ok(())
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
    /// // Get the device of the tensor
    /// let device = tensor.device().unwrap();
    /// println!("The tensor is on device: {:?}", device);
    /// ```
    pub fn device(&self) -> Result<Device, TensorError> {
        let device = self.data.device.read()?;
        Ok(device.clone())
    }

    /// Convert the tensor to the specified dtype.
    ///
    /// # Notes
    /// * The gradient (if present) is also converted to the same dtype to maintain consistency.
    ///
    /// # Arguments
    /// * `dtype` - The dtype to convert the tensor to.
    ///
    /// # Returns
    /// * `Ok(())` - If the tensor is successfully converted to the dtype.
    /// * `Err(TensorError)` - The error when converting the tensor to the dtype.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, DType, Tensor, TensorError};
    /// let cpu = Device::cpu();
    /// let mut tensor = Tensor::from_data(&[1.0f32, 2.0f32], &cpu, false).unwrap();
    ///
    /// // Convert the tensor to F64 dtype
    /// match tensor.to_dtype(&DType::F64) {
    ///     Ok(()) => println!("Tensor has been converted to F64 dtype"),
    ///     Err(err) => println!("Error converting tensor to F64 dtype: {:?}", err),
    /// }
    /// ```
    pub fn to_dtype(&self, dtype: &DType) -> Result<(), TensorError> {
        // Check current dtype first to avoid unnecessary conversion
        let current_dtype = {
            let inner = self.data.inner.read()?;
            match &*inner {
                TensorInner::Tensor(tensor) => tensor.dtype(),
                TensorInner::Var(var) => var.dtype(),
            }
        };

        // If already the target dtype, return Ok
        if current_dtype == *dtype {
            return Ok(());
        }

        // Convert the inner to the dtype
        let mut inner = self.data.inner.write()?;
        match &mut *inner {
            TensorInner::Tensor(tensor) => {
                *tensor = tensor.to_dtype(*dtype)?;
            }
            TensorInner::Var(var) => {
                *var = candle_core::Var::from_tensor(&var.to_dtype(*dtype)?)?;
            }
        }

        // Convert the gradient to the dtype
        let mut grad = self.data.grad.write()?;
        if let Some(grad) = grad.as_mut() {
            *grad = grad.to_dtype(*dtype)?;
        }

        Ok(())
    }

    /// Get the dtype of the tensor.
    ///
    /// # Returns
    /// * `Ok(Dtype)` - The dtype of the tensor.
    /// * `Err(TensorError)` - The error when getting the dtype of the tensor.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let cpu = Device::cpu();
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32], &cpu, false).unwrap();
    ///
    /// // Get the dtype of the tensor
    /// let dtype = tensor.dtype().unwrap();
    /// println!("The dtype of the tensor is: {:?}", dtype);
    /// ```
    pub fn dtype(&self) -> Result<DType, TensorError> {
        let inner = self.data.inner.read()?;
        let dtype = match &*inner {
            TensorInner::Tensor(tensor) => tensor.dtype(),
            TensorInner::Var(var) => var.dtype(),
        };
        Ok(dtype)
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
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32], &cpu, false).unwrap();
    ///
    /// // Get the shape of the tensor
    /// let shape = tensor.shape().unwrap();
    /// println!("The shape of the tensor is: {:?}", shape);
    /// ```
    pub fn shape(&self) -> Result<Shape, TensorError> {
        let inner = self.data.inner.read()?;
        let shape = match &*inner {
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
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32], &cpu, false).unwrap();
    ///
    /// // Get the number of dimensions of the tensor
    /// let num_dim = tensor.num_dim().unwrap();
    /// println!("The number of dimensions of the tensor is : {:?}", num_dim);
    /// ```
    pub fn num_dim(&self) -> Result<usize, TensorError> {
        let shape = self.shape()?;
        Ok(shape.rank())
    }
}
