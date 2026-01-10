use crate::tensor::{DType, Device, Shape, Tensor, TensorError, tensor::TensorInner};

impl Tensor {
    /// Move the tensor to the specified device.
    ///
    /// # Notes
    /// * If the tensor is already on the specified device, an error(TensorError::AlreadyOnDevice) is returned.
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
    /// let cpu = Device::get_cpu();
    /// let mut tensor = Tensor::from_data(&[1.0f32, 2.0f32], &cpu, false).unwrap();
    ///
    /// // Move the tensor to the CPU(It already on the CPU, so it will return an error)
    /// match tensor.to_device(&cpu) {
    ///     Ok(()) => println!("Tensor has been moved to CPU"),
    ///     Err(TensorError::AlreadyOnDevice) => println!("Tensor is already on CPU"),
    ///     Err(err) => println!("Error moving tensor to CPU: {:?}", err),
    /// }
    /// ```
    pub fn to_device(&mut self, device: &Device) -> Result<(), TensorError> {
        // Check the device
        let current_device = self.get_device()?;
        if current_device == *device {
            return Err(TensorError::AlreadyOnDevice);
        }

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
    /// let cpu = Device::get_cpu();
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32], &cpu, false).unwrap();
    ///
    /// // Get the device of the tensor
    /// let device = tensor.get_device().unwrap();
    /// println!("The tensor is on device: {:?}", device);
    /// ```
    pub fn get_device(&self) -> Result<Device, TensorError> {
        let device = self.data.device.read()?;
        Ok(device.clone())
    }

    /// Convert the tensor to the specified dtype.
    ///
    /// # Notes
    /// * If the element dtype of the tensor is already the specified dtype, an error(TensorError::AlreadyDtype) is returned.
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
    /// let cpu = Device::get_cpu();
    /// let mut tensor = Tensor::from_data(&[1.0f32, 2.0f32], &cpu, false).unwrap();
    ///
    /// // Convert the tensor to F64 dtype
    /// match tensor.to_dtype(&DType::F64) {
    ///     Ok(()) => println!("Tensor has been converted to F64 dtype"),
    ///     Err(TensorError::AlreadyDtype) => println!("Tensor is already F64 dtype"),
    ///     Err(err) => println!("Error converting tensor to F64 dtype: {:?}", err),
    /// }
    /// ```
    pub fn to_dtype(&mut self, dtype: &DType) -> Result<(), TensorError> {
        // Check current dtype first to avoid unnecessary conversion
        let current_dtype = {
            let inner = self.data.inner.read()?;
            match &*inner {
                TensorInner::Tensor(tensor) => tensor.dtype(),
                TensorInner::Var(var) => var.dtype(),
            }
        };

        // If already the target dtype, return error
        if current_dtype == *dtype {
            return Err(TensorError::AlreadyDtype);
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
    /// let cpu = Device::get_cpu();
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32], &cpu, false).unwrap();
    ///
    /// // Get the dtype of the tensor
    /// let dtype = tensor.get_dtype().unwrap();
    /// println!("The dtype of the tensor is: {:?}", dtype);
    /// ```
    pub fn get_dtype(&self) -> Result<DType, TensorError> {
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
    /// let cpu = Device::get_cpu();
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32], &cpu, false).unwrap();
    ///
    /// // Get the shape of the tensor
    /// let shape = tensor.get_shape().unwrap();
    /// println!("The shape of the tensor is: {:?}", shape);
    /// ```
    pub fn get_shape(&self) -> Result<Shape, TensorError> {
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
    /// * `Ok(dim_num)` - The number of dimensions of the tensor.
    /// * `Err(TensorError)` - The error when getting the number of dimensions of the tensor.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let cpu = Device::get_cpu();
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32], &cpu, false).unwrap();
    ///
    /// // Get the number of dimensions of the tensor
    /// let dim_num = tensor.get_dim_num().unwrap();
    /// println!("The number of dimensions of the tensor is : {:?}", dim_num);
    /// ```
    pub fn get_dim_num(&self) -> Result<usize, TensorError> {
        let shape = self.get_shape()?;
        Ok(shape.rank())
    }
}
