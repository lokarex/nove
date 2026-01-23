use std::sync::{Arc, RwLock};

use crate::{
    DType, Device, Shape, Tensor, TensorError,
    tensor::{TensorData, TensorInner},
};

impl Tensor {
    /// Move the tensor to the specified device inplace.
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
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32], &cpu, false).unwrap();
    ///
    /// // Move the tensor to the CPU
    /// match tensor.to_device_inplace(&cpu) {
    ///     Ok(()) => println!("Tensor has been moved to CPU"),
    ///     Err(err) => println!("Error moving tensor to CPU: {:?}", err),
    /// }
    /// ```
    pub fn to_device_inplace(&self, device: &Device) -> Result<(), TensorError> {
        // Check the device
        if self.device()? == *device {
            return Ok(());
        }

        // Create the new inner
        let new_inner = match &*self.data.inner.read()? {
            TensorInner::Tensor(tensor) => TensorInner::Tensor(tensor.to_device(device)?),
            TensorInner::Var(var) => {
                TensorInner::Var(candle_core::Var::from_tensor(&var.to_device(device)?)?)
            }
        };

        // Create the new gradient
        let new_grad = match &*self.data.grad.read()? {
            Some(grad) => Some(grad.to_device(device)?),
            None => None,
        };

        {
            let mut inner_write = self.data.inner.write()?;
            let mut grad_write = self.data.grad.write()?;
            let mut device_write = self.data.device.write()?;

            // Update the inner
            *inner_write = new_inner;
            // Update the gradient
            *grad_write = new_grad;
            // Update the device
            *device_write = device.clone();
        }
        Ok(())
    }

    /// Create a new tensor like the current tensor, but on the specified device.
    ///
    /// # Arguments
    /// * `device` - The device to move the tensor to.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The new tensor on the specified device.
    /// * `Err(TensorError)` - The error when moving the tensor to the device.
    pub fn to_device(&self, device: &Device) -> Result<Tensor, TensorError> {
        // Check the device, if the device is the same, return the original tensor
        if self.device()? == *device {
            return Ok(self.clone());
        }

        // Create the new inner
        let new_inner = match &*self.data.inner.read()? {
            TensorInner::Tensor(tensor) => TensorInner::Tensor(tensor.to_device(device)?),
            TensorInner::Var(var) => {
                TensorInner::Var(candle_core::Var::from_tensor(&var.to_device(device)?)?)
            }
        };

        // Create the new gradient
        let new_grad = match &*self.data.grad.read()? {
            Some(grad) => Some(grad.to_device(device)?),
            None => None,
        };

        // Create the new tensor
        Ok(Tensor {
            data: Arc::new(TensorData {
                inner: RwLock::new(new_inner),
                grad: RwLock::new(new_grad),
                device: RwLock::new(device.clone()),
                parents: RwLock::new(vec![self.clone()]),
            }),
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
    /// // Get the device of the tensor
    /// let device = tensor.device().unwrap();
    /// println!("The tensor is on device: {:?}", device);
    /// ```
    pub fn device(&self) -> Result<Device, TensorError> {
        let device = self.data.device.read()?;
        Ok(device.clone())
    }

    /// Convert the tensor to the specified dtype inplace.
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
    /// match tensor.to_dtype_inplace(&DType::F64) {
    ///     Ok(()) => println!("Tensor has been converted to F64 dtype"),
    ///     Err(err) => println!("Error converting tensor to F64 dtype: {:?}", err),
    /// }
    /// ```
    pub fn to_dtype_inplace(&self, dtype: &DType) -> Result<(), TensorError> {
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

        // Create the new inner
        let new_inner = match &*self.data.inner.read()? {
            TensorInner::Tensor(tensor) => TensorInner::Tensor(tensor.to_dtype(*dtype)?),
            TensorInner::Var(var) => {
                TensorInner::Var(candle_core::Var::from_tensor(&var.to_dtype(*dtype)?)?)
            }
        };

        // Create the new gradient
        let new_grad = match &*self.data.grad.read()? {
            Some(grad) => Some(grad.to_dtype(*dtype)?),
            None => None,
        };

        {
            let mut inner_write = self.data.inner.write()?;
            let mut grad_write = self.data.grad.write()?;

            // Update the inner
            *inner_write = new_inner;
            // Update the gradient
            *grad_write = new_grad;
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

    /// Reshape the tensor inplace to the specified shape.
    ///
    /// # Arguments
    /// * `shape` - The shape to reshape the tensor to.
    ///
    /// # Returns
    /// * `Ok(())` - If the tensor is successfully reshaped.
    /// * `Err(TensorError)` - The error when reshaping the tensor.
    pub fn to_shape_inplace(&self, shape: &Shape) -> Result<(), TensorError> {
        let new_inner = match &*self.data.inner.read()? {
            TensorInner::Tensor(tensor) => TensorInner::Tensor(tensor.reshape(shape)?),
            TensorInner::Var(var) => {
                TensorInner::Var(candle_core::Var::from_tensor(&var.reshape(shape)?)?)
            }
        };

        let new_grad = match &*self.data.grad.read()? {
            Some(grad) => Some(grad.reshape(shape)?),
            None => None,
        };

        {
            let mut inner_write = self.data.inner.write()?;
            let mut grad_write = self.data.grad.write()?;

            // Update the inner
            *inner_write = new_inner;
            // Update the gradient
            *grad_write = new_grad;
        }
        Ok(())
    }

    /// Create a new tensor like the current tensor with the specified shape.
    ///
    /// # Arguments
    /// * `shape` - The shape to reshape the tensor to.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The new tensor with the specified shape.
    /// * `Err(TensorError)` - The error when reshaping the tensor.
    pub fn to_shape(&self, shape: &Shape) -> Result<Tensor, TensorError> {
        let new_inner = match &*self.data.inner.read()? {
            TensorInner::Tensor(tensor) => TensorInner::Tensor(tensor.reshape(shape)?),
            TensorInner::Var(var) => {
                TensorInner::Var(candle_core::Var::from_tensor(&var.reshape(shape)?)?)
            }
        };

        let new_grad = match &*self.data.grad.read()? {
            Some(grad) => Some(grad.reshape(shape)?),
            None => None,
        };

        Ok(Tensor {
            data: Arc::new(TensorData {
                inner: RwLock::new(new_inner),
                grad: RwLock::new(new_grad),
                device: RwLock::new(self.data.device.read()?.clone()),
                parents: RwLock::new(vec![self.clone()]),
            }),
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
