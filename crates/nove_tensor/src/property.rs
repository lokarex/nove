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
    pub fn to_device(&self, device: &Device) -> Result<Tensor, TensorError> {
        // Check the device, if the device is the same, return the original tensor
        if self.device()? == *device {
            return Ok(self.clone());
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
                parents: vec![self.clone()],
                name: None,
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
    /// // Get the device of the tensor
    /// let device = tensor.device().unwrap();
    /// println!("The tensor is on device: {:?}", device);
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
            return Ok(self.clone());
        }

        let new_inner = match &self.data.read()?.inner {
            TensorInner::Tensor(tensor) => TensorInner::Tensor(tensor.to_dtype(*dtype)?),
            TensorInner::Var(var) => {
                TensorInner::Var(candle_core::Var::from_tensor(&var.to_dtype(*dtype)?)?)
            }
        };

        let new_grad = match &self.data.read()?.grad {
            Some(grad) => Some(grad.to_dtype(*dtype)?),
            None => None,
        };

        Ok(Tensor {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                device: self.data.read()?.device.clone(),
                grad: new_grad,
                parents: vec![self.clone()],
                name: None,
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
    /// use nove::tensor::{Device, Tensor};
    /// let cpu = Device::cpu();
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32], &cpu, false).unwrap();
    ///
    /// // Get the dtype of the tensor
    /// let dtype = tensor.dtype().unwrap();
    /// println!("The dtype of the tensor is: {:?}", dtype);
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
    /// # Arguments
    /// * `shape` - The shape to reshape the tensor to.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The new tensor with the specified shape.
    /// * `Err(TensorError)` - The error when reshaping the tensor.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let cpu = Device::cpu();
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32], &cpu, false).unwrap();
    ///
    /// // Reshape the tensor
    /// let reshaped_tensor = tensor.reshape(&Shape::from(&[2, 1])).unwrap();
    ///
    /// // Get the shape of the reshaped tensor
    /// let shape = reshaped_tensor.shape().unwrap();
    /// println!("The shape of the reshaped tensor is: {:?}", shape);
    /// ```
    pub fn reshape(&self, shape: &Shape) -> Result<Tensor, TensorError> {
        let new_inner = match &self.data.read()?.inner {
            TensorInner::Tensor(tensor) => TensorInner::Tensor(tensor.reshape(shape)?),
            TensorInner::Var(var) => {
                TensorInner::Var(candle_core::Var::from_tensor(&var.reshape(shape)?)?)
            }
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
                parents: vec![self.clone()],
                name: None,
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
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32], &cpu, false).unwrap();
    ///
    /// // Get the shape of the tensor
    /// let shape = tensor.shape().unwrap();
    /// println!("The shape of the tensor is: {:?}", shape);
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
    /// let mut tensor = Tensor::from_data(&[1.0f32, 2.0f32], &cpu, false).unwrap();
    ///
    /// // Set the name of the tensor
    /// tensor.set_name("my_tensor".to_string()).unwrap();
    ///
    /// // Get the name of the tensor
    /// let name = tensor.name().unwrap();
    /// println!("The name of the tensor is: {:?}", name);
    /// ```
    pub fn name(&self) -> Result<Option<String>, TensorError> {
        let data = self.data.read()?;
        Ok(data.name.clone())
    }

    /// Set the name of the tensor.
    ///
    /// # Arguments
    /// * `name` - The name to set for the tensor.
    ///
    /// # Returns
    /// * `Ok(())` - If the name is successfully set.
    /// * `Err(TensorError)` - The error when setting the name of the tensor.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let cpu = Device::cpu();
    /// let mut tensor = Tensor::from_data(&[1.0f32, 2.0f32], &cpu, false).unwrap();
    ///
    /// // Set the name of the tensor
    /// tensor.set_name("my_tensor".to_string()).unwrap();
    ///
    /// // Get the name of the tensor
    /// let name = tensor.name().unwrap();
    /// println!("The name of the tensor is: {:?}", name);
    /// ```
    pub fn set_name(&mut self, name: String) -> Result<(), TensorError> {
        let mut data = self.data.write()?;
        data.name = Some(name);
        Ok(())
    }
}
