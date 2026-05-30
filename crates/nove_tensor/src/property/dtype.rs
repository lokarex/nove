use crate::{DType, Tensor, TensorError, backpropagation::graph::OpKind};

impl Tensor {
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
    /// let device = Device::default();
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32, 3.0f32], &device, false).unwrap();
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
    /// let device = Device::default();
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32, 3.0f32, 4.0f32], &device, false).unwrap();
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
        let data = self.data.read()?;
        let current_dtype = data.storage.dtype()?;
        if current_dtype == *dtype {
            return Ok(self.copy());
        }

        let storage = data.storage.to_dtype(*dtype)?;
        let grad = match &data.grad {
            Some(grad) => Some(grad.to_dtype(dtype)?),
            None => None,
        };
        let device = data.device.clone();
        let requires_grad = data.requires_grad;
        let name = data.name.clone();
        drop(data);

        Ok(Tensor::from_backend_parts(
            storage,
            device,
            requires_grad,
            vec![self.copy()],
            OpKind::ToDType {
                from: current_dtype,
                to: *dtype,
            },
            grad,
            name,
        ))
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
    /// let device = Device::default();
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32], &device, false).unwrap();
    ///
    /// let dtype = tensor.dtype().unwrap();
    /// assert_eq!(dtype, DType::F32);
    /// ```
    pub fn dtype(&self) -> Result<DType, TensorError> {
        Ok(self.data.read()?.storage.dtype()?)
    }
}
