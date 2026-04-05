use crate::{
    DType, Tensor, TensorError,
    tensor::{TensorData, TensorInner},
};
use std::sync::{Arc, RwLock};

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
}
