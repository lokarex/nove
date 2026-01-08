use std::sync::{Arc, RwLock};

use crate::tensor::{
    Device, Tensor, TensorError,
    tensor::{TensorData, TensorInner},
};

impl Tensor {
    /// Create a new tensor from an array.
    ///
    /// # Notes
    /// * The element type of the array supported by this function includes `f32`, `f64`, `i64`, `u32`, `u8`.
    ///
    /// # Arguments
    /// * `array` - The array to create the tensor from.
    /// * `device` - The device to place the tensor on.
    /// * `grad_enabled` - Whether to enable gradient tracking for the tensor.
    ///
    /// # Returns
    /// * `Ok(tensor)` - The created tensor if successful.
    /// * `Err(TensorError)` - The error when creating the tensor.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::Device;
    /// use nove::tensor::Tensor;
    /// let device = Device::get_cuda_if_available(0);
    /// let tensor = Tensor::from_array(&[1f32, 2f32, 3f32], &device, false).unwrap();
    /// println!("{:?}", tensor);
    /// let tensor = Tensor::from_array(&[[4i64, 5i64, 6i64], [7i64, 8i64, 9i64]], &device, false).unwrap();
    /// println!("{:?}", tensor);
    /// ```
    pub fn from_array<A>(array: A, device: &Device, grad_enabled: bool) -> Result<Self, TensorError>
    where
        A: candle_core::NdArray,
    {
        let inner = match grad_enabled {
            true => TensorInner::Var(candle_core::Var::new(array, &device)?),
            false => TensorInner::Tensor(candle_core::Tensor::new(array, &device)?),
        };

        let grad = match &inner {
            TensorInner::Var(var) => Some(var.zeros_like()?),
            TensorInner::Tensor(_) => None,
        };

        Ok(Self {
            data: Arc::new(TensorData {
                inner: RwLock::new(inner),
                parents: RwLock::new(vec![]),
                grad: RwLock::new(grad),
            }),
        })
    }

    /// Create a new tensor from a scalar.
    ///
    /// # Notes
    /// * This function is an alias of `from_array` but accepts a scalar value with more explicit type.
    /// * The element type of the scalar supported by this function includes `f32`, `f64`, `i64`, `u32`, `u8`.
    ///
    /// # Arguments
    /// * `scalar` - The scalar to create the tensor from. It should be a single value not a vector or array.
    /// * `device` - The device to place the tensor on.
    /// * `grad_enabled` - Whether to enable gradient tracking for the tensor.
    ///
    /// # Returns
    /// * `Ok(tensor)` - The created tensor if successful.
    /// * `Err(TensorError)` - The error when creating the tensor.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::Device;
    /// use nove::tensor::Tensor;
    /// let device = Device::get_cuda_if_available(0);
    /// let tensor = Tensor::from_scalar(1.0f64, &device, false).unwrap();
    /// assert_eq!(tensor.to_vec::<f64>().unwrap(), vec![1.0f64]);
    /// let tensor = Tensor::from_scalar(1.0f32, &device, false).unwrap();
    /// assert_eq!(tensor.to_scalar::<f32>().unwrap(), 1.0f32);
    /// ```
    pub fn from_scalar<S>(
        scalar: S,
        device: &Device,
        grad_enabled: bool,
    ) -> Result<Self, TensorError>
    where
        S: candle_core::NdArray + candle_core::WithDType,
    {
        Self::from_array(scalar, device, grad_enabled)
    }

    /// Convert the tensor to a scalar.
    ///
    /// # Generic Type Parameters
    /// * `S` - The element type of the scalar. It supports `f32`, `f64`, `i64`, `u32`, `u8`.
    ///
    /// # Notes
    /// * The tensor must only have one element.
    /// * The dimension number of the tensor must be 0 or 1.
    ///
    /// # Returns
    /// * `Ok(scalar)` - The scalar value if the tensor is a scalar.
    /// * `Err(TensorError)` - The error when converting the tensor to a scalar.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::Device;
    /// use nove::tensor::Tensor;
    /// let device = Device::get_cuda_if_available(0);
    /// let tensor = Tensor::from_scalar(1.0f32, &device, false).unwrap();
    /// assert_eq!(tensor.to_scalar::<f32>().unwrap(), 1.0f32);
    /// let tensor = Tensor::from_scalar(1.0f64, &device, false).unwrap();
    /// assert_eq!(tensor.to_scalar::<f64>().unwrap(), 1.0f64);
    /// let tensor = Tensor::from_scalar(1i64, &device, false).unwrap();
    /// assert_eq!(tensor.to_scalar::<i64>().unwrap(), 1i64);
    /// let tensor = Tensor::from_scalar(1u32, &device, false).unwrap();
    /// assert_eq!(tensor.to_scalar::<u32>().unwrap(), 1u32);
    /// let tensor = Tensor::from_scalar(1u8, &device, false).unwrap();
    /// assert_eq!(tensor.to_scalar::<u8>().unwrap(), 1u8);
    /// ```
    pub fn to_scalar<S>(&self) -> Result<S, TensorError>
    where
        S: candle_core::WithDType,
    {
        // Get the inner tensor.
        let inner = self.data.inner.read()?;
        let inner_tensor = match &*inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        // If the tensor dimension number is 0, it is a scalar tensor.
        // We can directly convert it to a scalar.
        // Otherwise, we need to squeeze the tensor to remove the dimension.
        let dim_num = self.get_dim_num()?;
        match dim_num {
            0 => Ok(inner_tensor.to_scalar::<S>()?),
            _ => {
                let squeezed = inner_tensor.squeeze(0)?;
                Ok(squeezed.to_scalar::<S>()?)
            }
        }
    }

    /// Convert the tensor to a one-dimensional vector.
    ///
    /// # Generic Type Parameters
    /// * `S` - The element type of the vector. It supports `f32`, `f64`, `i64`, `u32`, `u8`.
    ///
    /// # Notes
    /// * The tensor could be any shape, and it will be flattened to a one-dimensional vector.
    ///
    /// # Returns
    /// * `Ok(vec)` - The vector value if the tensor can be converted to a vector.
    /// * `Err(TensorError)` - The error when converting the tensor to a vector.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::Device;
    /// use nove::tensor::Tensor;
    /// let device = Device::get_cuda_if_available(0);
    /// let tensor = Tensor::from_array(&[1.0f64, 2.0f64, 3.0f64], &device, false).unwrap();
    /// assert_eq!(tensor.to_vec::<f64>().unwrap(), vec![1.0f64, 2.0f64, 3.0f64]);
    /// let tensor = Tensor::from_array(&[[1.0f64, 2.0f64, 3.0f64], [4.0f64, 5.0f64, 6.0f64]], &device, false).unwrap();
    /// assert_eq!(tensor.to_vec::<f64>().unwrap(), vec![1.0f64, 2.0f64, 3.0f64, 4.0f64, 5.0f64, 6.0f64]);
    /// let tensor = Tensor::from_array(1.0f64, &device, false).unwrap();
    /// assert_eq!(tensor.to_vec::<f64>().unwrap(), vec![1.0f64]);
    /// ```
    ///
    pub fn to_vec<S>(&self) -> Result<Vec<S>, TensorError>
    where
        S: candle_core::WithDType,
    {
        let inner = self.data.inner.read()?;
        let vec = match &*inner {
            TensorInner::Tensor(tensor) => tensor.flatten_all()?.to_vec1::<S>()?,
            TensorInner::Var(var) => var.flatten_all()?.to_vec1::<S>()?,
        };
        Ok(vec)
    }
}
