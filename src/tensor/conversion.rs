use std::sync::{Arc, RwLock};

use crate::tensor::{
    Device, Tensor, TensorError,
    tensor::{TensorData, TensorInner},
};

impl Tensor {
    /// Create a new tensor from the given data.
    ///
    /// # Notes
    /// * The type of data supported by this function includes scalar, vector, and slice.
    /// * The element type of the data supported by this function includes `f32`, `f64`, `i64`, `u32`, `u8`.
    ///
    /// # Arguments
    /// * `data` - The data to create the tensor from.
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
    /// let device = Device::get_cpu();
    ///
    /// // Create a tensor from a scalar
    /// let tensor = Tensor::from_data(1.0f32, &device, false).unwrap();
    /// println!("{:?}", tensor);
    ///
    /// // Create a tensor from a 2D vector
    /// let vec = vec![vec![1.0f32, 2.0f32, 3.0f32], vec![4.0f32, 5.0f32, 6.0f32]];
    /// let tensor = Tensor::from_data(vec, &device, false).unwrap();
    /// println!("{:?}", tensor);
    ///
    /// // Create a tensor from a 2D slice
    /// let slice = &[[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]];
    /// let tensor = Tensor::from_data(slice, &device, false).unwrap();
    /// println!("{:?}", tensor);
    /// ```
    pub fn from_data<A>(data: A, device: &Device, grad_enabled: bool) -> Result<Self, TensorError>
    where
        A: candle_core::NdArray,
    {
        let inner = match grad_enabled {
            true => TensorInner::Var(candle_core::Var::new(data, &device)?),
            false => TensorInner::Tensor(candle_core::Tensor::new(data, &device)?),
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
    /// * This function is an alias of `from_data` but accepts a scalar value with more explicit type.
    /// * The type of the scalar supported by this function includes `f32`, `f64`, `i64`, `u32`, `u8`.
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
    /// let device = Device::get_cpu();
    ///
    /// let tensor = Tensor::from_scalar(1.0f32, &device, false).unwrap();
    /// println!("{:?}", tensor);
    /// ```
    pub fn from_scalar<S>(
        scalar: S,
        device: &Device,
        grad_enabled: bool,
    ) -> Result<Self, TensorError>
    where
        S: candle_core::NdArray + candle_core::WithDType,
    {
        Self::from_data(scalar, device, grad_enabled)
    }

    /// Convert the tensor to a scalar.
    ///
    /// # Generic Type Parameters
    /// * `S` - The element type of the scalar. It supports `f32`, `f64`, `i64`, `u32`, `u8`.
    ///
    /// # Notes
    /// * The tensor must only have one element.
    ///
    /// # Returns
    /// * `Ok(scalar)` - The scalar value if the tensor is a scalar.
    /// * `Err(TensorError)` - The error when converting the tensor to a scalar.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::Device;
    /// use nove::tensor::Tensor;
    /// let device = Device::get_cpu();
    ///
    /// let tensor = Tensor::from_scalar(1.0f32, &device, false).unwrap();
    /// println!("{:?}", tensor.to_scalar::<f32>().unwrap());
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
    /// let device = Device::get_cpu();
    ///
    /// let tensor = Tensor::from_data(&[[1.0f64, 2.0f64, 3.0f64], [4.0f64, 5.0f64, 6.0f64]], &device, false).unwrap();
    /// println!("{:?}", tensor.to_vec::<f64>().unwrap());
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
