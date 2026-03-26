use std::sync::{Arc, RwLock};

use crate::{
    DType, Device, Shape, Tensor, TensorError,
    tensor::{TensorData, TensorInner},
};

impl Tensor {
    /// Create a new tensor with random values uniformly distributed in the specified range.
    ///
    /// # Parameters
    /// * `low` - The lower bound of the uniform distribution.
    /// * `high` - The upper bound of the uniform distribution.
    /// * `shape` - The shape of the tensor.
    /// * `device` - The device to store the tensor.
    /// * `grad_enabled` - Whether to enable gradient tracking for the tensor.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The new tensor if successful.
    /// * `Err(TensorError)` - The error when creating the tensor.
    ///
    /// # Examples
    /// ```
    /// use nove_tensor::{Device, Shape, Tensor};
    ///
    /// // Create a 2x3 tensor with random values uniformly distributed between 0.0 and 1.0
    /// let device = Device::cpu();
    /// let shape = Shape::from(&[2, 3]);
    ///
    /// // Create the tensor without gradient tracking
    /// let tensor = Tensor::rand(0.0f32, 1.0f32, &shape, &device, false).unwrap();
    ///
    /// // Verify tensor properties
    /// assert_eq!(tensor.shape().unwrap(), shape);
    /// assert_eq!(tensor.device().unwrap(), device);
    ///
    /// // All values should be in the range [0.0, 1.0)
    /// let data = tensor.to_vec::<f32>().unwrap();
    /// for &value in &data {
    ///     assert!(value >= 0.0 && value < 1.0);
    /// }
    /// // Note: This method generates random values. The actual values will vary between runs
    /// // unless the random number generator seed is fixed externally.
    /// ```
    pub fn rand<T>(
        low: T,
        high: T,
        shape: &Shape,
        device: &Device,
        grad_enabled: bool,
    ) -> Result<Self, TensorError>
    where
        T: candle_core::FloatDType,
    {
        let inner = match grad_enabled {
            true => TensorInner::Var(candle_core::Var::rand(low, high, shape, device)?),
            false => TensorInner::Tensor(candle_core::Tensor::rand(low, high, shape, device)?),
        };

        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner,
                device: device.clone(),
                parents: vec![],
                grad: None,
                name: None,
            })),
        })
    }

    /// Create a new tensor with random values normally distributed with mean `mean` and standard deviation `std`.
    ///
    /// # Parameters
    /// * `mean` - The mean of the normal distribution.
    /// * `std` - The standard deviation of the normal distribution.
    /// * `shape` - The shape of the tensor.
    /// * `device` - The device to store the tensor.
    /// * `grad_enabled` - Whether to enable gradient tracking for the tensor.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The new tensor if successful.
    /// * `Err(TensorError)` - The error when creating the tensor.
    ///
    /// # Examples
    /// ```
    /// use nove_tensor::{Device, Shape, Tensor};
    ///
    /// // Create a 2x3 tensor with random values from normal distribution (mean=0.0, std=1.0)
    /// let device = Device::cpu();
    /// let shape = Shape::from(&[2, 3]);
    ///
    /// // Create the tensor without gradient tracking
    /// let tensor = Tensor::randn(0.0f32, 1.0f32, &shape, &device, false).unwrap();
    ///
    /// // Verify tensor properties
    /// assert_eq!(tensor.shape().unwrap(), shape);
    /// assert_eq!(tensor.device().unwrap(), device);
    ///
    /// // Note: This method generates random values from a normal distribution.
    /// // The actual values will vary between runs unless the random number generator
    /// // seed is fixed externally. We can verify the tensor has the expected size.
    /// let data = tensor.to_vec::<f32>().unwrap();
    /// assert_eq!(data.len(), 6); // 2x3 = 6 elements
    /// ```
    pub fn randn<T>(
        mean: T,
        std: T,
        shape: &Shape,
        device: &Device,
        grad_enabled: bool,
    ) -> Result<Self, TensorError>
    where
        T: candle_core::FloatDType,
    {
        let inner = match grad_enabled {
            true => TensorInner::Var(candle_core::Var::randn(mean, std, shape, device)?),
            false => TensorInner::Tensor(candle_core::Tensor::randn(mean, std, shape, device)?),
        };

        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner,
                device: device.clone(),
                parents: vec![],
                grad: None,
                name: None,
            })),
        })
    }

    /// Create a new tensor filled with zeros.
    ///
    /// # Parameters
    /// * `shape` - The shape of the tensor.
    /// * `dtype` - The data type of the tensor.
    /// * `device` - The device to store the tensor.
    /// * `grad_enabled` - Whether to enable gradient tracking for the tensor.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The new tensor if successful.
    /// * `Err(TensorError)` - The error when creating the tensor.
    ///
    /// # Examples
    /// ```
    /// use nove_tensor::{DType, Device, Shape, Tensor};
    ///
    /// // Create a 2x3 tensor of zeros with f32 data type on CPU
    /// let device = Device::cpu();
    /// let shape = Shape::from(&[2, 3]);
    /// let dtype = DType::F32;
    ///
    /// // Create the tensor without gradient tracking
    /// let tensor = Tensor::zeros(&shape, &dtype, &device, false).unwrap();
    ///
    /// // Verify tensor properties
    /// assert_eq!(tensor.shape().unwrap(), shape);
    /// assert_eq!(tensor.dtype().unwrap(), dtype);
    /// assert_eq!(tensor.device().unwrap(), device);
    ///
    /// // The tensor should contain all zeros
    /// let data = tensor.to_vec::<f32>().unwrap();
    /// assert_eq!(data, vec![0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32]);
    /// ```
    pub fn zeros(
        shape: &Shape,
        dtype: &DType,
        device: &Device,
        grad_enabled: bool,
    ) -> Result<Self, TensorError> {
        let inner = match grad_enabled {
            true => TensorInner::Var(candle_core::Var::zeros(shape, *dtype, device)?),
            false => TensorInner::Tensor(candle_core::Tensor::zeros(shape, *dtype, device)?),
        };
        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner,
                device: device.clone(),
                parents: vec![],
                grad: None,
                name: None,
            })),
        })
    }

    /// Create a new tensor filled with ones.
    ///
    /// # Parameters
    /// * `shape` - The shape of the tensor.
    /// * `dtype` - The data type of the tensor.
    /// * `device` - The device to store the tensor.
    /// * `grad_enabled` - Whether to enable gradient tracking for the tensor.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The new tensor if successful.
    /// * `Err(TensorError)` - The error when creating the tensor.
    ///
    /// # Examples
    /// ```
    /// use nove_tensor::{DType, Device, Shape, Tensor};
    ///
    /// // Create a 2x3 tensor of ones with f32 data type on CPU
    /// let device = Device::cpu();
    /// let shape = Shape::from(&[2, 3]);
    /// let dtype = DType::F32;
    ///
    /// // Create the tensor without gradient tracking
    /// let tensor = Tensor::ones(&shape, &dtype, &device, false).unwrap();
    ///
    /// // Verify tensor properties
    /// assert_eq!(tensor.shape().unwrap(), shape);
    /// assert_eq!(tensor.dtype().unwrap(), dtype);
    /// assert_eq!(tensor.device().unwrap(), device);
    ///
    /// // The tensor should contain all ones
    /// let data = tensor.to_vec::<f32>().unwrap();
    /// assert_eq!(data, vec![1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32]);
    /// ```
    pub fn ones(
        shape: &Shape,
        dtype: &DType,
        device: &Device,
        grad_enabled: bool,
    ) -> Result<Self, TensorError> {
        let inner = match grad_enabled {
            true => TensorInner::Var(candle_core::Var::ones(shape, *dtype, device)?),
            false => TensorInner::Tensor(candle_core::Tensor::ones(shape, *dtype, device)?),
        };
        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner,
                device: device.clone(),
                parents: vec![],
                grad: None,
                name: None,
            })),
        })
    }
}
