use crate::{
    DType, Device, Shape, Tensor, TensorError,
    backend::{BackendStorage, FloatTensorElement},
    backpropagation::graph::OpKind,
};

impl Tensor {
    /// Create a new tensor with random values uniformly distributed in the specified range.
    ///
    /// # Parameters
    /// * `low` - The lower bound of the uniform distribution.
    /// * `high` - The upper bound of the uniform distribution.
    /// * `shape` - The shape of the tensor.
    /// * `device` - The device to store the tensor.
    /// * `requires_grad` - Whether to enable gradient tracking for the tensor.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The new tensor if successful.
    /// * `Err(TensorError)` - The error when creating the tensor.
    ///
    /// # Examples
    /// * Create a 2x3 tensor with random values uniformly distributed between 0.0 and 1.0 (requires_grad=false)
    /// ```
    /// use nove::tensor::{Device, DType, Shape, Tensor};
    ///
    /// let device = Device::default();
    /// let shape = Shape::from(&[2, 3]);
    /// let dtype = DType::F32;
    ///
    /// let tensor = Tensor::rand(0.0f32, 1.0f32, &shape, &device, false).unwrap();
    ///
    /// assert_eq!(tensor.shape().unwrap(), shape);
    /// assert_eq!(tensor.dtype().unwrap(), dtype);
    /// assert_eq!(tensor.device().unwrap(), device);
    ///
    /// let data = tensor.to_vec::<f32>().unwrap();
    /// assert_eq!(data.len(), 6);
    /// for &value in &data {
    ///     assert!(value >= 0.0 && value < 1.0);
    /// }
    /// ```
    /// * Create a random tensor with gradient tracking and verify backward propagation
    /// ```
    /// use nove::tensor::{Device, DType, Shape, Tensor};
    ///
    /// let device = Device::default();
    /// let shape = Shape::from(&[2, 3]);
    /// let dtype = DType::F32;
    ///
    /// let tensor = Tensor::rand(0.0f32, 1.0f32, &shape, &device, true).unwrap();
    ///
    /// assert_eq!(tensor.shape().unwrap(), shape);
    /// assert_eq!(tensor.dtype().unwrap(), dtype);
    /// assert_eq!(tensor.device().unwrap(), device);
    ///
    /// let data = tensor.to_vec::<f32>().unwrap();
    /// assert_eq!(data.len(), 6);
    /// for &value in &data {
    ///     assert!(value >= 0.0 && value < 1.0);
    /// }
    ///
    /// let sum = tensor.sum(None).unwrap();
    /// sum.backward().unwrap();
    ///
    /// let grad = tensor.grad().unwrap().unwrap();
    /// assert_eq!(grad.shape().unwrap(), shape);
    /// assert_eq!(grad.to_vec::<f32>().unwrap(), vec![1.0; 6]);
    /// ```
    pub fn rand<T>(
        low: T,
        high: T,
        shape: &Shape,
        device: &Device,
        requires_grad: bool,
    ) -> Result<Self, TensorError>
    where
        T: FloatTensorElement,
    {
        let storage = BackendStorage::rand(low, high, shape, device, requires_grad)?;
        Ok(Self::from_backend_storage(
            storage,
            device.clone(),
            requires_grad,
            vec![],
            OpKind::Leaf,
        ))
    }

    /// Create a new tensor with random values normally distributed with mean `mean` and standard deviation `std`.
    ///
    /// # Parameters
    /// * `mean` - The mean of the normal distribution.
    /// * `std` - The standard deviation of the normal distribution.
    /// * `shape` - The shape of the tensor.
    /// * `device` - The device to store the tensor.
    /// * `requires_grad` - Whether to enable gradient tracking for the tensor.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The new tensor if successful.
    /// * `Err(TensorError)` - The error when creating the tensor.
    ///
    /// # Examples
    /// * Create a 2x3 tensor with random values from normal distribution (requires_grad=false)
    /// ```
    /// use nove::tensor::{Device, DType, Shape, Tensor};
    ///
    /// let device = Device::default();
    /// let shape = Shape::from(&[2, 3]);
    /// let dtype = DType::F32;
    ///
    /// let tensor = Tensor::randn(0.0f32, 1.0f32, &shape, &device, false).unwrap();
    ///
    /// assert_eq!(tensor.shape().unwrap(), shape);
    /// assert_eq!(tensor.dtype().unwrap(), dtype);
    /// assert_eq!(tensor.device().unwrap(), device);
    ///
    /// let data = tensor.to_vec::<f32>().unwrap();
    /// assert_eq!(data.len(), 6);
    /// ```
    /// * Create a random tensor from normal distribution with gradient tracking and verify backward propagation
    /// ```
    /// use nove::tensor::{Device, DType, Shape, Tensor};
    ///
    /// let device = Device::default();
    /// let shape = Shape::from(&[2, 3]);
    /// let dtype = DType::F32;
    ///
    /// let tensor = Tensor::randn(0.0f32, 1.0f32, &shape, &device, true).unwrap();
    ///
    /// assert_eq!(tensor.shape().unwrap(), shape);
    /// assert_eq!(tensor.dtype().unwrap(), dtype);
    /// assert_eq!(tensor.device().unwrap(), device);
    ///
    /// let data = tensor.to_vec::<f32>().unwrap();
    /// assert_eq!(data.len(), 6);
    ///
    /// let sum = tensor.sum(None).unwrap();
    /// sum.backward().unwrap();
    ///
    /// let grad = tensor.grad().unwrap().unwrap();
    /// assert_eq!(grad.shape().unwrap(), shape);
    /// assert_eq!(grad.to_vec::<f32>().unwrap(), vec![1.0; 6]);
    /// ```
    pub fn randn<T>(
        mean: T,
        std: T,
        shape: &Shape,
        device: &Device,
        requires_grad: bool,
    ) -> Result<Self, TensorError>
    where
        T: FloatTensorElement,
    {
        let storage = BackendStorage::randn(mean, std, shape, device, requires_grad)?;
        Ok(Self::from_backend_storage(
            storage,
            device.clone(),
            requires_grad,
            vec![],
            OpKind::Leaf,
        ))
    }

    /// Create a new tensor filled with zeros.
    ///
    /// # Parameters
    /// * `shape` - The shape of the tensor.
    /// * `dtype` - The data type of the tensor.
    /// * `device` - The device to store the tensor.
    /// * `requires_grad` - Whether to enable gradient tracking for the tensor.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The new tensor if successful.
    /// * `Err(TensorError)` - The error when creating the tensor.
    ///
    /// # Examples
    /// * Create a 2x3 tensor of zeros with f32 data type (requires_grad=false)
    /// ```
    /// use nove::tensor::{Device, DType, Shape, Tensor};
    ///
    /// let device = Device::default();
    /// let shape = Shape::from(&[2, 3]);
    /// let dtype = DType::F32;
    ///
    /// let tensor = Tensor::zeros(&shape, &dtype, &device, false).unwrap();
    ///
    /// assert_eq!(tensor.shape().unwrap(), shape);
    /// assert_eq!(tensor.dtype().unwrap(), dtype);
    /// assert_eq!(tensor.device().unwrap(), device);
    ///
    /// let data = tensor.to_vec::<f32>().unwrap();
    /// assert_eq!(data, vec![0.0f32; 6]);
    /// ```
    /// * Create a tensor of zeros with gradient tracking and verify backward propagation
    /// ```
    /// use nove::tensor::{Device, DType, Shape, Tensor};
    ///
    /// let device = Device::default();
    /// let shape = Shape::from(&[2, 3]);
    /// let dtype = DType::F32;
    ///
    /// let tensor = Tensor::zeros(&shape, &dtype, &device, true).unwrap();
    ///
    /// assert_eq!(tensor.shape().unwrap(), shape);
    /// assert_eq!(tensor.dtype().unwrap(), dtype);
    /// assert_eq!(tensor.device().unwrap(), device);
    ///
    /// let data = tensor.to_vec::<f32>().unwrap();
    /// assert_eq!(data, vec![0.0f32; 6]);
    ///
    /// let sum = tensor.sum(None).unwrap();
    /// sum.backward().unwrap();
    ///
    /// let grad = tensor.grad().unwrap().unwrap();
    /// assert_eq!(grad.shape().unwrap(), shape);
    /// assert_eq!(grad.to_vec::<f32>().unwrap(), vec![1.0; 6]);
    /// ```
    pub fn zeros(
        shape: &Shape,
        dtype: &DType,
        device: &Device,
        requires_grad: bool,
    ) -> Result<Self, TensorError> {
        let storage = BackendStorage::zeros(shape, *dtype, device, requires_grad)?;
        Ok(Self::from_backend_storage(
            storage,
            device.clone(),
            requires_grad,
            vec![],
            OpKind::Leaf,
        ))
    }

    /// Create a new tensor filled with ones.
    ///
    /// # Parameters
    /// * `shape` - The shape of the tensor.
    /// * `dtype` - The data type of the tensor.
    /// * `device` - The device to store the tensor.
    /// * `requires_grad` - Whether to enable gradient tracking for the tensor.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The new tensor if successful.
    /// * `Err(TensorError)` - The error when creating the tensor.
    ///
    /// # Examples
    /// * Create a 2x3 tensor of ones with f32 data type (requires_grad=false)
    /// ```
    /// use nove::tensor::{Device, DType, Shape, Tensor};
    ///
    /// let device = Device::default();
    /// let shape = Shape::from(&[2, 3]);
    /// let dtype = DType::F32;
    ///
    /// let tensor = Tensor::ones(&shape, &dtype, &device, false).unwrap();
    ///
    /// assert_eq!(tensor.shape().unwrap(), shape);
    /// assert_eq!(tensor.dtype().unwrap(), dtype);
    /// assert_eq!(tensor.device().unwrap(), device);
    ///
    /// let data = tensor.to_vec::<f32>().unwrap();
    /// assert_eq!(data, vec![1.0f32; 6]);
    /// ```
    /// * Create a tensor of ones with gradient tracking and verify backward propagation
    /// ```
    /// use nove::tensor::{Device, DType, Shape, Tensor};
    ///
    /// let device = Device::default();
    /// let shape = Shape::from(&[2, 3]);
    /// let dtype = DType::F32;
    ///
    /// let tensor = Tensor::ones(&shape, &dtype, &device, true).unwrap();
    ///
    /// assert_eq!(tensor.shape().unwrap(), shape);
    /// assert_eq!(tensor.dtype().unwrap(), dtype);
    /// assert_eq!(tensor.device().unwrap(), device);
    ///
    /// let data = tensor.to_vec::<f32>().unwrap();
    /// assert_eq!(data, vec![1.0f32; 6]);
    ///
    /// let sum = tensor.sum(None).unwrap();
    /// sum.backward().unwrap();
    ///
    /// let grad = tensor.grad().unwrap().unwrap();
    /// assert_eq!(grad.shape().unwrap(), shape);
    /// assert_eq!(grad.to_vec::<f32>().unwrap(), vec![1.0; 6]);
    /// ```
    pub fn ones(
        shape: &Shape,
        dtype: &DType,
        device: &Device,
        requires_grad: bool,
    ) -> Result<Self, TensorError> {
        let storage = BackendStorage::ones(shape, *dtype, device, requires_grad)?;
        Ok(Self::from_backend_storage(
            storage,
            device.clone(),
            requires_grad,
            vec![],
            OpKind::Leaf,
        ))
    }

    /// Create a new tensor with the same shape and device as the input tensor, filled with zeros.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The new tensor filled with zeros.
    /// * `Err(TensorError)` - The error when creating the tensor.
    ///
    /// # Examples
    /// * Create a tensor of zeros with the same shape as the input tensor
    /// ```
    /// use nove::tensor::{Device, Tensor};
    ///
    /// let device = Device::default();
    /// let tensor = Tensor::from_data(vec![1.0f32, 2.0, 3.0], &device, false).unwrap();
    ///
    /// let zeros = tensor.zeros_like().unwrap();
    ///
    /// assert_eq!(zeros.shape().unwrap(), tensor.shape().unwrap());
    /// assert_eq!(zeros.dtype().unwrap(), tensor.dtype().unwrap());
    /// assert_eq!(zeros.device().unwrap(), tensor.device().unwrap());
    ///
    /// let data = zeros.to_vec::<f32>().unwrap();
    /// assert_eq!(data, vec![0.0f32, 0.0f32, 0.0f32]);
    /// ```
    /// * Create a zeros_like tensor from a tensor with gradient tracking and verify backward propagation
    /// ```
    /// use nove::tensor::{Device, Tensor};
    ///
    /// let device = Device::default();
    /// let tensor = Tensor::from_data(vec![1.0f32, 2.0, 3.0], &device, true).unwrap();
    ///
    /// let zeros = tensor.zeros_like().unwrap();
    ///
    /// assert_eq!(zeros.shape().unwrap(), tensor.shape().unwrap());
    /// assert_eq!(zeros.dtype().unwrap(), tensor.dtype().unwrap());
    /// assert_eq!(zeros.device().unwrap(), tensor.device().unwrap());
    ///
    /// let data = zeros.to_vec::<f32>().unwrap();
    /// assert_eq!(data, vec![0.0f32, 0.0f32, 0.0f32]);
    ///
    /// let result = tensor.add(&zeros).unwrap();
    /// result.backward().unwrap();
    ///
    /// let grad = tensor.grad().unwrap().unwrap();
    /// assert_eq!(grad.shape().unwrap(), tensor.shape().unwrap());
    /// assert_eq!(grad.to_vec::<f32>().unwrap(), vec![1.0; 3]);
    /// ```
    pub fn zeros_like(&self) -> Result<Self, TensorError> {
        let data = self.data.read()?;
        let storage = data.storage.zeros_like()?;
        let device = data.device.clone();
        drop(data);

        Ok(Self::op_result_with_kind(
            storage,
            device,
            vec![self.copy()],
            OpKind::ZerosLike,
        ))
    }

    /// Create a new tensor with the same shape and device as the input tensor, filled with ones.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The new tensor filled with ones.
    /// * `Err(TensorError)` - The error when creating the tensor.
    ///
    /// # Examples
    /// * Create a tensor of ones with the same shape as the input tensor
    /// ```
    /// use nove::tensor::{Device, Tensor};
    ///
    /// let device = Device::default();
    /// let tensor = Tensor::from_data(vec![1.0f32, 2.0, 3.0], &device, false).unwrap();
    ///
    /// let ones = tensor.ones_like().unwrap();
    ///
    /// assert_eq!(ones.shape().unwrap(), tensor.shape().unwrap());
    /// assert_eq!(ones.dtype().unwrap(), tensor.dtype().unwrap());
    /// assert_eq!(ones.device().unwrap(), tensor.device().unwrap());
    ///
    /// let data = ones.to_vec::<f32>().unwrap();
    /// assert_eq!(data, vec![1.0f32, 1.0f32, 1.0f32]);
    /// ```
    /// * Create a ones_like tensor from a tensor with gradient tracking and verify backward propagation
    /// ```
    /// use nove::tensor::{Device, Tensor};
    ///
    /// let device = Device::default();
    /// let tensor = Tensor::from_data(vec![1.0f32, 2.0, 3.0], &device, true).unwrap();
    ///
    /// let ones = tensor.ones_like().unwrap();
    ///
    /// assert_eq!(ones.shape().unwrap(), tensor.shape().unwrap());
    /// assert_eq!(ones.dtype().unwrap(), tensor.dtype().unwrap());
    /// assert_eq!(ones.device().unwrap(), tensor.device().unwrap());
    ///
    /// let data = ones.to_vec::<f32>().unwrap();
    /// assert_eq!(data, vec![1.0f32, 1.0f32, 1.0f32]);
    ///
    /// // ones_like creates a new tensor that doesn't have gradient tracking by itself
    /// // but it can be used in computations with other gradient-enabled tensors
    /// let result = tensor.mul(&ones).unwrap();
    /// result.backward().unwrap();
    ///
    /// let grad = tensor.grad().unwrap().unwrap();
    /// assert_eq!(grad.shape().unwrap(), tensor.shape().unwrap());
    /// assert_eq!(grad.to_vec::<f32>().unwrap(), vec![1.0; 3]);
    /// ```
    pub fn ones_like(&self) -> Result<Self, TensorError> {
        let data = self.data.read()?;
        let storage = data.storage.ones_like()?;
        let device = data.device.clone();
        drop(data);

        Ok(Self::op_result_with_kind(
            storage,
            device,
            vec![self.copy()],
            OpKind::OnesLike,
        ))
    }
}
