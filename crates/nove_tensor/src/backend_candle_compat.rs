use crate::{Device, Tensor, TensorError, backpropagation::graph::OpKind};

impl Tensor {
    /// Create a Nove tensor from a Candle tensor.
    ///
    /// # Notes
    /// The Candle tensor is copied to the target Nove device and detached from
    /// Candle autograd. Gradient tracking is represented only by Nove tensor
    /// metadata and Nove's computation graph.
    ///
    /// # Arguments
    /// * `tensor` - The Candle tensor to create the tensor from.
    /// * `device` - The Nove device descriptor to place the tensor on.
    /// * `requires_grad` - Whether to enable gradient tracking for the tensor.
    ///
    /// # Returns
    /// * `Ok(Self)` - The created tensor if successful.
    /// * `Err(TensorError)` - The error when creating the tensor.
    ///
    /// # Examples
    /// ```
    /// use candle_core::Tensor as CandleTensor;
    /// use nove::tensor::{Device, Tensor};
    ///
    /// let device = Device::default();
    /// let candle_device = device.to_candle_device().unwrap();
    ///
    /// let candle_tensor = CandleTensor::from_slice(
    ///     &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
    ///     &[2, 3],
    ///     &candle_device,
    /// )
    /// .unwrap();
    ///
    /// let nove_tensor = Tensor::from_candle_tensor(candle_tensor, &device, false).unwrap();
    ///
    /// assert_eq!(nove_tensor.shape().unwrap().dims(), &[2, 3]);
    /// assert_eq!(
    ///     nove_tensor.to_vec::<f32>().unwrap(),
    ///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    /// );
    /// ```
    ///
    /// # See Also
    /// * [`Tensor::to_candle_tensor`] - Convert a Nove tensor into a detached Candle tensor.
    pub fn from_candle_tensor(
        tensor: candle_core::Tensor,
        device: &Device,
        requires_grad: bool,
    ) -> Result<Self, TensorError> {
        let storage = nove_backend::backend::candle::storage_from_candle_tensor(tensor, device)?;
        Ok(Self::from_backend_storage(
            storage,
            device.clone(),
            requires_grad,
            vec![],
            OpKind::Leaf,
        ))
    }

    /// Convert the tensor to a detached `candle_core::Tensor`.
    ///
    /// # Notes
    /// This is an interoperability boundary for raw Candle tensors only. Nove
    /// does not expose Candle autograd state or Candle gradient stores.
    ///
    /// # Returns
    /// * `Ok(candle_core::Tensor)` - The Candle tensor if successful.
    /// * `Err(TensorError)` - The error when converting the tensor.
    ///
    /// # Examples
    /// ```
    /// use candle_core::Tensor as CandleTensor;
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::default();
    /// let nove_tensor = Tensor::from_slice(
    ///     &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
    ///     &Shape::from(&[2, 3]),
    ///     &device,
    ///     false,
    /// )
    /// .unwrap();
    ///
    /// let candle_tensor: CandleTensor = nove_tensor.to_candle_tensor().unwrap();
    ///
    /// assert_eq!(candle_tensor.dims(), &[2, 3]);
    /// assert_eq!(
    ///     candle_tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
    ///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    /// );
    /// ```
    ///
    /// # See Also
    /// * [`Tensor::from_candle_tensor`] - Create a Nove tensor from a detached Candle tensor.
    pub fn to_candle_tensor(&self) -> Result<candle_core::Tensor, TensorError> {
        let storage = self.backend_storage()?;
        Ok(nove_backend::backend::candle::storage_to_candle_tensor(
            &storage,
        )?)
    }
}
