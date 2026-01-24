use std::sync::{Arc, RwLock};

use crate::{
    Device, Shape, Tensor, TensorError,
    tensor::{TensorData, TensorInner},
};

impl Tensor {
    /// Deep clone the data from this tensor to a new tensor.
    ///
    /// # Notes
    /// * Because the cloned tensor is a new tensor, it will not be connected to the previous computation graph.
    ///
    /// # Returns
    /// * `Ok(tensor)` - The cloned tensor if successful.
    /// * `Err(TensorError)` - The error when cloning the tensor.
    pub fn deep_clone(&self) -> Result<Self, TensorError> {
        let inner = match &*self.data.inner.read()? {
            TensorInner::Tensor(tensor) => TensorInner::Tensor(tensor.detach()),
            TensorInner::Var(var) => {
                TensorInner::Var(candle_core::Var::from_tensor(&var.detach())?)
            }
        };
        let device = self.data.device.read()?.clone();
        let grad = match &*self.data.grad.read()? {
            Some(grad) => Some(grad.detach()),
            None => None,
        };
        Ok(Self {
            data: Arc::new(TensorData {
                inner: RwLock::new(inner),
                device: RwLock::new(device),
                grad: RwLock::new(grad),
                parents: RwLock::new(vec![]),
            }),
        })
    }

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
    /// * `Ok(tensor)` - The new tensor if successful.
    /// * `Err(TensorError)` - The error when creating the tensor.
    pub fn from_random_uniform<T>(
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
            data: Arc::new(TensorData {
                inner: RwLock::new(inner),
                device: RwLock::new(device.clone()),
                parents: RwLock::new(vec![]),
                grad: RwLock::new(None),
            }),
        })
    }
}
