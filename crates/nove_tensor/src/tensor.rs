use crate::{
    DType, Device, Shape,
    backend::{BackendError, BackendStorage},
    backpropagation::graph::{GraphNode, OpKind},
};
use std::{
    fmt::Display,
    sync::{Arc, RwLock},
};
use thiserror::Error;

/// Error type for tensor operations.
#[derive(Error, Debug)]
pub enum TensorError {
    #[error("Backend error: {0}")]
    BackendError(#[from] BackendError),

    #[error("Cannot perform gradient operation: tensor has gradients disabled")]
    GradientDisabled,

    #[error("Gradient for tensor not found in gradient store")]
    GradientStoreMissing,

    #[error("Tensor does not have a computed gradient")]
    NoTensorGradient,

    #[error("RwLock poisoned: {0}")]
    RwLockPoisoned(String),

    #[error("Shape mismatch: expected {0:?}, found {1:?}")]
    ShapeMismatch(Shape, Shape),

    #[error("DType mismatch: expected {0:?}, found {1:?}")]
    DTypeMismatch(DType, DType),

    #[error("Device mismatch: expected {0:?}, found {1:?}")]
    DeviceMismatch(Box<Device>, Box<Device>),

    #[error("Invalid dimension: {0}, must be >= -1")]
    InvalidDimension(isize),

    #[error("Invalid tensor operation: {0}")]
    InvalidOperation(String),
}

impl<T> From<std::sync::PoisonError<T>> for TensorError {
    fn from(error: std::sync::PoisonError<T>) -> Self {
        TensorError::RwLockPoisoned(error.to_string())
    }
}

impl PartialEq for TensorError {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (TensorError::GradientDisabled, TensorError::GradientDisabled) => true,
            (TensorError::GradientStoreMissing, TensorError::GradientStoreMissing) => true,
            (TensorError::NoTensorGradient, TensorError::NoTensorGradient) => true,
            (TensorError::RwLockPoisoned(a), TensorError::RwLockPoisoned(b)) => a == b,
            (TensorError::ShapeMismatch(a1, b1), TensorError::ShapeMismatch(a2, b2)) => {
                a1 == a2 && b1 == b2
            }
            (TensorError::DTypeMismatch(a1, b1), TensorError::DTypeMismatch(a2, b2)) => {
                a1 == a2 && b1 == b2
            }
            (TensorError::DeviceMismatch(a1, b1), TensorError::DeviceMismatch(a2, b2)) => {
                a1 == a2 && b1 == b2
            }
            (TensorError::InvalidDimension(a), TensorError::InvalidDimension(b)) => a == b,
            (TensorError::InvalidOperation(a), TensorError::InvalidOperation(b)) => a == b,
            _ => false,
        }
    }
}

/// Inner tensor state.
#[derive(Debug)]
pub(crate) struct TensorData {
    pub(crate) storage: BackendStorage,
    pub(crate) device: Device,
    pub(crate) graph: GraphNode,
    pub(crate) grad: Option<Tensor>,
    pub(crate) name: Option<String>,
    pub(crate) requires_grad: bool,
}

/// The tensor struct.
#[derive(Debug)]
pub struct Tensor {
    pub(crate) data: Arc<RwLock<TensorData>>,
}

impl Clone for Tensor {
    /// Clone the tensor.
    ///
    /// This is a convenience method that calls [`try_clone`](crate::tensor::Tensor::try_clone)
    /// and unwraps the result. If cloning fails, it will panic.
    ///
    /// For error-safe cloning, use [`try_clone`](crate::tensor::Tensor::try_clone) instead.
    ///
    /// # See Also
    /// * [`try_clone`](crate::tensor::Tensor::try_clone) - The fallible version that returns `Result`.
    /// * [`copy`](crate::tensor::Tensor::copy) - Shallow copy that shares underlying data.
    fn clone(&self) -> Self {
        self.try_clone().unwrap()
    }
}

impl Tensor {
    pub(crate) fn from_backend_storage(
        storage: BackendStorage,
        device: Device,
        requires_grad: bool,
        parents: Vec<Tensor>,
        op: OpKind,
    ) -> Self {
        Self::from_backend_parts(storage, device, requires_grad, parents, op, None, None)
    }

    pub(crate) fn from_backend_parts(
        storage: BackendStorage,
        device: Device,
        requires_grad: bool,
        parents: Vec<Tensor>,
        op: OpKind,
        grad: Option<Tensor>,
        name: Option<String>,
    ) -> Self {
        Tensor {
            data: Arc::new(RwLock::new(TensorData {
                storage,
                device,
                graph: GraphNode::new(op, parents),
                grad,
                name,
                requires_grad,
            })),
        }
    }

    pub(crate) fn backend_storage(&self) -> Result<BackendStorage, TensorError> {
        Ok(self.data.read()?.storage.clone())
    }

    pub(crate) fn op_result_with_kind(
        storage: BackendStorage,
        device: Device,
        parents: Vec<Tensor>,
        op: OpKind,
    ) -> Self {
        let requires_grad = parents.iter().any(|parent| {
            parent
                .data
                .read()
                .map(|data| data.requires_grad)
                .unwrap_or(false)
        });
        Self::from_backend_storage(storage, device, requires_grad, parents, op)
    }

    /// Create a shallow copy of the tensor.
    ///
    /// This method only clones the `Arc` reference to the underlying data,
    /// creating a new tensor that shares the same underlying data with the original.
    /// Unlike [`try_clone`](crate::tensor::Tensor::try_clone), this does NOT perform
    /// a deep copy of the actual tensor data.
    ///
    /// The copied tensor will:
    /// - Share the same underlying data allocation (shallow copy)
    /// - Share the same computational graph
    /// - Share the same gradient
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::default();
    ///
    /// // Create a tensor with gradient tracking enabled
    /// let data = [1.0f32, 2.0, 3.0, 4.0];
    /// let original = Tensor::from_slice(&data, &Shape::from(&[2, 2]), &device, true).unwrap();
    ///
    /// // Create a shallow copy
    /// let shallow_copy = original.copy();
    ///
    /// // Modify the original tensor's gradient through backpropagation
    /// let scalar = Tensor::from_scalar(2.0f32, &device, true).unwrap();
    /// let result = original.mul(&scalar).unwrap();
    /// result.backward().unwrap();
    ///
    /// // Both original and shallow copy share the same gradient
    /// let original_grad = original.grad().unwrap().unwrap();
    /// let copy_grad = shallow_copy.grad().unwrap().unwrap();
    /// assert_eq!(original_grad.to_vec::<f32>().unwrap(), copy_grad.to_vec::<f32>().unwrap());
    /// assert_eq!(original_grad.shape().unwrap(), copy_grad.shape().unwrap());
    /// ```
    ///
    /// # See Also
    /// * [`try_clone`](crate::tensor::Tensor::try_clone) - The fallible version for cloning a tensor that returns `Result`.
    /// * [`clone`](crate::tensor::Tensor::clone) - The unfallible version for cloning a tensor that panics on failure.
    pub fn copy(&self) -> Self {
        Tensor {
            data: Arc::clone(&self.data),
        }
    }

    /// Try to clone the tensor.
    ///
    /// This method performs a deep copy of the tensor data, creating a completely
    /// independent tensor. The cloned tensor will:
    /// - Have a new underlying data allocation
    /// - Be disconnected from the computational graph
    /// - Have a gradient tensor if the original tensor had one (gradient also calls `try_clone`)
    /// - Preserve the device, dtype, shape, and name from the original tensor
    ///
    /// # Returns
    /// * `Ok(Tensor)` - A new tensor with cloned data if successful.
    /// * `Err(TensorError)` - The error when cloning the tensor data.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::default();
    ///
    /// // Create a tensor with gradient tracking enabled
    /// let data = [1.0f32, 2.0, 3.0, 4.0];
    /// let original = Tensor::from_slice(&data, &Shape::from(&[2, 2]), &device, true).unwrap();
    ///
    /// // Perform computation to create gradient
    /// let scalar = Tensor::from_scalar(2.0f32, &device, true).unwrap();
    /// let result = original.mul(&scalar).unwrap();
    /// result.backward().unwrap();
    ///
    /// // Create a deep copy
    /// let deep_copy = original.try_clone().unwrap();
    ///
    /// // Deep copy is independent: clearing gradient on original doesn't affect the copy
    /// let mut original_mut = original.copy();
    /// original_mut.clear_grad().unwrap();
    /// assert!(original_mut.grad().unwrap().is_none());
    ///
    /// // Deep copy still has its gradient
    /// let deep_copy_grad = deep_copy.grad().unwrap();
    /// assert!(deep_copy_grad.is_some());
    ///
    /// // Deep copy has same shape and data
    /// assert_eq!(original.shape().unwrap(), deep_copy.shape().unwrap());
    ///
    /// // Deep copy is disconnected from computational graph
    /// // Any operations on original won't affect the deep copy
    /// ```
    ///
    /// # See Also
    /// * [`clone`](crate::tensor::Tensor::clone) - The unfallible version that panics on failure.
    /// * [`copy`](crate::tensor::Tensor::copy) - Shallow copy that shares underlying data.
    pub fn try_clone(&self) -> Result<Self, TensorError> {
        let data = self.data.read()?;
        Ok(Tensor::from_backend_parts(
            data.storage.try_clone()?,
            data.device.clone(),
            data.requires_grad,
            vec![],
            OpKind::Clone,
            match &data.grad {
                Some(grad) => Some(grad.try_clone()?),
                None => None,
            },
            data.name.clone(),
        ))
    }
}

impl AsRef<Tensor> for Tensor {
    fn as_ref(&self) -> &Self {
        self
    }
}

impl Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let data = self.data.read().map_err(|_| std::fmt::Error)?;
        data.storage.fmt_backend(f)
    }
}

impl Tensor {
    /// Update the tensor's backend storage from another tensor.
    ///
    /// # Notes
    /// * If the tensor has enabled gradients, the method will update the tensor's backend storage
    ///   without disconnecting it from the computational graph.
    /// * If the tensor does not have enabled gradients, the method will update the tensor's backend storage
    ///   and disconnect it from the computational graph.
    /// * The source must have the same backend, actual device, dtype and shape.
    /// * The tensor's identity, name, device and gradient setting are preserved.
    ///   The source's name, gradient setting and computational graph are not copied.
    /// * If the metadata does not match, the tensor remains unchanged.
    /// * This method does not check whether the existing graph contains stale forward values.
    ///
    /// # Arguments
    /// * `other` - The tensor to update the backend storage from.
    ///
    /// # Returns
    /// * `Ok(())` - The tensor's backend storage is successfully updated.
    /// * `Err(TensorError)` - The error when updating the tensor's backend storage.
    pub fn update_from_tensor(&self, other: &Tensor) -> Result<(), TensorError> {
        // Release the source read lock before acquiring the target write lock.
        let (other_storage, other_device) = {
            let other_data = other.data.read()?;
            (other_data.storage.clone(), other_data.device.clone())
        };
        let mut self_data = self.data.write()?;
        let expected = self_data.storage.backend_kind()?;
        let found = other_storage.backend_kind()?;
        if expected != found {
            return Err(BackendError::BackendMismatch { expected, found }.into());
        }
        if self_data.device != other_device || !self_data.storage.same_device(&other_storage) {
            return Err(TensorError::DeviceMismatch(
                Box::new(self_data.device.clone()),
                Box::new(other_device),
            ));
        }
        let expected = self_data.storage.dtype()?;
        let found = other_storage.dtype()?;
        if expected != found {
            return Err(TensorError::DTypeMismatch(expected, found));
        }
        let expected = self_data.storage.shape()?;
        let found = other_storage.shape()?;
        if expected != found {
            return Err(TensorError::ShapeMismatch(expected, found));
        }
        let requires_grad = self_data.requires_grad;
        self_data.storage = other_storage;
        if !requires_grad {
            self_data.grad = None;
            self_data.graph.clear_parents();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backpropagation::graph::TensorId;

    fn enabled_cpu_devices() -> Vec<Device> {
        vec![
            #[cfg(feature = "candle-cpu")]
            crate::device::candle::cpu().unwrap(),
            #[cfg(feature = "native-cpu")]
            crate::device::native::cpu().unwrap(),
        ]
    }

    #[test]
    fn try_clone_copies_value_and_gradient_buffers() -> Result<(), TensorError> {
        for device in enabled_cpu_devices() {
            let original =
                Tensor::from_data(vec![1.0f32, 2.0], &device, true)?.require_name("x")?;
            original.sum(None)?.backward()?;
            let copy = original.try_clone()?;
            original.data.write()?.storage.zero_set()?;
            original.grad()?.unwrap().data.write()?.storage.zero_set()?;
            assert_eq!(original.to_vec::<f32>()?, vec![0.0, 0.0]);
            assert_eq!(copy.to_vec::<f32>()?, vec![1.0, 2.0]);
            assert_eq!(copy.grad()?.unwrap().to_vec::<f32>()?, vec![1.0, 1.0]);
            assert_eq!(copy.name()?, Some("x".into()));
            assert_eq!(copy.dtype()?, original.dtype()?);
            assert_eq!(copy.shape()?, original.shape()?);
            assert_eq!(copy.device()?, device);
            assert!(copy.grad_enabled()?);
            assert_ne!(copy.graph_id()?, original.graph_id()?);
            assert!(copy.data.read()?.graph.parents.is_empty());
        }
        Ok(())
    }

    type GraphState = (TensorId, OpKind, Vec<TensorId>, Option<TensorId>);

    fn graph_state(tensor: &Tensor) -> Result<GraphState, TensorError> {
        let data = tensor.data.read()?;
        Ok((
            data.graph.id,
            data.graph.op.clone(),
            data.graph
                .parents
                .iter()
                .map(Tensor::graph_id)
                .collect::<Result<_, _>>()?,
            data.grad.as_ref().map(Tensor::graph_id).transpose()?,
        ))
    }

    #[test]
    fn update_failure_preserves_graph_and_gradient_state() -> Result<(), TensorError> {
        for device in enabled_cpu_devices() {
            for requires_grad in [false, true] {
                let leaf = Tensor::from_data(vec![1.0f32, 2.0], &device, true)?;
                leaf.sum(None)?.backward()?;
                let target = leaf.affine(2.0, 0.0)?;
                {
                    let mut data = target.data.write()?;
                    data.requires_grad = requires_grad;
                    data.grad = leaf.grad()?;
                    data.name = Some("target".into());
                }
                let before = graph_state(&target)?;
                let bad = Tensor::from_data(vec![7.0f32], &device, false)?;
                assert!(matches!(
                    target.update_from_tensor(&bad),
                    Err(TensorError::ShapeMismatch(..))
                ));
                assert_eq!(graph_state(&target)?, before);
                assert_eq!(target.to_vec::<f32>()?, vec![2.0, 4.0]);
                assert_eq!(target.grad()?.unwrap().to_vec::<f32>()?, vec![1.0, 1.0]);
                let good = Tensor::from_data(vec![7.0f32, 8.0], &device, false)?;
                target.update_from_tensor(&good)?;
                assert_eq!(target.to_vec::<f32>()?, vec![7.0, 8.0]);
                assert_eq!(target.name()?, Some("target".into()));
                assert_eq!(target.device()?, device);
                assert_eq!(target.grad_enabled()?, requires_grad);
                if requires_grad {
                    assert_eq!(graph_state(&target)?, before);
                } else {
                    let after = graph_state(&target)?;
                    assert_eq!(after.0, before.0);
                    assert_eq!(after.1, before.1);
                    assert!(after.2.is_empty());
                    assert!(after.3.is_none());
                }
            }
        }
        Ok(())
    }
}
