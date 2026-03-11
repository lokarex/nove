use crate::{DType, Device, Shape};
use crossbeam::queue::SegQueue;
use dashmap::DashMap;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::{
    fmt::Display,
    sync::{Arc, RwLock},
};
use thiserror::Error;

/// Error type for tensor operations.
#[derive(Error, Debug)]
pub enum TensorError {
    #[error("Error from Candle: {0}")]
    CandleError(#[from] candle_core::Error),

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
            _ => false,
        }
    }
}

/// Inner representation of tensor data.
#[derive(Clone, Debug)]
pub(crate) enum TensorInner {
    Tensor(candle_core::Tensor),
    Var(candle_core::Var),
}

/// The data structure for a tensor.
///
/// # Fields
/// * `inner` - The inner representation of the tensor data.
/// * `parents` - The list of parent tensors used to backpropagate gradients.
/// * `grad` - The gradient of the tensor data.
/// * `name` - The name of the tensor.
#[derive(Debug)]
pub(crate) struct TensorData {
    pub(crate) inner: TensorInner,
    pub(crate) device: Device,
    pub(crate) parents: Vec<Tensor>,
    pub(crate) grad: Option<Tensor>,
    pub(crate) name: Option<String>,
}

/// The tensor struct.
///
/// # Fields
/// * `data` - The data structure of the tensor.
///
/// # See Also
/// * [`try_clone`](crate::tensor::Tensor::try_clone) - The fallible version for cloning a tensor that returns `Result`.
/// * [`clone`](crate::tensor::Tensor::clone) - The unfallible version for cloning a tensor that panics on failure.
/// * [`copy`](crate::tensor::Tensor::copy) - Shallow copy that shares underlying data.
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
    /// # See Also
    /// * [`clone`](crate::tensor::Tensor::clone) - The unfallible version that panics on failure.
    /// * [`copy`](crate::tensor::Tensor::copy) - Shallow copy that shares underlying data.
    pub fn try_clone(&self) -> Result<Self, TensorError> {
        let data = self.data.read()?;
        Ok(Tensor {
            data: Arc::new(RwLock::new(TensorData {
                inner: match &data.inner {
                    TensorInner::Tensor(tensor) => TensorInner::Tensor(tensor.copy()?),
                    TensorInner::Var(var) => {
                        TensorInner::Var(candle_core::Var::from_tensor(&var.copy()?)?)
                    }
                },
                device: data.device.clone(),
                grad: match &data.grad {
                    Some(grad) => Some(grad.try_clone()?),
                    None => None,
                },
                parents: vec![],
                name: data.name.clone(),
            })),
        })
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
        let inner_tensor = match &data.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var.as_tensor(),
        };
        inner_tensor.fmt(f)?;
        Ok(())
    }
}

impl Tensor {
    pub fn detach(&self) -> Result<Tensor, TensorError> {
        let inner = self.data.read()?;
        let inner_tensor = match &inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = TensorInner::Tensor(inner_tensor.detach());

        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                device: self.data.read()?.device.clone(),
                parents: vec![],
                grad: None,
                name: None,
            })),
        })
    }

    /// Get the gradient enabled status of the tensor.
    ///
    /// # Returns
    /// * `Ok(true)` - The tensor has gradient enabled.
    /// * `Ok(false)` - The tensor has gradient disabled.
    /// * `Err(TensorError)` - The error when getting the tensor's gradient enabled status.
    pub fn grad_enabled(&self) -> Result<bool, TensorError> {
        let data = self.data.read()?;
        Ok(match &data.inner {
            TensorInner::Var(_) => true,
            TensorInner::Tensor(_) => false,
        })
    }

    /// Create a new tensor likes the current tensor, but with the desired gradient status.
    ///
    /// # Notes
    /// * If the tensor already has the desired gradient status, the method will return the current tensor.
    /// * Switching the gradient enabled status will disconnect the tensor from the computational graph.
    ///
    /// # Arguments
    /// * `grad_enabled` - The desired gradient enabled status.
    ///
    /// # Returns
    /// * `Ok(())` - The tensor's gradient enabled status is successfully set.
    /// * `Err(TensorError)` - The error when setting the tensor's gradient enabled status.
    pub fn require_grad(&mut self, grad_enabled: bool) -> Result<Tensor, TensorError> {
        // Check if the gradient status is already as required
        if self.grad_enabled()? == grad_enabled {
            return Ok(self.copy());
        }

        let inner_tensor = {
            let data = self.data.read()?;
            match &data.inner {
                TensorInner::Tensor(tensor) => tensor.clone(),
                TensorInner::Var(var) => var.as_tensor().clone(),
            }
        }
        .copy()?;

        let new_inner = match grad_enabled {
            true => {
                // Convert to Var type
                let var = candle_core::Var::from_tensor(&inner_tensor)?;
                TensorInner::Var(var)
            }

            false => {
                // Convert to Tensor type
                TensorInner::Tensor(inner_tensor)
            }
        };

        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                device: self.device()?.clone(),
                parents: vec![],
                grad: None,
                name: self.data.read()?.name.clone(),
            })),
        })
    }

    /// Get the gradient tensor of the tensor.
    ///
    /// # Returns
    /// * `Ok(Some(Tensor))` - The tensor's gradient tensor.
    /// * `Ok(None)` - The tensor has no gradient tensor.
    /// * `Err(TensorError)` - The error when getting the tensor's gradient tensor.
    pub fn grad(&self) -> Result<Option<Tensor>, TensorError> {
        let data = self.data.read()?;
        Ok(data.grad.as_ref().map(|grad| grad.copy()))
    }

    /// Zero the gradient tensor of the tensor.
    ///
    /// # Returns
    /// * `Ok(())` - The tensor's gradient tensor is successfully set to zero.
    /// * `Err(TensorError)` - The error when setting the tensor's gradient tensor to zero.
    pub fn zero_grad(&mut self) -> Result<(), TensorError> {
        let mut data = self.data.write()?;
        if let Some(grad) = &mut data.grad {
            match &mut grad.data.write()?.inner {
                TensorInner::Var(var) => var.zero_set()?,
                TensorInner::Tensor(tensor) => tensor.zero_set()?,
            }
        }
        Ok(())
    }

    /// Clear the gradient tensor of the tensor.
    ///
    /// # Returns
    /// * `Ok(())` - The tensor's gradient tensor is successfully cleared.
    /// * `Err(TensorError)` - The error when clearing the tensor's gradient tensor.
    pub fn clear_grad(&mut self) -> Result<(), TensorError> {
        let mut data = self.data.write()?;
        data.grad = None;
        Ok(())
    }

    /// Backpropagate the gradient of the tensor to its parent tensors.
    ///
    /// # Returns
    /// * `Ok(())` - The tensor's gradient is successfully backpropagated.
    /// * `Err(TensorError)` - The error when backpropagating the tensor's gradient.
    pub fn backward(&self) -> Result<(), TensorError> {
        // Get grad_store
        let grad_store = {
            let data = self.data.read()?;
            match &data.inner {
                TensorInner::Var(var) => {
                    // Clone the var to release the read lock before calling backward
                    let var_clone = var.clone();
                    drop(data);
                    var_clone.backward()?
                }
                TensorInner::Tensor(tensor) => {
                    // Clone the tensor to release the read lock before calling backward
                    let tensor_clone = tensor.clone();
                    drop(data);
                    tensor_clone.backward()?
                }
            }
        };

        // The parent tensors(including indirect parents) that have gradient enabled
        let mut parents = Vec::new();
        // The queue for pending tensors
        let queue = SegQueue::new();
        // The visited tensors
        let visited = DashMap::new();

        // Add the self tensor to the queue and mark it as visited
        queue.push(self.copy());
        visited.insert(Arc::as_ptr(&self.data) as usize, true);

        while let Some(current) = queue.pop() {
            // Get the parent tensors of the current tensor
            let current_parents = {
                let data = current.data.read()?;
                data.parents
                    .iter()
                    .map(|parent| parent.copy())
                    .collect::<Vec<_>>()
            };

            // Filter out the gradient enabled parent tensors
            let grad_enabled_parents = current_parents
                .par_iter()
                .filter_map(|parent| {
                    let parent_id = Arc::as_ptr(&parent.data) as usize;
                    match visited.insert(parent_id, true).is_some() {
                        true => None,
                        false => {
                            // Add the parent tensor to the queue for further processing
                            queue.push(parent.copy());

                            match parent.grad_enabled().unwrap_or(false) {
                                true => Some(parent.copy()),
                                false => None,
                            }
                        }
                    }
                })
                .collect::<Vec<_>>();

            // Add the parent tensors that have gradient enabled to the list of parents
            parents.extend(grad_enabled_parents);
        }

        // Update the gradient of each parent tensor
        parents
            .par_iter()
            .map(|parent| {
                let mut parent_write = parent.data.write()?;
                let inner_tensor = match &parent_write.inner {
                    TensorInner::Tensor(tensor) => tensor.clone(),
                    TensorInner::Var(var) => var.as_tensor().clone(),
                };

                // Get the new gradient of the parent tensor from the grad_store
                let new_grad = grad_store
                    .get(&inner_tensor)
                    .ok_or(TensorError::NoTensorGradient)?;

                match &parent_write.grad {
                    // If the parent tensor already has a gradient, add the new gradient to it
                    Some(parent_grad) => {
                        let mut parent_grad_write = parent_grad.data.write()?;
                        let parent_grad_inner_tensor = match &parent_grad_write.inner {
                            TensorInner::Tensor(tensor) => tensor.clone(),
                            TensorInner::Var(var) => var.as_tensor().clone(),
                        };
                        parent_grad_write.inner =
                            TensorInner::Tensor(new_grad.add(&parent_grad_inner_tensor)?.detach());
                    }
                    // If the parent tensor does not have a gradient, set it to the new gradient
                    None => {
                        parent_write.grad = Some(Tensor::from_candle_tensor(
                            new_grad.clone(),
                            &parent_write.device,
                            false,
                        )?);
                    }
                };

                Ok(())
            })
            .collect::<Result<(), TensorError>>()?;

        Ok(())
    }

    /// Update the tensor's inner data from another tensor.
    ///
    /// # Notes
    /// * If the tensor has enabled gradients, the method will update the tensor's inner data
    ///   without disconnecting it from the computational graph.
    /// * If the tensor does not have enabled gradients, the method will update the tensor's inner data
    ///   and disconnect it from the computational graph.
    ///
    /// # Arguments
    /// * `other` - The tensor to update the inner data from.
    ///
    /// # Returns
    /// * `Ok(())` - The tensor's inner data is successfully updated.
    /// * `Err(TensorError)` - The error when updating the tensor's inner data.
    pub fn update_from_tensor(&self, other: &Tensor) -> Result<(), TensorError> {
        let other_data = other.data.read()?;
        let other_inner_tensor = match &other_data.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };
        let mut self_data = self.data.write()?;
        match &mut self_data.inner {
            TensorInner::Tensor(_) => {
                self_data.inner = TensorInner::Tensor(other_inner_tensor.copy()?);
                self_data.grad = None;
                self_data.parents.clear();
            }
            TensorInner::Var(var) => {
                var.set(other_inner_tensor)?;
            }
        }

        Ok(())
    }

    /// Clear the computational graph.
    ///
    /// # Returns
    /// * `Ok(())` - The tensor's computational graph is successfully cleared.
    /// * `Err(TensorError)` - The error when clearing the tensor's computational graph.
    pub fn clear_graph(&self) -> Result<(), TensorError> {
        let queue = SegQueue::new();
        let visited = DashMap::new();

        queue.push(self.copy());
        visited.insert(Arc::as_ptr(&self.data) as usize, true);

        while let Some(current) = queue.pop() {
            let current_parents = {
                let mut data = current.data.write()?;
                if let Some(grad) = data.grad.as_ref() {
                    queue.push(grad.copy());
                }
                std::mem::take(&mut data.parents)
            };

            for parent in current_parents {
                let parent_id = Arc::as_ptr(&parent.data) as usize;
                if visited.insert(parent_id, true).is_none() {
                    queue.push(parent);
                }
            }
        }

        Ok(())
    }
}
