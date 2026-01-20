use crate::tensor::{DType, Device, Shape};
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
    DeviceMismatch(Device, Device),
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
#[derive(Debug)]
pub(crate) struct TensorData {
    pub(crate) inner: RwLock<TensorInner>,
    pub(crate) device: RwLock<Device>,
    pub(crate) parents: RwLock<Vec<Tensor>>,
    pub(crate) grad: RwLock<Option<candle_core::Tensor>>,
}

/// The tensor struct.
///
/// # Notes
/// * The `clone` method is cheap because the inner data is wrapped in an `Arc`.
///
/// # Fields
/// * `data` - The data structure of the tensor.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub(crate) data: Arc<TensorData>,
}

impl AsRef<Tensor> for Tensor {
    fn as_ref(&self) -> &Self {
        self
    }
}

impl Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self
            .data
            .inner
            .read()
            .map_err(|_| std::fmt::Error::default())?;
        let inner_tensor = match &*inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var.as_tensor(),
        };
        inner_tensor.fmt(f)?;
        Ok(())
    }
}

impl Tensor {
    /// Set the gradient enabled status of the tensor.
    ///
    /// # Notes
    /// * This method will truncate the graph.
    /// * When enabling gradients, a zero tensor used as the initial gradient will be created if no gradient exists.
    /// * When disabling gradients, the current gradient will be discarded.
    ///
    /// # Arguments
    /// * `grad_enabled` - The desired gradient enabled status.
    ///
    /// # Returns
    /// * `Ok(())` - The tensor's gradient enabled status is successfully set.
    /// * `Err(TensorError)` - The error when setting the tensor's gradient enabled status.
    pub fn set_grad_enabled(&self, grad_enabled: bool) -> Result<(), TensorError> {
        // Check if the gradient status is already as required
        if self.grad_enabled()? == grad_enabled {
            return Ok(());
        }

        // Get the inner tensor and gradient
        let (inner_tensor, inner_grad) = {
            let grad = self.data.grad.read()?;
            let inner = self.data.inner.read()?;
            let inner_tensor = match &*inner {
                TensorInner::Tensor(tensor) => tensor.clone(),
                TensorInner::Var(var) => var.as_tensor().clone(),
            };

            (inner_tensor, grad.clone())
        };

        // Create new inner and grad based on grad_enabled
        let (new_inner, new_grad) = match grad_enabled {
            true => {
                // Convert to Var type
                let var = candle_core::Var::from_tensor(&inner_tensor)?;
                // Get the current gradient tensor, or create a zero tensor if None
                let grad = match inner_grad {
                    Some(grad) => grad.clone(),
                    None => var.zeros_like()?,
                };
                (TensorInner::Var(var), Some(grad))
            }
            false => {
                // Convert to Tensor type
                (TensorInner::Tensor(inner_tensor), None)
            }
        };

        // Update the inner tensor
        *self.data.inner.write()? = new_inner;
        // Update the gradient tensor
        *self.data.grad.write()? = new_grad;
        // Truncate the graph by clearing parents
        *self.data.parents.write()? = Vec::new();

        Ok(())
    }

    /// Get the gradient enabled status of the tensor.
    ///
    /// # Returns
    /// * `Ok(true)` - The tensor has gradient enabled.
    /// * `Ok(false)` - The tensor has gradient disabled.
    /// * `Err(TensorError)` - The error when getting the tensor's gradient enabled status.
    pub fn grad_enabled(&self) -> Result<bool, TensorError> {
        let inner = self.data.inner.read()?;
        Ok(match &*inner {
            TensorInner::Var(_) => true,
            TensorInner::Tensor(_) => false,
        })
    }

    /// Set the gradient tensor of the tensor.
    ///
    /// # Arguments
    /// * `grad` - The gradient tensor to be set.
    ///
    /// # Returns
    /// * `Ok(())` - The tensor's gradient tensor is successfully set.
    /// * `Err(TensorError)` - The error when setting the tensor's gradient tensor.
    pub fn set_grad(&self, grad: Tensor) -> Result<(), TensorError> {
        // Check if the gradient status is enabled
        if !self.grad_enabled()? {
            return Err(TensorError::NoTensorGradient);
        }

        // Check if the gradient shape is the same as the tensor shape
        let grad_shape = grad.shape()?;
        let self_shape = self.shape()?;
        if grad_shape != self_shape {
            return Err(TensorError::ShapeMismatch(self_shape, grad_shape));
        }

        // Check if the gradient dtype is the same as the tensor dtype
        let grad_dtype = grad.dtype()?;
        let self_dtype = self.dtype()?;
        if grad_dtype != self_dtype {
            return Err(TensorError::DTypeMismatch(self_dtype, grad_dtype));
        }

        // Get the inner tensor of the gradient
        let grad_inner = grad.data.inner.read()?.clone();
        let grad_tensor = match grad_inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var.as_tensor().clone(),
        };

        // Set the gradient tensor
        *self.data.grad.write()? = Some(grad_tensor);
        Ok(())
    }

    /// Get the gradient tensor of the tensor.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The tensor's gradient tensor.
    /// * `Err(TensorError)` - The error when getting the tensor's gradient tensor.
    pub fn grad(&self) -> Result<Tensor, TensorError> {
        // Check if the gradient status is enabled
        if !self.grad_enabled()? {
            return Err(TensorError::GradientDisabled);
        }

        let new_inner = TensorInner::Tensor(
            self.data
                .grad
                .read()
                .unwrap()
                .clone()
                .ok_or(TensorError::NoTensorGradient)?,
        );

        let device = self.data.device.read()?.clone();

        Ok(Self {
            data: Arc::new(TensorData {
                inner: RwLock::new(new_inner),
                device: RwLock::new(device),
                grad: RwLock::new(None),
                parents: RwLock::new(Vec::new()),
            }),
        })
    }

    /// Backpropagate the gradient of the tensor to its parent tensors.
    ///
    /// # Returns
    /// * `Ok(())` - The tensor's gradient is successfully backpropagated.
    /// * `Err(TensorError)` - The error when backpropagating the tensor's gradient.
    pub fn backward(&self) -> Result<(), TensorError> {
        // Get grad_store
        let grad_store = {
            let inner = self.data.inner.read().unwrap();
            match &*inner {
                TensorInner::Var(var) => {
                    // Clone the var to release the read lock before calling backward
                    let var_clone = var.clone();
                    drop(inner); // Explicitly release the read lock
                    var_clone.backward()?
                }
                TensorInner::Tensor(tensor) => {
                    // Clone the tensor to release the read lock before calling backward
                    let tensor_clone = tensor.clone();
                    drop(inner); // Explicitly release the read lock
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
        queue.push(self.clone());
        visited.insert(Arc::as_ptr(&self.data) as usize, true);

        while let Some(current) = queue.pop() {
            // Get the parent tensors of the current tensor
            let current_parents = {
                let parents = current.data.parents.read()?;
                parents.clone()
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
                            queue.push(parent.clone());

                            match parent.grad_enabled().unwrap_or_else(|_| false) {
                                true => Some(parent.clone()),
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
                let inner = parent.data.inner.read()?;
                let inner_tensor = match &*inner {
                    TensorInner::Tensor(tensor) => tensor,
                    TensorInner::Var(var) => var,
                };

                // Get the new gradient of the parent tensor from the grad_store
                let grad = grad_store
                    .get(inner_tensor)
                    .ok_or(TensorError::NoTensorGradient)?;
                // Get the original gradient of the parent tensor
                let parent_grad = parent
                    .data
                    .grad
                    .read()?
                    .clone()
                    .ok_or(TensorError::NoTensorGradient)?;
                // Update the gradient of the parent tensor
                *parent.data.grad.write()? = Some(grad.add(&parent_grad)?);
                Ok(())
            })
            .collect::<Result<(), TensorError>>()?;

        Ok(())
    }

    pub fn update_from_tensor(&self, other: &Tensor) -> Result<(), TensorError> {
        let other_inner = other.data.inner.read()?;
        let other_inner_tensor = match &*other_inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };
        let mut self_inner = self.data.inner.write()?;
        match &mut *self_inner {
            TensorInner::Tensor(_tensor) => {
                todo!()
            }
            TensorInner::Var(var) => {
                var.set(other_inner_tensor)?;
            }
        }

        Ok(())
    }
}
