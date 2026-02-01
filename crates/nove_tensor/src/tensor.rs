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
/// * `name` - The name of the tensor.
#[derive(Debug)]
pub(crate) struct TensorData {
    pub(crate) inner: RwLock<TensorInner>,
    pub(crate) device: RwLock<Device>,
    pub(crate) parents: RwLock<Vec<Tensor>>,
    pub(crate) grad: RwLock<Option<candle_core::Tensor>>,
    pub(crate) name: RwLock<Option<String>>,
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
        let inner = self.data.inner.read().map_err(|_| std::fmt::Error)?;
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
    /// # Arguments
    /// * `grad_enabled` - The desired gradient enabled status.
    ///
    /// # Returns
    /// * `Ok(())` - The tensor's gradient enabled status is successfully set.
    /// * `Err(TensorError)` - The error when setting the tensor's gradient enabled status.
    pub fn set_grad_enabled(&mut self, grad_enabled: bool) -> Result<(), TensorError> {
        // Check if the gradient status is already as required
        if self.grad_enabled()? == grad_enabled {
            return Ok(());
        }

        // Get the inner tensor
        let inner_tensor = {
            let inner = self.data.inner.read()?;
            match &*inner {
                TensorInner::Tensor(tensor) => tensor.clone(),
                TensorInner::Var(var) => var.as_tensor().clone(),
            }
        };

        // Create new inner
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

        {
            let mut inner_write = self.data.inner.write()?;
            // Update the inner tensor
            *inner_write = new_inner;
        }

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
    pub fn set_grad(&mut self, grad: Tensor) -> Result<(), TensorError> {
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

        // Check if the gradient device is the same as the tensor device
        let grad_device = grad.device()?;
        let self_device = self.device()?;
        if grad_device != self_device {
            return Err(TensorError::DeviceMismatch(self_device, grad_device));
        }

        let new_grad = match &*grad.data.inner.read()? {
            TensorInner::Tensor(tensor) => tensor.copy()?,
            TensorInner::Var(var) => var.as_tensor().copy()?,
        };

        // Set the gradient tensor
        *self.data.grad.write()? = Some(new_grad);
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
                name: RwLock::new(None),
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
            let inner = self.data.inner.read()?;
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

                            match parent.grad_enabled().unwrap_or(false) {
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
                let inner_tensor = match &*parent.data.inner.read()? {
                    TensorInner::Tensor(tensor) => tensor.clone(),
                    TensorInner::Var(var) => var.as_tensor().clone(),
                };

                // Get the new gradient of the parent tensor from the grad_store
                let grad = grad_store
                    .get(&inner_tensor)
                    .ok_or(TensorError::NoTensorGradient)?;

                let mut grad_write = parent.data.grad.write()?;
                match &*grad_write {
                    // If the parent tensor already has a gradient, add the new gradient to it
                    Some(parent_grad) => {
                        *grad_write = Some(grad.add(parent_grad)?);
                    }
                    // If the parent tensor does not have a gradient, set it to the new gradient
                    None => {
                        *grad_write = Some(grad.clone());
                    }
                };

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
            TensorInner::Tensor(_) => {
                *self_inner = TensorInner::Tensor(other_inner_tensor.clone());
            }
            TensorInner::Var(var) => {
                var.set(other_inner_tensor)?;
            }
        }

        Ok(())
    }
}
