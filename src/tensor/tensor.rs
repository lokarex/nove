use crate::tensor::{DType, Device, Shape};
use std::sync::{Arc, RwLock};
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

    #[error("Cannot enable gradients: tensor already has gradients enabled")]
    AlreadyGradientEnabled,

    #[error("Cannot disable gradients: tensor already has gradients disabled")]
    AlreadyGradientDisabled,

    #[error("Tensor is already of dtype")]
    AlreadyDtype,

    #[error("Tensor is already on device")]
    AlreadyOnDevice,

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
            (TensorError::AlreadyGradientEnabled, TensorError::AlreadyGradientEnabled) => true,
            (TensorError::AlreadyGradientDisabled, TensorError::AlreadyGradientDisabled) => true,
            (TensorError::AlreadyDtype, TensorError::AlreadyDtype) => true,
            (TensorError::AlreadyOnDevice, TensorError::AlreadyOnDevice) => true,
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

impl Tensor {
    /// Add two tensors element-wise.
    ///
    /// # Arguments
    /// * `other` - The tensor to add.
    ///
    /// # Returns
    /// * `Ok(tensor)` - The result tensor after addition.
    /// * `Err(TensorError)` - The error when adding the tensors.
    pub fn add(&self, other: &Self) -> Result<Self, TensorError> {
        // Get the inner tensors
        let inner1 = self.data.inner.read()?;
        let inner1_tensor = match &*inner1 {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };
        let inner2 = other.data.inner.read()?;
        let inner2_tensor = match &*inner2 {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        // Get the device from the first tensor
        let device = self.data.device.read()?.clone();

        // Create the inner tensor
        let new_inner = TensorInner::Tensor(inner1_tensor.add(inner2_tensor)?);

        // Set the parents
        let parents = vec![self.clone(), other.clone()];

        Ok(Self {
            data: Arc::new(TensorData {
                inner: RwLock::new(new_inner),
                device: RwLock::new(device),
                parents: RwLock::new(parents),
                grad: RwLock::new(None),
            }),
        })
    }

    /// Stack a list of tensors along a new dimension.
    ///
    /// # Arguments
    /// * `tensors` - The list of tensors to stack.
    /// * `dim` - The dimension along which to stack the tensors.
    ///
    /// # Returns
    /// * `Ok(tensor)` - The result tensor after stacking.
    /// * `Err(TensorError)` - The error when stacking the tensors.
    pub fn stack<A, D>(tensors: &[A], dim: D) -> Result<Self, TensorError>
    where
        A: AsRef<Tensor> + std::clone::Clone,
        D: candle_core::shape::Dim,
    {
        // Get the inner tensors
        let inner_tensors = tensors
            .iter()
            .map(|tensor| {
                let inner = tensor.as_ref().data.inner.read()?;
                match &*inner {
                    TensorInner::Tensor(tensor) => Ok(tensor.clone()),
                    TensorInner::Var(var) => Ok(var.as_tensor().clone()),
                }
            })
            .collect::<Result<Vec<_>, TensorError>>()?;
        // Stack the inner tensors
        let new_inner_tensor = candle_core::Tensor::stack(&inner_tensors, dim)?;

        // Get the device from the first tensor
        let device = tensors
            .first()
            .map(|t| t.as_ref().data.device.read().unwrap().clone())
            .unwrap();

        // Create the new inner
        let new_inner = TensorInner::Tensor(new_inner_tensor);

        //  Set the parents
        let parents = tensors
            .iter()
            .map(|tensor| tensor.as_ref().clone())
            .collect::<Vec<_>>();

        Ok(Self {
            data: Arc::new(TensorData {
                inner: RwLock::new(new_inner),
                device: RwLock::new(device),
                parents: RwLock::new(parents),
                grad: RwLock::new(None),
            }),
        })
    }
}

impl Tensor {
    /// Set the gradient enabled status of the tensor.
    ///
    /// # Arguments
    /// * `requires_grad` - The desired gradient enabled status.
    ///
    /// # Returns
    /// * `Ok(())` - The tensor's gradient enabled status is successfully set.
    /// * `Err(TensorError)` - The error when setting the tensor's gradient enabled status.
    pub fn set_grad_enabled(&mut self, requires_grad: bool) -> Result<(), TensorError> {
        // Check if the gradient status is already as required
        if self.get_grad_enabled() == requires_grad && requires_grad == true {
            return Err(TensorError::AlreadyGradientEnabled);
        }
        if self.get_grad_enabled() == requires_grad && requires_grad == false {
            return Err(TensorError::AlreadyGradientDisabled);
        }

        // Get current inner tensor and release the read lock immediately
        let inner_tensor = {
            let inner = self.data.inner.read().unwrap();
            match &*inner {
                TensorInner::Tensor(tensor) => tensor.clone(),
                TensorInner::Var(var) => var.as_tensor().clone(),
            }
        };

        // Create new inner and grad based on requires_grad
        let (new_inner, new_grad) = match requires_grad {
            true => {
                // Convert to Var type
                let var = candle_core::Var::from_tensor(&inner_tensor)?;
                // Create zero tensor with the same shape
                let zero_grad = var.zeros_like()?;
                (TensorInner::Var(var), Some(zero_grad))
            }
            false => {
                // Convert to Tensor type
                (TensorInner::Tensor(inner_tensor), None)
            }
        };

        // Update the inner tensor
        *self.data.inner.write().unwrap() = new_inner;
        // Update the gradient tensor
        *self.data.grad.write().unwrap() = new_grad;

        Ok(())
    }

    /// Get the gradient enabled status of the tensor.
    ///
    /// # Returns
    /// * `true` - The tensor has gradient enabled.
    /// * `false` - The tensor has gradient disabled.
    pub fn get_grad_enabled(&self) -> bool {
        self.data.grad.read().unwrap().is_some()
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
        if !self.get_grad_enabled() {
            return Err(TensorError::NoTensorGradient);
        }

        // Check if the gradient shape is the same as the tensor shape
        let grad_shape = grad.get_shape()?;
        let self_shape = self.get_shape()?;
        if grad_shape != self_shape {
            return Err(TensorError::ShapeMismatch(self_shape, grad_shape));
        }

        // Check if the gradient dtype is the same as the tensor dtype
        let grad_dtype = grad.get_dtype()?;
        let self_dtype = self.get_dtype()?;
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
    pub fn get_grad(&self) -> Result<Tensor, TensorError> {
        // Check if the gradient status is enabled
        if !self.get_grad_enabled() {
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

        // Get the parent tensors that have gradient enabled
        let mut parents = Vec::new();
        let mut stack = vec![self.clone()];
        while let Some(tensor) = stack.pop() {
            for parent in tensor.data.parents.read()?.iter() {
                if parent.get_grad_enabled() {
                    parents.push(parent.clone());
                }
                stack.push(parent.clone());
            }
        }

        // Update the gradient of each parent tensor
        for parent in parents {
            let inner = parent.data.inner.read()?;
            let inner_tensor = match &*inner {
                TensorInner::Tensor(tensor) => tensor,
                TensorInner::Var(var) => var,
            };

            let grad = grad_store
                .get(inner_tensor)
                .ok_or(TensorError::NoTensorGradient)?;
            let parent_grad = parent
                .data
                .grad
                .read()?
                .clone()
                .ok_or(TensorError::NoTensorGradient)?;
            *parent.data.grad.write()? = Some(grad.add(&parent_grad)?);
        }

        Ok(())
    }
}
