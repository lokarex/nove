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

    #[error("RwLock poisoned: {0}")]
    RwLockPoisoned(String),

    #[error("Shape mismatch: expected {0:?}, found {1:?}")]
    ShapeMismatch(Shape, Shape),

    #[error("DType mismatch: expected {0:?}, found {1:?}")]
    DTypeMismatch(DType, DType),
}

impl<T> From<std::sync::PoisonError<T>> for TensorError {
    fn from(error: std::sync::PoisonError<T>) -> Self {
        TensorError::RwLockPoisoned(error.to_string())
    }
}

/// Inner representation of tensor data.
#[derive(Clone)]
enum TensorInner {
    Tensor(candle_core::Tensor),
    Var(candle_core::Var),
}

/// The data structure for a tensor.
///
/// # Fields
/// * `inner` - The inner representation of the tensor data.
/// * `parents` - The list of parent tensors used to backpropagate gradients.
/// * `grad` - The gradient of the tensor data.
pub struct TensorData {
    inner: RwLock<TensorInner>,
    parents: RwLock<Vec<Tensor>>,
    grad: RwLock<Option<candle_core::Tensor>>,
}

/// The tensor struct.
///
/// # Notes
/// * The `clone` method is cheap because the inner data is wrapped in an `Arc`.
///
/// # Fields
/// * `data` - The data structure of the tensor.
#[derive(Clone)]
pub struct Tensor {
    data: Arc<TensorData>,
}

impl AsRef<Tensor> for Tensor {
    fn as_ref(&self) -> &Self {
        self
    }
}

impl Tensor {
    /// Create a new tensor from an array.
    ///
    /// # Arguments
    /// * `array` - The array to create the tensor from.
    /// * `device` - The device to place the tensor on.
    /// * `grad_enabled` - Whether to enable gradient tracking for the tensor.
    ///
    /// # Returns
    /// * `Ok(tensor)` - The created tensor if successful.
    /// * `Err(TensorError)` - The error when creating the tensor.
    pub fn from_array<A>(array: A, device: &Device, grad_enabled: bool) -> Result<Self, TensorError>
    where
        A: candle_core::NdArray,
    {
        let inner = match grad_enabled {
            true => TensorInner::Var(candle_core::Var::new(array, &device)?),
            false => TensorInner::Tensor(candle_core::Tensor::new(array, &device)?),
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

    /// Convert the tensor to a scalar.
    ///
    /// # Returns
    /// * `Ok(scalar)` - The scalar value if the tensor is a scalar.
    /// * `Err(TensorError)` - The error when converting the tensor to a scalar.
    pub fn to_scalar<S>(&self) -> Result<S, TensorError>
    where
        S: candle_core::WithDType,
    {
        let inner = self.data.inner.read()?;
        let scalar = match &*inner {
            TensorInner::Tensor(tensor) => {
                let squeezed = tensor.squeeze(0)?;
                squeezed.to_scalar::<S>()?
            }
            TensorInner::Var(var) => {
                let squeezed = var.squeeze(0)?;
                squeezed.to_scalar::<S>()?
            }
        };

        Ok(scalar)
    }

    /// Convert the tensor to a one-dimensional vector.
    ///
    /// # Notes
    /// * The tensor could be any shape, and it will be flattened to a one-dimensional vector.
    ///
    /// # Returns
    /// * `Ok(vec)` - The vector value if the tensor can be converted to a vector.
    /// * `Err(TensorError)` - The error when converting the tensor to a vector.
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

impl Tensor {
    /// Move the tensor to the specified device.
    ///
    /// # Arguments
    /// * `device` - The device to move the tensor to.
    ///
    /// # Returns
    /// * `Ok(())` - If the tensor is successfully moved to the device.
    /// * `Err(TensorError)` - The error when moving the tensor to the device.
    pub fn to_device(&mut self, device: &Device) -> Result<(), TensorError> {
        // Move the inner to the device
        let mut inner = self.data.inner.write()?;
        match &mut *inner {
            TensorInner::Tensor(tensor) => {
                *tensor = tensor.to_device(device)?;
            }
            TensorInner::Var(var) => {
                *var = candle_core::Var::from_tensor(&var.to_device(device)?)?;
            }
        }

        // Move the gradient to the device
        let mut grad = self.data.grad.write()?;
        if let Some(grad) = grad.as_mut() {
            *grad = grad.to_device(device)?;
        }

        // Clear the parents
        let mut parents = self.data.parents.write()?;
        parents.clear();
        Ok(())
    }

    /// Convert the tensor to the specified dtype.
    ///
    /// # Arguments
    /// * `dtype` - The dtype to convert the tensor to.
    ///
    /// # Notes
    /// * If the tensor is already of the specified dtype, an error is returned.
    /// * The gradient (if present) is also converted to the same dtype to maintain consistency.
    ///
    /// # Returns
    /// * `Ok(())` - If the tensor is successfully converted to the dtype.
    /// * `Err(TensorError)` - The error when converting the tensor to the dtype.
    pub fn to_dtype(&mut self, dtype: DType) -> Result<(), TensorError> {
        // Check current dtype first to avoid unnecessary conversion
        let current_dtype = {
            let inner = self.data.inner.read()?;
            match &*inner {
                TensorInner::Tensor(tensor) => tensor.dtype(),
                TensorInner::Var(var) => var.dtype(),
            }
        };

        // If already the target dtype, return error
        if current_dtype == dtype {
            return Err(TensorError::AlreadyDtype);
        }

        // Convert the inner to the dtype
        let mut inner = self.data.inner.write()?;
        match &mut *inner {
            TensorInner::Tensor(tensor) => {
                *tensor = tensor.to_dtype(dtype)?;
            }
            TensorInner::Var(var) => {
                *var = candle_core::Var::from_tensor(&var.to_dtype(dtype)?)?;
            }
        }

        // Convert the gradient to the dtype
        let mut grad = self.data.grad.write()?;
        if let Some(grad) = grad.as_mut() {
            *grad = grad.to_dtype(dtype)?;
        }

        Ok(())
    }

    /// Get the dtype of the tensor.
    ///
    /// # Returns
    /// * `Ok(dtype)` - The dtype of the tensor.
    /// * `Err(TensorError)` - The error when getting the dtype of the tensor.
    pub fn get_dtype(&self) -> Result<DType, TensorError> {
        let inner = self.data.inner.read()?;
        let dtype = match &*inner {
            TensorInner::Tensor(tensor) => tensor.dtype(),
            TensorInner::Var(var) => var.dtype(),
        };
        Ok(dtype)
    }

    /// Get the shape of the tensor.
    ///
    /// # Returns
    /// * `Ok(shape)` - The shape of the tensor.
    /// * `Err(TensorError)` - The error when getting the shape of the tensor.
    pub fn get_shape(&self) -> Result<Shape, TensorError> {
        let inner = self.data.inner.read()?;
        let shape = match &*inner {
            TensorInner::Tensor(tensor) => tensor.shape(),
            TensorInner::Var(var) => var.shape(),
        };
        Ok(Shape::from(shape))
    }

    /// Get the number of dimensions of the tensor.
    ///
    /// # Returns
    /// * `Ok(dim_num)` - The number of dimensions of the tensor.
    /// * `Err(TensorError)` - The error when getting the number of dimensions of the tensor.
    pub fn get_dim_num(&self) -> Result<usize, TensorError> {
        let shape = self.get_shape()?;
        Ok(shape.rank())
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

        // Create the inner tensor
        let new_inner = TensorInner::Tensor(inner1_tensor.add(inner2_tensor)?);

        // Set the parents
        let parents = vec![self.clone(), other.clone()];

        Ok(Self {
            data: Arc::new(TensorData {
                inner: RwLock::new(new_inner),
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

        Ok(Self {
            data: Arc::new(TensorData {
                inner: RwLock::new(new_inner),
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
