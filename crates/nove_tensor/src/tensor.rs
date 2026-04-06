use crate::{DType, Device, Shape};
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

    #[error("Invalid dimension: {0}, must be >= -1")]
    InvalidDimension(isize),
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
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
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
    /// let device = Device::cpu();
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
        Ok(Tensor {
            data: Arc::new(RwLock::new(TensorData {
                inner: match &data.inner {
                    TensorInner::Tensor(tensor) => TensorInner::Tensor(tensor.copy()?.detach()),
                    TensorInner::Var(var) => {
                        TensorInner::Var(candle_core::Var::from_tensor(&var.copy()?.detach())?)
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
    /// Update the tensor's inner data from another tensor.
    ///
    /// # Notes
    /// * If the tensor has enabled gradients, the method will update the tensor's inner data
    ///   without disconnecting it from the computational graph.
    /// * If the tensor does not have enabled gradients, the method will update the tensor's inner data
    ///   and disconnect it from the computational graph.
    /// * Always used with [`Tensor::detach`].
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
}
