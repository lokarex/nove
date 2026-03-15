use std::{
    collections::HashMap,
    fmt::Display,
    sync::atomic::{AtomicUsize, Ordering},
};

use nove_tensor::Tensor;

use crate::{Model, ModelError};

static ID: AtomicUsize = AtomicUsize::new(0);

/// Rectified Linear Unit (ReLU) layer.
///
/// <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" id="MathJax-script" async></script>
///
/// The Rectified Linear Unit (ReLU) is an activation function that introduces non-linearity
/// to the model. It is defined as the positive part of its input.
///
/// The ReLU function is computed as:
///
/// $$ \text{ReLU}(x) = \max(0, x) $$
///
/// Where:
/// - x is the input value
/// - max(0, x) returns x if x is positive, otherwise returns 0
///
/// # Notes
/// * The `ReLU` is now only created by the `ReLU::new()` function.
///
/// # Fields
/// * `id` - The unique ID of the ReLU layer.
///
/// # Examples
/// ```
/// use nove::tensor::{Device, Tensor};
/// use nove::model::layer::ReLU;
/// use nove::model::Model;
///
/// let mut relu = ReLU::new();
/// println!("{}", relu);
///
/// let input = Tensor::from_data(&[0.0f32, 1.0f32, 2.0f32], &Device::cpu(), false).unwrap();
/// let output = relu.forward(input).unwrap();
/// println!("{}", output);
/// ```
#[derive(Debug, Clone)]
pub struct ReLU {
    id: usize,
}

impl ReLU {
    /// Create a new Rectified Linear Unit (ReLU) layer.
    ///
    /// # Returns
    /// * `ReLU` - The new ReLU layer.
    pub fn new() -> Self {
        Self {
            id: ID.fetch_add(1, Ordering::Relaxed),
        }
    }
}

impl Default for ReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl Model for ReLU {
    type Input = Tensor;

    type Output = Tensor;

    /// Apply the Rectified Linear Unit (ReLU) layer to the input tensor.
    ///
    /// # Arguments
    /// * `input: Tensor` - The input tensor.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The tensor after applying the ReLU activation function.
    /// * `Err(ModelError)` - The error when applying the ReLU activation function.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// use nove::model::layer::ReLU;
    /// use nove::model::Model;
    ///
    /// let mut relu = ReLU::new();
    /// let input = Tensor::from_data(vec![-1.0, 2.0, -3.0, 4.0], &Device::cpu(), false).unwrap();
    /// let output = relu.forward(input).unwrap();
    /// assert_eq!(output.to_vec::<f64>().unwrap(), vec![0.0, 2.0, 0.0, 4.0]);
    /// ```
    fn forward(&mut self, input: Self::Input) -> Result<Self::Output, crate::ModelError> {
        Ok(input.relu()?)
    }

    fn require_grad(&mut self, _: bool) -> Result<(), crate::ModelError> {
        Ok(())
    }

    fn to_device(&mut self, _: &nove_tensor::Device) -> Result<(), crate::ModelError> {
        Ok(())
    }

    fn to_dtype(&mut self, _: &nove_tensor::DType) -> Result<(), crate::ModelError> {
        Ok(())
    }

    fn parameters(&self) -> Result<Vec<nove_tensor::Tensor>, crate::ModelError> {
        Ok(vec![])
    }

    fn named_parameters(&self) -> Result<HashMap<String, Tensor>, ModelError> {
        Ok(HashMap::new())
    }
}

impl Display for ReLU {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "relu.{}()", self.id)
    }
}
