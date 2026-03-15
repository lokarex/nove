use std::{
    collections::HashMap,
    fmt::Display,
    sync::atomic::{AtomicUsize, Ordering},
};

use nove_tensor::Tensor;

use crate::{Model, ModelError};

static ID: AtomicUsize = AtomicUsize::new(0);

/// Sigmoid Linear Unit (SiLU) layer.
///
/// <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" id="MathJax-script" async></script>
///
/// The Sigmoid Linear Unit (SiLU), also known as the Swish activation function,
/// is a smooth, non-monotonic activation function that has shown improved performance
/// over traditional activation functions like ReLU in some deep learning models.
///
/// The SiLU function is computed as:
///
/// $$ \text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}} $$
///
/// Where:
/// - x is the input value
/// - σ(x) is the sigmoid function: σ(x) = 1 / (1 + e^(-x))
///
/// # Notes
/// * The `SiLU` is now only created by the `SiLU::new()` function.
///
/// # Fields
/// * `id` - The unique ID of the SiLU layer.
///
/// # Examples
/// ```
/// use nove::tensor::{Device, Tensor};
/// use nove::model::layer::SiLU;
/// use nove::model::Model;
///
/// let mut silu = SiLU::new();
/// println!("{}", silu);
///
/// let input = Tensor::from_data(&[-1.0f32, 0.0f32, 1.0f32], &Device::cpu(), false).unwrap();
/// let output = silu.forward(input).unwrap();
/// println!("{}", output);
/// ```
#[derive(Debug, Clone)]
pub struct SiLU {
    id: usize,
}

impl SiLU {
    /// Create a new Sigmoid Linear Unit (SiLU) layer.
    ///
    /// # Returns
    /// * `SiLU` - The new SiLU layer.
    pub fn new() -> Self {
        Self {
            id: ID.fetch_add(1, Ordering::Relaxed),
        }
    }
}

impl Default for SiLU {
    fn default() -> Self {
        Self::new()
    }
}

impl Model for SiLU {
    type Input = Tensor;

    type Output = Tensor;

    /// Apply the Sigmoid Linear Unit (SiLU) layer to the input tensor.
    ///
    /// # Arguments
    /// * `input: Tensor` - The input tensor.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The tensor after applying the SiLU activation function.
    /// * `Err(ModelError)` - The error when applying the SiLU activation function.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// use nove::model::layer::SiLU;
    /// use nove::model::Model;
    ///
    /// let mut silu = SiLU::new();
    /// let input = Tensor::from_data(vec![-1.0, 0.0, 1.0], &Device::cpu(), false).unwrap();
    /// let output = silu.forward(input).unwrap();
    /// assert_eq!(output.to_vec::<f64>().unwrap(), vec![-0.2689414213699951, 0.0, 0.7310585786300049]);
    /// ```
    fn forward(&mut self, input: Self::Input) -> Result<Self::Output, crate::ModelError> {
        Ok(input.silu()?)
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

impl Display for SiLU {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "silu.{}()", self.id)
    }
}
