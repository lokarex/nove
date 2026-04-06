use std::{
    collections::HashMap,
    fmt::Display,
    sync::atomic::{AtomicUsize, Ordering},
};

use nove_tensor::Tensor;

use crate::{Model, ModelError};

static ID: AtomicUsize = AtomicUsize::new(0);

/// Sigmoid layer.
///
/// <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" id="MathJax-script" async></script>
///
/// The sigmoid function is a smooth, non-linear activation function that maps
/// input values to the range (0, 1). It is commonly used in binary classification
/// problems and as the output activation function in neural networks where
/// probability-like outputs are needed.
///
/// The Sigmoid function is computed as:
///
/// $$ \text{Sigmoid}(x) = \frac{1}{1 + e^{-x}} = \frac{e^x}{e^x + 1} $$
///
/// Where:
/// - x is the input value
/// - e is the base of the natural logarithm
///
/// # Notes
/// * The `Sigmoid` is now only created by the `Sigmoid::new()` function.
///
/// # Fields
/// * `id` - The unique ID of the Sigmoid layer.
///
/// # Examples
/// ```
/// use nove::tensor::{Device, Tensor};
/// use nove::model::nn::Sigmoid;
/// use nove::model::Model;
///
/// let mut sigmoid = Sigmoid::new();
/// println!("{}", sigmoid);
///
/// let input = Tensor::from_data(&[-1.0f32, 0.0f32, 1.0f32], &Device::cpu(), false).unwrap();
/// let output = sigmoid.forward(input).unwrap();
/// println!("{}", output);
/// ```
#[derive(Debug, Clone)]
pub struct Sigmoid {
    id: usize,
}

impl Sigmoid {
    /// Create a new sigmoid layer.
    ///
    /// # Returns
    /// * `Sigmoid` - The new Sigmoid layer.
    pub fn new() -> Self {
        Self {
            id: ID.fetch_add(1, Ordering::Relaxed),
        }
    }
}

impl Default for Sigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl Model for Sigmoid {
    type Input = Tensor;

    type Output = Tensor;

    /// Apply the sigmoid layer to the input tensor.
    ///
    /// # Arguments
    /// * `input: Tensor` - The input tensor.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The tensor after applying the sigmoid activation function.
    /// * `Err(ModelError)` - The error when applying the sigmoid activation function.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// use nove::model::nn::Sigmoid;
    /// use nove::model::Model;
    ///
    /// let mut sigmoid = Sigmoid::new();
    /// let input = Tensor::from_data(vec![-1.0, 0.0, 1.0], &Device::cpu(), false).unwrap();
    /// let output = sigmoid.forward(input).unwrap();
    /// assert_eq!(output.to_vec::<f64>().unwrap(), vec![0.2689414213699951, 0.5, 0.7310585786300049]);
    /// ```
    fn forward(&mut self, input: Self::Input) -> Result<Self::Output, crate::ModelError> {
        Ok(input.sigmoid()?)
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

impl Display for Sigmoid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sigmoid.{}()", self.id)
    }
}
