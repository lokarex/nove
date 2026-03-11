use std::{
    collections::HashMap,
    fmt::Display,
    sync::atomic::{AtomicUsize, Ordering},
};

use nove_tensor::Tensor;

use crate::{Model, ModelError};

static ID: AtomicUsize = AtomicUsize::new(0);

/// Hyperbolic tangent (Tanh) layer.
///
/// <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" id="MathJax-script" async></script>
///
/// The hyperbolic tangent (Tanh) is a smooth, non-linear activation function that maps
/// input values to the range (-1, 1). It is commonly used in recurrent neural networks
/// (RNNs) and as an alternative to the sigmoid function when outputs need to be
/// centered around zero.
///
/// The Tanh function is computed as:
///
/// $$ \text{Tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = \frac{1 - e^{-2x}}{1 + e^{-2x}} $$
///
/// Where:
/// - x is the input value
/// - e is the base of the natural logarithm
///
/// # Notes
/// * The `Tanh` is now only created by the `Tanh::new()` function.
///
/// # Fields
/// * `id` - The unique ID of the Tanh layer.
///
/// # Examples
/// ```
/// use nove::tensor::{Device, Tensor};
/// use nove::model::layer::Tanh;
/// use nove::model::Model;
///
/// let mut tanh = Tanh::new();
/// println!("{}", tanh);
///
/// let input = Tensor::from_data(&[-1.0f32, 0.0f32, 1.0f32], &Device::cpu(), false).unwrap();
/// let output = tanh.forward(input).unwrap();
/// println!("{}", output);
/// ```
#[derive(Debug, Clone)]
pub struct Tanh {
    id: usize,
}

impl Tanh {
    /// Create a new hyperbolic tangent (Tanh) layer.
    ///
    /// # Returns
    /// * `Tanh` - The new Tanh layer.
    pub fn new() -> Self {
        Self {
            id: ID.fetch_add(1, Ordering::Relaxed),
        }
    }
}

impl Default for Tanh {
    fn default() -> Self {
        Self::new()
    }
}

impl Model for Tanh {
    type Input = Tensor;

    type Output = Tensor;

    /// Apply the hyperbolic tangent (Tanh) layer to the input tensor.
    ///
    /// # Arguments
    /// * `input: Tensor` - The input tensor.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The tensor after applying the Tanh activation function.
    /// * `Err(ModelError)` - The error when applying the Tanh activation function.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// use nove::model::layer::Tanh;
    /// use nove::model::Model;
    ///
    /// let mut tanh = Tanh::new();
    /// let input = Tensor::from_data(vec![-1.0, 0.0, 1.0], &Device::cpu(), false).unwrap();
    /// let output = tanh.forward(input).unwrap();
    /// assert_eq!(output.to_vec::<f64>().unwrap(), vec![-0.7615941559557649, 0.0, 0.7615941559557649]);
    /// ```
    fn forward(&mut self, input: Self::Input) -> Result<Self::Output, crate::ModelError> {
        Ok(input.tanh()?)
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

impl Display for Tanh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "tanh.{}()", self.id)
    }
}
