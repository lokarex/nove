use std::{
    collections::HashMap,
    fmt::Display,
    sync::atomic::{AtomicUsize, Ordering},
};

use nove_tensor::Tensor;

use crate::{Model, ModelError};

static ID: AtomicUsize = AtomicUsize::new(0);

/// Gaussian Error Linear Unit (GELU) layer.
///
/// <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" id="MathJax-script" async></script>
///
/// The Gaussian Error Linear Unit (GELU) is a smooth, non-linear activation function
/// that is used in state-of-the-art transformer models like BERT and GPT. It combines
/// the properties of dropout and zoneout, providing better performance than ReLU
/// in many tasks.
///
/// The GELU function is computed as:
///
/// $$ \text{GELU}(x) = x \cdot \Phi(x) $$
///
/// A common approximation used in practice is:
///
/// $$ \text{GELU}(x) \approx 0.5 \cdot x \cdot \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \cdot \left(x + 0.044715 \cdot x^3\right)\right)\right) $$
///
/// Where:
/// - x is the input value
/// - Φ(x) is the cumulative distribution function of the standard normal distribution:
///
/// $$ \Phi(x) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{x} e^{-\frac{t^2}{2}} dt = \frac{1}{2} \left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right] $$
///
/// # Notes
/// * The `GELU` is now only created by the `GELU::new()` function.
///
/// # Fields
/// * `id` - The unique ID of the GELU layer.
///
/// # Examples
/// ```
/// use nove::tensor::{Device, Tensor};
/// use nove::model::layer::GELU;
/// use nove::model::Model;
///
/// let mut gelu = GELU::new();
/// println!("{}", gelu);
///
/// let input = Tensor::from_data(&[-1.0f32, 0.0f32, 1.0f32], &Device::cpu(), false).unwrap();
/// let output = gelu.forward(input).unwrap();
/// println!("{}", output);
/// ```
#[derive(Debug, Clone)]
pub struct GELU {
    id: usize,
}

impl GELU {
    /// Create a new Gaussian Error Linear Unit (GELU) layer.
    ///
    /// # Returns
    /// * `GELU` - The new GELU layer.
    pub fn new() -> Self {
        Self {
            id: ID.fetch_add(1, Ordering::Relaxed),
        }
    }
}

impl Default for GELU {
    fn default() -> Self {
        Self::new()
    }
}

impl Model for GELU {
    type Input = Tensor;

    type Output = Tensor;

    /// Apply the Gaussian Error Linear Unit (GELU) layer to the input tensor.
    ///
    /// # Arguments
    /// * `input: Tensor` - The input tensor.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The tensor after applying the GELU activation function.
    /// * `Err(ModelError)` - The error when applying the GELU activation function.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// use nove::model::layer::GELU;
    /// use nove::model::Model;
    ///
    /// let mut gelu = GELU::new();
    /// let input = Tensor::from_data(vec![-1.0, 0.0, 1.0], &Device::cpu(), false).unwrap();
    /// let output = gelu.forward(input).unwrap();
    /// assert_eq!(output.to_vec::<f64>().unwrap(), vec![-0.15880800939172324, 0.0, 0.8411919906082768]);
    /// ```
    fn forward(&mut self, input: Self::Input) -> Result<Self::Output, crate::ModelError> {
        Ok(input.gelu()?)
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

impl Display for GELU {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "gelu.{}()", self.id)
    }
}
