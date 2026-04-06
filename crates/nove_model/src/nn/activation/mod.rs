mod gelu;
pub use gelu::GELU;

mod relu;
pub use relu::ReLU;

mod silu;
pub use silu::SiLU;

mod tanh;
pub use tanh::Tanh;

mod sigmoid;
pub use sigmoid::Sigmoid;

use crate::{Model, ModelError};
use nove_tensor::{DType, Device, Tensor};
use std::{collections::HashMap, fmt::Display};

/// Activation function layer.
///
/// This enum provides a unified interface for different activation functions
/// in the model layer.
///
/// # Variants
/// * `ReLU` - Rectified Linear Unit activation layer.
/// * `GELU` - Gaussian Error Linear Unit activation layer.
/// * `SiLU` - Sigmoid Linear Unit (Swish) activation layer.
/// * `Tanh` - Hyperbolic Tangent activation layer.
/// * `Sigmoid` - Sigmoid activation layer.
///
/// # Examples
/// ```no_run
/// use nove::tensor::{Device, Tensor};
/// use nove::model::nn::Activation;
/// use nove::model::Model;
///
/// let mut relu = Activation::relu();
/// println!("{}", relu);
///
/// let input = Tensor::from_data(&[0.0f32, 1.0f32, 2.0f32], &Device::cpu(), false).unwrap();
/// let output = relu.forward(input).unwrap();
/// println!("{}", output);
/// ```
#[derive(Debug, Clone)]
pub enum Activation {
    ReLU(ReLU),
    GELU(GELU),
    SiLU(SiLU),
    Tanh(Tanh),
    Sigmoid(Sigmoid),
}

impl Activation {
    /// Create a new Rectified Linear Unit (ReLU) activation layer.
    ///
    /// # Returns
    /// * `Activation` - The new ReLU activation layer.
    pub fn relu() -> Self {
        Self::ReLU(ReLU::new())
    }

    /// Create a new Gaussian Error Linear Unit (GELU) activation layer.
    ///
    /// # Returns
    /// * `Activation` - The new GELU activation layer.
    pub fn gelu() -> Self {
        Self::GELU(GELU::new())
    }

    /// Create a new Sigmoid Linear Unit (SiLU) activation layer.
    ///
    /// # Returns
    /// * `Activation` - The new SiLU activation layer.
    pub fn silu() -> Self {
        Self::SiLU(SiLU::new())
    }

    /// Create a new Hyperbolic Tangent (Tanh) activation layer.
    ///
    /// # Returns
    /// * `Activation` - The new Tanh activation layer.
    pub fn tanh() -> Self {
        Self::Tanh(Tanh::new())
    }

    /// Create a new Sigmoid activation layer.
    ///
    /// # Returns
    /// * `Activation` - The new Sigmoid activation layer.
    pub fn sigmoid() -> Self {
        Self::Sigmoid(Sigmoid::new())
    }
}

impl Model for Activation {
    type Input = Tensor;

    type Output = Tensor;

    fn forward(&mut self, input: Self::Input) -> Result<Self::Output, crate::ModelError> {
        match self {
            Activation::ReLU(layer) => layer.forward(input),
            Activation::GELU(layer) => layer.forward(input),
            Activation::SiLU(layer) => layer.forward(input),
            Activation::Tanh(layer) => layer.forward(input),
            Activation::Sigmoid(layer) => layer.forward(input),
        }
    }

    fn require_grad(&mut self, grad_enabled: bool) -> Result<(), crate::ModelError> {
        match self {
            Activation::ReLU(layer) => layer.require_grad(grad_enabled),
            Activation::GELU(layer) => layer.require_grad(grad_enabled),
            Activation::SiLU(layer) => layer.require_grad(grad_enabled),
            Activation::Tanh(layer) => layer.require_grad(grad_enabled),
            Activation::Sigmoid(layer) => layer.require_grad(grad_enabled),
        }
    }

    fn to_device(&mut self, device: &Device) -> Result<(), crate::ModelError> {
        match self {
            Activation::ReLU(layer) => layer.to_device(device),
            Activation::GELU(layer) => layer.to_device(device),
            Activation::SiLU(layer) => layer.to_device(device),
            Activation::Tanh(layer) => layer.to_device(device),
            Activation::Sigmoid(layer) => layer.to_device(device),
        }
    }

    fn to_dtype(&mut self, dtype: &DType) -> Result<(), crate::ModelError> {
        match self {
            Activation::ReLU(layer) => layer.to_dtype(dtype),
            Activation::GELU(layer) => layer.to_dtype(dtype),
            Activation::SiLU(layer) => layer.to_dtype(dtype),
            Activation::Tanh(layer) => layer.to_dtype(dtype),
            Activation::Sigmoid(layer) => layer.to_dtype(dtype),
        }
    }

    fn parameters(&self) -> Result<Vec<nove_tensor::Tensor>, crate::ModelError> {
        match self {
            Activation::ReLU(layer) => layer.parameters(),
            Activation::GELU(layer) => layer.parameters(),
            Activation::SiLU(layer) => layer.parameters(),
            Activation::Tanh(layer) => layer.parameters(),
            Activation::Sigmoid(layer) => layer.parameters(),
        }
    }

    fn named_parameters(&self) -> Result<HashMap<String, Tensor>, ModelError> {
        match self {
            Activation::ReLU(layer) => layer.named_parameters(),
            Activation::GELU(layer) => layer.named_parameters(),
            Activation::SiLU(layer) => layer.named_parameters(),
            Activation::Tanh(layer) => layer.named_parameters(),
            Activation::Sigmoid(layer) => layer.named_parameters(),
        }
    }
}

impl Display for Activation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Activation::ReLU(layer) => write!(f, "{}", layer),
            Activation::GELU(layer) => write!(f, "{}", layer),
            Activation::SiLU(layer) => write!(f, "{}", layer),
            Activation::Tanh(layer) => write!(f, "{}", layer),
            Activation::Sigmoid(layer) => write!(f, "{}", layer),
        }
    }
}
