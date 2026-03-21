mod avg_pool1d;
mod avg_pool2d;
mod max_pool1d;
mod max_pool2d;

pub use avg_pool1d::AvgPool1d;
pub use avg_pool2d::AvgPool2d;
pub use max_pool1d::MaxPool1d;
pub use max_pool2d::MaxPool2d;

use crate::{Model, ModelError};
use nove_tensor::{DType, Device, Tensor};
use std::collections::HashMap;

/// 1D pooling layer enum.
///
/// This enum provides a unified interface for different 1D pooling functions
/// in the model layer.
///
/// # Variants
/// * `AvgPool1d` - 1D average pooling layer.
/// * `MaxPool1d` - 1D max pooling layer.
///
/// # Examples
/// ```no_run
/// use nove::tensor::{Device, Shape, Tensor};
/// use nove::model::layer::{Pool1d, AvgPool1d};
/// use nove::model::Model;
///
/// let mut avg_pool = Pool1d::AvgPool1d(AvgPool1d::new(2, None).unwrap());
/// println!("{}", avg_pool);
///
/// let input = Tensor::randn(0.0, 1.0, &Shape::from(&[1, 1, 10]), &Device::cpu(), false).unwrap();
/// let output = avg_pool.forward(input).unwrap();
/// println!("{}", output);
/// ```
#[derive(Debug, Clone)]
pub enum Pool1d {
    AvgPool1d(AvgPool1d),
    MaxPool1d(MaxPool1d),
}

impl Model for Pool1d {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&mut self, input: Self::Input) -> Result<Self::Output, ModelError> {
        match self {
            Pool1d::AvgPool1d(layer) => layer.forward(input),
            Pool1d::MaxPool1d(layer) => layer.forward(input),
        }
    }

    fn require_grad(&mut self, grad_enabled: bool) -> Result<(), ModelError> {
        match self {
            Pool1d::AvgPool1d(layer) => layer.require_grad(grad_enabled),
            Pool1d::MaxPool1d(layer) => layer.require_grad(grad_enabled),
        }
    }

    fn to_device(&mut self, device: &Device) -> Result<(), ModelError> {
        match self {
            Pool1d::AvgPool1d(layer) => layer.to_device(device),
            Pool1d::MaxPool1d(layer) => layer.to_device(device),
        }
    }

    fn to_dtype(&mut self, dtype: &DType) -> Result<(), ModelError> {
        match self {
            Pool1d::AvgPool1d(layer) => layer.to_dtype(dtype),
            Pool1d::MaxPool1d(layer) => layer.to_dtype(dtype),
        }
    }

    fn parameters(&self) -> Result<Vec<Tensor>, ModelError> {
        match self {
            Pool1d::AvgPool1d(layer) => layer.parameters(),
            Pool1d::MaxPool1d(layer) => layer.parameters(),
        }
    }

    fn named_parameters(&self) -> Result<HashMap<String, Tensor>, ModelError> {
        match self {
            Pool1d::AvgPool1d(layer) => layer.named_parameters(),
            Pool1d::MaxPool1d(layer) => layer.named_parameters(),
        }
    }
}

impl std::fmt::Display for Pool1d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Pool1d::AvgPool1d(layer) => write!(f, "{}", layer),
            Pool1d::MaxPool1d(layer) => write!(f, "{}", layer),
        }
    }
}

/// 2D pooling layer enum.
///
/// This enum provides a unified interface for both average and max pooling operations.
/// It wraps the concrete pooling layer implementations and delegates all operations
/// to the underlying layer.
///
/// # Variants
/// * `AvgPool2d` - 2D average pooling layer.
/// * `MaxPool2d` - 2D max pooling layer.
///
/// # Examples
/// ```no_run
/// use nove::model::layer::{Pool2d, AvgPool2d, MaxPool2d};
///
/// // Create an average pooling layer
/// let avg_pool = Pool2d::AvgPool2d(AvgPool2d::new((2, 2), None).unwrap());
///
/// // Create a max pooling layer
/// let max_pool = Pool2d::MaxPool2d(MaxPool2d::new((2, 2), None).unwrap());
/// ```
#[derive(Debug, Clone)]
pub enum Pool2d {
    /// 2D average pooling layer.
    AvgPool2d(AvgPool2d),
    /// 2D max pooling layer.
    MaxPool2d(MaxPool2d),
}

impl crate::Model for Pool2d {
    type Input = nove_tensor::Tensor;
    type Output = nove_tensor::Tensor;

    fn forward(&mut self, input: Self::Input) -> Result<Self::Output, crate::ModelError> {
        match self {
            Pool2d::AvgPool2d(layer) => layer.forward(input),
            Pool2d::MaxPool2d(layer) => layer.forward(input),
        }
    }

    fn require_grad(&mut self, grad_enabled: bool) -> Result<(), crate::ModelError> {
        match self {
            Pool2d::AvgPool2d(layer) => layer.require_grad(grad_enabled),
            Pool2d::MaxPool2d(layer) => layer.require_grad(grad_enabled),
        }
    }

    fn to_device(&mut self, device: &nove_tensor::Device) -> Result<(), crate::ModelError> {
        match self {
            Pool2d::AvgPool2d(layer) => layer.to_device(device),
            Pool2d::MaxPool2d(layer) => layer.to_device(device),
        }
    }

    fn to_dtype(&mut self, dtype: &nove_tensor::DType) -> Result<(), crate::ModelError> {
        match self {
            Pool2d::AvgPool2d(layer) => layer.to_dtype(dtype),
            Pool2d::MaxPool2d(layer) => layer.to_dtype(dtype),
        }
    }

    fn parameters(&self) -> Result<Vec<nove_tensor::Tensor>, crate::ModelError> {
        match self {
            Pool2d::AvgPool2d(layer) => layer.parameters(),
            Pool2d::MaxPool2d(layer) => layer.parameters(),
        }
    }

    fn named_parameters(
        &self,
    ) -> Result<std::collections::HashMap<String, nove_tensor::Tensor>, crate::ModelError> {
        match self {
            Pool2d::AvgPool2d(layer) => layer.named_parameters(),
            Pool2d::MaxPool2d(layer) => layer.named_parameters(),
        }
    }
}

impl std::fmt::Display for Pool2d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Pool2d::AvgPool2d(layer) => write!(f, "{}", layer),
            Pool2d::MaxPool2d(layer) => write!(f, "{}", layer),
        }
    }
}
