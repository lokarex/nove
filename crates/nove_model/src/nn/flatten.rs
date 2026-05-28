use std::{
    collections::HashMap,
    fmt::Display,
    sync::atomic::{AtomicUsize, Ordering},
};

use nove_tensor::{DType, Device, Tensor};

use crate::{Model, ModelError};

static ID: AtomicUsize = AtomicUsize::new(1);

/// Flatten layer that reshapes a tensor by merging multiple dimensions into one.
///
/// # Notes
/// * Uses 0-based indexing, consistent with Tensor::flatten.
/// * Default start_dim=1, end_dim=-1.
/// * Supports negative indexing: -1 means the last dimension, -2 means second last, etc.
///
/// # Fields
/// * `start_dim` - The 0-based dimension index to start flattening (default: 1).
/// * `end_dim` - The 0-based dimension index to stop flattening, inclusive (default: -1, meaning last dimension).
/// * `id` - The unique ID of the flatten layer.
///
/// # Examples
/// ```
/// use nove::tensor::{Device, DType, Shape, Tensor};
/// use nove::model::nn::FlattenBuilder;
/// use nove::model::Model;
///
/// // Default: flatten from dimension 1 to last dimension (preserves first dimension)
/// let mut flatten = FlattenBuilder::new().build();
/// println!("{}", flatten);
///
/// // Input: [batch, channels, height, width] e.g. [2, 3, 4, 5]
/// let input = Tensor::ones(&Shape::from_dims(&[2, 3, 4, 5]), &DType::F32, &nove::device::candle::cpu().unwrap(), false).unwrap();
/// let output = flatten.forward(input.copy()).unwrap();
/// // Output: [batch, channels * height * width] e.g. [2, 60]
///
/// // Flatten all dimensions
/// let mut flatten = FlattenBuilder::new()
///     .start_dim(0)
///     .end_dim(-1)
///     .build();
/// let output = flatten.forward(input).unwrap();
/// // Output: [120]
/// ```
#[derive(Debug, Clone)]
pub struct Flatten {
    start_dim: isize,
    end_dim: isize,
    id: usize,
}

/// The builder for the Flatten layer.
///
/// # Notes
/// * The `FlattenBuilder` is created using [`FlattenBuilder::new()`] with no required arguments.
/// * Default values: `start_dim=1`, `end_dim=-1` (PyTorch defaults).
///
/// # Optional Arguments
/// * `start_dim` - The 0-based dimension index to start flattening. Default is 1.
/// * `end_dim` - The 0-based dimension index to stop flattening (inclusive). Default is -1.
///
/// # Examples
/// ```
/// use nove::model::nn::FlattenBuilder;
///
/// // Default: flatten from dimension 1 to the last dimension
/// let flatten = FlattenBuilder::new().build();
///
/// // Flatten all dimensions
/// let flatten = FlattenBuilder::new()
///     .start_dim(0)
///     .end_dim(-1)
///     .build();
/// ```
pub struct FlattenBuilder {
    start_dim: isize,
    end_dim: isize,
}

impl FlattenBuilder {
    /// Create a new FlattenBuilder with default values.
    ///
    /// # Returns
    /// * `FlattenBuilder` - The builder with default start_dim=1, end_dim=-1.
    pub fn new() -> Self {
        Self {
            start_dim: 1,
            end_dim: -1,
        }
    }

    /// Configure the starting dimension (0-based).
    ///
    /// # Arguments
    /// * `start_dim` - The 0-based dimension index to start flattening.
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured starting dimension.
    pub fn start_dim(&mut self, start_dim: isize) -> &mut Self {
        self.start_dim = start_dim;
        self
    }

    /// Configure the ending dimension (0-based, inclusive).
    ///
    /// # Arguments
    /// * `end_dim` - The 0-based dimension index to stop flattening (inclusive).
    /// Use -1 for the last dimension.
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured ending dimension.
    pub fn end_dim(&mut self, end_dim: isize) -> &mut Self {
        self.end_dim = end_dim;
        self
    }

    /// Build the Flatten layer.
    ///
    /// # Returns
    /// * `Flatten` - The built Flatten layer.
    pub fn build(&self) -> Flatten {
        Flatten {
            start_dim: self.start_dim,
            end_dim: self.end_dim,
            id: ID.fetch_add(1, Ordering::Relaxed),
        }
    }
}

impl Default for FlattenBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Model for Flatten {
    type Input = Tensor;
    type Output = Tensor;

    /// Apply the Flatten layer to the input tensor.
    ///
    /// # Arguments
    /// * `input` - The input tensor.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The flattened tensor.
    /// * `Err(ModelError)` - The error when flattening the tensor.
    fn forward(&mut self, input: Self::Input) -> Result<Self::Output, ModelError> {
        let output = input.flatten(Some(self.start_dim), Some(self.end_dim))?;
        Ok(output)
    }

    fn require_grad(&mut self, _: bool) -> Result<(), ModelError> {
        Ok(())
    }

    fn to_device(&mut self, _: &Device) -> Result<(), ModelError> {
        Ok(())
    }

    fn to_dtype(&mut self, _: &DType) -> Result<(), ModelError> {
        Ok(())
    }

    fn parameters(&self) -> Result<Vec<Tensor>, ModelError> {
        Ok(vec![])
    }

    fn named_parameters(&self) -> Result<HashMap<String, Tensor>, ModelError> {
        Ok(HashMap::new())
    }
}

impl Display for Flatten {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "flatten.{}(start_dim={}, end_dim={})",
            self.id, self.start_dim, self.end_dim
        )
    }
}
