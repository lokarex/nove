use std::{
    collections::HashMap,
    fmt::Display,
    sync::atomic::{AtomicUsize, Ordering},
};

use nove_tensor::{DType, Device, Shape, Tensor};

use crate::{Model, ModelError};

static ID: AtomicUsize = AtomicUsize::new(0);

/// Embedding layer.
///
/// # Notes
/// * The `Embedding` is now only created by the `EmbeddingBuilder`.
///
/// # Fields
/// * `weight` - The weight tensor with shape `[num_embeddings, embedding_dim]`.
/// * `num_embeddings` - The number of embeddings (vocabulary size).
/// * `embedding_dim` - The dimensionality of the embeddings.
/// * `padding_idx` - Optional index to pad with zeros.
/// * `id` - The unique ID of the embedding layer.
///
/// # Examples
/// ```
/// use nove::model::nn::EmbeddingBuilder;
/// use nove::tensor::{Device, DType};
///
/// let embedding = EmbeddingBuilder::new(1000, 512)  // Required: num_embeddings, embedding_dim
///     .padding_idx(Some(0))     // Optional, default is None
///     .device(Device::cpu())    // Optional, default is cpu
///     .dtype(DType::F32)        // Optional, default is F32
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct Embedding {
    weight: Tensor,
    num_embeddings: usize,
    embedding_dim: usize,
    padding_idx: Option<usize>,
    id: usize,
}

impl Embedding {
    /// Get the weight tensor in the embedding layer.
    ///
    /// # Returns
    /// * `Tensor` - The weight tensor.
    pub fn weight(&self) -> Tensor {
        self.weight.copy()
    }

    /// Get the number of embeddings in the embedding layer.
    ///
    /// # Returns
    /// * `usize` - The number of embeddings.
    pub fn num_embeddings(&self) -> usize {
        self.num_embeddings
    }

    /// Get the embedding dimension in the embedding layer.
    ///
    /// # Returns
    /// * `usize` - The embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Get the padding index in the embedding layer.
    ///
    /// # Returns
    /// * `Option<usize>` - The padding index if set, otherwise None.
    pub fn padding_idx(&self) -> Option<usize> {
        self.padding_idx
    }

    /// Get the unique ID of the embedding layer.
    ///
    /// # Returns
    /// * `usize` - The unique ID.
    pub fn id(&self) -> usize {
        self.id
    }
}

impl Model for Embedding {
    type Input = Tensor;
    type Output = Tensor;

    /// Apply the embedding layer to the input tensor.
    ///
    /// # Arguments
    /// * `input` - The input tensor with indices of dtype I64.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The output tensor with shape `input_shape + [embedding_dim]`.
    /// * `Err(ModelError)` - The error when applying the embedding layer to the input tensor.
    fn forward(&mut self, input: Self::Input) -> Result<Self::Output, ModelError> {
        // Validate input dtype is I64
        let input_dtype = input.dtype()?;
        if input_dtype != DType::I64 {
            return Err(ModelError::InvalidArgument(format!(
                "Embedding input must have dtype I64, got {:?}",
                input_dtype
            )));
        }

        // Get input shape
        let input_shape = input.shape()?;
        let input_dims = input_shape.dims();

        // Flatten input to 1D for index_select
        let flattened_size = input_dims.iter().product::<usize>();
        let flat_input = input.reshape(&Shape::from_dims(&[flattened_size]))?;

        // Perform embedding lookup using index_select along dimension 0
        let embeddings = self.weight.index_select(&flat_input, 0)?;

        // Reshape embeddings to input_shape + [embedding_dim]
        let mut output_shape = input_dims.to_vec();
        output_shape.push(self.embedding_dim);
        let mut embeddings = embeddings.reshape(&Shape::from_dims(&output_shape))?;

        // Apply padding_idx masking if set
        if let Some(padding_idx) = self.padding_idx {
            // Create mask where input == padding_idx (same shape as input)
            let padding_tensor = Tensor::from_scalar(padding_idx as i64, &input.device()?, false)?;
            let mask = input.eq(&padding_tensor)?;

            // Expand mask to match embedding dimensions: add embedding dimension
            let mask = mask.unsqueeze(mask.shape()?.dims().len())?;
            let mask = mask.broadcast(&Shape::from_dims(&output_shape))?;

            // Zero out positions where mask is true (input == padding_idx)
            let zeros = Tensor::zeros(
                &embeddings.shape()?,
                &embeddings.dtype()?,
                &embeddings.device()?,
                false,
            )?;
            embeddings = Tensor::where_cond(&mask, &zeros, &embeddings)?;
        }

        Ok(embeddings)
    }

    fn require_grad(&mut self, grad_enabled: bool) -> Result<(), ModelError> {
        self.weight = self.weight.require_grad(grad_enabled)?;
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<(), ModelError> {
        self.weight = self.weight.to_device(device)?;
        Ok(())
    }

    fn to_dtype(&mut self, dtype: &DType) -> Result<(), ModelError> {
        self.weight = self.weight.to_dtype(dtype)?;
        Ok(())
    }

    fn parameters(&self) -> Result<Vec<Tensor>, ModelError> {
        Ok(vec![self.weight.copy()])
    }

    fn named_parameters(&self) -> Result<HashMap<String, Tensor>, ModelError> {
        self.parameters()?
            .into_iter()
            .map(|t| match t.name()? {
                Some(name) => Ok((name, t)),
                None => Err(ModelError::ParameterMissingName),
            })
            .collect::<Result<HashMap<_, _>, ModelError>>()
    }
}

impl Display for Embedding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "embedding.{}(num_embeddings={}, embedding_dim={}, padding_idx={:?})",
            self.id, self.num_embeddings, self.embedding_dim, self.padding_idx
        )
    }
}

/// The builder for the Embedding layer.
///
/// # Notes
/// * The `EmbeddingBuilder` must be created using [`EmbeddingBuilder::new()`] with required `num_embeddings` and `embedding_dim` arguments.
/// * The `weight` tensor in the embedding layer is initialized with a normal distribution (`mean=0.0`, `std=1.0`).
///
/// # Required Arguments
/// * `num_embeddings` - The number of embeddings (vocabulary size) (passed to `new()`).
/// * `embedding_dim` - The dimensionality of the embeddings (passed to `new()`).
///
/// # Optional Arguments
/// * `padding_idx` - Optional index to pad with zeros. Default is `None`.
/// * `device` - The device to use for the layer. Default is `Device::cpu()`.
/// * `dtype` - The data type to use for the layer. Default is `DType::F32`.
/// * `grad_enabled` - Whether to enable the gradient computation. Default is `true`.
///
/// # Fields
/// * `num_embeddings` - The number of embeddings.
/// * `embedding_dim` - The dimensionality of the embeddings.
/// * `padding_idx` - Optional index to pad with zeros.
/// * `device` - The device to use for the layer.
/// * `dtype` - The data type to use for the layer.
/// * `grad_enabled` - Whether to enable the gradient computation.
///
/// # Examples
/// ```
/// use nove::model::nn::EmbeddingBuilder;
/// use nove::tensor::{Device, DType};
///
/// let embedding = EmbeddingBuilder::new(1000, 512)  // Required: num_embeddings, embedding_dim
///     .padding_idx(Some(0))     // Optional, default is None
///     .device(Device::cpu())    // Optional, default is cpu
///     .dtype(DType::F32)        // Optional, default is F32
///     .grad_enabled(true)       // Optional, default is true
///     .build();
/// ```
pub struct EmbeddingBuilder {
    num_embeddings: usize,
    embedding_dim: usize,
    padding_idx: Option<usize>,
    device: Device,
    dtype: DType,
    grad_enabled: bool,
}

impl EmbeddingBuilder {
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        Self {
            num_embeddings,
            embedding_dim,
            padding_idx: None,
            device: Device::cpu(),
            dtype: DType::F32,
            grad_enabled: true,
        }
    }
}

impl EmbeddingBuilder {
    /// Configure the number of embeddings.
    ///
    /// # Arguments
    /// * `num_embeddings` - The number of embeddings.
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured number of embeddings.
    pub fn num_embeddings(&mut self, num_embeddings: usize) -> &mut Self {
        self.num_embeddings = num_embeddings;
        self
    }

    /// Configure the embedding dimension.
    ///
    /// # Arguments
    /// * `embedding_dim` - The embedding dimension.
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured embedding dimension.
    pub fn embedding_dim(&mut self, embedding_dim: usize) -> &mut Self {
        self.embedding_dim = embedding_dim;
        self
    }

    /// Configure the padding index.
    ///
    /// # Arguments
    /// * `padding_idx` - Optional index to pad with zeros.
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured padding index.
    pub fn padding_idx(&mut self, padding_idx: Option<usize>) -> &mut Self {
        self.padding_idx = padding_idx;
        self
    }

    /// Configure the device to use for the layer.
    ///
    /// # Arguments
    /// * `device` - The device to use for the layer.
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured device.
    pub fn device(&mut self, device: Device) -> &mut Self {
        self.device = device;
        self
    }

    /// Configure the data type to use for the layer.
    ///
    /// # Arguments
    /// * `dtype` - The data type to use for the layer.
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured data type.
    pub fn dtype(&mut self, dtype: DType) -> &mut Self {
        self.dtype = dtype;
        self
    }

    /// Configure whether to enable the gradient computation.
    ///
    /// # Arguments
    /// * `grad_enabled` - Whether to enable the gradient computation.
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured gradient computation.
    pub fn grad_enabled(&mut self, grad_enabled: bool) -> &mut Self {
        self.grad_enabled = grad_enabled;
        self
    }

    /// Build the embedding layer.
    ///
    /// # Returns
    /// * `Ok(Embedding)` - The built embedding layer.
    /// * `Err(ModelError)` - The error when building the embedding layer.
    pub fn build(&self) -> Result<Embedding, ModelError> {
        let num_embeddings = self.num_embeddings;
        let embedding_dim = self.embedding_dim;
        let padding_idx = self.padding_idx;

        // Validate parameters
        if num_embeddings == 0 {
            return Err(ModelError::InvalidArgument(
                "num_embeddings must be greater than 0".to_string(),
            ));
        }

        if embedding_dim == 0 {
            return Err(ModelError::InvalidArgument(
                "embedding_dim must be greater than 0".to_string(),
            ));
        }

        if let Some(padding_idx) = padding_idx {
            if padding_idx >= num_embeddings {
                return Err(ModelError::InvalidArgument(format!(
                    "padding_idx {} is out of range [0, {})",
                    padding_idx, num_embeddings
                )));
            }
        }

        // Generate a unique ID for the embedding layer.
        let id = ID.fetch_add(1, Ordering::Relaxed);

        // Initialize the weight tensor with uniform distribution U(-sqrt(k), sqrt(k)) where k = 1/embedding_dim
        let k = 1.0 / embedding_dim as f64;
        let bound = k.sqrt();
        let weight = Tensor::rand(
            -bound,
            bound,
            &Shape::from_dims(&[num_embeddings, embedding_dim]),
            &self.device,
            self.grad_enabled,
        )?
        .to_dtype(&self.dtype)?
        .require_name(&format!("embedding.{}.weight", id))?;

        Ok(Embedding {
            weight,
            num_embeddings,
            embedding_dim,
            padding_idx,
            id,
        })
    }
}
