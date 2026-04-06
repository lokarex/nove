use std::{
    collections::HashMap,
    fmt::Display,
    sync::atomic::{AtomicUsize, Ordering},
};

use nove_tensor::{DType, Device, Shape, Tensor};

use crate::{Model, ModelError};

static ID: AtomicUsize = AtomicUsize::new(0);

/// Layer Normalization layer.
///
/// # Notes
/// * The `LayerNorm` is now only created by the `LayerNormBuilder`.
/// * This layer normalizes the input over the last N dimensions specified by `normalized_shape`.
/// * During both training and inference, it computes mean and variance per sample.
/// * No running statistics are maintained.
///
/// # Fields
/// * `gamma` - The learnable scale parameter (gamma) with shape `normalized_shape`.
/// * `beta` - The learnable shift parameter (beta) with shape `normalized_shape`.
/// * `normalized_shape` - The shape of the dimensions to normalize.
/// * `epsilon` - A small value added to the variance for numerical stability.
/// * `affine` - Whether to use learnable affine parameters (gamma and beta).
/// * `id` - The unique ID of the layer normalization layer.
///
/// # Examples
/// ```no_run
/// use nove::model::nn::LayerNormBuilder;
/// use nove::tensor::{Device, DType};
///
/// let ln = LayerNormBuilder::new(vec![768])       // Required
///     .epsilon(1e-5)                     // Optional, default is 1e-5
///     .affine(true)                      // Optional, default is true
///     .device(Device::cpu())             // Optional, default is cpu
///     .dtype(DType::F32)                 // Optional, default is F32
///     .build();
/// ```
///
/// # See Also
/// * [`LayerNormBuilder`] - The builder for the layer normalization layer.
#[derive(Debug, Clone)]
pub struct LayerNorm {
    gamma: Tensor,
    beta: Tensor,
    normalized_shape: Vec<usize>,
    epsilon: f64,
    id: usize,
    affine: bool,
}

impl LayerNorm {
    /// Get the gamma tensor (gamma) in the layer normalization layer.
    ///
    /// # Returns
    /// * `Tensor` - The gamma tensor.
    pub fn gamma(&self) -> Tensor {
        self.gamma.copy()
    }

    /// Get the beta tensor (beta) in the layer normalization layer.
    ///
    /// # Returns
    /// * `Tensor` - The beta tensor.
    pub fn beta(&self) -> Tensor {
        self.beta.copy()
    }
}

/// The builder for the layer normalization layer.
///
/// # Notes
/// * The `LayerNormBuilder` must be created using [`LayerNormBuilder::new()`] with a required `normalized_shape` argument.
/// * The `gamma` and `beta` tensors are initialized with ones and zeros respectively.
/// * When `affine` is `false`, `gamma` and `beta` are still initialized but with `grad_enabled` set to `false`.
///
/// # Required Arguments
/// * `normalized_shape` - The shape of the dimensions to normalize (e.g., `vec![768]`).
///
/// # Optional Arguments
/// * `epsilon` - A small value added to the variance for numerical stability. Default is `1e-5`.
/// * `affine` - Whether to use learnable affine parameters (gamma and beta). Default is `true`.
/// * `device` - The device to use for the layer. Default is `Device::cpu()`.
/// * `dtype` - The data type to use for the layer. Default is `DType::F32`.
///
/// # Fields
/// * `normalized_shape` - The shape of the dimensions to normalize.
/// * `epsilon` - A small value added to the variance for numerical stability.
/// * `affine` - Whether to use learnable affine parameters (gamma and beta).
/// * `device` - The device to use for the layer.
/// * `dtype` - The data type to use for the layer.
///
/// # Examples
/// ```no_run
/// use nove::model::nn::LayerNormBuilder;
/// use nove::tensor::{Device, DType};
///
/// let ln = LayerNormBuilder::new(vec![768])       // Required
///     .epsilon(1e-5)                     // Optional, default is 1e-5
///     .affine(true)                      // Optional, default is true
///     .device(Device::cpu())             // Optional, default is cpu
///     .dtype(DType::F32)                 // Optional, default is F32
///     .build();
/// ```
pub struct LayerNormBuilder {
    normalized_shape: Vec<usize>,
    epsilon: f64,
    affine: bool,
    device: Device,
    dtype: DType,
}

impl LayerNormBuilder {
    /// Create a new LayerNormBuilder with the specified normalized shape.
    ///
    /// # Arguments
    /// * `normalized_shape` - The shape of the dimensions to normalize.
    ///
    /// # Returns
    /// * `Self` - A new builder instance with the configured normalized shape.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::nn::LayerNormBuilder;
    /// let mut ln_builder = LayerNormBuilder::new(vec![768]);
    /// ```
    pub fn new(normalized_shape: Vec<usize>) -> Self {
        Self {
            normalized_shape,
            epsilon: 1e-5,
            affine: true,
            device: Device::cpu(),
            dtype: DType::F32,
        }
    }

    /// Configure the normalized shape.
    ///
    /// # Arguments
    /// * `normalized_shape` - The shape of the dimensions to normalize.
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured normalized shape.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::nn::LayerNormBuilder;
    /// let mut ln_builder = LayerNormBuilder::new(vec![768]);
    /// ln_builder.normalized_shape(vec![512]);
    /// ```
    pub fn normalized_shape(&mut self, normalized_shape: Vec<usize>) -> &mut Self {
        self.normalized_shape = normalized_shape;
        self
    }

    /// Configure the epsilon value for numerical stability.
    ///
    /// # Arguments
    /// * `epsilon` - A small value added to the variance for numerical stability.
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured epsilon value.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::nn::LayerNormBuilder;
    /// let mut ln_builder = LayerNormBuilder::new(vec![768]);
    /// ln_builder.epsilon(1e-5);
    /// ```
    pub fn epsilon(&mut self, epsilon: f64) -> &mut Self {
        self.epsilon = epsilon;
        self
    }

    /// Configure whether to use learnable affine parameters (gamma and beta).
    ///
    /// # Arguments
    /// * `affine` - Whether to use learnable affine parameters (gamma and beta).
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured affine setting.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::nn::LayerNormBuilder;
    /// let mut ln_builder = LayerNormBuilder::new(vec![768]);
    /// ln_builder.affine(true);
    /// ```
    pub fn affine(&mut self, affine: bool) -> &mut Self {
        self.affine = affine;
        self
    }

    /// Configure the device to use for the layer.
    ///
    /// # Arguments
    /// * `device` - The device to use for the layer.
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured device.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::nn::LayerNormBuilder;
    /// use nove::tensor::Device;
    /// let mut ln_builder = LayerNormBuilder::new(vec![768]);
    /// ln_builder.device(Device::cpu());
    /// ```
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
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::nn::LayerNormBuilder;
    /// use nove::tensor::DType;
    /// let mut ln_builder = LayerNormBuilder::new(vec![768]);
    /// ln_builder.dtype(DType::F32);
    /// ```
    pub fn dtype(&mut self, dtype: DType) -> &mut Self {
        self.dtype = dtype;
        self
    }

    /// Build the layer normalization layer.
    ///
    /// # Returns
    /// * `Ok(LayerNorm)` - The built layer normalization layer.
    /// * `Err(ModelError)` - The error when building the layer normalization layer.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::nn::LayerNormBuilder;
    /// let mut ln_builder = LayerNormBuilder::new(vec![768]);
    /// let ln = ln_builder.build().unwrap();
    /// ```
    pub fn build(&self) -> Result<LayerNorm, ModelError> {
        let normalized_shape = self.normalized_shape.clone();

        if normalized_shape.is_empty() {
            return Err(ModelError::InvalidArgument(
                "normalized_shape in LayerNormBuilder must not be empty".to_string(),
            ));
        }

        for &dim in &normalized_shape {
            if dim == 0 {
                return Err(ModelError::InvalidArgument(
                    "normalized_shape dimensions must be greater than 0".to_string(),
                ));
            }
        }

        if self.epsilon <= 0.0 {
            return Err(ModelError::InvalidArgument(
                "epsilon in LayerNormBuilder must be greater than 0".to_string(),
            ));
        }

        let id = ID.fetch_add(1, Ordering::Relaxed);

        let gamma = Tensor::ones(
            &Shape::from_dims(&normalized_shape),
            &self.dtype,
            &self.device,
            self.affine,
        )?
        .to_dtype(&self.dtype)?
        .require_name(&format!("layer_norm.{}.gamma", id))?;

        let beta = Tensor::zeros(
            &Shape::from_dims(&normalized_shape),
            &self.dtype,
            &self.device,
            self.affine,
        )?
        .to_dtype(&self.dtype)?
        .require_name(&format!("layer_norm.{}.beta", id))?;

        Ok(LayerNorm {
            gamma,
            beta,
            normalized_shape,
            epsilon: self.epsilon,
            id,
            affine: self.affine,
        })
    }
}

impl Display for LayerNorm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "layer_norm.{}(normalized_shape={:?}, epsilon={}, affine={})",
            self.id, self.normalized_shape, self.epsilon, self.affine
        )
    }
}

impl Model for LayerNorm {
    type Input = Tensor;
    type Output = Tensor;

    /// Apply the layer normalization layer to the input tensor.
    ///
    /// # Arguments
    /// * `input` - The input tensor with shape [..., *normalized_shape].
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The output tensor with the same shape as input if successful.
    /// * `Err(ModelError)` - The error when applying the layer normalization layer.
    fn forward(&mut self, input: Self::Input) -> Result<Self::Output, ModelError> {
        let input_shape = input.shape()?;
        let ndim = input_shape.dims().len();
        let normalized_ndim = self.normalized_shape.len();

        if ndim < normalized_ndim {
            return Err(ModelError::InvalidArgument(format!(
                "LayerNorm expects input with at least {} dimensions, got {} with shape {:?}",
                normalized_ndim,
                ndim,
                input_shape.dims()
            )));
        }

        // Check that the last `normalized_ndim` dimensions match `normalized_shape`
        for (i, &dim) in self.normalized_shape.iter().enumerate() {
            let input_dim = input_shape.dims()[ndim - normalized_ndim + i];
            if input_dim != dim {
                return Err(ModelError::InvalidArgument(format!(
                    "LayerNorm expects normalized_shape {:?} to match input shape {:?} at dimension {}",
                    self.normalized_shape,
                    input_shape.dims(),
                    ndim - normalized_ndim + i
                )));
            }
        }

        // Compute total size of normalized dimensions and rest dimensions
        let normalized_size: usize = self.normalized_shape.iter().product();
        let total_size = input_shape.dims().iter().product::<usize>();
        let rest_size = total_size / normalized_size;

        // Reshape to [rest_size, normalized_size] for statistics computation
        let reshaped = input.reshape(&Shape::from_dims(&[rest_size, normalized_size]))?;

        // Compute mean and variance over the normalized dimension (dimension 1)
        let mean = reshaped
            .mean(Some((1, true)))?
            .reshape(&Shape::from_dims(&[rest_size, 1]))?;
        let var = reshaped
            .var(1, false, true)?
            .reshape(&Shape::from_dims(&[rest_size, 1]))?;

        // Reshape mean and var to have shape [..., 1, 1, ...] with ones for normalized dimensions
        let mut broadcast_shape = input_shape.dims().to_vec();
        for i in 0..normalized_ndim {
            broadcast_shape[ndim - normalized_ndim + i] = 1;
        }
        let mean_broadcast = mean.reshape(&Shape::from_dims(&broadcast_shape))?;
        let var_broadcast = var.reshape(&Shape::from_dims(&broadcast_shape))?;

        // Normalize: (x - mean) / sqrt(var + epsilon)
        let normalized = Tensor::div(
            &input.sub(&mean_broadcast)?,
            &Tensor::sqrt(&var_broadcast.affine(1f64, self.epsilon)?)?,
        )?;

        // Apply affine transformation if enabled
        let output = if self.affine {
            normalized.mul(&self.gamma)?.add(&self.beta)?
        } else {
            normalized
        };

        Ok(output)
    }

    fn require_grad(&mut self, grad_enabled: bool) -> Result<(), ModelError> {
        self.gamma = self.gamma.require_grad(grad_enabled)?;
        self.beta = self.beta.require_grad(grad_enabled)?;
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<(), ModelError> {
        self.gamma = self.gamma.to_device(device)?;
        self.beta = self.beta.to_device(device)?;
        Ok(())
    }

    fn to_dtype(&mut self, dtype: &DType) -> Result<(), ModelError> {
        self.gamma = self.gamma.to_dtype(dtype)?;
        self.beta = self.beta.to_dtype(dtype)?;
        Ok(())
    }

    fn parameters(&self) -> Result<Vec<Tensor>, ModelError> {
        if self.affine {
            Ok(vec![self.gamma.copy(), self.beta.copy()])
        } else {
            Ok(vec![])
        }
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
