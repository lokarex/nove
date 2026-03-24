use std::{collections::HashMap, fmt::Display};

use nove_tensor::{DType, Device, Tensor};

use crate::{Model, ModelError};

use super::{Activation, BatchNorm1d, BatchNorm1dBuilder, Dropout, Linear, LinearBuilder};

static ID: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Linear (fully connected) block configuration.
///
/// # Notes
/// * The `LinearBlockBuilder` is used to configure a linear layer with optional
///   activation function and batch normalization (1D).
/// * The default configuration has bias enabled, activation disabled,
///   batch normalization (1D) disabled, and dropout disabled.
/// * **Important**: Only one activation function can be enabled at a time.
///   When you call `with_activation()` method, any previous activation function
///   will be replaced with the new one.
///
/// # Required Arguments
/// * `in_features` - The number of input features.
/// * `out_features` - The number of output features.
///
/// # Optional Arguments
/// * `activation` - Optional activation function after the linear layer. Default is `None`. (configured via `with_activation()`)
/// * `use_batch_norm1d` - Whether 1D batch normalization is enabled after the linear layer. Default is `false`. (configured via `with_batch_norm1d()`)
/// * `bias_enabled` - Whether the bias term is enabled. Default is `true`.
/// * `dropout_probability` - The dropout probability (0.0 means no dropout). Default is `0.0`. (configured via `with_dropout()`)
/// * `device` - The device to use for the layer. Default is `Device::cpu()`.
/// * `dtype` - The data type to use for the layer. Default is `DType::F32`.
/// * `grad_enabled` - Whether to enable the gradient computation. Default is `true`.
///
/// # Fields
/// * `in_features` - The number of input features.
/// * `out_features` - The number of output features.
/// * `activation` - Optional activation function after the linear layer.
/// * `use_batch_norm1d` - Whether 1D batch normalization is enabled after the linear layer.
/// * `bias_enabled` - Whether the bias term is enabled.
/// * `dropout_probability` - The dropout probability (0.0 means no dropout).
/// * `device` - The device to use for the layer.
/// * `dtype` - The data type to use for the layer.
/// * `grad_enabled` - Whether to enable the gradient computation.
///
/// # Examples
/// ```no_run
/// use nove::model::layer::{LinearBlockBuilder, Activation};
/// use nove::tensor::{Device, DType};
///
/// let block = LinearBlockBuilder::new(800, 10)
///     .with_activation(Activation::gelu())
///     .with_batch_norm1d()
///     .bias_enabled(true)
///     .device(Device::cpu())
///     .dtype(DType::F32)
///     .grad_enabled(true);
/// ```
#[derive(Debug, Clone)]
pub struct LinearBlockBuilder {
    in_features: usize,
    out_features: usize,
    activation: Option<Activation>,
    use_batch_norm1d: bool,
    bias_enabled: bool,
    dropout_probability: f32,
    device: Device,
    dtype: DType,
    grad_enabled: bool,
}

impl LinearBlockBuilder {
    /// Create a new linear block builder with default settings.
    ///
    /// # Arguments
    /// * `in_features` - The number of input features.
    /// * `out_features` - The number of output features.
    ///
    /// # Returns
    /// * `LinearBlockBuilder` - The new linear block builder.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::LinearBlockBuilder;
    /// let builder = LinearBlockBuilder::new(800, 10);
    /// ```
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            in_features,
            out_features,
            activation: None,
            use_batch_norm1d: false,
            bias_enabled: true,
            dropout_probability: 0.0,
            device: Device::cpu(),
            dtype: DType::F32,
            grad_enabled: true,
        }
    }

    /// Configure the number of input features.
    ///
    /// # Arguments
    /// * `in_features` - The number of input features.
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured input features.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::LinearBlockBuilder;
    /// let mut builder = LinearBlockBuilder::new(800, 10);
    /// builder.in_features(784);
    /// ```
    pub fn in_features(&mut self, in_features: usize) -> &mut Self {
        self.in_features = in_features;
        self
    }

    /// Configure the number of output features.
    ///
    /// # Arguments
    /// * `out_features` - The number of output features.
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured output features.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::LinearBlockBuilder;
    /// let mut builder = LinearBlockBuilder::new(800, 10);
    /// builder.out_features(100);
    /// ```
    pub fn out_features(&mut self, out_features: usize) -> &mut Self {
        self.out_features = out_features;
        self
    }

    /// Configure activation function after the linear layer.
    ///
    /// # Notes
    /// * Only one activation function can be active at a time. Calling this method
    ///   replaces any previously configured activation function.
    ///
    /// # Arguments
    /// * `activation` - The activation function to use.
    ///
    /// # Returns
    /// * `Self` - The linear block builder with the configured activation function.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::{LinearBlockBuilder, Activation};
    /// let builder = LinearBlockBuilder::new(800, 10).with_activation(Activation::relu());
    /// ```
    pub fn with_activation(mut self, activation: Activation) -> Self {
        self.activation = Some(activation);
        self
    }

    /// Configure without any activation functions.
    ///
    /// # Returns
    /// * `Self` - The linear block builder with all activation functions disabled.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::{LinearBlockBuilder, Activation};
    /// let builder = LinearBlockBuilder::new(800, 10).with_activation(Activation::relu()).without_activation();
    /// ```
    pub fn without_activation(mut self) -> Self {
        self.activation = None;
        self
    }

    /// Configure 1D batch normalization after the linear layer.
    ///
    /// # Returns
    /// * `Self` - The linear block builder with 1D batch normalization enabled.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::LinearBlockBuilder;
    /// let builder = LinearBlockBuilder::new(800, 10).with_batch_norm1d();
    /// ```
    pub fn with_batch_norm1d(mut self) -> Self {
        self.use_batch_norm1d = true;
        self
    }

    /// Configure without 1D batch normalization.
    ///
    /// # Returns
    /// * `Self` - The linear block builder with 1D batch normalization disabled.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::LinearBlockBuilder;
    /// let builder = LinearBlockBuilder::new(800, 10).with_batch_norm1d().without_batch_norm1d();
    /// ```
    pub fn without_batch_norm1d(mut self) -> Self {
        self.use_batch_norm1d = false;
        self
    }

    /// Configure whether to enable the bias term.
    ///
    /// # Arguments
    /// * `bias_enabled` - Whether to enable the bias term.
    ///
    /// # Returns
    /// * `Self` - The linear block builder with the configured bias term.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::LinearBlockBuilder;
    /// let builder = LinearBlockBuilder::new(800, 10).bias_enabled(false);
    /// ```
    pub fn bias_enabled(mut self, bias_enabled: bool) -> Self {
        self.bias_enabled = bias_enabled;
        self
    }

    /// Configure dropout after the linear layer.
    ///
    /// # Arguments
    /// * `probability` - The dropout probability (must be in range [0, 1)).
    ///
    /// # Returns
    /// * `Self` - The linear block builder with dropout enabled.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::LinearBlockBuilder;
    /// let builder = LinearBlockBuilder::new(800, 10).with_dropout(0.5);
    /// ```
    pub fn with_dropout(mut self, probability: f32) -> Self {
        self.dropout_probability = probability;
        self
    }

    /// Configure without dropout.
    ///
    /// # Returns
    /// * `Self` - The linear block builder with dropout disabled.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::LinearBlockBuilder;
    /// let builder = LinearBlockBuilder::new(800, 10).with_dropout(0.5).without_dropout();
    /// ```
    pub fn without_dropout(mut self) -> Self {
        self.dropout_probability = 0.0;
        self
    }

    /// Configure the device to use for the layer.
    ///
    /// # Arguments
    /// * `device` - The device to use for the layer.
    ///
    /// # Returns
    /// * `Self` - The linear block builder with the configured device.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::LinearBlockBuilder;
    /// use nove::tensor::Device;
    /// let builder = LinearBlockBuilder::new(800, 10).device(Device::cpu());
    /// ```
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Configure the data type to use for the layer.
    ///
    /// # Arguments
    /// * `dtype` - The data type to use for the layer.
    ///
    /// # Returns
    /// * `Self` - The linear block builder with the configured data type.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::LinearBlockBuilder;
    /// use nove::tensor::DType;
    /// let builder = LinearBlockBuilder::new(800, 10).dtype(DType::F32);
    /// ```
    pub fn dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }

    /// Configure whether to enable the gradient computation.
    ///
    /// # Arguments
    /// * `grad_enabled` - Whether to enable the gradient computation.
    ///
    /// # Returns
    /// * `Self` - The linear block builder with the configured gradient computation.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::LinearBlockBuilder;
    /// let builder = LinearBlockBuilder::new(800, 10).grad_enabled(true);
    /// ```
    pub fn grad_enabled(mut self, grad_enabled: bool) -> Self {
        self.grad_enabled = grad_enabled;
        self
    }

    /// Build the linear block.
    ///
    /// # Returns
    /// * `Ok(LinearBlock)` - The built linear block.
    /// * `Err(ModelError)` - The error when building the linear block.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::LinearBlockBuilder;
    /// let block = LinearBlockBuilder::new(800, 10).build().unwrap();
    /// ```
    pub fn build(self) -> Result<LinearBlock, ModelError> {
        let linear = LinearBuilder::new(self.in_features, self.out_features)
            .bias_enabled(self.bias_enabled)
            .device(self.device.clone())
            .dtype(self.dtype)
            .grad_enabled(self.grad_enabled)
            .build()?;

        let batch_norm1d = if self.use_batch_norm1d {
            Some(
                BatchNorm1dBuilder::new(self.out_features)
                    .device(self.device.clone())
                    .dtype(self.dtype)
                    .build()?,
            )
        } else {
            None
        };

        let activation = self.activation;

        let dropout = if self.dropout_probability > 0.0 {
            Some(Dropout::new(self.dropout_probability)?)
        } else {
            None
        };

        let id = ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Ok(LinearBlock {
            linear,
            batch_norm1d,
            activation,
            dropout,
            id,
        })
    }
}

/// Linear (fully connected) block.
///
/// # Notes
/// * The `LinearBlock` is a sequential block that chains a linear layer with optional
///   batch normalization (1D), activation function, and dropout.
///
/// # Fields
/// * `linear` - The linear layer.
/// * `batch_norm1d` - Optional 1D batch normalization layer.
/// * `activation` - Optional activation function.
/// * `dropout` - Optional dropout layer.
/// * `id` - The unique ID of the linear block.
///
/// # Examples
/// ```no_run
/// use nove::model::layer::{LinearBlockBuilder, Activation};
///
/// let mut block = LinearBlockBuilder::new(800, 10)
///     .with_activation(Activation::gelu())
///     .with_batch_norm1d()
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct LinearBlock {
    linear: Linear,
    batch_norm1d: Option<BatchNorm1d>,
    activation: Option<Activation>,
    dropout: Option<Dropout>,
    id: usize,
}

impl Model for LinearBlock {
    type Input = (Tensor, bool);
    type Output = Tensor;

    /// Apply the linear block to the input tensor.
    ///
    /// # Arguments
    /// * `input: (Tensor, bool)` - A tuple containing the input tensor and a boolean flag
    ///   indicating whether the block is in training mode. This is used by BatchNorm1d and Dropout.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The output tensor if successful.
    /// * `Err(ModelError)` - The error when applying the linear block to the input tensor.
    fn forward(&mut self, input: Self::Input) -> Result<Self::Output, ModelError> {
        let (input, training) = input;
        let mut output = self.linear.forward(input)?;

        if let Some(ref mut bn) = self.batch_norm1d {
            output = bn.forward((output, training))?;
        }

        if let Some(ref mut activation) = self.activation {
            output = activation.forward(output)?;
        }

        if let Some(ref mut dropout) = self.dropout {
            output = dropout.forward((output, training))?;
        }

        Ok(output)
    }

    fn require_grad(&mut self, grad_enabled: bool) -> Result<(), ModelError> {
        self.linear.require_grad(grad_enabled)?;
        if let Some(ref mut bn) = self.batch_norm1d {
            bn.require_grad(grad_enabled)?;
        }
        if let Some(ref mut activation) = self.activation {
            activation.require_grad(grad_enabled)?;
        }
        if let Some(ref mut dropout) = self.dropout {
            dropout.require_grad(grad_enabled)?;
        }
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<(), ModelError> {
        self.linear.to_device(device)?;
        if let Some(ref mut bn) = self.batch_norm1d {
            bn.to_device(device)?;
        }
        if let Some(ref mut activation) = self.activation {
            activation.to_device(device)?;
        }
        if let Some(ref mut dropout) = self.dropout {
            dropout.to_device(device)?;
        }
        Ok(())
    }

    fn to_dtype(&mut self, dtype: &DType) -> Result<(), ModelError> {
        self.linear.to_dtype(dtype)?;
        if let Some(ref mut bn) = self.batch_norm1d {
            bn.to_dtype(dtype)?;
        }
        if let Some(ref mut activation) = self.activation {
            activation.to_dtype(dtype)?;
        }
        if let Some(ref mut dropout) = self.dropout {
            dropout.to_dtype(dtype)?;
        }
        Ok(())
    }

    fn parameters(&self) -> Result<Vec<Tensor>, ModelError> {
        let mut params = Vec::new();
        params.extend(self.linear.parameters()?);
        if let Some(ref bn) = self.batch_norm1d {
            params.extend(bn.parameters()?);
        }
        if let Some(ref activation) = self.activation {
            params.extend(activation.parameters()?);
        }
        if let Some(ref dropout) = self.dropout {
            params.extend(dropout.parameters()?);
        }
        Ok(params)
    }

    fn named_parameters(&self) -> Result<HashMap<String, Tensor>, ModelError> {
        let mut params = HashMap::new();
        let prefix = format!("linear_block{}.", self.id);
        for (name, tensor) in self.linear.named_parameters()? {
            params.insert(format!("{}{}", prefix, name), tensor);
        }
        if let Some(ref bn) = self.batch_norm1d {
            for (name, tensor) in bn.named_parameters()? {
                params.insert(format!("{}{}", prefix, name), tensor);
            }
        }
        if let Some(ref activation) = self.activation {
            for (name, tensor) in activation.named_parameters()? {
                params.insert(format!("{}{}", prefix, name), tensor);
            }
        }
        if let Some(ref dropout) = self.dropout {
            for (name, tensor) in dropout.named_parameters()? {
                params.insert(format!("{}{}", prefix, name), tensor);
            }
        }
        Ok(params)
    }
}

impl Display for LinearBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "linear_block.{}(\n", self.id)?;
        write!(f, "  {},\n", self.linear)?;
        if let Some(ref bn) = self.batch_norm1d {
            write!(f, "  {},\n", bn)?;
        }
        if let Some(ref activation) = self.activation {
            write!(f, "  {},\n", activation)?;
        }
        if let Some(ref dropout) = self.dropout {
            write!(f, "  {},\n", dropout)?;
        }
        write!(f, ")")
    }
}
