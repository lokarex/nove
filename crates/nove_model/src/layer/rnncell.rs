use std::{
    collections::HashMap,
    fmt::Display,
    sync::atomic::{AtomicUsize, Ordering},
};

use nove_tensor::{DType, Device, Shape, Tensor};

use crate::layer::Activation;
use crate::{Model, ModelError};

static ID: AtomicUsize = AtomicUsize::new(0);

/// Recurrent Neural Network Cell layer.
///
/// # Notes
/// * The `RNNCell` is now only created by the [`RNNCellBuilder`].
/// * This cell implements the basic RNN computation:
///   \( h' = \text{activation}(W_{ih} x + b_{ih} + W_{hh} h + b_{hh}) \)
/// * The forward pass expects a tuple of (input, hidden) tensors and returns
///   the new hidden state.
///
/// # Fields
/// * `weight_ih` - The weight tensor for input to hidden transformation with shape [hidden_size, input_size].
/// * `weight_hh` - The weight tensor for hidden to hidden transformation with shape [hidden_size, hidden_size].
/// * `bias_ih` - The bias tensor for input to hidden transformation with shape [hidden_size] (optional).
/// * `bias_hh` - The bias tensor for hidden to hidden transformation with shape [hidden_size] (optional).
/// * `input_size` - The number of input features.
/// * `hidden_size` - The number of hidden features.
/// * `activation` - The activation function to use.
/// * `id` - The unique ID of the RNN cell.
///
/// # Examples
/// ```no_run
/// use nove::model::layer::RNNCellBuilder;
/// use nove::model::layer::Activation;
/// use nove::tensor::{Device, DType};
///
/// let rnn_cell = RNNCellBuilder::new(10, 20)
///     .activation(Activation::tanh())
///     .bias_enabled(true)
///     .device(Device::cpu())
///     .dtype(DType::F32)
///     .build();
/// ```
///
/// # See Also
/// * [`RNNCellBuilder`] - The builder for the RNNCell layer.
#[derive(Debug, Clone)]
pub struct RNNCell {
    weight_ih: Tensor,
    weight_hh: Tensor,
    bias_ih: Option<Tensor>,
    bias_hh: Option<Tensor>,
    input_size: usize,
    hidden_size: usize,
    activation: Activation,
    id: usize,
}

impl RNNCell {
    /// Get the input-to-hidden weight tensor.
    ///
    /// # Returns
    /// * `Tensor` - The weight_ih tensor.
    pub fn weight_ih(&self) -> Tensor {
        self.weight_ih.copy()
    }

    /// Get the hidden-to-hidden weight tensor.
    ///
    /// # Returns
    /// * `Tensor` - The weight_hh tensor.
    pub fn weight_hh(&self) -> Tensor {
        self.weight_hh.copy()
    }

    /// Get the input-to-hidden bias tensor.
    ///
    /// # Returns
    /// * `Option<Tensor>` - The bias_ih tensor if enabled, otherwise None.
    pub fn bias_ih(&self) -> Option<Tensor> {
        self.bias_ih.as_ref().map(|b| b.copy())
    }

    /// Get the hidden-to-hidden bias tensor.
    ///
    /// # Returns
    /// * `Option<Tensor>` - The bias_hh tensor if enabled, otherwise None.
    pub fn bias_hh(&self) -> Option<Tensor> {
        self.bias_hh.as_ref().map(|b| b.copy())
    }

    /// Get the input size.
    ///
    /// # Returns
    /// * `usize` - The input size.
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Get the hidden size.
    ///
    /// # Returns
    /// * `usize` - The hidden size.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Get the activation function.
    ///
    /// # Returns
    /// * `Activation` - The activation function.
    pub fn activation(&self) -> Activation {
        self.activation.clone()
    }
}

impl Model for RNNCell {
    type Input = (Tensor, Tensor);
    type Output = Tensor;

    /// Apply the RNN cell to the input tensor and hidden state.
    ///
    /// # Arguments
    /// * `input` - A tuple containing (input_tensor, hidden_state).
    ///   - input_tensor: Tensor with shape [batch_size, input_size]
    ///   - hidden_state: Tensor with shape [batch_size, hidden_size]
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The new hidden state with shape [batch_size, hidden_size].
    /// * `Err(ModelError)` - The error when applying the RNN cell.
    fn forward(&mut self, input: Self::Input) -> Result<Self::Output, ModelError> {
        let (input_tensor, hidden_state) = input;

        let input_shape = input_tensor.shape()?;
        let hidden_shape = hidden_state.shape()?;

        if input_shape.dims().len() != 2 {
            return Err(ModelError::InvalidArgument(format!(
                "RNNCell expects input tensor with 2 dimensions [batch_size, input_size], got shape {:?}",
                input_shape.dims()
            )));
        }

        if hidden_shape.dims().len() != 2 {
            return Err(ModelError::InvalidArgument(format!(
                "RNNCell expects hidden state tensor with 2 dimensions [batch_size, hidden_size], got shape {:?}",
                hidden_shape.dims()
            )));
        }

        let batch_size = input_shape.dims()[0];
        let input_size_from_tensor = input_shape.dims()[1];
        let hidden_size_from_tensor = hidden_shape.dims()[1];

        if input_size_from_tensor != self.input_size {
            return Err(ModelError::InvalidArgument(format!(
                "RNNCell expects input size {}, got {}",
                self.input_size, input_size_from_tensor
            )));
        }

        if hidden_size_from_tensor != self.hidden_size {
            return Err(ModelError::InvalidArgument(format!(
                "RNNCell expects hidden size {}, got {}",
                self.hidden_size, hidden_size_from_tensor
            )));
        }

        if hidden_shape.dims()[0] != batch_size {
            return Err(ModelError::InvalidArgument(format!(
                "RNNCell expects batch size {} for hidden state, got {}",
                batch_size,
                hidden_shape.dims()[0]
            )));
        }

        let input_part = input_tensor.matmul(&self.weight_ih.transpose(0, 1)?)?;
        let hidden_part = hidden_state.matmul(&self.weight_hh.transpose(0, 1)?)?;

        let mut combined = input_part.add(&hidden_part)?;

        if let Some(bias_ih) = &self.bias_ih {
            combined = combined.add(bias_ih)?;
        }

        if let Some(bias_hh) = &self.bias_hh {
            combined = combined.add(bias_hh)?;
        }

        self.activation.forward(combined)
    }

    fn require_grad(&mut self, grad_enabled: bool) -> Result<(), ModelError> {
        self.weight_ih = self.weight_ih.require_grad(grad_enabled)?;
        self.weight_hh = self.weight_hh.require_grad(grad_enabled)?;

        if let Some(bias_ih) = &mut self.bias_ih {
            self.bias_ih = Some(bias_ih.require_grad(grad_enabled)?);
        }

        if let Some(bias_hh) = &mut self.bias_hh {
            self.bias_hh = Some(bias_hh.require_grad(grad_enabled)?);
        }

        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<(), ModelError> {
        self.weight_ih = self.weight_ih.to_device(device)?;
        self.weight_hh = self.weight_hh.to_device(device)?;

        if let Some(bias_ih) = &mut self.bias_ih {
            self.bias_ih = Some(bias_ih.to_device(device)?);
        }

        if let Some(bias_hh) = &mut self.bias_hh {
            self.bias_hh = Some(bias_hh.to_device(device)?);
        }

        Ok(())
    }

    fn to_dtype(&mut self, dtype: &DType) -> Result<(), ModelError> {
        self.weight_ih = self.weight_ih.to_dtype(dtype)?;
        self.weight_hh = self.weight_hh.to_dtype(dtype)?;

        if let Some(bias_ih) = &mut self.bias_ih {
            self.bias_ih = Some(bias_ih.to_dtype(dtype)?);
        }

        if let Some(bias_hh) = &mut self.bias_hh {
            self.bias_hh = Some(bias_hh.to_dtype(dtype)?);
        }

        Ok(())
    }

    fn parameters(&self) -> Result<Vec<Tensor>, ModelError> {
        let mut params = vec![self.weight_ih.copy(), self.weight_hh.copy()];

        if let Some(bias_ih) = &self.bias_ih {
            params.push(bias_ih.copy());
        }

        if let Some(bias_hh) = &self.bias_hh {
            params.push(bias_hh.copy());
        }

        Ok(params)
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

impl Display for RNNCell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bias_enabled = self.bias_ih.is_some(); // bias_ih and bias_hh are both enabled or disabled together
        write!(
            f,
            "rnncell.{}(input_size={}, hidden_size={}, bias_enabled={}, activation={})",
            self.id, self.input_size, self.hidden_size, bias_enabled, self.activation,
        )
    }
}

/// The builder for the RNNCell layer.
///
/// # Notes
/// * The `RNNCellBuilder` provides [`RNNCellBuilder::new()`] method to create a builder with required parameters.
/// * The weight tensors are initialized with uniform distribution `U(-sqrt(k), sqrt(k))` where `k = 1 / hidden_size`,
///   following PyTorch's RNNCell initialization.
///   The bias tensors are initialized with zeros (if enabled).
///
/// # Required Arguments
/// * `input_size` - The number of input features.
/// * `hidden_size` - The number of hidden features.
/// * `activation` - The activation function to use.
///
/// # Optional Arguments
/// * `bias_enabled` - Whether to enable the bias terms. Default is `true`.
/// * `device` - The device to use for the layer. Default is `Device::cpu()`.
/// * `dtype` - The data type to use for the layer. Default is `DType::F32`.
/// * `grad_enabled` - Whether to enable the gradient computation. Default is `true`.
///
/// # Fields
/// * `input_size` - The number of input features.
/// * `hidden_size` - The number of hidden features.
/// * `activation` - The activation function to use.
/// * `bias_enabled` - Whether to enable the bias terms.
/// * `device` - The device to use for the layer.
/// * `dtype` - The data type to use for the layer.
/// * `grad_enabled` - Whether to enable the gradient computation.
///
/// # Examples
/// ```no_run
/// use nove::model::layer::RNNCellBuilder;
/// use nove::model::layer::Activation;
/// use nove::tensor::{Device, DType};
///
/// let rnn_cell = RNNCellBuilder::new(10, 20)
///     .activation(Activation::tanh())
///     .bias_enabled(true)
///     .device(Device::cpu())
///     .dtype(DType::F32)
///     .grad_enabled(true)
///     .build();
/// ```
pub struct RNNCellBuilder {
    input_size: usize,
    hidden_size: usize,
    bias_enabled: bool,
    activation: Activation,
    device: Device,
    dtype: DType,
    grad_enabled: bool,
}

impl RNNCellBuilder {
    /// Create a new RNNCellBuilder with required input_size and hidden_size.
    ///
    /// # Arguments
    /// * `input_size` - The number of expected features in the input x
    /// * `hidden_size` - The number of features in the hidden state h
    ///
    /// # Returns
    /// * `Self` - A new builder instance
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        Self {
            input_size,
            hidden_size,
            bias_enabled: true,
            activation: Activation::tanh(),
            device: Device::cpu(),
            dtype: DType::F32,
            grad_enabled: true,
        }
    }

    /// Configure the activation function.
    ///
    /// # Arguments
    /// * `activation` - The activation function (e.g., `Activation::Tanh`, `Activation::ReLU`)
    ///
    /// # Returns
    /// * `&mut Self` - The builder with configured activation
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::RNNCellBuilder;
    /// use nove::model::layer::Activation;
    /// let mut builder = RNNCellBuilder::new(10, 20);
    /// builder.activation(Activation::tanh());
    /// ```
    pub fn activation(&mut self, activation: Activation) -> &mut Self {
        self.activation = activation;
        self
    }

    /// Configure whether to enable the bias terms.
    ///
    /// # Arguments
    /// * `bias_enabled` - Whether to enable the bias terms.
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured bias terms.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::RNNCellBuilder;
    /// let mut builder = RNNCellBuilder::new(10, 20);
    /// builder.bias_enabled(false);
    /// ```
    pub fn bias_enabled(&mut self, bias_enabled: bool) -> &mut Self {
        self.bias_enabled = bias_enabled;
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
    /// use nove::model::layer::RNNCellBuilder;
    /// use nove::tensor::Device;
    /// let mut builder = RNNCellBuilder::new(10, 20);
    /// builder.device(Device::cpu());
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
    /// use nove::model::layer::RNNCellBuilder;
    /// use nove::tensor::DType;
    /// let mut builder = RNNCellBuilder::new(10, 20);
    /// builder.dtype(DType::F32);
    /// ```
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
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::RNNCellBuilder;
    /// let mut builder = RNNCellBuilder::new(10, 20);
    /// builder.grad_enabled(false);
    /// ```
    pub fn grad_enabled(&mut self, grad_enabled: bool) -> &mut Self {
        self.grad_enabled = grad_enabled;
        self
    }

    /// Build the RNN cell layer.
    ///
    /// # Returns
    /// * `Ok(RNNCell)` - The built RNN cell layer.
    /// * `Err(ModelError)` - The error when building the RNN cell layer.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::RNNCellBuilder;
    /// use nove::model::layer::Activation;
    /// let mut builder = RNNCellBuilder::new(10, 20);
    /// builder.activation(Activation::tanh());
    /// let rnn_cell = builder.build().unwrap();
    /// ```
    pub fn build(&self) -> Result<RNNCell, ModelError> {
        if self.input_size == 0 {
            return Err(ModelError::InvalidArgument(
                "input_size in RNNCellBuilder must be greater than 0".to_string(),
            ));
        }

        if self.hidden_size == 0 {
            return Err(ModelError::InvalidArgument(
                "hidden_size in RNNCellBuilder must be greater than 0".to_string(),
            ));
        }

        let id = ID.fetch_add(1, Ordering::Relaxed);

        // PyTorch RNNCell weight initialization: uniform distribution U(-sqrt(k), sqrt(k)) where k = 1 / hidden_size
        let k = 1.0 / self.hidden_size as f64;
        let bound = k.sqrt();

        let weight_ih = Tensor::rand(
            -bound,
            bound,
            &Shape::from_dims(&[self.hidden_size, self.input_size]),
            &self.device,
            self.grad_enabled,
        )?
        .to_dtype(&self.dtype)?
        .require_name(&format!("rnn_cell.{}.weight_ih", id))?;

        let weight_hh = Tensor::rand(
            -bound,
            bound,
            &Shape::from_dims(&[self.hidden_size, self.hidden_size]),
            &self.device,
            self.grad_enabled,
        )?
        .to_dtype(&self.dtype)?
        .require_name(&format!("rnn_cell.{}.weight_hh", id))?;

        let bias_ih = if self.bias_enabled {
            let bias = Tensor::zeros(
                &Shape::from_dims(&[self.hidden_size]),
                &self.dtype,
                &self.device,
                self.grad_enabled,
            )?
            .require_name(&format!("rnn_cell.{}.bias_ih", id))?;
            Some(bias)
        } else {
            None
        };

        let bias_hh = if self.bias_enabled {
            let bias = Tensor::zeros(
                &Shape::from_dims(&[self.hidden_size]),
                &self.dtype,
                &self.device,
                self.grad_enabled,
            )?
            .require_name(&format!("rnn_cell.{}.bias_hh", id))?;
            Some(bias)
        } else {
            None
        };

        Ok(RNNCell {
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            input_size: self.input_size,
            hidden_size: self.hidden_size,
            activation: self.activation.clone(),
            id,
        })
    }
}
