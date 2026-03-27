use std::{
    collections::HashMap,
    fmt::Display,
    sync::atomic::{AtomicUsize, Ordering},
};

use nove_tensor::{DType, Device, Shape, Tensor};

use crate::layer::Activation;
use crate::{Model, ModelError};

static ID: AtomicUsize = AtomicUsize::new(0);

/// Gated Recurrent Unit Cell layer.
///
/// <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" id="MathJax-script" async></script>
///
/// The GruCell (Gated Recurrent Unit Cell) is a type of recurrent neural network
/// cell that implements gated recurrence with reset and update gates.
/// It is a simpler alternative to LSTM cells with fewer parameters.
///
/// The GRU computes the following:
///
/// $$ r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{t-1} + b_{hr}) $$
///
/// $$ z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{t-1} + b_{hz}) $$
///
/// $$ n_t = \tanh(W_{in} x_t + b_{in} + r_t \odot (W_{hn} h_{t-1} + b_{hn})) $$
///
/// $$ h_t = (1 - z_t) \odot n_t + z_t \odot h_{t-1} $$
///
/// Where:
/// - \( x_t \) is the input at time step t
/// - \( h_{t-1} \) is the hidden state from previous time step
/// - \( r_t \) is the reset gate
/// - \( z_t \) is the update gate
/// - \( n_t \) is the candidate hidden state
/// - \( h_t \) is the hidden state at time step t
/// - \( \sigma \) is the sigmoid function
/// - \( \tanh \) is the hyperbolic tangent function
/// - \( \odot \) denotes element-wise multiplication
///
/// # Notes
/// * The `GruCell` is now only created by the [`GruCellBuilder`].
/// * The forward pass expects a tuple of (input, hidden) tensors and returns
///   the new hidden state.
///
/// # Fields
/// * `weight_ih` - The weight tensor for input to hidden transformation with shape [3*hidden_size, input_size].
/// * `weight_hh` - The weight tensor for hidden to hidden transformation with shape [3*hidden_size, hidden_size].
/// * `bias_ih` - The bias tensor for input to hidden transformation with shape \[3*hidden_size\] (optional).
/// * `bias_hh` - The bias tensor for hidden to hidden transformation with shape \[3*hidden_size\] (optional).
/// * `input_size` - The number of input features.
/// * `hidden_size` - The number of hidden features.
/// * `id` - The unique ID of the GRU cell.
///
/// # Examples
/// ```no_run
/// use nove::model::layer::GruCellBuilder;
/// use nove::tensor::{Device, DType};
///
/// let gru_cell = GruCellBuilder::new(10, 20)
///     .bias_enabled(true)
///     .device(Device::cpu())
///     .dtype(DType::F32)
///     .grad_enabled(true)
///     .build();
/// ```
///
/// # See Also
/// * [`GruCellBuilder`] - The builder for the GruCell layer.
#[derive(Debug, Clone)]
pub struct GruCell {
    weight_ih: Tensor,
    weight_hh: Tensor,
    bias_ih: Option<Tensor>,
    bias_hh: Option<Tensor>,
    input_size: usize,
    hidden_size: usize,
    id: usize,
}

impl GruCell {
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

    /// Get the unique ID.
    ///
    /// # Returns
    /// * `usize` - The unique ID.
    pub fn id(&self) -> usize {
        self.id
    }
}

impl Display for GruCell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GruCell(id: {}, input_size: {}, hidden_size: {}, bias: {})",
            self.id,
            self.input_size,
            self.hidden_size,
            self.bias_ih.is_some() && self.bias_hh.is_some()
        )
    }
}

impl Model for GruCell {
    type Input = (Tensor, Tensor);
    type Output = Tensor;

    /// Apply the GRU cell to the input tensor and hidden state.
    ///
    /// # Arguments
    /// * `input` - A tuple containing (input_tensor, hidden_state).
    ///   - input_tensor: Tensor with shape [batch_size, input_size]
    ///   - hidden_state: Tensor with shape [batch_size, hidden_size]
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The new hidden state with shape [batch_size, hidden_size].
    /// * `Err(ModelError)` - The error when applying the GRU cell.
    fn forward(&mut self, input: Self::Input) -> Result<Self::Output, ModelError> {
        let (input_tensor, hidden_state) = input;

        let input_shape = input_tensor.shape()?;
        let hidden_shape = hidden_state.shape()?;

        if input_shape.dims().len() != 2 {
            return Err(ModelError::InvalidArgument(format!(
                "GruCell expects input tensor with 2 dimensions [batch_size, input_size], got shape {:?}",
                input_shape.dims()
            )));
        }

        if hidden_shape.dims().len() != 2 {
            return Err(ModelError::InvalidArgument(format!(
                "GruCell expects hidden state tensor with 2 dimensions [batch_size, hidden_size], got shape {:?}",
                hidden_shape.dims()
            )));
        }

        let batch_size = input_shape.dims()[0];
        let input_size_from_tensor = input_shape.dims()[1];
        let hidden_size_from_tensor = hidden_shape.dims()[1];

        if input_size_from_tensor != self.input_size {
            return Err(ModelError::InvalidArgument(format!(
                "GruCell expects input size {}, got {}",
                self.input_size, input_size_from_tensor
            )));
        }

        if hidden_size_from_tensor != self.hidden_size {
            return Err(ModelError::InvalidArgument(format!(
                "GruCell expects hidden size {}, got {}",
                self.hidden_size, hidden_size_from_tensor
            )));
        }

        if hidden_shape.dims()[0] != batch_size {
            return Err(ModelError::InvalidArgument(format!(
                "GruCell expects batch size {} for hidden state, got {}",
                batch_size,
                hidden_shape.dims()[0]
            )));
        }

        // Compute linear transformations: gates = input @ weight_ih.t() + hidden @ weight_hh.t()
        let input_part = input_tensor.matmul(&self.weight_ih.transpose(0, 1)?)?;
        let hidden_part = hidden_state.matmul(&self.weight_hh.transpose(0, 1)?)?;

        let mut gates = input_part.add(&hidden_part)?;

        // Add biases if enabled
        if let Some(bias_ih) = &self.bias_ih {
            gates = gates.add(bias_ih)?;
        }

        if let Some(bias_hh) = &self.bias_hh {
            gates = gates.add(bias_hh)?;
        }

        // Split gates into three parts: reset, update, candidate
        // gates shape: [batch_size, 3 * hidden_size]
        let chunk_size = self.hidden_size;
        let (reset_gate, update_gate, candidate) = (
            gates.narrow(1, 0 * chunk_size, chunk_size)?,
            gates.narrow(1, 1 * chunk_size, chunk_size)?,
            gates.narrow(1, 2 * chunk_size, chunk_size)?,
        );

        // Apply sigmoid activation to reset and update gates
        let reset_gate = Activation::sigmoid().forward(reset_gate)?;
        let update_gate = Activation::sigmoid().forward(update_gate)?;

        // Compute candidate hidden state: n_t = tanh(W_in x_t + b_in + r_t * (W_hn h_{t-1} + b_hn))
        // The 'candidate' variable contains: input @ W_in.t() + b_in + hidden @ W_hn.t() + b_hn
        // We need to separate the hidden part, multiply by reset_gate, then combine with input part

        // First, compute the hidden candidate part: hidden @ W_hn.t() + b_hn
        let hidden_candidate_part = hidden_state.matmul(
            &self
                .weight_hh
                .transpose(0, 1)?
                .narrow(1, 2 * chunk_size, chunk_size)?,
        )?;

        // Add bias for candidate part if enabled
        let mut hidden_candidate_with_bias = hidden_candidate_part;
        if let Some(bias_hh) = &self.bias_hh {
            let candidate_bias = bias_hh.narrow(0, 2 * chunk_size, chunk_size)?;
            hidden_candidate_with_bias = hidden_candidate_with_bias.add(&candidate_bias)?;
        }

        // Apply reset gate: reset_gate * hidden_candidate_with_bias
        let reset_hidden = reset_gate.mul(&hidden_candidate_with_bias)?;

        // Extract input part: candidate - hidden_candidate_with_bias
        let input_part = candidate.sub(&hidden_candidate_with_bias)?;

        // Combine: input_part + reset_gate * hidden_candidate_with_bias
        let candidate_with_reset = input_part.add(&reset_hidden)?;

        // Apply tanh activation to get candidate hidden state
        let candidate_hidden = Activation::tanh().forward(candidate_with_reset)?;

        // Compute new hidden state: (1 - update_gate) * candidate_hidden + update_gate * hidden_state
        let one_minus_update_gate = update_gate.affine(-1.0, 1.0)?;
        let new_hidden = one_minus_update_gate
            .mul(&candidate_hidden)?
            .add(&update_gate.mul(&hidden_state)?)?;

        Ok(new_hidden)
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

/// The builder for the GruCell layer.
///
/// # Notes
/// * The `GruCellBuilder` provides [`GruCellBuilder::new()`] method to create a builder with required parameters.
/// * The weight tensors are initialized with uniform distribution `U(-sqrt(k), sqrt(k))` where `k = 1 / hidden_size`,
///   following PyTorch's GruCell initialization.
///   The bias tensors are initialized with zeros (if enabled).
///
/// # Required Arguments
/// * `input_size` - The number of input features.
/// * `hidden_size` - The number of hidden features.
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
/// * `bias_enabled` - Whether to enable the bias terms.
/// * `device` - The device to use for the layer.
/// * `dtype` - The data type to use for the layer.
/// * `grad_enabled` - Whether to enable the gradient computation.
///
/// # Examples
/// ```no_run
/// use nove::model::layer::GruCellBuilder;
/// use nove::tensor::{Device, DType};
///
/// let gru_cell = GruCellBuilder::new(10, 20)
///     .bias_enabled(true)
///     .device(Device::cpu())
///     .dtype(DType::F32)
///     .grad_enabled(true)
///     .build();
/// ```
pub struct GruCellBuilder {
    input_size: usize,
    hidden_size: usize,
    bias_enabled: bool,
    device: Device,
    dtype: DType,
    grad_enabled: bool,
}

impl GruCellBuilder {
    /// Create a new GruCellBuilder with required input_size and hidden_size.
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
            device: Device::cpu(),
            dtype: DType::F32,
            grad_enabled: true,
        }
    }

    /// Configure the number of expected features in the input x.
    ///
    /// # Arguments
    /// * `input_size` - The number of expected features in the input x
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured input size.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::GruCellBuilder;
    /// let mut builder = GruCellBuilder::new(10, 20);
    /// builder.input_size(10);
    /// ```
    pub fn input_size(&mut self, input_size: usize) -> &mut Self {
        self.input_size = input_size;
        self
    }

    /// Configure the number of features in the hidden state h.
    ///
    /// # Arguments
    /// * `hidden_size` - The number of features in the hidden state h
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured hidden size.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::GruCellBuilder;
    /// let mut builder = GruCellBuilder::new(10, 20);
    /// builder.hidden_size(20);
    /// ```
    pub fn hidden_size(&mut self, hidden_size: usize) -> &mut Self {
        self.hidden_size = hidden_size;
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
    /// use nove::model::layer::GruCellBuilder;
    /// let mut builder = GruCellBuilder::new(10, 20);
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
    /// use nove::model::layer::GruCellBuilder;
    /// use nove::tensor::Device;
    /// let mut builder = GruCellBuilder::new(10, 20);
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
    /// use nove::model::layer::GruCellBuilder;
    /// use nove::tensor::DType;
    /// let mut builder = GruCellBuilder::new(10, 20);
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
    /// use nove::model::layer::GruCellBuilder;
    /// let mut builder = GruCellBuilder::new(10, 20);
    /// builder.grad_enabled(false);
    /// ```
    pub fn grad_enabled(&mut self, grad_enabled: bool) -> &mut Self {
        self.grad_enabled = grad_enabled;
        self
    }

    /// Build the GRU cell layer.
    ///
    /// # Returns
    /// * `Ok(GruCell)` - The built GRU cell layer.
    /// * `Err(ModelError)` - The error when building the GRU cell layer.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::GruCellBuilder;
    /// let mut builder = GruCellBuilder::new(10, 20);
    /// let gru_cell = builder.build().unwrap();
    /// ```
    pub fn build(&self) -> Result<GruCell, ModelError> {
        if self.input_size == 0 {
            return Err(ModelError::InvalidArgument(
                "input_size in GruCellBuilder must be greater than 0".to_string(),
            ));
        }

        if self.hidden_size == 0 {
            return Err(ModelError::InvalidArgument(
                "hidden_size in GruCellBuilder must be greater than 0".to_string(),
            ));
        }

        let id = ID.fetch_add(1, Ordering::Relaxed);

        // PyTorch GruCell weight initialization: uniform distribution U(-sqrt(k), sqrt(k)) where k = 1 / hidden_size
        let k = 1.0 / self.hidden_size as f64;
        let bound = k.sqrt();

        let weight_ih = Tensor::rand(
            -bound,
            bound,
            &Shape::from_dims(&[3 * self.hidden_size, self.input_size]),
            &self.device,
            self.grad_enabled,
        )?
        .to_dtype(&self.dtype)?
        .require_name(&format!("gru_cell.{}.weight_ih", id))?;

        let weight_hh = Tensor::rand(
            -bound,
            bound,
            &Shape::from_dims(&[3 * self.hidden_size, self.hidden_size]),
            &self.device,
            self.grad_enabled,
        )?
        .to_dtype(&self.dtype)?
        .require_name(&format!("gru_cell.{}.weight_hh", id))?;

        let bias_ih = if self.bias_enabled {
            let bias = Tensor::zeros(
                &Shape::from_dims(&[3 * self.hidden_size]),
                &self.dtype,
                &self.device,
                self.grad_enabled,
            )?
            .require_name(&format!("gru_cell.{}.bias_ih", id))?;
            Some(bias)
        } else {
            None
        };

        let bias_hh = if self.bias_enabled {
            let bias = Tensor::zeros(
                &Shape::from_dims(&[3 * self.hidden_size]),
                &self.dtype,
                &self.device,
                self.grad_enabled,
            )?
            .require_name(&format!("gru_cell.{}.bias_hh", id))?;
            Some(bias)
        } else {
            None
        };

        Ok(GruCell {
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            input_size: self.input_size,
            hidden_size: self.hidden_size,
            id,
        })
    }
}
