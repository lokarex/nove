use std::{
    collections::HashMap,
    fmt::Display,
    sync::atomic::{AtomicUsize, Ordering},
};

use nove_tensor::{DType, Device, Shape, Tensor};

use crate::{Model, ModelError};

static ID: AtomicUsize = AtomicUsize::new(0);

/// Long Short-Term Memory Cell layer.
///
/// <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" id="MathJax-script" async></script>
///
/// The LstmCell (Long Short-Term Memory Cell) is a type of recurrent neural network
/// cell that implements gated recurrence. It is capable of learning long-term
/// dependencies through its gating mechanisms.
///
/// The LSTM computes the following:
///
/// $$ i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) $$
///
/// $$ f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) $$
///
/// $$ o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) $$
///
/// $$ g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) $$
///
/// $$ c_t = f_t \odot c_{t-1} + i_t \odot g_t $$
///
/// $$ h_t = o_t \odot \tanh(c_t) $$
///
/// Where:
/// - \( x_t \) is the input at time step t
/// - \( h_{t-1} \) is the hidden state from previous time step
/// - \( c_{t-1} \) is the cell state from previous time step
/// - \( i_t \) is the input gate
/// - \( f_t \) is the forget gate
/// - \( o_t \) is the output gate
/// - \( g_t \) is the candidate cell state
/// - \( c_t \) is the cell state at time step t
/// - \( h_t \) is the hidden state at time step t
/// - \( \sigma \) is the sigmoid function
/// - \( \odot \) denotes element-wise multiplication
///
/// # Notes
/// * The `LstmCell` is now only created by the [`LstmCellBuilder`].
/// * The forward pass expects a tuple of (input, (hidden, cell)) tensors and returns
///   a tuple of (new_hidden, new_cell).
///
/// # Fields
/// * `weight_ih` - The weight tensor for input to hidden transformation with shape [4*hidden_size, input_size].
/// * `weight_hh` - The weight tensor for hidden to hidden transformation with shape [4*hidden_size, hidden_size].
/// * `bias_ih` - The bias tensor for input to hidden transformation with shape \[4*hidden_size\] (optional).
/// * `bias_hh` - The bias tensor for hidden to hidden transformation with shape \[4*hidden_size\] (optional).
/// * `input_size` - The number of input features.
/// * `hidden_size` - The number of hidden features.
/// * `id` - The unique ID of the LSTM cell.
///
/// # Examples
/// ```no_run
/// use nove::model::nn::LstmCellBuilder;
/// use nove::tensor::{Device, DType};
///
/// let lstm_cell = LstmCellBuilder::new(10, 20)
///     .bias_enabled(true)
///     .device(Device::cpu())
///     .dtype(DType::F32)
///     .grad_enabled(true)
///     .build();
/// ```
///
/// # See Also
/// * [`LstmCellBuilder`] - The builder for the LstmCell layer.
#[derive(Debug, Clone)]
pub struct LstmCell {
    weight_ih: Tensor,
    weight_hh: Tensor,
    bias_ih: Option<Tensor>,
    bias_hh: Option<Tensor>,
    input_size: usize,
    hidden_size: usize,
    id: usize,
}

impl LstmCell {
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

    /// Get the unique ID of the LSTM cell.
    ///
    /// # Returns
    /// * `usize` - The unique ID.
    pub fn id(&self) -> usize {
        self.id
    }
}

impl Display for LstmCell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bias_enabled = self.bias_ih.is_some(); // bias_ih and bias_hh are both enabled or disabled together
        write!(
            f,
            "lstm_cell.{}(input_size={}, hidden_size={}, bias_enabled={})",
            self.id, self.input_size, self.hidden_size, bias_enabled,
        )
    }
}

/// The builder for the LstmCell layer.
///
/// # Notes
/// * The `LstmCellBuilder` provides [`LstmCellBuilder::new()`] method to create a builder with required parameters.
/// * The weight tensors are initialized with uniform distribution `U(-sqrt(k), sqrt(k))` where `k = 1 / hidden_size`,
///   following PyTorch's LstmCell initialization.
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
/// use nove::model::nn::LstmCellBuilder;
/// use nove::tensor::{Device, DType};
///
/// let lstm_cell = LstmCellBuilder::new(10, 20)
///     .bias_enabled(true)
///     .device(Device::cpu())
///     .dtype(DType::F32)
///     .grad_enabled(true)
///     .build();
/// ```
pub struct LstmCellBuilder {
    input_size: usize,
    hidden_size: usize,
    bias_enabled: bool,
    device: Device,
    dtype: DType,
    grad_enabled: bool,
}

impl LstmCellBuilder {
    /// Create a new LstmCellBuilder with required input_size and hidden_size.
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
    /// use nove::model::nn::LstmCellBuilder;
    /// let mut builder = LstmCellBuilder::new(10, 20);
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
    /// use nove::model::nn::LstmCellBuilder;
    /// let mut builder = LstmCellBuilder::new(10, 20);
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
    /// use nove::model::nn::LstmCellBuilder;
    /// let mut builder = LstmCellBuilder::new(10, 20);
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
    /// use nove::model::nn::LstmCellBuilder;
    /// use nove::tensor::Device;
    /// let mut builder = LstmCellBuilder::new(10, 20);
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
    /// use nove::model::nn::LstmCellBuilder;
    /// use nove::tensor::DType;
    /// let mut builder = LstmCellBuilder::new(10, 20);
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
    /// use nove::model::nn::LstmCellBuilder;
    /// let mut builder = LstmCellBuilder::new(10, 20);
    /// builder.grad_enabled(false);
    /// ```
    pub fn grad_enabled(&mut self, grad_enabled: bool) -> &mut Self {
        self.grad_enabled = grad_enabled;
        self
    }

    /// Build the LSTM cell layer.
    ///
    /// # Returns
    /// * `Ok(LstmCell)` - The built LSTM cell layer.
    /// * `Err(ModelError)` - The error when building the LSTM cell layer.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::nn::LstmCellBuilder;
    /// let mut builder = LstmCellBuilder::new(10, 20);
    /// let lstm_cell = builder.build().unwrap();
    /// ```
    pub fn build(&self) -> Result<LstmCell, ModelError> {
        if self.input_size == 0 {
            return Err(ModelError::InvalidArgument(
                "input_size in LstmCellBuilder must be greater than 0".to_string(),
            ));
        }

        if self.hidden_size == 0 {
            return Err(ModelError::InvalidArgument(
                "hidden_size in LstmCellBuilder must be greater than 0".to_string(),
            ));
        }

        let id = ID.fetch_add(1, Ordering::Relaxed);

        // PyTorch LstmCell weight initialization: uniform distribution U(-sqrt(k), sqrt(k)) where k = 1 / hidden_size
        let k = 1.0 / self.hidden_size as f64;
        let bound = k.sqrt();

        let weight_ih = Tensor::rand(
            -bound,
            bound,
            &Shape::from_dims(&[4 * self.hidden_size, self.input_size]),
            &self.device,
            self.grad_enabled,
        )?
        .to_dtype(&self.dtype)?
        .require_name(&format!("lstm_cell.{}.weight_ih", id))?;

        let weight_hh = Tensor::rand(
            -bound,
            bound,
            &Shape::from_dims(&[4 * self.hidden_size, self.hidden_size]),
            &self.device,
            self.grad_enabled,
        )?
        .to_dtype(&self.dtype)?
        .require_name(&format!("lstm_cell.{}.weight_hh", id))?;

        let bias_ih = if self.bias_enabled {
            let bias = Tensor::zeros(
                &Shape::from_dims(&[4 * self.hidden_size]),
                &self.dtype,
                &self.device,
                self.grad_enabled,
            )?
            .require_name(&format!("lstm_cell.{}.bias_ih", id))?;
            Some(bias)
        } else {
            None
        };

        let bias_hh = if self.bias_enabled {
            let bias = Tensor::zeros(
                &Shape::from_dims(&[4 * self.hidden_size]),
                &self.dtype,
                &self.device,
                self.grad_enabled,
            )?
            .require_name(&format!("lstm_cell.{}.bias_hh", id))?;
            Some(bias)
        } else {
            None
        };

        Ok(LstmCell {
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

impl Model for LstmCell {
    type Input = (Tensor, (Tensor, Tensor));
    type Output = (Tensor, Tensor);

    /// Apply the LSTM cell to the input tensor, hidden state, and cell state.
    ///
    /// # Arguments
    /// * `input` - A tuple containing (input_tensor, (hidden_state, cell_state)).
    ///   - input_tensor: Tensor with shape [batch_size, input_size]
    ///   - hidden_state: Tensor with shape [batch_size, hidden_size]
    ///   - cell_state: Tensor with shape [batch_size, hidden_size]
    ///
    /// # Returns
    /// * `Ok((Tensor, Tensor))` - The new hidden state and cell state, both with shape [batch_size, hidden_size].
    /// * `Err(ModelError)` - The error when applying the LSTM cell.
    fn forward(&mut self, input: Self::Input) -> Result<Self::Output, ModelError> {
        let (input_tensor, (hidden_state, cell_state)) = input;

        let input_shape = input_tensor.shape()?;
        let hidden_shape = hidden_state.shape()?;
        let cell_shape = cell_state.shape()?;

        if input_shape.dims().len() != 2 {
            return Err(ModelError::InvalidArgument(format!(
                "LstmCell expects input tensor with 2 dimensions [batch_size, input_size], got shape {:?}",
                input_shape.dims()
            )));
        }

        if hidden_shape.dims().len() != 2 {
            return Err(ModelError::InvalidArgument(format!(
                "LstmCell expects hidden state tensor with 2 dimensions [batch_size, hidden_size], got shape {:?}",
                hidden_shape.dims()
            )));
        }

        if cell_shape.dims().len() != 2 {
            return Err(ModelError::InvalidArgument(format!(
                "LstmCell expects cell state tensor with 2 dimensions [batch_size, hidden_size], got shape {:?}",
                cell_shape.dims()
            )));
        }

        let batch_size = input_shape.dims()[0];
        let input_size_from_tensor = input_shape.dims()[1];
        let hidden_size_from_tensor = hidden_shape.dims()[1];
        let cell_size_from_tensor = cell_shape.dims()[1];

        if input_size_from_tensor != self.input_size {
            return Err(ModelError::InvalidArgument(format!(
                "LstmCell expects input size {}, got {}",
                self.input_size, input_size_from_tensor
            )));
        }

        if hidden_size_from_tensor != self.hidden_size {
            return Err(ModelError::InvalidArgument(format!(
                "LstmCell expects hidden size {}, got {}",
                self.hidden_size, hidden_size_from_tensor
            )));
        }

        if cell_size_from_tensor != self.hidden_size {
            return Err(ModelError::InvalidArgument(format!(
                "LstmCell expects cell size {}, got {}",
                self.hidden_size, cell_size_from_tensor
            )));
        }

        if hidden_shape.dims()[0] != batch_size {
            return Err(ModelError::InvalidArgument(format!(
                "LstmCell expects batch size {} for hidden state, got {}",
                batch_size,
                hidden_shape.dims()[0]
            )));
        }

        if cell_shape.dims()[0] != batch_size {
            return Err(ModelError::InvalidArgument(format!(
                "LstmCell expects batch size {} for cell state, got {}",
                batch_size,
                cell_shape.dims()[0]
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

        // Split gates into four parts: input, forget, output, candidate cell
        // gates shape: [batch_size, 4 * hidden_size]
        let chunk_size = self.hidden_size;
        let (input_gate, forget_gate, output_gate, candidate_cell) = (
            gates.narrow(1, 0 * chunk_size, chunk_size)?,
            gates.narrow(1, 1 * chunk_size, chunk_size)?,
            gates.narrow(1, 2 * chunk_size, chunk_size)?,
            gates.narrow(1, 3 * chunk_size, chunk_size)?,
        );

        // Apply activation functions
        use crate::nn::Activation;
        let input_gate = Activation::sigmoid().forward(input_gate)?;
        let forget_gate = Activation::sigmoid().forward(forget_gate)?;
        let output_gate = Activation::sigmoid().forward(output_gate)?;
        let candidate_cell = Activation::tanh().forward(candidate_cell)?;

        // Update cell state: cell_new = forget_gate * cell + input_gate * candidate_cell
        let cell_new = forget_gate
            .mul(&cell_state)?
            .add(&input_gate.mul(&candidate_cell)?)?;

        // Update hidden state: hidden_new = output_gate * tanh(cell_new)
        let hidden_new = output_gate.mul(&Activation::tanh().forward(cell_new.copy())?)?;

        Ok((hidden_new, cell_new))
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
