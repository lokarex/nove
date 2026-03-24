use std::{
    collections::HashMap,
    fmt::Display,
    sync::atomic::{AtomicUsize, Ordering},
};

use nove_tensor::{DType, Device, Tensor};

use crate::layer::{Activation, Dropout, RnnCell, RnnCellBuilder};
use crate::{Model, ModelError};

static ID: AtomicUsize = AtomicUsize::new(0);

/// Recurrent Neural Network layer.
///
/// <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" id="MathJax-script" async></script>
///
/// The Rnn layer is a multi-layer recurrent neural network that applies a      
/// transformation to the input and previous hidden state across multiple layers.
///
/// For each time step and each layer, the RNN computes the following:
///
/// $$ h_t' = \text{activation}(W_{ih} x_t + b_{ih} + W_{hh} h_{t-1} + b_{hh}) $$
///
/// Where:
/// - \( x_t \) is the input tensor at time step t
/// - \( h_{t-1} \) is the hidden state from previous time step
/// - \( W_{ih} \) is the weight matrix from input to hidden
/// - \( b_{ih} \) is the bias vector for input to hidden
/// - \( W_{hh} \) is the weight matrix from hidden to hidden
/// - \( b_{hh} \) is the bias vector for hidden to hidden
/// - \( \text{activation} \) is the activation function (tanh or ReLU)
///
/// # Notes
/// * The `Rnn` is now only created by the [`RnnBuilder`].
/// * The forward pass expects an input tensor and returns the output tensor and hidden states.
/// * The Rnn layer can be bidirectional, processing the input sequence in both forward and backward directions.
/// * Dropout can be applied between RNN layers (except the last layer) to prevent overfitting.
///
/// # Fields
/// * `cells` - The RNN cells for each layer and direction (forward/backward).  
/// * `dropout` - The dropout layer applied between RNN layers (optional).      
/// * `input_size` - The number of input features.
/// * `hidden_size` - The number of hidden features.
/// * `num_layers` - The number of recurrent layers.
/// * `nonlinearity` - The activation function to use.
/// * `bias` - Whether bias terms are enabled.
/// * `batch_first` - Whether the input tensor has batch dimension first (batch, seq, feature) or last (seq, batch, feature).
/// * `dropout_rate` - The dropout probability between RNN layers (except the last layer).
/// * `bidirectional` - Whether the RNN is bidirectional.
/// * `num_directions` - The number of directions (1 for unidirectional, 2 for bidirectional).
/// * `id` - The unique ID of the RNN layer.
///
/// # Examples
/// ```no_run
/// use nove::model::layer::RnnBuilder;
/// use nove::model::layer::Activation;
/// use nove::tensor::{Device, DType};
///
/// let rnn = RnnBuilder::new(10, 20)
///     .num_layers(2)
///     .nonlinearity(Activation::tanh())
///     .bias(true)
///     .batch_first(true)
///     .dropout(0.5)
///     .bidirectional(false)
///     .device(Device::cpu())
///     .dtype(DType::F32)
///     .grad_enabled(true)
///     .build();
/// ```
///
/// # See Also
/// * [`RnnBuilder`] - The builder for the Rnn layer.
/// * [`RnnCell`] - The basic RNN cell used by the Rnn layer.
#[derive(Debug, Clone)]
pub struct Rnn {
    cells: Vec<Vec<RnnCell>>,
    dropout: Option<Dropout>,
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    nonlinearity: Activation,
    bias: bool,
    batch_first: bool,
    dropout_rate: f64,
    bidirectional: bool,
    num_directions: usize,
    id: usize,
}

impl Rnn {
    /// Get the input size.
    ///
    /// # Returns
    /// * `usize` - The number of input features.
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Get the hidden size.
    ///
    /// # Returns
    /// * `usize` - The number of hidden features.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Get the number of layers.
    ///
    /// # Returns
    /// * `usize` - The number of recurrent layers.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Get the activation function.
    ///
    /// # Returns
    /// * `Activation` - The activation function.
    pub fn nonlinearity(&self) -> Activation {
        self.nonlinearity.clone()
    }

    /// Get whether bias is enabled.
    ///
    /// # Returns
    /// * `bool` - Whether bias terms are enabled.
    pub fn bias(&self) -> bool {
        self.bias
    }

    /// Get whether batch dimension is first.
    ///
    /// # Returns
    /// * `bool` - Whether the input tensor has batch dimension first.
    pub fn batch_first(&self) -> bool {
        self.batch_first
    }

    /// Get the dropout rate.
    ///
    /// # Returns
    /// * `f64` - The dropout probability between RNN layers.
    pub fn dropout_rate(&self) -> f64 {
        self.dropout_rate
    }

    /// Get whether the RNN is bidirectional.
    ///
    /// # Returns
    /// * `bool` - Whether the RNN is bidirectional.
    pub fn bidirectional(&self) -> bool {
        self.bidirectional
    }

    /// Get the number of directions.
    ///
    /// # Returns
    /// * `usize` - The number of directions (1 for unidirectional, 2 for bidirectional).
    pub fn num_directions(&self) -> usize {
        self.num_directions
    }

    /// Get the unique ID of the RNN layer.
    ///
    /// # Returns
    /// * `usize` - The unique ID.
    pub fn id(&self) -> usize {
        self.id
    }
}

/// The builder for the Rnn layer.
///
/// # Notes
/// * The `RnnBuilder` provides [`RnnBuilder::new()`] method to create a builder with required parameters.
/// * The RNN cells are initialized with uniform distribution `U(-sqrt(k), sqrt(k))` where `k = 1 / hidden_size`,
///   following PyTorch's RNN initialization.
///   The bias tensors are initialized with zeros (if enabled).
///
/// # Required Arguments
/// * `input_size` - The number of input features.
/// * `hidden_size` - The number of hidden features.
///
/// # Optional Arguments
/// * `num_layers` - The number of recurrent layers. Default is `1`.
/// * `nonlinearity` - The activation function to use. Default is `Activation::Tanh`.
/// * `bias` - Whether to enable the bias terms. Default is `true`.
/// * `batch_first` - Whether the input tensor has batch dimension first. Default is `false`.
/// * `dropout` - The dropout probability between RNN layers (except the last layer). Default is `0.0`.
/// * `bidirectional` - Whether the RNN is bidirectional. Default is `false`.   
/// * `device` - The device to use for the layer. Default is `Device::cpu()`.   
/// * `dtype` - The data type to use for the layer. Default is `DType::F32`.    
/// * `grad_enabled` - Whether to enable the gradient computation. Default is `true`.
///
/// # Fields
/// * `input_size` - The number of input features.
/// * `hidden_size` - The number of hidden features.
/// * `num_layers` - The number of recurrent layers.
/// * `nonlinearity` - The activation function to use.
/// * `bias` - Whether to enable the bias terms.
/// * `batch_first` - Whether the input tensor has batch dimension first.       
/// * `dropout` - The dropout probability between RNN layers (except the last layer).
/// * `bidirectional` - Whether the RNN is bidirectional.
/// * `device` - The device to use for the layer.
/// * `dtype` - The data type to use for the layer.
/// * `grad_enabled` - Whether to enable the gradient computation.
///
/// # Examples
/// ```no_run
/// use nove::model::layer::RnnBuilder;
/// use nove::model::layer::Activation;
/// use nove::tensor::{Device, DType};
///
/// let rnn = RnnBuilder::new(10, 20)
///     .num_layers(2)
///     .nonlinearity(Activation::tanh())
///     .bias(true)
///     .batch_first(true)
///     .dropout(0.5)
///     .bidirectional(false)
///     .device(Device::cpu())
///     .dtype(DType::F32)
///     .grad_enabled(true)
///     .build();
/// ```
pub struct RnnBuilder {
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    nonlinearity: Activation,
    bias: bool,
    batch_first: bool,
    dropout: f64,
    bidirectional: bool,
    device: Device,
    dtype: DType,
    grad_enabled: bool,
}

impl RnnBuilder {
    /// Create a new RnnBuilder with required input_size and hidden_size.
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
            num_layers: 1,
            nonlinearity: Activation::tanh(),
            bias: true,
            batch_first: false,
            dropout: 0.0,
            bidirectional: false,
            device: Device::cpu(),
            dtype: DType::F32,
            grad_enabled: true,
        }
    }

    /// Configure the number of recurrent layers.
    ///
    /// # Arguments
    /// * `num_layers` - The number of recurrent layers.
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured number of layers.
    pub fn num_layers(&mut self, num_layers: usize) -> &mut Self {
        self.num_layers = num_layers;
        self
    }

    /// Configure the activation function.
    ///
    /// # Arguments
    /// * `nonlinearity` - The activation function (e.g., `Activation::Tanh`, `Activation::ReLU`)
    ///
    /// # Returns
    /// * `&mut Self` - The builder with configured activation
    pub fn nonlinearity(&mut self, nonlinearity: Activation) -> &mut Self {
        self.nonlinearity = nonlinearity;
        self
    }

    /// Configure whether to enable the bias terms.
    ///
    /// # Arguments
    /// * `bias` - Whether to enable the bias terms.
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured bias terms.
    pub fn bias(&mut self, bias: bool) -> &mut Self {
        self.bias = bias;
        self
    }

    /// Configure whether the input tensor has batch dimension first.
    ///
    /// # Arguments
    /// * `batch_first` - Whether the input tensor has batch dimension first.
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured batch dimension ordering.
    pub fn batch_first(&mut self, batch_first: bool) -> &mut Self {
        self.batch_first = batch_first;
        self
    }

    /// Configure the dropout probability between RNN layers (except the last layer).
    ///
    /// # Arguments
    /// * `dropout` - The dropout probability (must be in range [0.0, 1.0]).
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured dropout.
    pub fn dropout(&mut self, dropout: f64) -> &mut Self {
        self.dropout = dropout;
        self
    }

    /// Configure whether the RNN is bidirectional.
    ///
    /// # Arguments
    /// * `bidirectional` - Whether the RNN is bidirectional.
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured directionality.
    pub fn bidirectional(&mut self, bidirectional: bool) -> &mut Self {
        self.bidirectional = bidirectional;
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

    /// Build the RNN layer.
    ///
    /// # Returns
    /// * `Ok(Rnn)` - The built RNN layer.
    /// * `Err(ModelError)` - The error when building the RNN layer.
    pub fn build(&self) -> Result<Rnn, ModelError> {
        if self.input_size == 0 {
            return Err(ModelError::InvalidArgument(
                "input_size in RnnBuilder must be greater than 0".to_string(),
            ));
        }

        if self.hidden_size == 0 {
            return Err(ModelError::InvalidArgument(
                "hidden_size in RnnBuilder must be greater than 0".to_string(),
            ));
        }

        if self.num_layers == 0 {
            return Err(ModelError::InvalidArgument(
                "num_layers in RnnBuilder must be greater than 0".to_string(),
            ));
        }

        if !(0.0..=1.0).contains(&self.dropout) {
            return Err(ModelError::InvalidArgument(
                "dropout in RnnBuilder must be in range [0.0, 1.0]".to_string(),
            ));
        }

        let num_directions = if self.bidirectional { 2 } else { 1 };
        let mut cells = Vec::with_capacity(self.num_layers);
        let dropout_layer = if self.dropout > 0.0 {
            Some(Dropout::new(self.dropout as f32)?)
        } else {
            None
        };

        for l in 0..self.num_layers {
            let mut layer_cells = Vec::with_capacity(num_directions);
            for _d in 0..num_directions {
                let input_size_for_layer = if l == 0 {
                    self.input_size
                } else {
                    self.hidden_size * num_directions
                };

                let mut cell_builder = RnnCellBuilder::new(input_size_for_layer, self.hidden_size);
                cell_builder
                    .activation(self.nonlinearity.clone())
                    .bias_enabled(self.bias)
                    .device(self.device.clone())
                    .dtype(self.dtype.clone())
                    .grad_enabled(self.grad_enabled);

                let cell = cell_builder.build()?;
                layer_cells.push(cell);
            }
            cells.push(layer_cells);
        }

        let id = ID.fetch_add(1, Ordering::Relaxed);

        Ok(Rnn {
            cells,
            dropout: dropout_layer,
            input_size: self.input_size,
            hidden_size: self.hidden_size,
            num_layers: self.num_layers,
            nonlinearity: self.nonlinearity.clone(),
            bias: self.bias,
            batch_first: self.batch_first,
            dropout_rate: self.dropout,
            bidirectional: self.bidirectional,
            num_directions,
            id,
        })
    }
}

impl Model for Rnn {
    type Input = Tensor;
    type Output = (Tensor, Tensor);

    /// Apply the RNN layer to the input tensor.
    ///
    /// # Arguments
    /// * `input` - The input tensor with shape [batch_size, seq_len, input_size] if `batch_first` is true,
    ///   or [seq_len, batch_size, input_size] if `batch_first` is false.
    ///
    /// # Returns
    /// * `Ok((Tensor, Tensor))` - A tuple containing:
    ///   - The output tensor with shape [batch_size, seq_len, hidden_size * num_directions] if `batch_first` is true,
    ///     or [seq_len, batch_size, hidden_size * num_directions] if `batch_first` is false.
    ///   - The hidden state tensor with shape [num_layers * num_directions, batch_size, hidden_size].
    /// * `Err(ModelError)` - The error when applying the RNN layer to the input tensor.
    fn forward(&mut self, input: Self::Input) -> Result<Self::Output, ModelError> {
        let input_shape = input.shape()?;
        let dims = input_shape.dims();

        if dims.len() != 3 {
            return Err(ModelError::InvalidArgument(format!(
                "Rnn expects 3D input tensor, got shape {:?}",
                dims
            )));
        }

        let mut processed_input = input;

        // Handle batch_first parameter
        if self.batch_first {
            // Transpose from [batch_size, seq_len, input_size] to [seq_len, batch_size, input_size]
            processed_input = processed_input.transpose(0, 1)?;
        }

        let transposed_shape = processed_input.shape()?;
        let transposed_dims = transposed_shape.dims();

        let seq_len = transposed_dims[0];
        let batch_size = transposed_dims[1];
        let input_size = transposed_dims[2];

        // Validate input size
        if input_size != self.input_size {
            return Err(ModelError::InvalidArgument(format!(
                "Rnn expects input size {}, got {}",
                self.input_size, input_size
            )));
        }

        // Get dtype and device from processed_input before moving it
        let dtype = processed_input.dtype()?;
        let device = processed_input.device()?;

        // Initialize hidden states for each layer and direction
        let mut layer_input = processed_input;
        let mut final_hidden_states = Vec::with_capacity(self.num_layers * self.num_directions);

        for layer_idx in 0..self.num_layers {
            // Initialize hidden states for this layer for each direction
            let mut layer_hidden_states = Vec::with_capacity(self.num_directions);
            let mut layer_outputs = Vec::with_capacity(self.num_directions);

            for direction_idx in 0..self.num_directions {
                // Initialize zero hidden state for this direction
                let hidden_state = Tensor::zeros(
                    &nove_tensor::Shape::from_dims(&[batch_size, self.hidden_size]),
                    &dtype,
                    &device,
                    false,
                )?;
                layer_hidden_states.push(hidden_state);

                // Process sequence for this direction
                let cell = &mut self.cells[layer_idx][direction_idx];
                let mut direction_outputs = Vec::with_capacity(seq_len);

                // Determine sequence order based on direction
                let sequence_indices: Vec<usize> = if direction_idx == 0 {
                    // Forward direction
                    (0..seq_len).collect()
                } else {
                    // Backward direction
                    (0..seq_len).rev().collect()
                };

                let mut current_hidden_state = layer_hidden_states[direction_idx].copy();

                for t in sequence_indices {
                    // Get input for this time step
                    let input_slice = layer_input.narrow(0, t, 1)?.squeeze(Some(0))?;

                    // Forward through RNN cell
                    current_hidden_state = cell.forward((input_slice, current_hidden_state))?;
                    direction_outputs.push(current_hidden_state.unsqueeze(0)?);
                }

                // Concatenate outputs along time dimension
                let direction_output = Tensor::cat(&direction_outputs, 0)?;
                layer_outputs.push(direction_output);

                // Store final hidden state for this direction
                final_hidden_states.push(current_hidden_state);
            }

            // Concatenate direction outputs along feature dimension
            let layer_output = if self.num_directions == 1 {
                layer_outputs[0].copy()
            } else {
                // Concatenate along hidden_size dimension (dimension 2)
                let mut concatenated = Vec::new();
                for output in layer_outputs {
                    concatenated.push(output);
                }
                Tensor::cat(&concatenated, 2)?
            };

            // Apply dropout between layers (except after last layer)
            let next_layer_input = if layer_idx < self.num_layers - 1 && self.dropout_rate > 0.0 {
                if let Some(dropout) = &mut self.dropout {
                    dropout.forward((layer_output, true))?
                } else {
                    layer_output
                }
            } else {
                layer_output
            };

            layer_input = next_layer_input;
        }

        // If batch_first was true, transpose output back to [batch_size, seq_len, hidden_size * num_directions]
        let output = if self.batch_first {
            layer_input.transpose(0, 1)?
        } else {
            layer_input
        };

        // Collect final hidden states into tensor of shape [num_layers * num_directions, batch_size, hidden_size]
        let mut hidden_states_tensors = Vec::new();
        for hidden_state in final_hidden_states {
            hidden_states_tensors.push(hidden_state.unsqueeze(0)?);
        }
        let hidden_states = Tensor::cat(&hidden_states_tensors, 0)?;

        Ok((output, hidden_states))
    }

    fn require_grad(&mut self, grad_enabled: bool) -> Result<(), ModelError> {
        for layer in &mut self.cells {
            for cell in layer {
                cell.require_grad(grad_enabled)?;
            }
        }
        if let Some(dropout) = &mut self.dropout {
            dropout.require_grad(grad_enabled)?;
        }
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<(), ModelError> {
        for layer in &mut self.cells {
            for cell in layer {
                cell.to_device(device)?;
            }
        }
        if let Some(dropout) = &mut self.dropout {
            dropout.to_device(device)?;
        }
        Ok(())
    }

    fn to_dtype(&mut self, dtype: &DType) -> Result<(), ModelError> {
        for layer in &mut self.cells {
            for cell in layer {
                cell.to_dtype(dtype)?;
            }
        }
        if let Some(dropout) = &mut self.dropout {
            dropout.to_dtype(dtype)?;
        }
        Ok(())
    }

    fn parameters(&self) -> Result<Vec<Tensor>, ModelError> {
        let mut params = Vec::new();
        for layer in &self.cells {
            for cell in layer {
                params.extend(cell.parameters()?);
            }
        }
        if let Some(dropout) = &self.dropout {
            params.extend(dropout.parameters()?);
        }
        Ok(params)
    }

    fn named_parameters(&self) -> Result<HashMap<String, Tensor>, ModelError> {
        let mut params = HashMap::new();
        for (l, layer) in self.cells.iter().enumerate() {
            for (d, cell) in layer.iter().enumerate() {
                let prefix = format!("rnn.{}.layer{}.dir{}", self.id, l, d);
                let cell_params = cell.named_parameters()?;
                for (name, tensor) in cell_params {
                    params.insert(format!("{}.{}", prefix, name), tensor);
                }
            }
        }
        if let Some(dropout) = &self.dropout {
            let dropout_params = dropout.named_parameters()?;
            params.extend(dropout_params);
        }
        Ok(params)
    }
}

impl Display for Rnn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "rnn.{}(input_size={}, hidden_size={}, num_layers={}, nonlinearity={}, bias={}, batch_first={}, dropout_rate={}, bidirectional={})",
            self.id,
            self.input_size,
            self.hidden_size,
            self.num_layers,
            self.nonlinearity,
            self.bias,
            self.batch_first,
            self.dropout_rate,
            self.bidirectional,
        )
    }
}
