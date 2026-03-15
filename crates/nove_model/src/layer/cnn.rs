use std::{
    collections::HashMap,
    fmt::{Display, Write},
    sync::atomic::{AtomicUsize, Ordering},
};

use nove_tensor::{DType, Device, Tensor};

use crate::{Model, ModelError};

use super::{
    AvgPool2d, BatchNorm2d, BatchNorm2dBuilder, Conv2d, Conv2dBuilder, GELU, Linear, LinearBuilder,
    MaxPool2d, ReLU, SiLU, Sigmoid, Tanh,
};

static ID: AtomicUsize = AtomicUsize::new(0);

/// Layer enum representing different types of CNN layers.
///
/// # Notes
/// * The `CNNLayer` enum wraps different layer types and provides a unified interface
///   through the `Model` trait implementation.
///
/// # Variants
/// * `Conv2d` - 2D convolution layer.
/// * `ReLU` - Rectified Linear Unit activation layer.
/// * `GELU` - Gaussian Error Linear Unit activation layer.
/// * `SiLU` - Sigmoid Linear Unit activation layer.
/// * `Tanh` - Hyperbolic tangent activation layer.
/// * `Sigmoid` - Sigmoid activation layer.
/// * `MaxPool2d` - 2D max pooling layer.
/// * `AvgPool2d` - 2D average pooling layer.
/// * `BatchNorm2d` - 2D batch normalization layer.
/// * `Linear` - Fully connected linear layer.
#[derive(Debug, Clone)]
pub enum CNNLayer {
    Conv2d(Conv2d),
    ReLU(ReLU),
    GELU(GELU),
    SiLU(SiLU),
    Tanh(Tanh),
    Sigmoid(Sigmoid),
    MaxPool2d(MaxPool2d),
    AvgPool2d(AvgPool2d),
    BatchNorm2d(BatchNorm2d),
    Linear(Linear),
}

impl Model for CNNLayer {
    type Input = (Tensor, bool);
    type Output = Tensor;

    /// Apply the layer to the input tensor.
    ///
    /// # Arguments
    /// * `input: (Tensor, bool)` - A tuple containing the input tensor and a boolean flag
    ///   indicating whether the layer is in training mode. This is used by BatchNorm2d.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The output tensor if successful.
    /// * `Err(ModelError)` - The error when applying the layer to the input tensor.
    fn forward(&mut self, input: Self::Input) -> Result<Self::Output, ModelError> {
        match self {
            CNNLayer::Conv2d(layer) => layer.forward(input.0),
            CNNLayer::ReLU(layer) => layer.forward(input.0),
            CNNLayer::GELU(layer) => layer.forward(input.0),
            CNNLayer::SiLU(layer) => layer.forward(input.0),
            CNNLayer::Tanh(layer) => layer.forward(input.0),
            CNNLayer::Sigmoid(layer) => layer.forward(input.0),
            CNNLayer::MaxPool2d(layer) => layer.forward(input.0),
            CNNLayer::AvgPool2d(layer) => layer.forward(input.0),
            CNNLayer::BatchNorm2d(layer) => layer.forward(input),
            CNNLayer::Linear(layer) => layer.forward(input.0),
        }
    }

    fn require_grad(&mut self, grad_enabled: bool) -> Result<(), ModelError> {
        match self {
            CNNLayer::Conv2d(layer) => layer.require_grad(grad_enabled),
            CNNLayer::ReLU(layer) => layer.require_grad(grad_enabled),
            CNNLayer::GELU(layer) => layer.require_grad(grad_enabled),
            CNNLayer::SiLU(layer) => layer.require_grad(grad_enabled),
            CNNLayer::Tanh(layer) => layer.require_grad(grad_enabled),
            CNNLayer::Sigmoid(layer) => layer.require_grad(grad_enabled),
            CNNLayer::MaxPool2d(layer) => layer.require_grad(grad_enabled),
            CNNLayer::AvgPool2d(layer) => layer.require_grad(grad_enabled),
            CNNLayer::BatchNorm2d(layer) => layer.require_grad(grad_enabled),
            CNNLayer::Linear(layer) => layer.require_grad(grad_enabled),
        }
    }

    fn to_device(&mut self, device: &Device) -> Result<(), ModelError> {
        match self {
            CNNLayer::Conv2d(layer) => layer.to_device(device),
            CNNLayer::ReLU(layer) => layer.to_device(device),
            CNNLayer::GELU(layer) => layer.to_device(device),
            CNNLayer::SiLU(layer) => layer.to_device(device),
            CNNLayer::Tanh(layer) => layer.to_device(device),
            CNNLayer::Sigmoid(layer) => layer.to_device(device),
            CNNLayer::MaxPool2d(layer) => layer.to_device(device),
            CNNLayer::AvgPool2d(layer) => layer.to_device(device),
            CNNLayer::BatchNorm2d(layer) => layer.to_device(device),
            CNNLayer::Linear(layer) => layer.to_device(device),
        }
    }

    fn to_dtype(&mut self, dtype: &DType) -> Result<(), ModelError> {
        match self {
            CNNLayer::Conv2d(layer) => layer.to_dtype(dtype),
            CNNLayer::ReLU(layer) => layer.to_dtype(dtype),
            CNNLayer::GELU(layer) => layer.to_dtype(dtype),
            CNNLayer::SiLU(layer) => layer.to_dtype(dtype),
            CNNLayer::Tanh(layer) => layer.to_dtype(dtype),
            CNNLayer::Sigmoid(layer) => layer.to_dtype(dtype),
            CNNLayer::MaxPool2d(layer) => layer.to_dtype(dtype),
            CNNLayer::AvgPool2d(layer) => layer.to_dtype(dtype),
            CNNLayer::BatchNorm2d(layer) => layer.to_dtype(dtype),
            CNNLayer::Linear(layer) => layer.to_dtype(dtype),
        }
    }

    fn parameters(&self) -> Result<Vec<Tensor>, ModelError> {
        match self {
            CNNLayer::Conv2d(layer) => layer.parameters(),
            CNNLayer::ReLU(layer) => layer.parameters(),
            CNNLayer::GELU(layer) => layer.parameters(),
            CNNLayer::SiLU(layer) => layer.parameters(),
            CNNLayer::Tanh(layer) => layer.parameters(),
            CNNLayer::Sigmoid(layer) => layer.parameters(),
            CNNLayer::MaxPool2d(layer) => layer.parameters(),
            CNNLayer::AvgPool2d(layer) => layer.parameters(),
            CNNLayer::BatchNorm2d(layer) => layer.parameters(),
            CNNLayer::Linear(layer) => layer.parameters(),
        }
    }

    fn named_parameters(&self) -> Result<HashMap<String, Tensor>, ModelError> {
        match self {
            CNNLayer::Conv2d(layer) => layer.named_parameters(),
            CNNLayer::ReLU(layer) => layer.named_parameters(),
            CNNLayer::GELU(layer) => layer.named_parameters(),
            CNNLayer::SiLU(layer) => layer.named_parameters(),
            CNNLayer::Tanh(layer) => layer.named_parameters(),
            CNNLayer::Sigmoid(layer) => layer.named_parameters(),
            CNNLayer::MaxPool2d(layer) => layer.named_parameters(),
            CNNLayer::AvgPool2d(layer) => layer.named_parameters(),
            CNNLayer::BatchNorm2d(layer) => layer.named_parameters(),
            CNNLayer::Linear(layer) => layer.named_parameters(),
        }
    }
}

impl Display for CNNLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CNNLayer::Conv2d(layer) => write!(f, "{}", layer),
            CNNLayer::ReLU(layer) => write!(f, "{}", layer),
            CNNLayer::GELU(layer) => write!(f, "{}", layer),
            CNNLayer::SiLU(layer) => write!(f, "{}", layer),
            CNNLayer::Tanh(layer) => write!(f, "{}", layer),
            CNNLayer::Sigmoid(layer) => write!(f, "{}", layer),
            CNNLayer::MaxPool2d(layer) => write!(f, "{}", layer),
            CNNLayer::AvgPool2d(layer) => write!(f, "{}", layer),
            CNNLayer::BatchNorm2d(layer) => write!(f, "{}", layer),
            CNNLayer::Linear(layer) => write!(f, "{}", layer),
        }
    }
}

/// Convolutional Neural Network (CNN) model.
///
/// # Notes
/// * The `CNN` is a sequential model that chains multiple layers together.
/// * The `CNN` is now only created by the `CNNBuilder`.
///
/// # Fields
/// * `layers` - The sequence of layers in the network.
/// * `id` - The unique ID of the CNN model.
///
/// # Examples
/// ```
/// use nove::model::layer::{CNNBuilder, CNNConvBlock, CNNLinearBlock};
///
/// let mut cnn = CNNBuilder::default()
///     .conv_block(CNNConvBlock::new(1, 16, (3, 3), 1, 1).use_max_pool((2, 2), (2, 2)))
///     .conv_block(CNNConvBlock::new(16, 32, (3, 3), 1, 1).use_avg_pool((2, 2), (2, 2)))
///     .linear_block(CNNLinearBlock::new(800, 10))
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct CNN {
    layers: Vec<CNNLayer>,
    id: usize,
}

impl CNN {
    /// Get the layers of the CNN model.
    ///
    /// # Returns
    /// * `&Vec<CNNLayer>` - The layers of the CNN model.
    pub fn layers(&self) -> &Vec<CNNLayer> {
        &self.layers
    }

    /// Get a mutable reference to the layers of the CNN model.
    ///
    /// # Returns
    /// * `&mut Vec<CNNLayer>` - A mutable reference to the layers of the CNN model.
    pub fn layers_mut(&mut self) -> &mut Vec<CNNLayer> {
        &mut self.layers
    }
}

impl Model for CNN {
    type Input = (Tensor, bool);
    type Output = Tensor;

    /// Apply the CNN model to the input tensor.
    ///
    /// # Arguments
    /// * `input` - A tuple containing the input tensor and a boolean flag.
    ///   - `Tensor` - The input tensor.
    ///   - `bool` - Whether the model is in training mode (used by BatchNorm2d).
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The output tensor if successful.
    /// * `Err(ModelError)` - The error when applying the CNN model to the input tensor.
    ///
    /// # Notes
    /// * The `bool` flag indicating training mode is passed to BatchNorm2d layers.
    /// * Other layers (Conv2d, ReLU, MaxPool2d, Linear) ignore this flag.
    fn forward(&mut self, input: Self::Input) -> Result<Self::Output, ModelError> {
        let (mut input, training) = input;
        for layer in &mut self.layers {
            // Flatten the input tensor before Linear layer
            if let CNNLayer::Linear(_) = layer {
                let shape = input.shape()?;
                if shape.dims().len() == 4 {
                    let batch_size = shape.dims()[0];
                    let flattened_size = shape.dims()[1] * shape.dims()[2] * shape.dims()[3];
                    input =
                        input.reshape(&nove_tensor::Shape::from(&[batch_size, flattened_size]))?;
                }
            }
            input = layer.forward((input, training))?;
        }
        Ok(input)
    }

    fn require_grad(&mut self, grad_enabled: bool) -> Result<(), ModelError> {
        for layer in &mut self.layers {
            layer.require_grad(grad_enabled)?;
        }
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<(), ModelError> {
        for layer in &mut self.layers {
            layer.to_device(device)?;
        }
        Ok(())
    }

    fn to_dtype(&mut self, dtype: &DType) -> Result<(), ModelError> {
        for layer in &mut self.layers {
            layer.to_dtype(dtype)?;
        }
        Ok(())
    }

    fn parameters(&self) -> Result<Vec<Tensor>, ModelError> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters()?);
        }
        Ok(params)
    }

    fn named_parameters(&self) -> Result<HashMap<String, Tensor>, ModelError> {
        let mut params = HashMap::new();
        for (i, layer) in self.layers.iter().enumerate() {
            let prefix = format!("cnn{}.", i);
            for (name, tensor) in layer.named_parameters()? {
                params.insert(format!("{}{}", prefix, name), tensor);
            }
        }
        Ok(params)
    }
}

impl Display for CNN {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = String::new();
        writeln!(s, "cnn.{}(", self.id)?;
        for layer in &self.layers {
            writeln!(s, "  {},", layer)?;
        }
        write!(s, ")")?;
        write!(f, "{}", s)
    }
}

/// Convolutional block configuration for building CNN models.
///
/// # Notes
/// * The `CNNConvBlock` is used to configure a convolutional layer with optional
///   activation function, batch normalization, and pooling.
/// * The kernel size, stride, and padding must be specified when creating a block.
/// * The default configuration uses ReLU activation enabled, batch normalization disabled,
///   and pooling disabled.
/// * **Important**: Only one activation function can be enabled at a time.
///   When you call any `use_xxx()` method (e.g., `use_relu()`, `use_gelu()`, etc.),
///   all other activation function flags will be automatically set to `false`
///   to ensure mutual exclusion.
/// * **Important**: Only one pooling type can be enabled at a time.
///   When you call any `use_xxx_pool()` method (e.g., `use_max_pool()`, `use_avg_pool()`),
///   all other pooling flags will be automatically set to `false` to ensure mutual exclusion.
///   The pooling kernel size and stride must be specified when enabling pooling.
///
/// # Fields
/// * `in_channels` - The number of input channels.
/// * `out_channels` - The number of output channels.
/// * `kernel_size` - The size of the convolution kernel (height, width).
/// * `stride` - The stride of the convolution.
/// * `padding` - The padding size.
/// * `use_relu` - Whether ReLU activation is enabled after convolution.
/// * `use_gelu` - Whether GELU activation is enabled after convolution.
/// * `use_silu` - Whether SiLU activation is enabled after convolution.
/// * `use_tanh` - Whether Tanh activation is enabled after convolution.
/// * `use_sigmoid` - Whether Sigmoid activation is enabled after convolution.
/// * `use_batch_norm` - Whether batch normalization is enabled after convolution.
/// * `use_max_pool` - Whether max pooling is enabled after activation/BatchNorm.
/// * `use_avg_pool` - Whether average pooling is enabled after activation/BatchNorm.
/// * `pool_kernel_size` - The size of the pooling kernel (height, width).
/// * `pool_stride` - The stride of the pooling operation (height, width).
///
/// # Examples
/// ```
/// use nove::model::layer::CNNConvBlock;
///
/// let block = CNNConvBlock::new(1, 16, (3, 3), 1, 1)
///     .use_gelu()
///     .use_batch_norm()
///     .use_avg_pool((2, 2), (2, 2));
/// ```
#[derive(Debug, Clone)]
pub struct CNNConvBlock {
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize),
    stride: usize,
    padding: usize,
    use_relu: bool,
    use_gelu: bool,
    use_silu: bool,
    use_tanh: bool,
    use_sigmoid: bool,
    use_batch_norm: bool,
    use_max_pool: bool,
    use_avg_pool: bool,
    pool_kernel_size: (usize, usize),
    pool_stride: (usize, usize),
}

impl CNNConvBlock {
    /// Create a new convolutional block with specified settings.
    ///
    /// # Arguments
    /// * `in_channels` - The number of input channels.
    /// * `out_channels` - The number of output channels.
    /// * `kernel_size` - The size of the convolution kernel (height, width).
    /// * `stride` - The stride of the convolution.
    /// * `padding` - The padding size.
    ///
    /// # Returns
    /// * `CNNConvBlock` - The new convolutional block.
    ///
    /// # Examples
    /// ```
    /// use nove::model::layer::CNNConvBlock;
    /// let block = CNNConvBlock::new(1, 16, (3, 3), 1, 1);
    /// ```
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: usize,
        padding: usize,
    ) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            use_relu: true,
            use_gelu: false,
            use_silu: false,
            use_tanh: false,
            use_sigmoid: false,
            use_batch_norm: false,
            use_max_pool: false,
            use_avg_pool: false,
            pool_kernel_size: (2, 2),
            pool_stride: (2, 2),
        }
    }

    /// Use ReLU activation after convolution.
    ///
    /// # Notes
    /// * When this method is called, all other activation function flags
    ///   (GELU, SiLU, Tanh, Sigmoid) will be automatically set to `false` to ensure
    ///   mutual exclusion. Only one activation function can be active at a time.
    ///
    /// # Returns
    /// * `Self` - The convolutional block with ReLU activation enabled.
    ///
    /// # Examples
    /// ```
    /// use nove::model::layer::CNNConvBlock;
    /// let block = CNNConvBlock::new(1, 16, (3, 3), 1, 1).use_relu();
    /// ```
    pub fn use_relu(mut self) -> Self {
        self.use_gelu = false;
        self.use_silu = false;
        self.use_tanh = false;
        self.use_sigmoid = false;
        self.use_relu = true;
        self
    }

    /// Use GELU activation after convolution.
    ///
    /// # Notes
    /// * When this method is called, all other activation function flags
    ///   (ReLU, SiLU, Tanh, Sigmoid) will be automatically set to `false` to ensure
    ///   mutual exclusion. Only one activation function can be active at a time.
    ///
    /// # Returns
    /// * `Self` - The convolutional block with GELU activation enabled.
    ///
    /// # Examples
    /// ```
    /// use nove::model::layer::CNNConvBlock;
    /// let block = CNNConvBlock::new(1, 16, (3, 3), 1, 1).use_gelu();
    /// ```
    pub fn use_gelu(mut self) -> Self {
        self.use_relu = false;
        self.use_silu = false;
        self.use_tanh = false;
        self.use_sigmoid = false;
        self.use_gelu = true;
        self
    }

    /// Use SiLU activation after convolution.
    ///
    /// # Notes
    /// * When this method is called, all other activation function flags
    ///   (ReLU, GELU, Tanh, Sigmoid) will be automatically set to `false` to ensure
    ///   mutual exclusion. Only one activation function can be active at a time.
    ///
    /// # Returns
    /// * `Self` - The convolutional block with SiLU activation enabled.
    ///
    /// # Examples
    /// ```
    /// use nove::model::layer::CNNConvBlock;
    /// let block = CNNConvBlock::new(1, 16, (3, 3), 1, 1).use_silu();
    /// ```
    pub fn use_silu(mut self) -> Self {
        self.use_relu = false;
        self.use_gelu = false;
        self.use_tanh = false;
        self.use_sigmoid = false;
        self.use_silu = true;
        self
    }

    /// Use Tanh activation after convolution.
    ///
    /// # Notes
    /// * When this method is called, all other activation function flags
    ///   (ReLU, GELU, SiLU, Sigmoid) will be automatically set to `false` to ensure
    ///   mutual exclusion. Only one activation function can be active at a time.
    ///
    /// # Returns
    /// * `Self` - The convolutional block with Tanh activation enabled.
    ///
    /// # Examples
    /// ```
    /// use nove::model::layer::CNNConvBlock;
    /// let block = CNNConvBlock::new(1, 16, (3, 3), 1, 1).use_tanh();
    /// ```
    pub fn use_tanh(mut self) -> Self {
        self.use_relu = false;
        self.use_gelu = false;
        self.use_silu = false;
        self.use_sigmoid = false;
        self.use_tanh = true;
        self
    }

    /// Use Sigmoid activation after convolution.
    ///
    /// # Notes
    /// * When this method is called, all other activation function flags
    ///   (ReLU, GELU, SiLU, Tanh) will be automatically set to `false` to ensure
    ///   mutual exclusion. Only one activation function can be active at a time.
    ///
    /// # Returns
    /// * `Self` - The convolutional block with Sigmoid activation enabled.
    ///
    /// # Examples
    /// ```
    /// use nove::model::layer::CNNConvBlock;
    /// let block = CNNConvBlock::new(1, 16, (3, 3), 1, 1).use_sigmoid();
    /// ```
    pub fn use_sigmoid(mut self) -> Self {
        self.use_relu = false;
        self.use_gelu = false;
        self.use_silu = false;
        self.use_tanh = false;
        self.use_sigmoid = true;
        self
    }

    /// Use batch normalization after convolution.
    ///
    /// # Returns
    /// * `Self` - The convolutional block with batch normalization enabled.
    ///
    /// # Examples
    /// ```
    /// use nove::model::layer::CNNConvBlock;
    /// let block = CNNConvBlock::new(1, 16, (3, 3), 1, 1).use_batch_norm();
    /// ```
    pub fn use_batch_norm(mut self) -> Self {
        self.use_batch_norm = true;
        self
    }

    /// Use max pooling after activation/BatchNorm.
    ///
    /// # Notes
    /// * When this method is called, average pooling flag
    ///   will be automatically set to `false` to ensure mutual exclusion.
    ///   Only one pooling type can be active at a time.
    /// * The pooling kernel size and stride must be specified.
    ///
    /// # Arguments
    /// * `pool_kernel_size` - The size of the pooling kernel (height, width).
    /// * `pool_stride` - The stride of the pooling operation (height, width).
    ///
    /// # Returns
    /// * `Self` - The convolutional block with max pooling enabled.
    ///
    /// # Examples
    /// ```
    /// use nove::model::layer::CNNConvBlock;
    /// let block = CNNConvBlock::new(1, 16, (3, 3), 1, 1).use_max_pool((2, 2), (2, 2));
    /// ```
    pub fn use_max_pool(
        mut self,
        pool_kernel_size: (usize, usize),
        pool_stride: (usize, usize),
    ) -> Self {
        self.use_avg_pool = false;
        self.pool_kernel_size = pool_kernel_size;
        self.pool_stride = pool_stride;
        self.use_max_pool = true;
        self
    }

    /// Use average pooling after activation/BatchNorm.
    ///
    /// # Notes
    /// * When this method is called, max pooling flag
    ///   will be automatically set to `false` to ensure mutual exclusion.
    ///   Only one pooling type can be active at a time.
    /// * The pooling kernel size and stride must be specified.
    ///
    /// # Arguments
    /// * `pool_kernel_size` - The size of the pooling kernel (height, width).
    /// * `pool_stride` - The stride of the pooling operation (height, width).
    ///
    /// # Returns
    /// * `Self` - The convolutional block with average pooling enabled.
    ///
    /// # Examples
    /// ```
    /// use nove::model::layer::CNNConvBlock;
    /// let block = CNNConvBlock::new(1, 16, (3, 3), 1, 1).use_avg_pool((2, 2), (2, 2));
    /// ```
    pub fn use_avg_pool(
        mut self,
        pool_kernel_size: (usize, usize),
        pool_stride: (usize, usize),
    ) -> Self {
        self.use_max_pool = false;
        self.pool_kernel_size = pool_kernel_size;
        self.pool_stride = pool_stride;
        self.use_avg_pool = true;
        self
    }

    /// Disable all activation functions.
    ///
    /// # Returns
    /// * `Self` - The convolutional block with all activation functions disabled.
    ///
    /// # Examples
    /// ```
    /// use nove::model::layer::CNNConvBlock;
    /// let block = CNNConvBlock::new(1, 16, (3, 3), 1, 1).use_relu().disable_activation();
    /// ```
    pub fn disable_activation(mut self) -> Self {
        self.use_relu = false;
        self.use_gelu = false;
        self.use_silu = false;
        self.use_tanh = false;
        self.use_sigmoid = false;
        self
    }

    /// Disable batch normalization.
    ///
    /// # Returns
    /// * `Self` - The convolutional block with batch normalization disabled.
    ///
    /// # Examples
    /// ```
    /// use nove::model::layer::CNNConvBlock;
    /// let block = CNNConvBlock::new(1, 16, (3, 3), 1, 1).use_batch_norm().disable_batch_norm();
    /// ```
    pub fn disable_batch_norm(mut self) -> Self {
        self.use_batch_norm = false;
        self
    }

    /// Disable all pooling operations.
    ///
    /// # Returns
    /// * `Self` - The convolutional block with all pooling operations disabled.
    ///
    /// # Examples
    /// ```
    /// use nove::model::layer::CNNConvBlock;
    /// let block = CNNConvBlock::new(1, 16, (3, 3), 1, 1).use_max_pool((2, 2), (2, 2)).disable_pooling();
    /// ```
    pub fn disable_pooling(mut self) -> Self {
        self.use_max_pool = false;
        self.use_avg_pool = false;
        self
    }
}

/// Linear (fully connected) block configuration for building CNN models.
///
/// # Notes
/// * The `CNNLinearBlock` is used to configure a linear layer with optional
///   activation function.
/// * The default configuration has bias enabled and ReLU activation disabled.
/// * **Important**: Only one activation function can be enabled at a time.
///   When you call any `use_xxx()` method (e.g., `use_relu()`, `use_gelu()`, etc.),
///   all other activation function flags will be automatically set to `false`
///   to ensure mutual exclusion.
///
/// # Fields
/// * `in_features` - The number of input features.
/// * `out_features` - The number of output features.
/// * `use_relu` - Whether ReLU activation is enabled after the linear layer.
/// * `use_gelu` - Whether GELU activation is enabled after the linear layer.
/// * `use_silu` - Whether SiLU activation is enabled after the linear layer.
/// * `use_tanh` - Whether Tanh activation is enabled after the linear layer.
/// * `use_sigmoid` - Whether Sigmoid activation is enabled after the linear layer.
/// * `bias_enabled` - Whether the bias term is enabled.
///
/// # Examples
/// ```
/// use nove::model::layer::CNNLinearBlock;
///
/// let block = CNNLinearBlock::new(800, 10)
///     .use_gelu()
///     .bias_enabled(true);
/// ```
#[derive(Debug, Clone)]
pub struct CNNLinearBlock {
    in_features: usize,
    out_features: usize,
    use_relu: bool,
    use_gelu: bool,
    use_silu: bool,
    use_tanh: bool,
    use_sigmoid: bool,
    bias_enabled: bool,
}

impl CNNLinearBlock {
    /// Create a new linear block with default settings.
    ///
    /// # Arguments
    /// * `in_features` - The number of input features.
    /// * `out_features` - The number of output features.
    ///
    /// # Returns
    /// * `CNNLinearBlock` - The new linear block.
    ///
    /// # Examples
    /// ```
    /// use nove::model::layer::CNNLinearBlock;
    /// let block = CNNLinearBlock::new(800, 10);
    /// ```
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            in_features,
            out_features,
            use_relu: false,
            use_gelu: false,
            use_silu: false,
            use_tanh: false,
            use_sigmoid: false,
            bias_enabled: true,
        }
    }

    /// Use ReLU activation after the linear layer.
    ///
    /// # Notes
    /// * When this method is called, all other activation function flags
    ///   (GELU, SiLU, Tanh, Sigmoid) will be automatically set to `false` to ensure
    ///   mutual exclusion. Only one activation function can be active at a time.
    ///
    /// # Returns
    /// * `Self` - The linear block with ReLU activation enabled.
    ///
    /// # Examples
    /// ```
    /// use nove::model::layer::CNNLinearBlock;
    /// let block = CNNLinearBlock::new(800, 10).use_relu();
    /// ```
    pub fn use_relu(mut self) -> Self {
        self.use_gelu = false;
        self.use_silu = false;
        self.use_tanh = false;
        self.use_sigmoid = false;
        self.use_relu = true;
        self
    }

    /// Use GELU activation after the linear layer.
    ///
    /// # Notes
    /// * When this method is called, all other activation function flags
    ///   (ReLU, SiLU, Tanh, Sigmoid) will be automatically set to `false` to ensure
    ///   mutual exclusion. Only one activation function can be active at a time.
    ///
    /// # Returns
    /// * `Self` - The linear block with GELU activation enabled.
    ///
    /// # Examples
    /// ```
    /// use nove::model::layer::CNNLinearBlock;
    /// let block = CNNLinearBlock::new(800, 10).use_gelu();
    /// ```
    pub fn use_gelu(mut self) -> Self {
        self.use_relu = false;
        self.use_silu = false;
        self.use_tanh = false;
        self.use_sigmoid = false;
        self.use_gelu = true;
        self
    }

    /// Use SiLU activation after the linear layer.
    ///
    /// # Notes
    /// * When this method is called, all other activation function flags
    ///   (ReLU, GELU, Tanh, Sigmoid) will be automatically set to `false` to ensure
    ///   mutual exclusion. Only one activation function can be active at a time.
    ///
    /// # Returns
    /// * `Self` - The linear block with SiLU activation enabled.
    ///
    /// # Examples
    /// ```
    /// use nove::model::layer::CNNLinearBlock;
    /// let block = CNNLinearBlock::new(800, 10).use_silu();
    /// ```
    pub fn use_silu(mut self) -> Self {
        self.use_relu = false;
        self.use_gelu = false;
        self.use_tanh = false;
        self.use_sigmoid = false;
        self.use_silu = true;
        self
    }

    /// Use Tanh activation after the linear layer.
    ///
    /// # Notes
    /// * When this method is called, all other activation function flags
    ///   (ReLU, GELU, SiLU, Sigmoid) will be automatically set to `false` to ensure
    ///   mutual exclusion. Only one activation function can be active at a time.
    ///
    /// # Returns
    /// * `Self` - The linear block with Tanh activation enabled.
    ///
    /// # Examples
    /// ```
    /// use nove::model::layer::CNNLinearBlock;
    /// let block = CNNLinearBlock::new(800, 10).use_tanh();
    /// ```
    pub fn use_tanh(mut self) -> Self {
        self.use_relu = false;
        self.use_gelu = false;
        self.use_silu = false;
        self.use_sigmoid = false;
        self.use_tanh = true;
        self
    }

    /// Use Sigmoid activation after the linear layer.
    ///
    /// # Notes
    /// * When this method is called, all other activation function flags
    ///   (ReLU, GELU, SiLU, Tanh) will be automatically set to `false` to ensure
    ///   mutual exclusion. Only one activation function can be active at a time.
    ///
    /// # Returns
    /// * `Self` - The linear block with Sigmoid activation enabled.
    ///
    /// # Examples
    /// ```
    /// use nove::model::layer::CNNLinearBlock;
    /// let block = CNNLinearBlock::new(800, 10).use_sigmoid();
    /// ```
    pub fn use_sigmoid(mut self) -> Self {
        self.use_relu = false;
        self.use_gelu = false;
        self.use_silu = false;
        self.use_tanh = false;
        self.use_sigmoid = true;
        self
    }

    /// Disable all activation functions.
    ///
    /// # Returns
    /// * `Self` - The linear block with all activation functions disabled.
    ///
    /// # Examples
    /// ```
    /// use nove::model::layer::CNNLinearBlock;
    /// let block = CNNLinearBlock::new(800, 10).use_relu().disable_activation();
    /// ```
    pub fn disable_activation(mut self) -> Self {
        self.use_relu = false;
        self.use_gelu = false;
        self.use_silu = false;
        self.use_tanh = false;
        self.use_sigmoid = false;
        self
    }

    /// Configure whether to enable the bias term.
    ///
    /// # Arguments
    /// * `bias_enabled` - Whether to enable the bias term.
    ///
    /// # Returns
    /// * `Self` - The linear block with the configured bias term.
    ///
    /// # Examples
    /// ```
    /// use nove::model::layer::CNNLinearBlock;
    /// let block = CNNLinearBlock::new(800, 10).bias_enabled(false);
    /// ```
    pub fn bias_enabled(mut self, bias_enabled: bool) -> Self {
        self.bias_enabled = bias_enabled;
        self
    }
}

/// The builder for the CNN model.
///
/// # Notes
/// * The `CNNBuilder` implements the `Default` trait, so you can
///   use `CNNBuilder::default()` to create a builder with default values.
///
/// # Required Arguments
/// * At least one `CNNConvBlock` or `CNNLinearBlock` must be added.
///
/// # Optional Arguments
/// * `device` - The device to use for the model. Default is `Device::cpu()`.
/// * `dtype` - The data type to use for the model. Default is `DType::F32`.
/// * `grad_enabled` - Whether to enable the gradient computation. Default is `true`.
///
/// # Fields
/// * `conv_blocks` - The convolutional blocks to add to the model.
/// * `linear_blocks` - The linear blocks to add to the model.
/// * `device` - The device to use for the model.
/// * `dtype` - The data type to use for the model.
/// * `grad_enabled` - Whether to enable the gradient computation.
///
/// # Examples
/// ```
/// use nove::model::layer::{CNNBuilder, CNNConvBlock, CNNLinearBlock};
/// use nove::tensor::{Device, DType};
///
/// let cnn = CNNBuilder::default()
///     .conv_block(CNNConvBlock::new(1, 16, (3, 3), 1, 1).use_max_pool((2, 2), (2, 2)))
///     .conv_block(CNNConvBlock::new(16, 32, (3, 3), 1, 1).use_avg_pool((2, 2), (2, 2)))
///     .linear_block(CNNLinearBlock::new(800, 10))
///     .device(Device::cpu())
///     .dtype(DType::F32)
///     .grad_enabled(true)
///     .build();
/// ```
pub struct CNNBuilder {
    conv_blocks: Vec<CNNConvBlock>,
    linear_blocks: Vec<CNNLinearBlock>,
    device: Device,
    dtype: DType,
    grad_enabled: bool,
}

impl Default for CNNBuilder {
    fn default() -> Self {
        Self {
            conv_blocks: Vec::new(),
            linear_blocks: Vec::new(),
            device: Device::cpu(),
            dtype: DType::F32,
            grad_enabled: true,
        }
    }
}

impl CNNBuilder {
    /// Add a convolutional block to the CNN model.
    ///
    /// # Arguments
    /// * `block` - The convolutional block to add.
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the added convolutional block.
    ///
    /// # Examples
    /// ```
    /// use nove::model::layer::{CNNBuilder, CNNConvBlock};
    /// let mut builder = CNNBuilder::default();
    /// builder.conv_block(CNNConvBlock::new(1, 16, (3, 3), 1, 1).use_max_pool((2, 2), (2, 2)));
    /// ```
    pub fn conv_block(&mut self, block: CNNConvBlock) -> &mut Self {
        self.conv_blocks.push(block);
        self
    }

    /// Add a linear block to the CNN model.
    ///
    /// # Arguments
    /// * `block` - The linear block to add.
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the added linear block.
    ///
    /// # Examples
    /// ```
    /// use nove::model::layer::{CNNBuilder, CNNLinearBlock};
    /// let mut builder = CNNBuilder::default();
    /// builder.linear_block(CNNLinearBlock::new(800, 10));
    /// ```
    pub fn linear_block(&mut self, block: CNNLinearBlock) -> &mut Self {
        self.linear_blocks.push(block);
        self
    }

    /// Configure the device to use for the model.
    ///
    /// # Arguments
    /// * `device` - The device to use for the model.
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured device.
    ///
    /// # Examples
    /// ```
    /// use nove::model::layer::CNNBuilder;
    /// use nove::tensor::Device;
    /// let mut builder = CNNBuilder::default();
    /// builder.device(Device::cpu());
    /// ```
    pub fn device(&mut self, device: Device) -> &mut Self {
        self.device = device;
        self
    }

    /// Configure the data type to use for the model.
    ///
    /// # Arguments
    /// * `dtype` - The data type to use for the model.
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured data type.
    ///
    /// # Examples
    /// ```
    /// use nove::model::layer::CNNBuilder;
    /// use nove::tensor::DType;
    /// let mut builder = CNNBuilder::default();
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
    /// ```
    /// use nove::model::layer::CNNBuilder;
    /// let mut builder = CNNBuilder::default();
    /// builder.grad_enabled(true);
    /// ```
    pub fn grad_enabled(&mut self, grad_enabled: bool) -> &mut Self {
        self.grad_enabled = grad_enabled;
        self
    }

    /// Build the CNN model.
    ///
    /// # Returns
    /// * `Ok(CNN)` - The built CNN model.
    /// * `Err(ModelError)` - The error when building the CNN model.
    ///
    /// # Examples
    /// ```
    /// use nove::model::layer::{CNNBuilder, CNNConvBlock, CNNLinearBlock};
    /// let mut builder = CNNBuilder::default();
    /// builder.conv_block(CNNConvBlock::new(1, 16, (3, 3), 1, 1).use_max_pool((2, 2), (2, 2)));
    /// builder.linear_block(CNNLinearBlock::new(800, 10));
    /// let cnn = builder.build().unwrap();
    /// ```
    pub fn build(&self) -> Result<CNN, ModelError> {
        let mut layers = Vec::new();

        for block in &self.conv_blocks {
            let conv = Conv2dBuilder::default()
                .in_channels(block.in_channels)
                .out_channels(block.out_channels)
                .kernel_size(block.kernel_size)
                .stride(block.stride)
                .padding(block.padding)
                .device(self.device.clone())
                .dtype(self.dtype)
                .grad_enabled(self.grad_enabled)
                .build()?;
            layers.push(CNNLayer::Conv2d(conv));

            if block.use_batch_norm {
                let bn = BatchNorm2dBuilder::default()
                    .num_features(block.out_channels)
                    .device(self.device.clone())
                    .dtype(self.dtype)
                    .affine(self.grad_enabled)
                    .build()?;
                layers.push(CNNLayer::BatchNorm2d(bn));
            }

            if block.use_relu {
                layers.push(CNNLayer::ReLU(ReLU::new()));
            } else if block.use_gelu {
                layers.push(CNNLayer::GELU(GELU::new()));
            } else if block.use_silu {
                layers.push(CNNLayer::SiLU(SiLU::new()));
            } else if block.use_tanh {
                layers.push(CNNLayer::Tanh(Tanh::new()));
            } else if block.use_sigmoid {
                layers.push(CNNLayer::Sigmoid(Sigmoid::new()));
            }

            if block.use_max_pool {
                let pool = MaxPool2d::new(block.pool_kernel_size, Some(block.pool_stride))?;
                layers.push(CNNLayer::MaxPool2d(pool));
            } else if block.use_avg_pool {
                let pool = AvgPool2d::new(block.pool_kernel_size, Some(block.pool_stride))?;
                layers.push(CNNLayer::AvgPool2d(pool));
            }
        }

        for block in &self.linear_blocks {
            let linear = LinearBuilder::default()
                .in_features(block.in_features)
                .out_features(block.out_features)
                .bias_enabled(block.bias_enabled)
                .device(self.device.clone())
                .dtype(self.dtype)
                .grad_enabled(self.grad_enabled)
                .build()?;
            layers.push(CNNLayer::Linear(linear));

            if block.use_relu {
                layers.push(CNNLayer::ReLU(ReLU::new()));
            } else if block.use_gelu {
                layers.push(CNNLayer::GELU(GELU::new()));
            } else if block.use_silu {
                layers.push(CNNLayer::SiLU(SiLU::new()));
            } else if block.use_tanh {
                layers.push(CNNLayer::Tanh(Tanh::new()));
            } else if block.use_sigmoid {
                layers.push(CNNLayer::Sigmoid(Sigmoid::new()));
            }
        }

        let id = ID.fetch_add(1, Ordering::Relaxed);

        Ok(CNN { layers, id })
    }
}
