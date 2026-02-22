use std::{
    collections::HashMap,
    fmt::{Display, Write},
    sync::atomic::{AtomicUsize, Ordering},
};

use nove_tensor::{DType, Device, Tensor};

use crate::{Model, ModelError};

use super::{Conv2d, Conv2dBuilder, Linear, LinearBuilder, MaxPool2d, MaxPool2dBuilder, ReLU};

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
/// * `MaxPool2d` - 2D max pooling layer.
/// * `Linear` - Fully connected linear layer.
#[derive(Debug, Clone)]
pub enum CNNLayer {
    Conv2d(Conv2d),
    ReLU(ReLU),
    MaxPool2d(MaxPool2d),
    Linear(Linear),
}

impl Model for CNNLayer {
    type Input = Tensor;
    type Output = Tensor;

    /// Apply the layer to the input tensor.
    ///
    /// # Arguments
    /// * `input: Tensor` - The input tensor.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The output tensor if successful.
    /// * `Err(ModelError)` - The error when applying the layer to the input tensor.
    fn forward(&mut self, input: Self::Input) -> Result<Self::Output, ModelError> {
        match self {
            CNNLayer::Conv2d(layer) => layer.forward(input),
            CNNLayer::ReLU(layer) => layer.forward(input),
            CNNLayer::MaxPool2d(layer) => layer.forward(input),
            CNNLayer::Linear(layer) => layer.forward(input),
        }
    }

    fn require_grad(&mut self, grad_enabled: bool) -> Result<(), ModelError> {
        match self {
            CNNLayer::Conv2d(layer) => layer.require_grad(grad_enabled),
            CNNLayer::ReLU(layer) => layer.require_grad(grad_enabled),
            CNNLayer::MaxPool2d(layer) => layer.require_grad(grad_enabled),
            CNNLayer::Linear(layer) => layer.require_grad(grad_enabled),
        }
    }

    fn to_device(&mut self, device: &Device) -> Result<(), ModelError> {
        match self {
            CNNLayer::Conv2d(layer) => layer.to_device(device),
            CNNLayer::ReLU(layer) => layer.to_device(device),
            CNNLayer::MaxPool2d(layer) => layer.to_device(device),
            CNNLayer::Linear(layer) => layer.to_device(device),
        }
    }

    fn to_dtype(&mut self, dtype: &DType) -> Result<(), ModelError> {
        match self {
            CNNLayer::Conv2d(layer) => layer.to_dtype(dtype),
            CNNLayer::ReLU(layer) => layer.to_dtype(dtype),
            CNNLayer::MaxPool2d(layer) => layer.to_dtype(dtype),
            CNNLayer::Linear(layer) => layer.to_dtype(dtype),
        }
    }

    fn parameters(&self) -> Result<Vec<Tensor>, ModelError> {
        match self {
            CNNLayer::Conv2d(layer) => layer.parameters(),
            CNNLayer::ReLU(layer) => layer.parameters(),
            CNNLayer::MaxPool2d(layer) => layer.parameters(),
            CNNLayer::Linear(layer) => layer.parameters(),
        }
    }

    fn named_parameters(&self) -> Result<HashMap<String, Tensor>, ModelError> {
        match self {
            CNNLayer::Conv2d(layer) => layer.named_parameters(),
            CNNLayer::ReLU(layer) => layer.named_parameters(),
            CNNLayer::MaxPool2d(layer) => layer.named_parameters(),
            CNNLayer::Linear(layer) => layer.named_parameters(),
        }
    }
}

impl Display for CNNLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CNNLayer::Conv2d(layer) => write!(f, "{}", layer),
            CNNLayer::ReLU(layer) => write!(f, "{}", layer),
            CNNLayer::MaxPool2d(layer) => write!(f, "{}", layer),
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
///     .conv_block(CNNConvBlock::new(1, 16).use_pool(true))
///     .conv_block(CNNConvBlock::new(16, 32).use_pool(true))
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
    ///   - `bool` - Whether the model is in training mode (ignored by CNN).
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The output tensor if successful.
    /// * `Err(ModelError)` - The error when applying the CNN model to the input tensor.
    ///
    /// # Notes
    /// * The `bool` flag indicating training mode is ignored by CNN.
    /// * CNN produces identical outputs in both training and non-training modes,
    ///   as it does not contain layers with different behaviors (e.g., Dropout, BatchNorm).
    fn forward(&mut self, input: Self::Input) -> Result<Self::Output, ModelError> {
        let (mut input, _training) = input;
        for layer in &mut self.layers {
            input = layer.forward(input)?;
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
        for layer in &self.layers {
            params.extend(layer.named_parameters()?);
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
///   ReLU activation and max pooling.
/// * The default configuration uses a 3x3 kernel with stride 1 and padding 1,
///   with ReLU activation enabled and pooling disabled.
///
/// # Fields
/// * `in_channels` - The number of input channels.
/// * `out_channels` - The number of output channels.
/// * `kernel_size` - The size of the convolution kernel (height, width).
/// * `stride` - The stride of the convolution.
/// * `padding` - The padding size.
/// * `use_relu` - Whether to use ReLU activation after convolution.
/// * `use_pool` - Whether to use max pooling after ReLU.
/// * `pool_kernel_size` - The size of the pooling kernel (height, width).
/// * `pool_stride` - The stride of the pooling operation (height, width).
///
/// # Examples
/// ```
/// use nove::model::layer::CNNConvBlock;
///
/// let block = CNNConvBlock::new(1, 16)
///     .kernel_size((3, 3))
///     .stride(1)
///     .padding(1)
///     .use_relu(true)
///     .use_pool(true)
///     .pool_kernel_size((2, 2))
///     .pool_stride((2, 2));
/// ```
#[derive(Debug, Clone)]
pub struct CNNConvBlock {
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize),
    stride: usize,
    padding: usize,
    use_relu: bool,
    use_pool: bool,
    pool_kernel_size: (usize, usize),
    pool_stride: (usize, usize),
}

impl CNNConvBlock {
    /// Create a new convolutional block with default settings.
    ///
    /// # Arguments
    /// * `in_channels` - The number of input channels.
    /// * `out_channels` - The number of output channels.
    ///
    /// # Returns
    /// * `CNNConvBlock` - The new convolutional block.
    ///
    /// # Examples
    /// ```
    /// use nove::model::layer::CNNConvBlock;
    /// let block = CNNConvBlock::new(1, 16);
    /// ```
    pub fn new(in_channels: usize, out_channels: usize) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_size: (3, 3),
            stride: 1,
            padding: 1,
            use_relu: true,
            use_pool: false,
            pool_kernel_size: (2, 2),
            pool_stride: (2, 2),
        }
    }

    /// Configure the size of the convolution kernel.
    ///
    /// # Arguments
    /// * `kernel_size` - The size of the convolution kernel (height, width).
    ///
    /// # Returns
    /// * `Self` - The convolutional block with the configured kernel size.
    ///
    /// # Examples
    /// ```
    /// use nove::model::layer::CNNConvBlock;
    /// let block = CNNConvBlock::new(1, 16).kernel_size((3, 3));
    /// ```
    pub fn kernel_size(mut self, kernel_size: (usize, usize)) -> Self {
        self.kernel_size = kernel_size;
        self
    }

    /// Configure the stride of the convolution.
    ///
    /// # Arguments
    /// * `stride` - The stride of the convolution.
    ///
    /// # Returns
    /// * `Self` - The convolutional block with the configured stride.
    ///
    /// # Examples
    /// ```
    /// use nove::model::layer::CNNConvBlock;
    /// let block = CNNConvBlock::new(1, 16).stride(1);
    /// ```
    pub fn stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    /// Configure the padding size.
    ///
    /// # Arguments
    /// * `padding` - The padding size.
    ///
    /// # Returns
    /// * `Self` - The convolutional block with the configured padding.
    ///
    /// # Examples
    /// ```
    /// use nove::model::layer::CNNConvBlock;
    /// let block = CNNConvBlock::new(1, 16).padding(1);
    /// ```
    pub fn padding(mut self, padding: usize) -> Self {
        self.padding = padding;
        self
    }

    /// Configure whether to use ReLU activation after convolution.
    ///
    /// # Arguments
    /// * `use_relu` - Whether to use ReLU activation.
    ///
    /// # Returns
    /// * `Self` - The convolutional block with the configured ReLU activation.
    ///
    /// # Examples
    /// ```
    /// use nove::model::layer::CNNConvBlock;
    /// let block = CNNConvBlock::new(1, 16).use_relu(true);
    /// ```
    pub fn use_relu(mut self, use_relu: bool) -> Self {
        self.use_relu = use_relu;
        self
    }

    /// Configure whether to use max pooling after ReLU.
    ///
    /// # Arguments
    /// * `use_pool` - Whether to use max pooling.
    ///
    /// # Returns
    /// * `Self` - The convolutional block with the configured pooling.
    ///
    /// # Examples
    /// ```
    /// use nove::model::layer::CNNConvBlock;
    /// let block = CNNConvBlock::new(1, 16).use_pool(true);
    /// ```
    pub fn use_pool(mut self, use_pool: bool) -> Self {
        self.use_pool = use_pool;
        self
    }

    /// Configure the size of the pooling kernel.
    ///
    /// # Arguments
    /// * `pool_kernel_size` - The size of the pooling kernel (height, width).
    ///
    /// # Returns
    /// * `Self` - The convolutional block with the configured pooling kernel size.
    ///
    /// # Examples
    /// ```
    /// use nove::model::layer::CNNConvBlock;
    /// let block = CNNConvBlock::new(1, 16).pool_kernel_size((2, 2));
    /// ```
    pub fn pool_kernel_size(mut self, pool_kernel_size: (usize, usize)) -> Self {
        self.pool_kernel_size = pool_kernel_size;
        self
    }

    /// Configure the stride of the pooling operation.
    ///
    /// # Arguments
    /// * `pool_stride` - The stride of the pooling operation (height, width).
    ///
    /// # Returns
    /// * `Self` - The convolutional block with the configured pooling stride.
    ///
    /// # Examples
    /// ```
    /// use nove::model::layer::CNNConvBlock;
    /// let block = CNNConvBlock::new(1, 16).pool_stride((2, 2));
    /// ```
    pub fn pool_stride(mut self, pool_stride: (usize, usize)) -> Self {
        self.pool_stride = pool_stride;
        self
    }
}

/// Linear (fully connected) block configuration for building CNN models.
///
/// # Notes
/// * The `CNNLinearBlock` is used to configure a linear layer with optional
///   ReLU activation.
/// * The default configuration has bias enabled and ReLU activation disabled.
///
/// # Fields
/// * `in_features` - The number of input features.
/// * `out_features` - The number of output features.
/// * `use_relu` - Whether to use ReLU activation after the linear layer.
/// * `bias_enabled` - Whether to enable the bias term.
///
/// # Examples
/// ```
/// use nove::model::layer::CNNLinearBlock;
///
/// let block = CNNLinearBlock::new(800, 10)
///     .use_relu(false)
///     .bias_enabled(true);
/// ```
#[derive(Debug, Clone)]
pub struct CNNLinearBlock {
    in_features: usize,
    out_features: usize,
    use_relu: bool,
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
            bias_enabled: true,
        }
    }

    /// Configure whether to use ReLU activation after the linear layer.
    ///
    /// # Arguments
    /// * `use_relu` - Whether to use ReLU activation.
    ///
    /// # Returns
    /// * `Self` - The linear block with the configured ReLU activation.
    ///
    /// # Examples
    /// ```
    /// use nove::model::layer::CNNLinearBlock;
    /// let block = CNNLinearBlock::new(800, 10).use_relu(true);
    /// ```
    pub fn use_relu(mut self, use_relu: bool) -> Self {
        self.use_relu = use_relu;
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
    /// let block = CNNLinearBlock::new(800, 10).bias_enabled(true);
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
///     .conv_block(CNNConvBlock::new(1, 16).use_pool(true))
///     .conv_block(CNNConvBlock::new(16, 32).use_pool(true))
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
    /// builder.conv_block(CNNConvBlock::new(1, 16).use_pool(true));
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
    /// builder.conv_block(CNNConvBlock::new(1, 16).use_pool(true));
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
                .dtype(self.dtype.clone())
                .grad_enabled(self.grad_enabled)
                .build()?;
            layers.push(CNNLayer::Conv2d(conv));

            if block.use_relu {
                layers.push(CNNLayer::ReLU(ReLU::new()));
            }

            if block.use_pool {
                let pool = MaxPool2dBuilder::default()
                    .kernel_size(block.pool_kernel_size)
                    .stride(block.pool_stride)
                    .build()?;
                layers.push(CNNLayer::MaxPool2d(pool));
            }
        }

        for block in &self.linear_blocks {
            let linear = LinearBuilder::default()
                .in_features(block.in_features)
                .out_features(block.out_features)
                .bias_enabled(block.bias_enabled)
                .device(self.device.clone())
                .dtype(self.dtype.clone())
                .grad_enabled(self.grad_enabled)
                .build()?;
            layers.push(CNNLayer::Linear(linear));

            if block.use_relu {
                layers.push(CNNLayer::ReLU(ReLU::new()));
            }
        }

        let id = ID.fetch_add(1, Ordering::Relaxed);

        Ok(CNN { layers, id })
    }
}
