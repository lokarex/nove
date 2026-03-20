use std::{collections::HashMap, fmt::Display};

use nove_tensor::{DType, Device, Tensor};

use crate::{Model, ModelError};

use super::{
    AvgPool2d, BatchNorm2d, BatchNorm2dBuilder, Conv2d, Conv2dBuilder, GELU, MaxPool2d, ReLU, SiLU,
    Sigmoid, Tanh,
};

static ID: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Conv2d block configuration.
///
/// # Notes
/// * The `Conv2dBlockBuilder` is used to configure a convolutional layer with optional
///   activation function, batch normalization (2D), and pooling.
/// * The kernel size, stride, and padding must be specified when creating a block.
/// * The default configuration uses activation function disabled, batch normalization (2D) disabled,
///   and pooling disabled.
/// * **Important**: Only one activation function can be enabled at a time.
///   When you call any `with_xxx()` method (e.g., `with_relu()`, `with_gelu()`, etc.),
///   all other activation function flags will be automatically set to `false`
///   to ensure mutual exclusion.
/// * **Important**: Only one pooling type can be enabled at a time.
///   When you call any `with_xxx_pool()` method (e.g., `with_max_pool()`, `with_avg_pool()`),
///   all other pooling flags will be automatically set to `false` to ensure mutual exclusion.
///   The pooling kernel size and stride must be specified when enabling pooling.
///
/// # Required Arguments
/// * `in_channels` - The number of input channels.
/// * `out_channels` - The number of output channels.
/// * `kernel_size` - The size of the convolution kernel (height, width).
/// * `stride` - The stride of the convolution.
/// * `padding` - The padding size.
///
/// # Optional Arguments
/// * `use_relu` - Whether ReLU activation is enabled after convolution. Default is `false`. (configured via `with_relu()`)
/// * `use_gelu` - Whether GELU activation is enabled after convolution. Default is `false`. (configured via `with_gelu()`)
/// * `use_silu` - Whether SiLU activation is enabled after convolution. Default is `false`. (configured via `with_silu()`)
/// * `use_tanh` - Whether Tanh activation is enabled after convolution. Default is `false`. (configured via `with_tanh()`)
/// * `use_sigmoid` - Whether Sigmoid activation is enabled after convolution. Default is `false`. (configured via `with_sigmoid()`)
/// * `use_batch_norm2d` - Whether 2D batch normalization is enabled after convolution. Default is `false`. (configured via `with_batch_norm2d()`)
/// * `use_max_pool` - Whether max pooling is enabled after activation/BatchNorm2d. Default is `false`. (configured via `with_max_pool()`)
/// * `use_avg_pool` - Whether average pooling is enabled after activation/BatchNorm2d. Default is `false`. (configured via `with_avg_pool()`)
/// * `pool_kernel_size` - The size of the pooling kernel (height, width). Default is `(2, 2)`. (configured together with `with_max_pool()` or `with_avg_pool()`)
/// * `pool_stride` - The stride of the pooling operation (height, width). Default is `(2, 2)`. (configured together with `with_max_pool()` or `with_avg_pool()`)
/// * `device` - The device to use for the layer. Default is `Device::cpu()`.
/// * `dtype` - The data type to use for the layer. Default is `DType::F32`.
/// * `grad_enabled` - Whether to enable the gradient computation. Default is `true`.
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
/// * `use_batch_norm2d` - Whether 2D batch normalization is enabled after convolution.
/// * `use_max_pool` - Whether max pooling is enabled after activation/BatchNorm2d.
/// * `use_avg_pool` - Whether average pooling is enabled after activation/BatchNorm2d.
/// * `pool_kernel_size` - The size of the pooling kernel (height, width).
/// * `pool_stride` - The stride of the pooling operation (height, width).
/// * `device` - The device to use for the layer.
/// * `dtype` - The data type to use for the layer.
/// * `grad_enabled` - Whether to enable the gradient computation.
///
/// # Examples
/// ```no_run
/// use nove::model::layer::Conv2dBlockBuilder;
/// use nove::tensor::{Device, DType};
///
/// let block = Conv2dBlockBuilder::new(1, 16, (3, 3), 1, 1)
///     .with_gelu()
///     .with_batch_norm2d()
///     .with_avg_pool((2, 2), (2, 2))
///     .device(Device::cpu())
///     .dtype(DType::F32)
///     .grad_enabled(true);
/// ```
#[derive(Debug, Clone)]
pub struct Conv2dBlockBuilder {
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
    use_batch_norm2d: bool,
    use_max_pool: bool,
    use_avg_pool: bool,
    pool_kernel_size: (usize, usize),
    pool_stride: (usize, usize),
    device: Device,
    dtype: DType,
    grad_enabled: bool,
}

impl Conv2dBlockBuilder {
    /// Create a new conv2d block builder with specified settings.
    ///
    /// # Arguments
    /// * `in_channels` - The number of input channels.
    /// * `out_channels` - The number of output channels.
    /// * `kernel_size` - The size of the convolution kernel (height, width).
    /// * `stride` - The stride of the convolution.
    /// * `padding` - The padding size.
    ///
    /// # Returns
    /// * `Conv2dBlockBuilder` - The new conv2d block builder.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::Conv2dBlockBuilder;
    /// let builder = Conv2dBlockBuilder::new(1, 16, (3, 3), 1, 1);
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
            use_relu: false,
            use_gelu: false,
            use_silu: false,
            use_tanh: false,
            use_sigmoid: false,
            use_batch_norm2d: false,
            use_max_pool: false,
            use_avg_pool: false,
            pool_kernel_size: (2, 2),
            pool_stride: (2, 2),
            device: Device::cpu(),
            dtype: DType::F32,
            grad_enabled: true,
        }
    }

    /// Configure ReLU activation after convolution.
    ///
    /// # Notes
    /// * When this method is called, all other activation function flags
    ///   (GELU, SiLU, Tanh, Sigmoid) will be automatically set to `false` to ensure
    ///   mutual exclusion. Only one activation function can be active at a time.
    ///
    /// # Returns
    /// * `Self` - The conv2d block builder with ReLU activation enabled.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::Conv2dBlockBuilder;
    /// let builder = Conv2dBlockBuilder::new(1, 16, (3, 3), 1, 1).with_relu();
    /// ```
    pub fn with_relu(mut self) -> Self {
        self.use_gelu = false;
        self.use_silu = false;
        self.use_tanh = false;
        self.use_sigmoid = false;
        self.use_relu = true;
        self
    }

    /// Configure GELU activation after convolution.
    ///
    /// # Notes
    /// * When this method is called, all other activation function flags
    ///   (ReLU, SiLU, Tanh, Sigmoid) will be automatically set to `false` to ensure
    ///   mutual exclusion. Only one activation function can be active at a time.
    ///
    /// # Returns
    /// * `Self` - The conv2d block builder with GELU activation enabled.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::Conv2dBlockBuilder;
    /// let builder = Conv2dBlockBuilder::new(1, 16, (3, 3), 1, 1).with_gelu();
    /// ```
    pub fn with_gelu(mut self) -> Self {
        self.use_relu = false;
        self.use_silu = false;
        self.use_tanh = false;
        self.use_sigmoid = false;
        self.use_gelu = true;
        self
    }

    /// Configure SiLU activation after convolution.
    ///
    /// # Notes
    /// * When this method is called, all other activation function flags
    ///   (ReLU, GELU, Tanh, Sigmoid) will be automatically set to `false` to ensure
    ///   mutual exclusion. Only one activation function can be active at a time.
    ///
    /// # Returns
    /// * `Self` - The conv2d block builder with SiLU activation enabled.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::Conv2dBlockBuilder;
    /// let builder = Conv2dBlockBuilder::new(1, 16, (3, 3), 1, 1).with_silu();
    /// ```
    pub fn with_silu(mut self) -> Self {
        self.use_relu = false;
        self.use_gelu = false;
        self.use_tanh = false;
        self.use_sigmoid = false;
        self.use_silu = true;
        self
    }

    /// Configure Tanh activation after convolution.
    ///
    /// # Notes
    /// * When this method is called, all other activation function flags
    ///   (ReLU, GELU, SiLU, Sigmoid) will be automatically set to `false` to ensure
    ///   mutual exclusion. Only one activation function can be active at a time.
    ///
    /// # Returns
    /// * `Self` - The conv2d block builder with Tanh activation enabled.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::Conv2dBlockBuilder;
    /// let builder = Conv2dBlockBuilder::new(1, 16, (3, 3), 1, 1).with_tanh();
    /// ```
    pub fn with_tanh(mut self) -> Self {
        self.use_relu = false;
        self.use_gelu = false;
        self.use_silu = false;
        self.use_sigmoid = false;
        self.use_tanh = true;
        self
    }

    /// Configure Sigmoid activation after convolution.
    ///
    /// # Notes
    /// * When this method is called, all other activation function flags
    ///   (ReLU, GELU, SiLU, Tanh) will be automatically set to `false` to ensure
    ///   mutual exclusion. Only one activation function can be active at a time.
    ///
    /// # Returns
    /// * `Self` - The conv2d block builder with Sigmoid activation enabled.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::Conv2dBlockBuilder;
    /// let builder = Conv2dBlockBuilder::new(1, 16, (3, 3), 1, 1).with_sigmoid();
    /// ```
    pub fn with_sigmoid(mut self) -> Self {
        self.use_relu = false;
        self.use_gelu = false;
        self.use_silu = false;
        self.use_tanh = false;
        self.use_sigmoid = true;
        self
    }

    /// Configure 2D batch normalization after convolution.
    ///
    /// # Returns
    /// * `Self` - The conv2d block builder with 2D batch normalization enabled.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::Conv2dBlockBuilder;
    /// let builder = Conv2dBlockBuilder::new(1, 16, (3, 3), 1, 1).with_batch_norm2d();
    /// ```
    pub fn with_batch_norm2d(mut self) -> Self {
        self.use_batch_norm2d = true;
        self
    }

    /// Configure max pooling after activation/BatchNorm2d.
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
    /// * `Self` - The conv2d block builder with max pooling enabled.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::Conv2dBlockBuilder;
    /// let builder = Conv2dBlockBuilder::new(1, 16, (3, 3), 1, 1).with_max_pool((2, 2), (2, 2));
    /// ```
    pub fn with_max_pool(
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

    /// Configure average pooling after activation/BatchNorm2d.
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
    /// * `Self` - The conv2d block builder with average pooling enabled.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::Conv2dBlockBuilder;
    /// let builder = Conv2dBlockBuilder::new(1, 16, (3, 3), 1, 1).with_avg_pool((2, 2), (2, 2));
    /// ```
    pub fn with_avg_pool(
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

    /// Configure without any activation functions.
    ///
    /// # Returns
    /// * `Self` - The conv2d block builder with all activation functions disabled.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::Conv2dBlockBuilder;
    /// let builder = Conv2dBlockBuilder::new(1, 16, (3, 3), 1, 1).with_relu().without_activation();
    /// ```
    pub fn without_activation(mut self) -> Self {
        self.use_relu = false;
        self.use_gelu = false;
        self.use_silu = false;
        self.use_tanh = false;
        self.use_sigmoid = false;
        self
    }

    /// Configure without 2D batch normalization.
    ///
    /// # Returns
    /// * `Self` - The conv2d block builder with 2D batch normalization disabled.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::Conv2dBlockBuilder;
    /// let builder = Conv2dBlockBuilder::new(1, 16, (3, 3), 1, 1).with_batch_norm2d().without_batch_norm2d();
    /// ```
    pub fn without_batch_norm2d(mut self) -> Self {
        self.use_batch_norm2d = false;
        self
    }

    /// Configure without any pooling operations.
    ///
    /// # Returns
    /// * `Self` - The conv2d block builder with all pooling operations disabled.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::Conv2dBlockBuilder;
    /// let builder = Conv2dBlockBuilder::new(1, 16, (3, 3), 1, 1).with_max_pool((2, 2), (2, 2)).without_pooling();
    /// ```
    pub fn without_pooling(mut self) -> Self {
        self.use_max_pool = false;
        self.use_avg_pool = false;
        self
    }

    /// Configure the device to use for the layer.
    ///
    /// # Arguments
    /// * `device` - The device to use for the layer.
    ///
    /// # Returns
    /// * `Self` - The conv2d block builder with the configured device.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::Conv2dBlockBuilder;
    /// use nove::tensor::Device;
    /// let builder = Conv2dBlockBuilder::new(1, 16, (3, 3), 1, 1).device(Device::cpu());
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
    /// * `Self` - The conv2d block builder with the configured data type.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::Conv2dBlockBuilder;
    /// use nove::tensor::DType;
    /// let builder = Conv2dBlockBuilder::new(1, 16, (3, 3), 1, 1).dtype(DType::F32);
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
    /// * `Self` - The conv2d block builder with the configured gradient computation.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::Conv2dBlockBuilder;
    /// let builder = Conv2dBlockBuilder::new(1, 16, (3, 3), 1, 1).grad_enabled(true);
    /// ```
    pub fn grad_enabled(mut self, grad_enabled: bool) -> Self {
        self.grad_enabled = grad_enabled;
        self
    }

    /// Build the conv2d block.
    ///
    /// # Returns
    /// * `Ok(Conv2dBlock)` - The built conv2d block.
    /// * `Err(ModelError)` - The error when building the conv2d block.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::Conv2dBlockBuilder;
    /// let block = Conv2dBlockBuilder::new(1, 16, (3, 3), 1, 1).build().unwrap();
    /// ```
    pub fn build(self) -> Result<Conv2dBlock, ModelError> {
        let conv = Conv2dBuilder::default()
            .in_channels(self.in_channels)
            .out_channels(self.out_channels)
            .kernel_size(self.kernel_size)
            .stride(self.stride)
            .padding(self.padding)
            .device(self.device.clone())
            .dtype(self.dtype)
            .grad_enabled(self.grad_enabled)
            .build()?;

        let batch_norm2d = if self.use_batch_norm2d {
            Some(
                BatchNorm2dBuilder::default()
                    .num_features(self.out_channels)
                    .device(self.device.clone())
                    .dtype(self.dtype)
                    .build()?,
            )
        } else {
            None
        };

        let activation = if self.use_relu {
            Some(Conv2dBlockActivation::ReLU(ReLU::new()))
        } else if self.use_gelu {
            Some(Conv2dBlockActivation::GELU(GELU::new()))
        } else if self.use_silu {
            Some(Conv2dBlockActivation::SiLU(SiLU::new()))
        } else if self.use_tanh {
            Some(Conv2dBlockActivation::Tanh(Tanh::new()))
        } else if self.use_sigmoid {
            Some(Conv2dBlockActivation::Sigmoid(Sigmoid::new()))
        } else {
            None
        };

        let pool = if self.use_max_pool {
            Some(Conv2dBlockPool::MaxPool2d(MaxPool2d::new(
                self.pool_kernel_size,
                Some(self.pool_stride),
            )?))
        } else if self.use_avg_pool {
            Some(Conv2dBlockPool::AvgPool2d(AvgPool2d::new(
                self.pool_kernel_size,
                Some(self.pool_stride),
            )?))
        } else {
            None
        };

        let id = ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Ok(Conv2dBlock {
            conv,
            batch_norm2d,
            activation,
            pool,
            id,
        })
    }
}

/// Conv2d block activation layer enum.
///
/// # Variants
/// * `ReLU` - Rectified Linear Unit activation layer.
/// * `GELU` - Gaussian Error Linear Unit activation layer.
/// * `SiLU` - Sigmoid Linear Unit activation layer.
/// * `Tanh` - Hyperbolic tangent activation layer.
/// * `Sigmoid` - Sigmoid activation layer.
#[derive(Debug, Clone)]
enum Conv2dBlockActivation {
    ReLU(ReLU),
    GELU(GELU),
    SiLU(SiLU),
    Tanh(Tanh),
    Sigmoid(Sigmoid),
}

impl Conv2dBlockActivation {
    fn forward(&mut self, input: Tensor) -> Result<Tensor, ModelError> {
        match self {
            Conv2dBlockActivation::ReLU(layer) => layer.forward(input),
            Conv2dBlockActivation::GELU(layer) => layer.forward(input),
            Conv2dBlockActivation::SiLU(layer) => layer.forward(input),
            Conv2dBlockActivation::Tanh(layer) => layer.forward(input),
            Conv2dBlockActivation::Sigmoid(layer) => layer.forward(input),
        }
    }

    fn require_grad(&mut self, grad_enabled: bool) -> Result<(), ModelError> {
        match self {
            Conv2dBlockActivation::ReLU(layer) => layer.require_grad(grad_enabled),
            Conv2dBlockActivation::GELU(layer) => layer.require_grad(grad_enabled),
            Conv2dBlockActivation::SiLU(layer) => layer.require_grad(grad_enabled),
            Conv2dBlockActivation::Tanh(layer) => layer.require_grad(grad_enabled),
            Conv2dBlockActivation::Sigmoid(layer) => layer.require_grad(grad_enabled),
        }
    }

    fn to_device(&mut self, device: &Device) -> Result<(), ModelError> {
        match self {
            Conv2dBlockActivation::ReLU(layer) => layer.to_device(device),
            Conv2dBlockActivation::GELU(layer) => layer.to_device(device),
            Conv2dBlockActivation::SiLU(layer) => layer.to_device(device),
            Conv2dBlockActivation::Tanh(layer) => layer.to_device(device),
            Conv2dBlockActivation::Sigmoid(layer) => layer.to_device(device),
        }
    }

    fn to_dtype(&mut self, dtype: &DType) -> Result<(), ModelError> {
        match self {
            Conv2dBlockActivation::ReLU(layer) => layer.to_dtype(dtype),
            Conv2dBlockActivation::GELU(layer) => layer.to_dtype(dtype),
            Conv2dBlockActivation::SiLU(layer) => layer.to_dtype(dtype),
            Conv2dBlockActivation::Tanh(layer) => layer.to_dtype(dtype),
            Conv2dBlockActivation::Sigmoid(layer) => layer.to_dtype(dtype),
        }
    }

    fn parameters(&self) -> Result<Vec<Tensor>, ModelError> {
        match self {
            Conv2dBlockActivation::ReLU(layer) => layer.parameters(),
            Conv2dBlockActivation::GELU(layer) => layer.parameters(),
            Conv2dBlockActivation::SiLU(layer) => layer.parameters(),
            Conv2dBlockActivation::Tanh(layer) => layer.parameters(),
            Conv2dBlockActivation::Sigmoid(layer) => layer.parameters(),
        }
    }

    fn named_parameters(&self) -> Result<HashMap<String, Tensor>, ModelError> {
        match self {
            Conv2dBlockActivation::ReLU(layer) => layer.named_parameters(),
            Conv2dBlockActivation::GELU(layer) => layer.named_parameters(),
            Conv2dBlockActivation::SiLU(layer) => layer.named_parameters(),
            Conv2dBlockActivation::Tanh(layer) => layer.named_parameters(),
            Conv2dBlockActivation::Sigmoid(layer) => layer.named_parameters(),
        }
    }
}

impl Display for Conv2dBlockActivation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Conv2dBlockActivation::ReLU(layer) => write!(f, "{}", layer),
            Conv2dBlockActivation::GELU(layer) => write!(f, "{}", layer),
            Conv2dBlockActivation::SiLU(layer) => write!(f, "{}", layer),
            Conv2dBlockActivation::Tanh(layer) => write!(f, "{}", layer),
            Conv2dBlockActivation::Sigmoid(layer) => write!(f, "{}", layer),
        }
    }
}

/// Conv2d block pool layer enum.
///
/// # Variants
/// * `MaxPool2d` - 2D max pooling layer.
/// * `AvgPool2d` - 2D average pooling layer.
#[derive(Debug, Clone)]
enum Conv2dBlockPool {
    MaxPool2d(MaxPool2d),
    AvgPool2d(AvgPool2d),
}

impl Conv2dBlockPool {
    fn forward(&mut self, input: Tensor) -> Result<Tensor, ModelError> {
        match self {
            Conv2dBlockPool::MaxPool2d(layer) => layer.forward(input),
            Conv2dBlockPool::AvgPool2d(layer) => layer.forward(input),
        }
    }

    fn require_grad(&mut self, grad_enabled: bool) -> Result<(), ModelError> {
        match self {
            Conv2dBlockPool::MaxPool2d(layer) => layer.require_grad(grad_enabled),
            Conv2dBlockPool::AvgPool2d(layer) => layer.require_grad(grad_enabled),
        }
    }

    fn to_device(&mut self, device: &Device) -> Result<(), ModelError> {
        match self {
            Conv2dBlockPool::MaxPool2d(layer) => layer.to_device(device),
            Conv2dBlockPool::AvgPool2d(layer) => layer.to_device(device),
        }
    }

    fn to_dtype(&mut self, dtype: &DType) -> Result<(), ModelError> {
        match self {
            Conv2dBlockPool::MaxPool2d(layer) => layer.to_dtype(dtype),
            Conv2dBlockPool::AvgPool2d(layer) => layer.to_dtype(dtype),
        }
    }

    fn parameters(&self) -> Result<Vec<Tensor>, ModelError> {
        match self {
            Conv2dBlockPool::MaxPool2d(layer) => layer.parameters(),
            Conv2dBlockPool::AvgPool2d(layer) => layer.parameters(),
        }
    }

    fn named_parameters(&self) -> Result<HashMap<String, Tensor>, ModelError> {
        match self {
            Conv2dBlockPool::MaxPool2d(layer) => layer.named_parameters(),
            Conv2dBlockPool::AvgPool2d(layer) => layer.named_parameters(),
        }
    }
}

impl Display for Conv2dBlockPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Conv2dBlockPool::MaxPool2d(layer) => write!(f, "{}", layer),
            Conv2dBlockPool::AvgPool2d(layer) => write!(f, "{}", layer),
        }
    }
}

/// Conv2d block.
///
/// # Notes
/// * The `Conv2dBlock` is a sequential block that chains a convolutional layer with optional
///   batch normalization (2D), activation function, and pooling.
///
/// # Fields
/// * `conv` - The convolutional layer.
/// * `batch_norm2d` - Optional 2D batch normalization layer.
/// * `activation` - Optional activation function.
/// * `pool` - Optional pooling layer.
/// * `id` - The unique ID of the conv2d block.
///
/// # Examples
/// ```no_run
/// use nove::model::layer::Conv2dBlockBuilder;
///
/// let mut block = Conv2dBlockBuilder::new(1, 16, (3, 3), 1, 1)
///     .with_gelu()
///     .with_batch_norm2d()
///     .with_avg_pool((2, 2), (2, 2))
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct Conv2dBlock {
    conv: Conv2d,
    batch_norm2d: Option<BatchNorm2d>,
    activation: Option<Conv2dBlockActivation>,
    pool: Option<Conv2dBlockPool>,
    id: usize,
}

impl Model for Conv2dBlock {
    type Input = (Tensor, bool);
    type Output = Tensor;

    /// Apply the conv2d block to the input tensor.
    ///
    /// # Arguments
    /// * `input: (Tensor, bool)` - A tuple containing the input tensor and a boolean flag
    ///   indicating whether the block is in training mode. This is used by BatchNorm2d.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The output tensor if successful.
    /// * `Err(ModelError)` - The error when applying the conv2d block to the input tensor.
    fn forward(&mut self, input: Self::Input) -> Result<Self::Output, ModelError> {
        let (input, training) = input;
        let mut output = self.conv.forward(input)?;

        if let Some(ref mut bn) = self.batch_norm2d {
            output = bn.forward((output, training))?;
        }

        if let Some(ref mut activation) = self.activation {
            output = activation.forward(output)?;
        }

        if let Some(ref mut pool) = self.pool {
            output = pool.forward(output)?;
        }

        Ok(output)
    }

    fn require_grad(&mut self, grad_enabled: bool) -> Result<(), ModelError> {
        self.conv.require_grad(grad_enabled)?;
        if let Some(ref mut bn) = self.batch_norm2d {
            bn.require_grad(grad_enabled)?;
        }
        if let Some(ref mut activation) = self.activation {
            activation.require_grad(grad_enabled)?;
        }
        if let Some(ref mut pool) = self.pool {
            pool.require_grad(grad_enabled)?;
        }
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<(), ModelError> {
        self.conv.to_device(device)?;
        if let Some(ref mut bn) = self.batch_norm2d {
            bn.to_device(device)?;
        }
        if let Some(ref mut activation) = self.activation {
            activation.to_device(device)?;
        }
        if let Some(ref mut pool) = self.pool {
            pool.to_device(device)?;
        }
        Ok(())
    }

    fn to_dtype(&mut self, dtype: &DType) -> Result<(), ModelError> {
        self.conv.to_dtype(dtype)?;
        if let Some(ref mut bn) = self.batch_norm2d {
            bn.to_dtype(dtype)?;
        }
        if let Some(ref mut activation) = self.activation {
            activation.to_dtype(dtype)?;
        }
        if let Some(ref mut pool) = self.pool {
            pool.to_dtype(dtype)?;
        }
        Ok(())
    }

    fn parameters(&self) -> Result<Vec<Tensor>, ModelError> {
        let mut params = Vec::new();
        params.extend(self.conv.parameters()?);
        if let Some(ref bn) = self.batch_norm2d {
            params.extend(bn.parameters()?);
        }
        if let Some(ref activation) = self.activation {
            params.extend(activation.parameters()?);
        }
        if let Some(ref pool) = self.pool {
            params.extend(pool.parameters()?);
        }
        Ok(params)
    }

    fn named_parameters(&self) -> Result<HashMap<String, Tensor>, ModelError> {
        let mut params = HashMap::new();
        let prefix = format!("conv2d_block{}.", self.id);
        for (name, tensor) in self.conv.named_parameters()? {
            params.insert(format!("{}{}", prefix, name), tensor);
        }
        if let Some(ref bn) = self.batch_norm2d {
            for (name, tensor) in bn.named_parameters()? {
                params.insert(format!("{}{}", prefix, name), tensor);
            }
        }
        if let Some(ref activation) = self.activation {
            for (name, tensor) in activation.named_parameters()? {
                params.insert(format!("{}{}", prefix, name), tensor);
            }
        }
        if let Some(ref pool) = self.pool {
            for (name, tensor) in pool.named_parameters()? {
                params.insert(format!("{}{}", prefix, name), tensor);
            }
        }
        Ok(params)
    }
}

impl Display for Conv2dBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "conv2d_block.{}(\n", self.id)?;
        write!(f, "  {},\n", self.conv)?;
        if let Some(ref bn) = self.batch_norm2d {
            write!(f, "  {},\n", bn)?;
        }
        if let Some(ref activation) = self.activation {
            write!(f, "  {},\n", activation)?;
        }
        if let Some(ref pool) = self.pool {
            write!(f, "  {},\n", pool)?;
        }
        write!(f, ")")
    }
}
