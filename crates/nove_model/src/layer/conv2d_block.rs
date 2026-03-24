use std::{collections::HashMap, fmt::Display};

use nove_tensor::{DType, Device, Tensor};

use crate::{Model, ModelError};

use super::{Activation, BatchNorm2d, BatchNorm2dBuilder, Conv2d, Conv2dBuilder, Pool2d};

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
///   When you call `with_activation()` method, it replaces any previously configured activation.
/// * **Important**: Only one pooling type can be enabled at a time.
///   When you call `with_pool2d()` method, it replaces any previously configured pooling.
///
/// # Required Arguments
/// * `in_channels` - The number of input channels.
/// * `out_channels` - The number of output channels.
/// * `kernel_size` - The size of the convolution kernel (height, width).
/// * `stride` - The stride of the convolution.
/// * `padding` - The padding size.
///
/// # Optional Arguments
/// * `activation` - Optional activation function after convolution. Default is `None`. (configured via `with_activation()`)
/// * `batch_norm2d` - Whether 2D batch normalization is enabled after convolution. Default is `false`. (configured via `with_batch_norm2d()`)
/// * `pool2d` - Optional pooling layer after activation/BatchNorm2d. Default is `None`. (configured via `with_pool2d()`)
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
/// * `activation` - Optional activation function after convolution.
/// * `use_batch_norm2d` - Whether 2D batch normalization is enabled after convolution.
/// * `pool2d` - Optional pooling layer after activation/BatchNorm2d.
/// * `device` - The device to use for the layer.
/// * `dtype` - The data type to use for the layer.
/// * `grad_enabled` - Whether to enable the gradient computation.
///
/// # Examples
/// ```no_run
/// use nove::model::layer::{Conv2dBlockBuilder, Activation, Pool2d};
/// use nove::tensor::{Device, DType};
///
/// let block = Conv2dBlockBuilder::new(1, 16, (3, 3), 1, 1)
///     .with_activation(Activation::gelu())
///     .with_batch_norm2d()
///     .with_pool2d(Pool2d::avg_pool2d((2, 2), None).unwrap())
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
    activation: Option<Activation>,
    use_batch_norm2d: bool,
    pool2d: Option<Pool2d>,
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
            activation: None,
            use_batch_norm2d: false,
            pool2d: None,
            device: Device::cpu(),
            dtype: DType::F32,
            grad_enabled: true,
        }
    }

    /// Configure the number of input channels.
    ///
    /// # Arguments
    /// * `in_channels` - The number of input channels.
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured input channels.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::Conv2dBlockBuilder;
    /// let mut builder = Conv2dBlockBuilder::new(1, 16, (3, 3), 1, 1);
    /// builder.in_channels(3);
    /// ```
    pub fn in_channels(&mut self, in_channels: usize) -> &mut Self {
        self.in_channels = in_channels;
        self
    }

    /// Configure the number of output channels.
    ///
    /// # Arguments
    /// * `out_channels` - The number of output channels.
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured output channels.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::Conv2dBlockBuilder;
    /// let mut builder = Conv2dBlockBuilder::new(1, 16, (3, 3), 1, 1);
    /// builder.out_channels(64);
    /// ```
    pub fn out_channels(&mut self, out_channels: usize) -> &mut Self {
        self.out_channels = out_channels;
        self
    }

    /// Configure the size of the convolution kernel.
    ///
    /// # Arguments
    /// * `kernel_size` - The size of the convolution kernel (height, width).
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured kernel size.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::Conv2dBlockBuilder;
    /// let mut builder = Conv2dBlockBuilder::new(1, 16, (3, 3), 1, 1);
    /// builder.kernel_size((5, 5));
    /// ```
    pub fn kernel_size(&mut self, kernel_size: (usize, usize)) -> &mut Self {
        self.kernel_size = kernel_size;
        self
    }

    /// Configure the stride of the convolution.
    ///
    /// # Arguments
    /// * `stride` - The stride of the convolution.
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured stride.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::Conv2dBlockBuilder;
    /// let mut builder = Conv2dBlockBuilder::new(1, 16, (3, 3), 1, 1);
    /// builder.stride(2);
    /// ```
    pub fn stride(&mut self, stride: usize) -> &mut Self {
        self.stride = stride;
        self
    }

    /// Configure the padding of the convolution.
    ///
    /// # Arguments
    /// * `padding` - The padding size.
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured padding.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::Conv2dBlockBuilder;
    /// let mut builder = Conv2dBlockBuilder::new(1, 16, (3, 3), 1, 1);
    /// builder.padding(2);
    /// ```
    pub fn padding(&mut self, padding: usize) -> &mut Self {
        self.padding = padding;
        self
    }

    /// Configure activation function after convolution.
    ///
    /// # Notes
    /// * When this method is called, it replaces any previously configured activation.
    /// * Only one activation function can be active at a time.
    ///
    /// # Arguments
    /// * `activation` - The activation function to use.
    ///
    /// # Returns
    /// * `Self` - The conv2d block builder with the specified activation function.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::{Conv2dBlockBuilder, Activation};
    /// let builder = Conv2dBlockBuilder::new(1, 16, (3, 3), 1, 1)
    ///     .with_activation(Activation::relu());
    /// ```
    pub fn with_activation(mut self, activation: Activation) -> Self {
        self.activation = Some(activation);
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

    /// Configure pooling layer after activation/BatchNorm2d.
    ///
    /// # Notes
    /// * When this method is called, it replaces any previously configured pooling layer.
    /// * Only one pooling type can be active at a time.
    ///
    /// # Arguments
    /// * `pool2d` - The pooling layer to use.
    ///
    /// # Returns
    /// * `Self` - The conv2d block builder with the specified pooling layer.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::{Conv2dBlockBuilder, Pool2d};
    /// let builder = Conv2dBlockBuilder::new(1, 16, (3, 3), 1, 1)
    ///     .with_pool2d(Pool2d::max_pool2d((2, 2), None).unwrap());
    /// ```
    pub fn with_pool2d(mut self, pool2d: Pool2d) -> Self {
        self.pool2d = Some(pool2d);
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
    /// use nove::model::layer::Activation;
    /// let builder = Conv2dBlockBuilder::new(1, 16, (3, 3), 1, 1)
    ///     .with_activation(Activation::relu())
    ///     .without_activation();
    /// ```
    pub fn without_activation(mut self) -> Self {
        self.activation = None;
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
    /// use nove::model::layer::{Conv2dBlockBuilder, Pool2d};
    /// let builder = Conv2dBlockBuilder::new(1, 16, (3, 3), 1, 1)
    ///     .with_pool2d(Pool2d::max_pool2d((2, 2), None).unwrap())
    ///     .without_pool2d();
    /// ```
    pub fn without_pool2d(mut self) -> Self {
        self.pool2d = None;
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
        let conv = Conv2dBuilder::new(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
        )
        .stride(self.stride)
        .padding(self.padding)
        .device(self.device.clone())
        .dtype(self.dtype)
        .grad_enabled(self.grad_enabled)
        .build()?;

        let batch_norm2d = if self.use_batch_norm2d {
            Some(
                BatchNorm2dBuilder::new(self.out_channels)
                    .device(self.device.clone())
                    .dtype(self.dtype)
                    .build()?,
            )
        } else {
            None
        };

        let activation = self.activation;
        let pool2d = self.pool2d;

        let id = ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Ok(Conv2dBlock {
            conv,
            batch_norm2d,
            activation,
            pool2d,
            id,
        })
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
/// * `pool2d` - Optional pooling layer.
/// * `id` - The unique ID of the conv2d block.
///
/// # Examples
/// ```no_run
/// use nove::model::layer::{Conv2dBlockBuilder, Activation, Pool2d};
///
/// let mut block = Conv2dBlockBuilder::new(1, 16, (3, 3), 1, 1)
///     .with_activation(Activation::gelu())
///     .with_batch_norm2d()
///     .with_pool2d(Pool2d::avg_pool2d((2, 2), None).unwrap())
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct Conv2dBlock {
    conv: Conv2d,
    batch_norm2d: Option<BatchNorm2d>,
    activation: Option<Activation>,
    pool2d: Option<Pool2d>,
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

        if let Some(ref mut pool2d) = self.pool2d {
            output = pool2d.forward(output)?;
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
        if let Some(ref mut pool2d) = self.pool2d {
            pool2d.require_grad(grad_enabled)?;
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
        if let Some(ref mut pool2d) = self.pool2d {
            pool2d.to_device(device)?;
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
        if let Some(ref mut pool2d) = self.pool2d {
            pool2d.to_dtype(dtype)?;
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
        if let Some(ref pool) = self.pool2d {
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
        if let Some(ref pool) = self.pool2d {
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
        if let Some(ref pool) = self.pool2d {
            write!(f, "  {},\n", pool)?;
        }
        write!(f, ")")
    }
}
