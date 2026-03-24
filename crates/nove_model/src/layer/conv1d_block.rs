use std::{collections::HashMap, fmt::Display};

use nove_tensor::{DType, Device, Tensor};

use crate::{Model, ModelError};

use super::{
    BatchNorm1d, BatchNorm1dBuilder, Conv1d, Conv1dBuilder, activation::Activation, pool::Pool1d,
};

static ID: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Conv1d block configuration.
///
/// # Notes
/// * The `Conv1dBlockBuilder` is used to configure a convolutional layer with optional
///   activation function, batch normalization (1D), and pooling.
/// * The kernel size, stride, and padding must be specified when creating a block.
/// * The default configuration uses activation function disabled, batch normalization (1D) disabled,
///   and pooling disabled.
/// * **Important**: Only one activation function can be enabled at a time.
/// * **Important**: Only one pooling type can be enabled at a time.
///
/// # Required Arguments
/// * `in_channels` - The number of input channels.
/// * `out_channels` - The number of output channels.
/// * `kernel_size` - The size of the convolution kernel (length).
/// * `stride` - The stride of the convolution.
/// * `padding` - The padding size.
///
/// # Optional Arguments
/// * `use_batch_norm1d` - Whether 1D batch normalization is enabled after convolution. Default is `false`. (configured via `with_batch_norm1d()`)
/// * `activation` - The activation function to use after convolution. Default is `None`. (configured via `with_activation()`)
/// * `pool1d` - The pooling layer to use after activation/BatchNorm1d. Default is `None`. (configured via `with_pool1d()`)
/// * `device` - The device to use for the layer. Default is `Device::cpu()`.
/// * `dtype` - The data type to use for the layer. Default is `DType::F32`.
/// * `grad_enabled` - Whether to enable the gradient computation. Default is `true`.
///
/// # Fields
/// * `in_channels` - The number of input channels.
/// * `out_channels` - The number of output channels.
/// * `kernel_size` - The size of the convolution kernel (length).
/// * `stride` - The stride of the convolution.
/// * `padding` - The padding size.
/// * `use_batch_norm1d` - Whether 1D batch normalization is enabled after convolution.
/// * `activation` - The optional activation function.
/// * `pool1d` - The optional pooling layer.
/// * `device` - The device to use for the layer.
/// * `dtype` - The data type to use for the layer.
/// * `grad_enabled` - Whether to enable the gradient computation.
///
/// # Examples
/// ```no_run
/// use nove::model::layer::{Conv1dBlockBuilder, Activation, Pool1d};
/// use nove::tensor::{Device, DType};
///
/// let block = Conv1dBlockBuilder::new(1, 16, 3, 1, 1)
///     .with_activation(Activation::relu())
///     .with_batch_norm1d()
///     .with_pool1d(Pool1d::max_pool1d(2, None).unwrap())
///     .device(Device::cpu())
///     .dtype(DType::F32)
///     .grad_enabled(true);
/// ```
#[derive(Debug, Clone)]
pub struct Conv1dBlockBuilder {
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    use_batch_norm1d: bool,
    activation: Option<Activation>,
    pool1d: Option<Pool1d>,
    device: Device,
    dtype: DType,
    grad_enabled: bool,
}

impl Conv1dBlockBuilder {
    /// Create a new conv1d block builder with specified settings.
    ///
    /// # Arguments
    /// * `in_channels` - The number of input channels.
    /// * `out_channels` - The number of output channels.
    /// * `kernel_size` - The size of the convolution kernel (length).
    /// * `stride` - The stride of the convolution.
    /// * `padding` - The padding size.
    ///
    /// # Returns
    /// * `Conv1dBlockBuilder` - The new conv1d block builder.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::Conv1dBlockBuilder;
    /// let builder = Conv1dBlockBuilder::new(1, 16, 3, 1, 1);
    /// ```
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            use_batch_norm1d: false,
            activation: None,
            pool1d: None,
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
    /// use nove::model::layer::Conv1dBlockBuilder;
    /// let mut builder = Conv1dBlockBuilder::new(1, 16, 3, 1, 1);
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
    /// use nove::model::layer::Conv1dBlockBuilder;
    /// let mut builder = Conv1dBlockBuilder::new(1, 16, 3, 1, 1);
    /// builder.out_channels(64);
    /// ```
    pub fn out_channels(&mut self, out_channels: usize) -> &mut Self {
        self.out_channels = out_channels;
        self
    }

    /// Configure the size of the convolution kernel.
    ///
    /// # Arguments
    /// * `kernel_size` - The size of the convolution kernel (length).
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured kernel size.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::Conv1dBlockBuilder;
    /// let mut builder = Conv1dBlockBuilder::new(1, 16, 3, 1, 1);
    /// builder.kernel_size(5);
    /// ```
    pub fn kernel_size(&mut self, kernel_size: usize) -> &mut Self {
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
    /// use nove::model::layer::Conv1dBlockBuilder;
    /// let mut builder = Conv1dBlockBuilder::new(1, 16, 3, 1, 1);
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
    /// use nove::model::layer::Conv1dBlockBuilder;
    /// let mut builder = Conv1dBlockBuilder::new(1, 16, 3, 1, 1);
    /// builder.padding(2);
    /// ```
    pub fn padding(&mut self, padding: usize) -> &mut Self {
        self.padding = padding;
        self
    }

    /// Configure no activation function after convolution.
    ///
    /// # Returns
    /// * `Self` - The conv1d block builder with activation function disabled.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::Conv1dBlockBuilder;
    /// let builder = Conv1dBlockBuilder::new(1, 16, 3, 1, 1).without_activation();
    /// ```
    pub fn without_activation(mut self) -> Self {
        self.activation = None;
        self
    }

    /// Configure 1D batch normalization after convolution.
    ///
    /// # Returns
    /// * `Self` - The conv1d block builder with 1D batch normalization enabled.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::Conv1dBlockBuilder;
    /// let builder = Conv1dBlockBuilder::new(1, 16, 3, 1, 1).with_batch_norm1d();
    /// ```
    pub fn with_batch_norm1d(mut self) -> Self {
        self.use_batch_norm1d = true;
        self
    }

    /// Configure no batch normalization after convolution.
    ///
    /// # Returns
    /// * `Self` - The conv1d block builder with 1D batch normalization disabled.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::Conv1dBlockBuilder;
    /// let builder = Conv1dBlockBuilder::new(1, 16, 3, 1, 1).without_batch_norm1d();
    /// ```
    pub fn without_batch_norm1d(mut self) -> Self {
        self.use_batch_norm1d = false;
        self
    }

    /// Configure an activation function after convolution.
    ///
    /// # Arguments
    /// * `activation` - The activation function to use.
    ///
    /// # Notes
    /// * Only one activation function can be active at a time. Calling this method
    ///   replaces any previously configured activation function.
    ///
    /// # Returns
    /// * `Self` - The conv1d block builder with the specified activation function.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::{Conv1dBlockBuilder, Activation};
    /// let builder = Conv1dBlockBuilder::new(1, 16, 3, 1, 1).with_activation(Activation::relu());
    /// ```
    pub fn with_activation(mut self, activation: Activation) -> Self {
        self.activation = Some(activation);
        self
    }

    /// Configure a pooling layer after activation/BatchNorm1d.
    ///
    /// # Arguments
    /// * `pool1d` - The pooling layer to use.
    ///
    /// # Notes
    /// * Only one pooling layer can be active at a time. Calling this method
    ///   replaces any previously configured pooling layer.
    ///
    /// # Returns
    /// * `Self` - The conv1d block builder with the specified pooling layer.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::{Conv1dBlockBuilder, Pool1d};
    /// let builder = Conv1dBlockBuilder::new(1, 16, 3, 1, 1)
    ///     .with_pool1d(Pool1d::max_pool1d(2, None).unwrap());
    /// ```
    pub fn with_pool1d(mut self, pool1d: Pool1d) -> Self {
        self.pool1d = Some(pool1d);
        self
    }

    /// Configure no pooling after activation/BatchNorm1d.
    ///
    /// # Returns
    /// * `Self` - The conv1d block builder with pooling disabled.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::Conv1dBlockBuilder;
    /// let builder = Conv1dBlockBuilder::new(1, 16, 3, 1, 1).without_pool1d();
    /// ```
    pub fn without_pool1d(mut self) -> Self {
        self.pool1d = None;
        self
    }

    /// Set the device for the conv1d block.
    ///
    /// # Arguments
    /// * `device` - The device to use for the layer.
    ///
    /// # Returns
    /// * `Self` - The conv1d block builder with the specified device.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::Conv1dBlockBuilder;
    /// use nove::tensor::Device;
    /// let builder = Conv1dBlockBuilder::new(1, 16, 3, 1, 1).device(Device::cpu());
    /// ```
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Set the data type for the conv1d block.
    ///
    /// # Arguments
    /// * `dtype` - The data type to use for the layer.
    ///
    /// # Returns
    /// * `Self` - The conv1d block builder with the specified data type.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::Conv1dBlockBuilder;
    /// use nove::tensor::DType;
    /// let builder = Conv1dBlockBuilder::new(1, 16, 3, 1, 1).dtype(DType::F32);
    /// ```
    pub fn dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }

    /// Set whether to enable gradient computation for the conv1d block.
    ///
    /// # Arguments
    /// * `grad_enabled` - Whether to enable the gradient computation.
    ///
    /// # Returns
    /// * `Self` - The conv1d block builder with the specified gradient computation setting.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::Conv1dBlockBuilder;
    /// let builder = Conv1dBlockBuilder::new(1, 16, 3, 1, 1).grad_enabled(true);
    /// ```
    pub fn grad_enabled(mut self, grad_enabled: bool) -> Self {
        self.grad_enabled = grad_enabled;
        self
    }

    /// Build the conv1d block with the current configuration.
    ///
    /// # Returns
    /// * `Ok(Conv1dBlock)` - The new conv1d block if successful.
    /// * `Err(ModelError)` - The error when building the conv1d block.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::Conv1dBlockBuilder;
    /// let block = Conv1dBlockBuilder::new(1, 16, 3, 1, 1).build().unwrap();
    /// ```
    pub fn build(self) -> Result<Conv1dBlock, ModelError> {
        if self.in_channels == 0 {
            return Err(ModelError::InvalidArgument(format!(
                "in_channels in Conv1dBlock must be greater than 0"
            )));
        }
        if self.out_channels == 0 {
            return Err(ModelError::InvalidArgument(format!(
                "out_channels in Conv1dBlock must be greater than 0"
            )));
        }
        if self.kernel_size == 0 {
            return Err(ModelError::InvalidArgument(format!(
                "kernel_size in Conv1dBlock must be greater than 0"
            )));
        }
        if self.stride == 0 {
            return Err(ModelError::InvalidArgument(format!(
                "stride in Conv1dBlock must be greater than 0"
            )));
        }

        let id = ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let conv = Conv1dBuilder::new(
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

        let batch_norm1d = if self.use_batch_norm1d {
            Some(
                BatchNorm1dBuilder::new(self.out_channels)
                    .device(self.device.clone())
                    .dtype(self.dtype)
                    .build()?,
            )
        } else {
            None
        };

        Ok(Conv1dBlock {
            conv,
            batch_norm1d,
            activation: self.activation,
            pool1d: self.pool1d,
            id,
        })
    }
}

/// 1D convolutional block.
///
/// # Notes
/// * The `Conv1dBlock` is built by the `Conv1dBlockBuilder`.
/// * It applies the following operations in order:
///   1. 1D convolution
///   2. Optional 1D batch normalization
///   3. Optional activation function (ReLU, GELU, SiLU, Tanh, or Sigmoid)
///   4. Optional pooling (max pooling 1D or average pooling 1D)
/// * During training, the batch normalization uses batch statistics; during inference,
///   it uses running statistics.
///
/// # Fields
/// * `conv` - The 1D convolutional layer.
/// * `batch_norm1d` - The optional 1D batch normalization layer.
/// * `activation` - The optional activation function.
/// * `pool1d` - The optional pooling layer.
/// * `id` - The unique ID of the conv1d block.
///
/// # Examples
/// ```no_run
/// use nove::model::layer::Conv1dBlockBuilder;
/// let block = Conv1dBlockBuilder::new(1, 16, 3, 1, 1).build().unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct Conv1dBlock {
    conv: Conv1d,
    batch_norm1d: Option<BatchNorm1d>,
    activation: Option<Activation>,
    pool1d: Option<Pool1d>,
    id: usize,
}

impl Model for Conv1dBlock {
    type Input = (Tensor, bool);
    type Output = Tensor;

    /// Apply the 1D convolutional block to the input tensor.
    ///
    /// # Arguments
    /// * `input: (Tensor, bool)` - The input tensor with shape [batch_size, channels, length] and a boolean flag indicating training mode.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The output tensor if successful.
    /// * `Err(ModelError)` - The error when applying the conv1d block to the input tensor.
    fn forward(&mut self, input: Self::Input) -> Result<Self::Output, ModelError> {
        let (x, is_training) = input;
        let mut y = self.conv.forward(x)?;

        if let Some(ref mut bn) = self.batch_norm1d {
            y = bn.forward((y, is_training))?;
        }

        if let Some(ref mut activation) = self.activation {
            y = activation.forward(y)?;
        }

        if let Some(ref mut pool1d) = self.pool1d {
            y = pool1d.forward(y)?;
        }

        Ok(y)
    }

    fn require_grad(&mut self, grad_enabled: bool) -> Result<(), ModelError> {
        self.conv.require_grad(grad_enabled)?;
        if let Some(ref mut bn) = self.batch_norm1d {
            bn.require_grad(grad_enabled)?;
        }
        if let Some(ref mut activation) = self.activation {
            activation.require_grad(grad_enabled)?;
        }
        if let Some(ref mut pool1d) = self.pool1d {
            pool1d.require_grad(grad_enabled)?;
        }
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<(), ModelError> {
        self.conv.to_device(device)?;
        if let Some(ref mut bn) = self.batch_norm1d {
            bn.to_device(device)?;
        }
        if let Some(ref mut activation) = self.activation {
            activation.to_device(device)?;
        }
        if let Some(ref mut pool1d) = self.pool1d {
            pool1d.to_device(device)?;
        }
        Ok(())
    }

    fn to_dtype(&mut self, dtype: &DType) -> Result<(), ModelError> {
        self.conv.to_dtype(dtype)?;
        if let Some(ref mut bn) = self.batch_norm1d {
            bn.to_dtype(dtype)?;
        }
        if let Some(ref mut activation) = self.activation {
            activation.to_dtype(dtype)?;
        }
        if let Some(ref mut pool1d) = self.pool1d {
            pool1d.to_dtype(dtype)?;
        }
        Ok(())
    }

    fn parameters(&self) -> Result<Vec<Tensor>, ModelError> {
        let mut params = self.conv.parameters()?;
        if let Some(ref bn) = self.batch_norm1d {
            params.extend(bn.parameters()?);
        }
        if let Some(ref activation) = self.activation {
            params.extend(activation.parameters()?);
        }
        if let Some(ref pool) = self.pool1d {
            params.extend(pool.parameters()?);
        }
        Ok(params)
    }

    fn named_parameters(&self) -> Result<HashMap<String, Tensor>, ModelError> {
        let mut params = self.conv.named_parameters()?;
        if let Some(ref bn) = self.batch_norm1d {
            for (k, v) in bn.named_parameters()? {
                params.insert(k, v);
            }
        }
        if let Some(ref activation) = self.activation {
            for (k, v) in activation.named_parameters()? {
                params.insert(k, v);
            }
        }
        if let Some(ref pool) = self.pool1d {
            for (k, v) in pool.named_parameters()? {
                params.insert(k, v);
            }
        }
        Ok(params)
    }
}

impl Display for Conv1dBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "conv1d_block.{}(\n", self.id)?;
        write!(f, "  {},\n", self.conv)?;
        if let Some(ref bn) = self.batch_norm1d {
            write!(f, "  {},\n", bn)?;
        }
        if let Some(ref activation) = self.activation {
            write!(f, "  {},\n", activation)?;
        }
        if let Some(ref pool) = self.pool1d {
            write!(f, "  {},\n", pool)?;
        }
        write!(f, ")")
    }
}
