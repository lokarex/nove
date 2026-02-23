use std::{
    collections::HashMap,
    fmt::Display,
    sync::atomic::{AtomicUsize, Ordering},
};

use nove_tensor::{DType, Device, Shape, Tensor};

use crate::{Model, ModelError};

static ID: AtomicUsize = AtomicUsize::new(0);

/// 2D convolution layer.
///
/// # Notes
/// * The `Conv2d` is now only created by the `Conv2dBuilder`.
///
/// # Fields
/// * `weight` - The weight tensor with shape [out_channels, in_channels/groups, kernel_height, kernel_width].
/// * `bias` - The bias tensor with shape \[out_channels\].
/// * `in_channels` - The number of input channels.
/// * `out_channels` - The number of output channels.
/// * `kernel_size` - The size of the convolution kernel (height, width).
/// * `padding` - The padding size.
/// * `stride` - The stride of the convolution.
/// * `dilation` - The dilation of the convolution.
/// * `groups` - The number of groups for grouped convolution.
/// * `id` - The unique ID of the convolution layer.
///
/// # Examples
/// ```
/// use nove::model::layer::Conv2dBuilder;
/// use nove::tensor::{Device, DType};
///
/// let conv = Conv2dBuilder::default()
///     .in_channels(3)           // Required
///     .out_channels(64)         // Required
///     .kernel_size((3, 3))      // Required
///     .padding(1)               // Optional, default is 0
///     .stride(1)                // Optional, default is 1
///     .dilation(1)              // Optional, default is 1
///     .groups(1)                // Optional, default is 1
///     .bias_enabled(true)       // Optional, default is true
///     .device(Device::cpu())    // Optional, default is cpu
///     .dtype(DType::F32)        // Optional, default is F32
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct Conv2d {
    weight: Tensor,
    bias: Option<Tensor>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize),
    padding: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
    id: usize,
}

impl Conv2d {
    /// Get the weight tensor in the convolution layer.
    ///
    /// # Returns
    /// * `Tensor` - The weight tensor with shape [out_channels, in_channels/groups, kernel_height, kernel_width].
    pub fn weight(&self) -> Tensor {
        self.weight.clone()
    }

    /// Get the bias tensor in the convolution layer.
    ///
    /// # Returns
    /// * `Option<Tensor>` - The bias tensor if enabled, otherwise None.
    pub fn bias(&self) -> Option<Tensor> {
        self.bias.clone()
    }
}

impl Model for Conv2d {
    type Input = Tensor;

    type Output = Tensor;

    /// Apply the 2D convolution layer to the input tensor.
    ///
    /// # Arguments
    /// * `input: Tensor` - The input tensor with shape [batch_size, in_channels, height, width].
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The output tensor with shape [batch_size, out_channels, out_height, out_width] if successful.
    /// * `Err(ModelError)` - The error when applying the convolution layer to the input tensor.
    fn forward(&mut self, input: Self::Input) -> Result<Self::Output, ModelError> {
        let y = input.conv2d(
            &self.weight,
            self.padding,
            self.stride,
            self.dilation,
            self.groups,
        )?;

        let y = if let Some(bias) = &self.bias {
            y.add(bias)?
        } else {
            y
        };

        Ok(y)
    }

    fn require_grad(&mut self, grad_enabled: bool) -> Result<(), ModelError> {
        self.weight = self.weight.require_grad(grad_enabled)?;
        if let Some(bias) = &mut self.bias {
            self.bias = Some(bias.require_grad(grad_enabled)?);
        }
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<(), ModelError> {
        self.weight = self.weight.to_device(device)?;
        if let Some(bias) = &mut self.bias {
            self.bias = Some(bias.to_device(device)?);
        }
        Ok(())
    }

    fn to_dtype(&mut self, dtype: &DType) -> Result<(), ModelError> {
        self.weight = self.weight.to_dtype(dtype)?;
        if let Some(bias) = &mut self.bias {
            self.bias = Some(bias.to_dtype(dtype)?);
        }
        Ok(())
    }

    fn parameters(&self) -> Result<Vec<Tensor>, ModelError> {
        match &self.bias {
            Some(bias) => Ok(vec![self.weight.clone(), bias.clone()]),
            None => Ok(vec![self.weight.clone()]),
        }
    }

    fn named_parameters(&self) -> Result<HashMap<String, Tensor>, ModelError> {
        Ok(self
            .parameters()?
            .into_iter()
            .map(|t| match t.name()? {
                Some(name) => Ok((name, t)),
                None => Err(ModelError::ParameterMissingName),
            })
            .collect::<Result<HashMap<_, _>, ModelError>>()?)
    }
}

impl Display for Conv2d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "conv2d.{}(in_channels={}, out_channels={}, kernel_size={:?}, stride={}, padding={}, dilation={}, groups={}, bias_enabled={})",
            self.id,
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.bias.is_some(),
        )
    }
}

/// The builder for the 2D convolution layer.
///
/// # Notes
/// * The `Conv2dBuilder` implements the `Default` trait, so you can
///   use `Conv2dBuilder::default()` to create a builder with default values.
/// * The `weight` tensor in the convolution layer is initialized with the Kaiming normal distribution(`fan_in=(in_channels / groups) * kernel_size.0 * kernel_size.1`,`mean=0.0`, `std=sqrt(2 / fan_in)`).
///   The `bias`` tensor is initialized with zeros(if enabled).
///
/// # Required Arguments
/// * `in_channels` - The number of input channels.
/// * `out_channels` - The number of output channels.
/// * `kernel_size` - The size of the convolution kernel (height, width).
///
/// # Optional Arguments
/// * `padding` - The padding size. Default is `0`.
/// * `stride` - The stride of the convolution. Default is `1`.
/// * `dilation` - The dilation of the convolution. Default is `1`.
/// * `groups` - The number of groups for grouped convolution. Default is `1`.
/// * `bias_enabled` - Whether to enable the bias term. Default is `true`.
/// * `device` - The device to use for the layer. Default is `Device::cpu()`.
/// * `dtype` - The data type to use for the layer. Default is `DType::F32`.
/// * `grad_enabled` - Whether to enable the gradient computation. Default is `true`.
///
/// # Fields
/// * `in_channels` - The number of input channels.
/// * `out_channels` - The number of output channels.
/// * `kernel_size` - The size of the convolution kernel (height, width).
/// * `padding` - The padding size.
/// * `stride` - The stride of the convolution.
/// * `dilation` - The dilation of the convolution.
/// * `groups` - The number of groups for grouped convolution.
/// * `bias_enabled` - Whether to enable the bias term.
/// * `device` - The device to use for the layer.
/// * `dtype` - The data type to use for the layer.
/// * `grad_enabled` - Whether to enable the gradient computation.
///
/// # Examples
/// ```
/// use nove::model::layer::Conv2dBuilder;
/// use nove::tensor::{Device, DType};
///
/// let conv = Conv2dBuilder::default()
///     .in_channels(3)           // Required
///     .out_channels(64)          // Required
///     .kernel_size((3, 3))       // Required
///     .padding(1)               // Optional, default is 0
///     .stride(1)                // Optional, default is 1
///     .dilation(1)              // Optional, default is 1
///     .groups(1)                // Optional, default is 1
///     .bias_enabled(true)       // Optional, default is true
///     .device(Device::cpu())    // Optional, default is cpu
///     .dtype(DType::F32)        // Optional, default is F32
///     .grad_enabled(true)       // Optional, default is true
///     .build();
/// ```
pub struct Conv2dBuilder {
    in_channels: Option<usize>,
    out_channels: Option<usize>,
    kernel_size: Option<(usize, usize)>,
    padding: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
    bias_enabled: bool,
    device: Device,
    dtype: DType,
    grad_enabled: bool,
}

impl Default for Conv2dBuilder {
    fn default() -> Self {
        Self {
            in_channels: None,
            out_channels: None,
            kernel_size: None,
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
            bias_enabled: true,
            device: Device::cpu(),
            dtype: DType::F32,
            grad_enabled: true,
        }
    }
}

impl Conv2dBuilder {
    /// Configure the number of input channels.
    ///
    /// # Arguments
    /// * `in_channels` - The number of input channels.
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured number of input channels.
    ///
    /// # Examples
    /// ```
    /// use nove::model::layer::Conv2dBuilder;
    /// let mut conv_builder = Conv2dBuilder::default();
    /// conv_builder.in_channels(3);
    /// ```
    pub fn in_channels(&mut self, in_channels: usize) -> &mut Self {
        self.in_channels = Some(in_channels);
        self
    }

    /// Configure the number of output channels.
    ///
    /// # Arguments
    /// * `out_channels` - The number of output channels.
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured number of output channels.
    ///
    /// # Examples
    /// ```
    /// use nove::model::layer::Conv2dBuilder;
    /// let mut conv_builder = Conv2dBuilder::default();
    /// conv_builder.out_channels(64);
    /// ```
    pub fn out_channels(&mut self, out_channels: usize) -> &mut Self {
        self.out_channels = Some(out_channels);
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
    /// ```
    /// use nove::model::layer::Conv2dBuilder;
    /// let mut conv_builder = Conv2dBuilder::default();
    /// conv_builder.kernel_size((3, 3));
    /// ```
    pub fn kernel_size(&mut self, kernel_size: (usize, usize)) -> &mut Self {
        self.kernel_size = Some(kernel_size);
        self
    }

    /// Configure the padding size.
    ///
    /// # Arguments
    /// * `padding` - The padding size.
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured padding size.
    ///
    /// # Examples
    /// ```
    /// use nove::model::layer::Conv2dBuilder;
    /// let mut conv_builder = Conv2dBuilder::default();
    /// conv_builder.padding(1);
    /// ```
    pub fn padding(&mut self, padding: usize) -> &mut Self {
        self.padding = padding;
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
    /// ```
    /// use nove::model::layer::Conv2dBuilder;
    /// let mut conv_builder = Conv2dBuilder::default();
    /// conv_builder.stride(1);
    /// ```
    pub fn stride(&mut self, stride: usize) -> &mut Self {
        self.stride = stride;
        self
    }

    /// Configure the dilation of the convolution.
    ///
    /// # Arguments
    /// * `dilation` - The dilation of the convolution.
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured dilation.
    ///
    /// # Examples
    /// ```
    /// use nove::model::layer::Conv2dBuilder;
    /// let mut conv_builder = Conv2dBuilder::default();
    /// conv_builder.dilation(1);
    /// ```
    pub fn dilation(&mut self, dilation: usize) -> &mut Self {
        self.dilation = dilation;
        self
    }

    /// Configure the number of groups for grouped convolution.
    ///
    /// # Arguments
    /// * `groups` - The number of groups for grouped convolution.
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured number of groups.
    ///
    /// # Examples
    /// ```
    /// use nove::model::layer::Conv2dBuilder;
    /// let mut conv_builder = Conv2dBuilder::default();
    /// conv_builder.groups(1);
    /// ```
    pub fn groups(&mut self, groups: usize) -> &mut Self {
        self.groups = groups;
        self
    }

    /// Configure whether to enable the bias term.
    ///
    /// # Arguments
    /// * `bias_enabled` - Whether to enable the bias term.
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured bias term.
    ///
    /// # Examples
    /// ```
    /// use nove::model::layer::Conv2dBuilder;
    /// let mut conv_builder = Conv2dBuilder::default();
    /// conv_builder.bias_enabled(true);
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
    /// ```
    /// use nove::model::layer::Conv2dBuilder;
    /// use nove::tensor::Device;
    /// let mut conv_builder = Conv2dBuilder::default();
    /// conv_builder.device(Device::cpu());
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
    /// ```
    /// use nove::model::layer::Conv2dBuilder;
    /// use nove::tensor::DType;
    /// let mut conv_builder = Conv2dBuilder::default();
    /// conv_builder.dtype(DType::F32);
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
    /// use nove::model::layer::Conv2dBuilder;
    /// let mut conv_builder = Conv2dBuilder::default();
    /// conv_builder.grad_enabled(true);
    /// ```
    pub fn grad_enabled(&mut self, grad_enabled: bool) -> &mut Self {
        self.grad_enabled = grad_enabled;
        self
    }

    /// Build the 2D convolution layer.
    ///
    /// # Returns
    /// * `Ok(Conv2d)` - The built 2D convolution layer.
    /// * `Err(ModelError)` - The error when building the 2D convolution layer.
    ///
    /// # Examples
    /// ```
    /// use nove::model::layer::Conv2dBuilder;
    /// let mut conv_builder = Conv2dBuilder::default();
    /// conv_builder.in_channels(3);
    /// conv_builder.out_channels(64);
    /// conv_builder.kernel_size((3, 3));
    /// let conv = conv_builder.build().unwrap();
    /// ```
    pub fn build(&self) -> Result<Conv2d, ModelError> {
        let in_channels = self.in_channels.ok_or(ModelError::MissingArgument(
            "in_channels in Conv2dBuilder".to_string(),
        ))?;
        let out_channels = self.out_channels.ok_or(ModelError::MissingArgument(
            "out_channels in Conv2dBuilder".to_string(),
        ))?;
        let kernel_size = self.kernel_size.ok_or(ModelError::MissingArgument(
            "kernel_size in Conv2dBuilder".to_string(),
        ))?;

        if kernel_size.0 == 0 || kernel_size.1 == 0 {
            return Err(ModelError::InvalidArgument(
                "kernel_size in Conv2dBuilder must be greater than 0".to_string(),
            ));
        }
        if in_channels % self.groups != 0 {
            return Err(ModelError::InvalidArgument(
                "in_channels must be divisible by groups".to_string(),
            ));
        }
        if out_channels % self.groups != 0 {
            return Err(ModelError::InvalidArgument(
                "out_channels must be divisible by groups".to_string(),
            ));
        }
        if self.stride == 0 {
            return Err(ModelError::InvalidArgument(
                "stride in Conv2dBuilder must be greater than 0".to_string(),
            ));
        }
        if self.dilation == 0 {
            return Err(ModelError::InvalidArgument(
                "dilation in Conv2dBuilder must be greater than 0".to_string(),
            ));
        }
        if self.groups == 0 {
            return Err(ModelError::InvalidArgument(
                "groups in Conv2dBuilder must be greater than 0".to_string(),
            ));
        }

        let id = ID.fetch_add(1, Ordering::Relaxed);

        let fan_in = (in_channels / self.groups) * kernel_size.0 * kernel_size.1;
        let std = (2.0 / fan_in as f32).sqrt();

        let weight = Tensor::randn(
            0.0,
            std,
            &Shape::from_dims(&[
                out_channels,
                in_channels / self.groups,
                kernel_size.0,
                kernel_size.1,
            ]),
            &self.device,
            self.grad_enabled,
        )?
        .to_dtype(&self.dtype)?
        .require_name(&format!("conv2d.{}.weight", id))?;

        let bias = if self.bias_enabled {
            let bias = Tensor::zeros(
                &Shape::from_dims(&[out_channels, 1, 1]),
                &self.dtype,
                &self.device,
                self.grad_enabled,
            )?
            .to_dtype(&self.dtype)?
            .require_name(&format!("conv2d.{}.bias", id))?;
            Some(bias)
        } else {
            None
        };

        Ok(Conv2d {
            weight,
            bias,
            in_channels,
            out_channels,
            kernel_size,
            padding: self.padding,
            stride: self.stride,
            dilation: self.dilation,
            groups: self.groups,
            id,
        })
    }
}
