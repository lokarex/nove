use std::{
    fmt::Display,
    sync::atomic::{AtomicUsize, Ordering},
};

use nove_tensor::{DType, Device, Tensor};

use crate::{Model, ModelError};

static ID: AtomicUsize = AtomicUsize::new(0);

/// 2D max pooling layer.
///
/// # Notes
/// * The `MaxPool2d` is now only created by the `MaxPool2dBuilder`.
///
/// # Fields
/// * `kernel_size` - The size of the pooling kernel (height, width).
/// * `stride` - The stride of the pooling operation (height, width).
/// * `id` - The unique ID of the pooling layer.
///
/// # Examples
/// ```
/// use nove::model::layer::MaxPool2dBuilder;
///
/// let pool = MaxPool2dBuilder::default()
///     .kernel_size((2, 2))      // Required
///     .stride((2, 2))           // Optional, default is kernel_size
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct MaxPool2d {
    kernel_size: (usize, usize),
    stride: (usize, usize),
    id: usize,
}

impl MaxPool2d {
    /// Get the kernel size of the pooling layer.
    ///
    /// # Returns
    /// * `(usize, usize)` - The kernel size (height, width).
    pub fn kernel_size(&self) -> (usize, usize) {
        self.kernel_size
    }

    /// Get the stride of the pooling layer.
    ///
    /// # Returns
    /// * `(usize, usize)` - The stride (height, width).
    pub fn stride(&self) -> (usize, usize) {
        self.stride
    }
}

impl Model for MaxPool2d {
    type Input = Tensor;

    type Output = Tensor;

    /// Apply the 2D max pooling layer to the input tensor.
    ///
    /// # Arguments
    /// * `input: Tensor` - The input tensor with shape [batch_size, channels, height, width].
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The output tensor with shape [batch_size, channels, out_height, out_width] if successful.
    /// * `Err(ModelError)` - The error when applying the max pooling layer to the input tensor.
    fn forward(&mut self, input: Self::Input) -> Result<Self::Output, ModelError> {
        let y = input.max_pool2d(self.kernel_size, self.stride)?;
        Ok(y)
    }

    fn require_grad(&mut self, _grad_enabled: bool) -> Result<(), ModelError> {
        Ok(())
    }

    fn to_device(&mut self, _device: &Device) -> Result<(), ModelError> {
        Ok(())
    }

    fn to_dtype(&mut self, _dtype: &DType) -> Result<(), ModelError> {
        Ok(())
    }

    fn parameters(&self) -> Result<Vec<Tensor>, ModelError> {
        Ok(vec![])
    }

    fn named_parameters(&self) -> Result<std::collections::HashMap<String, Tensor>, ModelError> {
        Ok(std::collections::HashMap::new())
    }
}

impl Display for MaxPool2d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "maxpool2d.{}(kernel_size={:?}, stride={:?})",
            self.id, self.kernel_size, self.stride,
        )
    }
}

/// The builder for the 2D max pooling layer.
///
/// # Notes
/// * The `MaxPool2dBuilder` implements the `Default` trait, so you can
///   use `MaxPool2dBuilder::default()` to create a builder with default values.
///
/// # Required Arguments
/// * `kernel_size` - The size of the pooling kernel (height, width).
///
/// # Optional Arguments
/// * `stride` - The stride of the pooling operation (height, width). Default is `kernel_size`.
///
/// # Fields
/// * `kernel_size` - The size of the pooling kernel (height, width).
/// * `stride` - The stride of the pooling operation (height, width).
///
/// # Examples
/// ```
/// use nove::model::layer::MaxPool2dBuilder;
///
/// let pool = MaxPool2dBuilder::default()
///     .kernel_size((2, 2))      // Required
///     .stride((2, 2))           // Optional, default is kernel_size
///     .build();
/// ```
pub struct MaxPool2dBuilder {
    kernel_size: Option<(usize, usize)>,
    stride: Option<(usize, usize)>,
}

impl Default for MaxPool2dBuilder {
    fn default() -> Self {
        Self {
            kernel_size: None,
            stride: None,
        }
    }
}

impl MaxPool2dBuilder {
    /// Configure the size of the pooling kernel.
    ///
    /// # Arguments
    /// * `kernel_size` - The size of the pooling kernel (height, width).
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured kernel size.
    ///
    /// # Examples
    /// ```
    /// use nove::model::layer::MaxPool2dBuilder;
    /// let mut pool_builder = MaxPool2dBuilder::default();
    /// pool_builder.kernel_size((2, 2));
    /// ```
    pub fn kernel_size(&mut self, kernel_size: (usize, usize)) -> &mut Self {
        self.kernel_size = Some(kernel_size);
        self
    }

    /// Configure the stride of the pooling operation.
    ///
    /// # Arguments
    /// * `stride` - The stride of the pooling operation (height, width).
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured stride.
    ///
    /// # Examples
    /// ```
    /// use nove::model::layer::MaxPool2dBuilder;
    /// let mut pool_builder = MaxPool2dBuilder::default();
    /// pool_builder.stride((2, 2));
    /// ```
    pub fn stride(&mut self, stride: (usize, usize)) -> &mut Self {
        self.stride = Some(stride);
        self
    }

    /// Build the 2D max pooling layer.
    ///
    /// # Returns
    /// * `Ok(MaxPool2d)` - The built 2D max pooling layer.
    /// * `Err(ModelError)` - The error when building the 2D max pooling layer.
    ///
    /// # Examples
    /// ```
    /// use nove::model::layer::MaxPool2dBuilder;
    /// let mut pool_builder = MaxPool2dBuilder::default();
    /// pool_builder.kernel_size((2, 2));
    /// let pool = pool_builder.build().unwrap();
    /// ```
    pub fn build(&self) -> Result<MaxPool2d, ModelError> {
        let kernel_size = self.kernel_size.ok_or(ModelError::MissingArgument(
            "kernel_size in MaxPool2dBuilder".to_string(),
        ))?;

        if kernel_size.0 == 0 || kernel_size.1 == 0 {
            return Err(ModelError::InvalidArgument(
                "kernel_size in MaxPool2dBuilder must be greater than 0".to_string(),
            ));
        }

        let stride = self.stride.unwrap_or(kernel_size);

        if stride.0 == 0 || stride.1 == 0 {
            return Err(ModelError::InvalidArgument(
                "stride in MaxPool2dBuilder must be greater than 0".to_string(),
            ));
        }

        let id = ID.fetch_add(1, Ordering::Relaxed);

        Ok(MaxPool2d {
            kernel_size,
            stride,
            id,
        })
    }
}
