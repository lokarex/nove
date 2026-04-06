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
/// * The `MaxPool2d` is now only created by the [`MaxPool2d::new()`] method.
///
/// # Fields
/// * `kernel_size` - The size of the pooling kernel (height, width).
/// * `stride` - The stride of the pooling operation (height, width).
/// * `id` - The unique ID of the pooling layer.
///
/// # Examples
/// ```
/// use nove::model::nn::MaxPool2d;
///
/// let max_pool2d = MaxPool2d::new((2, 2), None).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct MaxPool2d {
    kernel_size: (usize, usize),
    stride: (usize, usize),
    id: usize,
}

impl MaxPool2d {
    /// Create a new 2D max pooling layer.
    ///
    /// # Arguments
    /// * `kernel_size` - The size of the pooling kernel (height, width).
    /// * `stride` - The stride of the pooling operation (height, width).
    ///   Default is `kernel_size` when `None`.
    ///
    /// # Returns
    /// * `Ok(MaxPool2d)` - The new max pooling layer if successful.
    /// * `Err(ModelError)` - The error when creating the max pooling layer.
    pub fn new(
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
    ) -> Result<Self, ModelError> {
        Self::validate_positive(kernel_size, "kernel_size")?;

        let stride = stride.unwrap_or(kernel_size);
        Self::validate_positive(stride, "stride")?;

        let id = ID.fetch_add(1, Ordering::Relaxed);

        Ok(Self {
            kernel_size,
            stride,
            id,
        })
    }

    fn validate_positive(size: (usize, usize), name: &str) -> Result<(), ModelError> {
        if size.0 == 0 || size.1 == 0 {
            return Err(ModelError::InvalidArgument(format!(
                "{} in MaxPool2d must be greater than 0",
                name
            )));
        }
        Ok(())
    }

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
