use crate::{Model, ModelError};
use nove_tensor::{DType, Device, Tensor};
use std::{
    fmt::Display,
    sync::atomic::{AtomicUsize, Ordering},
};

static ID: AtomicUsize = AtomicUsize::new(0);

/// 1D average pooling layer.
///
/// # Notes
/// * The `AvgPool1d` is now only created by the [`AvgPool1d::new()`] method.
///
/// # Fields
/// * `kernel_size` - The size of the pooling kernel.
/// * `stride` - The stride of the pooling operation.
/// * `id` - The unique ID of the pooling layer.
///
/// # Examples
/// ```
/// use nove::model::nn::AvgPool1d;
///
/// let avg_pool1d = AvgPool1d::new(2, None).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct AvgPool1d {
    kernel_size: usize,
    stride: usize,
    id: usize,
}

impl AvgPool1d {
    /// Create a new 1D average pooling layer.
    ///
    /// # Arguments
    /// * `kernel_size` - The size of the pooling kernel.
    /// * `stride` - The stride of the pooling operation.
    ///   Default is `kernel_size` when `None`.
    ///
    /// # Returns
    /// * `Ok(AvgPool1d)` - The new average pooling layer if successful.
    /// * `Err(ModelError)` - The error when creating the average pooling layer.
    pub fn new(kernel_size: usize, stride: Option<usize>) -> Result<Self, ModelError> {
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

    fn validate_positive(size: usize, name: &str) -> Result<(), ModelError> {
        if size == 0 {
            return Err(ModelError::InvalidArgument(format!(
                "{} in AvgPool1d must be greater than 0",
                name
            )));
        }
        Ok(())
    }

    /// Get the kernel size of the pooling layer.
    ///
    /// # Returns
    /// * `usize` - The kernel size.
    pub fn kernel_size(&self) -> usize {
        self.kernel_size
    }

    /// Get the stride of the pooling layer.
    ///
    /// # Returns
    /// * `usize` - The stride.
    pub fn stride(&self) -> usize {
        self.stride
    }
}

impl Model for AvgPool1d {
    type Input = Tensor;

    type Output = Tensor;

    /// Apply the 1D average pooling layer to the input tensor.
    ///
    /// # Arguments
    /// * `input: Tensor` - The input tensor with shape [batch_size, channels, length].
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The output tensor with shape [batch_size, channels, out_length] if successful.
    /// * `Err(ModelError)` - The error when applying the average pooling layer to the input tensor.
    fn forward(&mut self, input: Self::Input) -> Result<Self::Output, ModelError> {
        let y = input.avg_pool1d(self.kernel_size, self.stride)?;
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

impl Display for AvgPool1d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "avgpool1d.{}(kernel_size={}, stride={})",
            self.id, self.kernel_size, self.stride,
        )
    }
}
