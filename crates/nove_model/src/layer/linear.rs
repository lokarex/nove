use std::{
    collections::HashMap,
    fmt::Display,
    sync::atomic::{AtomicUsize, Ordering},
};

use nove_tensor::{DType, Device, Shape, Tensor};

use crate::{Model, ModelError};

static ID: AtomicUsize = AtomicUsize::new(0);

/// Linear layer.
///
/// # Notes
/// * The `Linear` is now only created by the `LinearBuilder`.
///
/// # Fields
/// * `weight` - The weight tensor.
/// * `bias` - The bias tensor.
/// * `in_features` - The number of input features.
/// * `out_features` - The number of output features.
/// * `id` - The unique ID of the linear layer.
///
/// # Examples
/// ```
/// use nove::model::layer::LinearBuilder;
/// use nove::tensor::{Device, DType};
///
/// let linear = LinearBuilder::default()
///     .in_features(10)        // Required
///     .out_features(20)       // Required
///     .bias_enabled(true)     // Optional, default is true
///     .device(Device::cpu())  // Optional, default is cpu
///     .dtype(DType::F32)      // Optional, default is F32
///     .build();
/// ```
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
    in_features: usize,
    out_features: usize,
    id: usize,
}

impl Linear {
    /// Get the name of the weight tensor.
    ///
    /// # Returns
    /// * `String` - The name of the weight tensor.
    fn weight_name(&self) -> String {
        format!("linear.{}.weight", self.id)
    }

    /// Get the name of the bias tensor.
    ///
    /// # Returns
    /// * `String` - The name of the bias tensor.
    fn bias_name(&self) -> String {
        format!("linear.{}.bias", self.id)
    }

    /// Get the weight tensor in the linear layer.
    ///
    /// # Returns
    /// * `Tensor` - The weight tensor.
    pub fn weight(&self) -> Tensor {
        self.weight.clone()
    }

    /// Get the bias tensor in the linear layer.
    ///
    /// # Returns
    /// * `Option<Tensor>` - The bias tensor if enabled, otherwise None.
    pub fn bias(&self) -> Option<Tensor> {
        self.bias.clone()
    }
}

impl Model for Linear {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&mut self, input: Self::Input) -> Result<Self::Output, ModelError> {
        let y = input.matmul(&self.weight)?;

        let y = if let Some(bias) = &self.bias {
            y.add(bias)?
        } else {
            y
        };

        Ok(y)
    }

    fn set_grad_enabled(&mut self, grad_enabled: bool) -> Result<(), ModelError> {
        self.weight.set_grad_enabled(grad_enabled)?;
        if let Some(bias) = &mut self.bias {
            bias.set_grad_enabled(grad_enabled)?;
        }
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<(), ModelError> {
        self.weight.to_device(device)?;
        if let Some(bias) = &mut self.bias {
            bias.to_device(device)?;
        }
        Ok(())
    }

    fn to_dtype(&mut self, dtype: &DType) -> Result<(), ModelError> {
        self.weight.to_dtype(dtype)?;
        if let Some(bias) = &mut self.bias {
            bias.to_dtype(dtype)?;
        }
        Ok(())
    }

    fn to_safetensors(&self) -> Result<HashMap<String, Tensor>, ModelError> {
        let mut tensors = HashMap::new();

        tensors.insert(self.weight_name(), self.weight.clone());

        if let Some(bias) = &self.bias {
            tensors.insert(self.bias_name(), bias.clone());
        }
        Ok(tensors)
    }

    fn load_from_safetensors(
        &mut self,
        tensors: HashMap<String, Tensor>,
    ) -> Result<(), ModelError> {
        let weight_name = self.weight_name();
        let bias_name = self.bias_name();

        // Update weight
        let new_weight = tensors
            .get(&weight_name)
            .ok_or(ModelError::MissingParameter(weight_name))?;
        self.weight.update_from_tensor(new_weight)?;

        // Update bias
        match (&self.bias, tensors.get(&bias_name)) {
            // If the old and new biases exist, update the old bias.
            (Some(bias), Some(new_bias)) => bias.update_from_tensor(new_bias)?,
            // If the old bias does not exist, but new bias is provided, return an error.
            (None, Some(_)) => {
                return Err(ModelError::UnexpectedParameter(bias_name));
            }
            // If the old bias exists, but new bias is not provided, return an error.
            (Some(_), None) => {
                return Err(ModelError::MissingParameter(bias_name));
            }
            // If the old and new biases do not exist, do nothing.
            (None, None) => {}
        }

        Ok(())
    }
}

impl Display for Linear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "linear.{} (in_features={}, out_features={}, bias_enabled={})",
            self.id,
            self.in_features,
            self.out_features,
            self.bias.is_some(),
        )
    }
}

/// The builder for the Linear layer.
///
/// # Notes
/// * The `LinearBuilder` implements the `Default` trait, so you can
///   use `LinearBuilder::default()` to create a builder with default values.
///
/// # Required Arguments
/// * `in_features` - The number of input features.
/// * `out_features` - The number of output features.
///
/// # Optional Arguments
/// * `bias_enabled` - Whether to enable the bias term. Default is `true`.
/// * `device` - The device to use for the layer. Default is `Device::cpu()`.
/// * `dtype` - The data type to use for the layer. Default is `DType::F32`.
/// * `grad_enabled` - Whether to enable the gradient computation. Default is `true`.
///
/// # Fields
/// * `in_features` - The number of input features.
/// * `out_features` - The number of output features.
/// * `bias_enabled` - Whether to enable the bias term.
/// * `device` - The device to use for the layer.
/// * `dtype` - The data type to use for the layer.
/// * `grad_enabled` - Whether to enable the gradient computation.
///
/// # Examples
/// ```
/// use nove::model::layer::LinearBuilder;
/// use nove::tensor::{Device, DType};
///
/// let linear = LinearBuilder::default()
///     .in_features(10)        // Required
///     .out_features(20)       // Required
///     .bias_enabled(true)     // Optional, default is true
///     .device(Device::cpu())  // Optional, default is cpu
///     .dtype(DType::F32)      // Optional, default is F32
///     .grad_enabled(true)     // Optional, default is true
///     .build();
/// ```
pub struct LinearBuilder {
    in_features: Option<usize>,
    out_features: Option<usize>,
    bias_enabled: bool,
    device: Device,
    dtype: DType,
    grad_enabled: bool,
}

impl Default for LinearBuilder {
    fn default() -> Self {
        Self {
            in_features: None,
            out_features: None,
            bias_enabled: true,
            device: Device::cpu(),
            dtype: DType::F32,
            grad_enabled: true,
        }
    }
}

impl LinearBuilder {
    /// Configure the number of input features.
    ///
    /// # Arguments
    /// * `in_features` - The number of input features.
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured number of input features.
    ///
    /// # Examples
    /// ```
    /// use nove::model::layer::LinearBuilder;
    /// let mut linear_builder = LinearBuilder::default();
    /// linear_builder.in_features(10);
    /// ```
    pub fn in_features(&mut self, in_features: usize) -> &mut Self {
        self.in_features = Some(in_features);
        self
    }

    /// Configure the number of output features.
    ///
    /// # Arguments
    /// * `out_features` - The number of output features.
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured number of output features.
    ///
    /// # Examples
    /// ```
    /// use nove::model::layer::LinearBuilder;
    /// let mut linear_builder = LinearBuilder::default();
    /// linear_builder.out_features(20);
    /// ```
    pub fn out_features(&mut self, out_features: usize) -> &mut Self {
        self.out_features = Some(out_features);
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
    /// use nove::model::layer::LinearBuilder;
    /// let mut linear_builder = LinearBuilder::default();
    /// linear_builder.bias_enabled(false);
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
    /// use nove::model::layer::LinearBuilder;
    /// use nove::tensor::Device;
    /// let mut linear_builder = LinearBuilder::default();
    /// linear_builder.device(Device::cpu());
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
    /// use nove::model::layer::LinearBuilder;
    /// use nove::tensor::DType;
    /// let mut linear_builder = LinearBuilder::default();
    /// linear_builder.dtype(DType::F32);
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
    /// use nove::model::layer::LinearBuilder;
    /// let mut linear_builder = LinearBuilder::default();
    /// linear_builder.grad_enabled(false);
    /// ```
    pub fn grad_enabled(&mut self, grad_enabled: bool) -> &mut Self {
        self.grad_enabled = grad_enabled;
        self
    }

    /// Build the linear layer.
    ///
    /// # Returns
    /// * `Ok(Linear)` - The built linear layer.
    /// * `Err(ModelError)` - The error when building the linear layer.
    ///
    /// # Examples
    /// ```
    /// use nove::model::layer::LinearBuilder;
    /// let mut linear_builder = LinearBuilder::default();
    /// linear_builder.in_features(10);
    /// linear_builder.out_features(20);
    /// let linear = linear_builder.build().unwrap();
    /// ```
    pub fn build(&self) -> Result<Linear, ModelError> {
        let in_features = self.in_features.ok_or(ModelError::MissingField(
            "in_features in LinearBuilder".to_string(),
        ))?;
        let out_features = self.out_features.ok_or(ModelError::MissingField(
            "out_features in LinearBuilder".to_string(),
        ))?;

        // Generate a unique ID for the linear layer.
        let id = ID.fetch_add(1, Ordering::Relaxed);

        // Determine the bounds for the weight and bias initialization.
        let bound: f32 = 1.0 / (self.in_features.unwrap() as f32).sqrt();
        let low: f32 = -bound;
        let high: f32 = bound;

        // Initialize the weight tensor.
        let mut weight = Tensor::rand(
            low,
            high,
            &Shape::from_dims(&[self.in_features.unwrap(), self.out_features.unwrap()]),
            &self.device,
            self.grad_enabled,
        )?
        .to_dtype(&self.dtype)?;
        weight.set_name(format!("linear.{}.weight", id))?;

        // Initialize the bias tensor if enabled.
        let bias = if self.bias_enabled {
            let mut bias = Tensor::rand(
                low,
                high,
                &Shape::from_dims(&[self.out_features.unwrap()]),
                &self.device,
                self.grad_enabled,
            )?
            .to_dtype(&self.dtype)?;
            bias.set_name(format!("linear.{}.bias", id))?;
            Some(bias)
        } else {
            None
        };

        Ok(Linear {
            weight,
            bias,
            in_features,
            out_features,
            id,
        })
    }
}
