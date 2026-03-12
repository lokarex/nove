use std::{
    collections::HashMap,
    fmt::Display,
    sync::atomic::{AtomicUsize, Ordering},
};

use nove_tensor::{DType, Device, Shape, Tensor};

use crate::{Model, ModelError};

static ID: AtomicUsize = AtomicUsize::new(0);

/// 2D Batch Normalization layer.
///
/// # Notes
/// * The `BatchNorm2d` is now only created by the `BatchNorm2dBuilder`.
/// * During training, this layer computes the mean and variance of each channel
///   over the batch and spatial dimensions, then normalizes the input.
/// * During inference, it uses the running mean and variance.
///
/// # Fields
/// * `gamma` - The learnable scale parameter (gamma) with shape [num_features].
/// * `beta` - The learnable shift parameter (beta) with shape [num_features].
/// * `running_mean` - The running mean with shape [num_features].
/// * `running_var` - The running variance with shape [num_features].
/// * `num_features` - The number of features (channels).
/// * `epsilon` - A small value added to the variance for numerical stability.
/// * `momentum` - The momentum for updating running statistics.
/// * `id` - The unique ID of the batch normalization layer.
/// * `affine` - Whether to use learnable affine parameters (gamma and beta).
///
/// # Examples
/// ```
/// use nove::model::layer::BatchNorm2dBuilder;
/// use nove::tensor::{Device, DType};
///
/// let bn = BatchNorm2dBuilder::default()
///     .num_features(64)       // Required
///     .epsilon(1e-5)          // Optional, default is 1e-5
///     .momentum(0.1)          // Optional, default is 0.1
///     .affine(true)           // Optional, default is true
///     .device(Device::cpu())  // Optional, default is cpu
///     .dtype(DType::F32)      // Optional, default is F32
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct BatchNorm2d {
    gamma: Tensor,
    beta: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    num_features: usize,
    epsilon: f64,
    momentum: f64,
    id: usize,
    affine: bool,
}

impl BatchNorm2d {
    /// Get the gamma tensor (gamma) in the batch normalization layer.
    ///
    /// # Returns
    /// * `Tensor` - The gamma tensor.
    pub fn gamma(&self) -> Tensor {
        self.gamma.copy()
    }

    /// Get the beta tensor (beta) in the batch normalization layer.
    ///
    /// # Returns
    /// * `Tensor` - The beta tensor.
    pub fn beta(&self) -> Tensor {
        self.beta.copy()
    }

    /// Get the running mean in the batch normalization layer.
    ///
    /// # Returns
    /// * `Tensor` - The running mean tensor.
    pub fn running_mean(&self) -> Tensor {
        self.running_mean.copy()
    }

    /// Get the running variance in the batch normalization layer.
    ///
    /// # Returns
    /// * `Tensor` - The running variance tensor.
    pub fn running_var(&self) -> Tensor {
        self.running_var.copy()
    }
}

impl Model for BatchNorm2d {
    type Input = (Tensor, bool);
    type Output = Tensor;

    /// Apply the 2D batch normalization layer to the input tensor.
    ///
    /// # Arguments
    /// * `input` - A tuple of (input_tensor, training) where:
    ///   - `input_tensor`: The input tensor with shape [batch_size, num_features, height, width].
    ///   - `training`: Whether the layer is in training mode.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The output tensor with the same shape as input if successful.
    /// * `Err(ModelError)` - The error when applying the batch normalization layer.
    fn forward(&mut self, input: Self::Input) -> Result<Self::Output, ModelError> {
        let (input, training) = input;
        match training {
            true => {
                let total_size = input.shape()?.dims().iter().product::<usize>();
                let reshaped_input = input.reshape(&Shape::from_dims(&[
                    self.num_features,
                    total_size / self.num_features,
                ]))?;

                let feature_mean = reshaped_input.mean(Some((1, false)))?;
                let feature_bias_var = reshaped_input.var(1, false, false)?;
                let feature_unbias_var = reshaped_input.var(1, false, true)?;

                self.running_mean = Tensor::add(
                    &self.running_mean.affine(1f64 - self.momentum, 0f64)?,
                    &feature_mean.affine(self.momentum, 0f64)?,
                )?
                .detach()?;
                self.running_var = Tensor::add(
                    &self.running_var.affine(1f64 - self.momentum, 0f64)?,
                    &feature_unbias_var.affine(self.momentum, 0f64)?,
                )?
                .detach()?;

                Ok(Tensor::batch_norm2d(
                    &input,
                    &feature_mean,
                    &feature_bias_var,
                    self.epsilon,
                    &self.gamma,
                    &self.beta,
                )?)
            }
            false => Ok(Tensor::batch_norm2d(
                &input,
                &self.running_mean,
                &self.running_var,
                self.epsilon,
                &self.gamma,
                &self.beta,
            )?),
        }
    }

    /// Enable or disable gradient computation for the learnable parameters (gamma and beta).
    ///
    /// This method is equivalent to the `affine` setting in the [`BatchNorm2dBuilder`].
    /// When gradient computation is disabled, the gamma and beta parameters are treated
    /// as fixed values and will not be updated during backpropagation.
    ///
    /// # Arguments
    /// * `grad_enabled` - Whether to enable gradient computation for gamma and beta.
    ///
    /// # Returns
    /// * `Ok(())` - If gradient computation is successfully configured.
    /// * `Err(ModelError)` - If there is an error configuring gradient computation.
    ///
    /// # See Also
    /// * [`BatchNorm2dBuilder::affine`] - Configure learnable affine parameters during construction.
    fn require_grad(&mut self, grad_enabled: bool) -> Result<(), ModelError> {
        self.gamma = self.gamma.require_grad(grad_enabled)?;
        self.beta = self.beta.require_grad(grad_enabled)?;
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<(), ModelError> {
        self.running_mean = self.running_mean.to_device(device)?;
        self.running_var = self.running_var.to_device(device)?;
        self.gamma = self.gamma.to_device(device)?;
        self.beta = self.beta.to_device(device)?;
        Ok(())
    }

    fn to_dtype(&mut self, dtype: &DType) -> Result<(), ModelError> {
        self.running_mean = self.running_mean.to_dtype(dtype)?;
        self.running_var = self.running_var.to_dtype(dtype)?;
        self.gamma = self.gamma.to_dtype(dtype)?;
        self.beta = self.beta.to_dtype(dtype)?;
        Ok(())
    }

    /// Get all parameters of the batch normalization layer.
    ///
    /// This includes both learnable and non-learnable parameters:
    /// - Learnable: `gamma` (scale), `beta` (shift)
    /// - Non-learnable: `running_mean`, `running_var`
    ///
    /// # Returns
    /// * `Ok(Vec<Tensor>)` - A vector containing gamma, beta, running_mean, and running_var tensors.
    /// * `Err(ModelError)` - The error when getting the parameters.
    fn parameters(&self) -> Result<Vec<Tensor>, ModelError> {
        Ok(vec![
            self.gamma.copy(),
            self.beta.copy(),
            self.running_mean.copy(),
            self.running_var.copy(),
        ])
    }

    /// Get all named parameters of the batch normalization layer.
    ///
    /// This includes both learnable and non-learnable parameters:
    /// - Learnable: `gamma` (scale), `beta` (shift)
    /// - Non-learnable: `running_mean`, `running_var`
    ///
    /// # Returns
    /// * `Ok(HashMap<String, Tensor>)` - A map from parameter names to tensors.
    /// * `Err(ModelError)` - The error when getting the named parameters.
    fn named_parameters(&self) -> Result<HashMap<String, Tensor>, ModelError> {
        self.parameters()?
            .into_iter()
            .map(|t| match t.name()? {
                Some(name) => Ok((name, t)),
                None => Err(ModelError::ParameterMissingName),
            })
            .collect::<Result<HashMap<_, _>, ModelError>>()
    }
}

impl Display for BatchNorm2d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "batch_norm2d.{}(num_features={}, epsilon={}, momentum={}, affine={})",
            self.id, self.num_features, self.epsilon, self.momentum, self.affine
        )
    }
}

/// The builder for the 2D batch normalization layer.
///
/// # Notes
/// * The `BatchNorm2dBuilder` implements the `Default` trait, so you can
///   use `BatchNorm2dBuilder::default()` to create a builder with default values.
/// * The `gamma` and `beta` tensors are initialized with ones and zeros respectively.
/// * The `running_mean` and `running_var` tensors are initialized with zeros and ones respectively.
/// * When `affine` is `false`, `gamma` and `beta` are still initialized but with `grad_enabled` set to `false`.
///
/// # Required Arguments
/// * `num_features` - The number of features (channels).
///
/// # Optional Arguments
/// * `epsilon` - A small value added to the variance for numerical stability. Default is `1e-5`.
/// * `momentum` - The momentum for updating running statistics. Default is `0.1`.
/// * `affine` - Whether to use learnable affine parameters (gamma and beta). Default is `true`.
/// * `device` - The device to use for the layer. Default is `Device::cpu()`.
/// * `dtype` - The data type to use for the layer. Default is `DType::F32`.
///
/// # Fields
/// * `num_features` - The number of features (channels).
/// * `epsilon` - A small value added to the variance for numerical stability.
/// * `momentum` - The momentum for updating running statistics.
/// * `affine` - Whether to use learnable affine parameters (gamma and beta).
/// * `device` - The device to use for the layer.
/// * `dtype` - The data type to use for the layer.
///
/// # Examples
/// ```
/// use nove::model::layer::BatchNorm2dBuilder;
/// use nove::tensor::{Device, DType};
///
/// let bn = BatchNorm2dBuilder::default()
///     .num_features(64)       // Required
///     .epsilon(1e-5)          // Optional, default is 1e-5
///     .momentum(0.1)          // Optional, default is 0.1
///     .affine(true)           // Optional, default is true
///     .device(Device::cpu())  // Optional, default is cpu
///     .dtype(DType::F32)      // Optional, default is F32
///     .build();
/// ```
pub struct BatchNorm2dBuilder {
    num_features: Option<usize>,
    epsilon: f64,
    momentum: f64,
    affine: bool,
    device: Device,
    dtype: DType,
}

impl Default for BatchNorm2dBuilder {
    fn default() -> Self {
        Self {
            num_features: None,
            epsilon: 1e-5,
            momentum: 0.1,
            affine: true,
            device: Device::cpu(),
            dtype: DType::F32,
        }
    }
}

impl BatchNorm2dBuilder {
    /// Configure the number of features (channels).
    ///
    /// # Arguments
    /// * `num_features` - The number of features (channels).
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured number of features.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::BatchNorm2dBuilder;
    /// let mut bn_builder = BatchNorm2dBuilder::default();
    /// bn_builder.num_features(64);
    /// ```
    pub fn num_features(&mut self, num_features: usize) -> &mut Self {
        self.num_features = Some(num_features);
        self
    }

    /// Configure the epsilon value for numerical stability.
    ///
    /// # Arguments
    /// * `epsilon` - A small value added to the variance for numerical stability.
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured epsilon value.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::BatchNorm2dBuilder;
    /// let mut bn_builder = BatchNorm2dBuilder::default();
    /// bn_builder.epsilon(1e-5);
    /// ```
    pub fn epsilon(&mut self, epsilon: f64) -> &mut Self {
        self.epsilon = epsilon;
        self
    }

    /// Configure the momentum for updating running statistics.
    ///
    /// # Arguments
    /// * `momentum` - The momentum for updating running statistics.
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured momentum.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::BatchNorm2dBuilder;
    /// let mut bn_builder = BatchNorm2dBuilder::default();
    /// bn_builder.momentum(0.1);
    /// ```
    pub fn momentum(&mut self, momentum: f64) -> &mut Self {
        self.momentum = momentum;
        self
    }

    /// Configure whether to use learnable affine parameters (gamma and beta).
    ///
    /// # Arguments
    /// * `affine` - Whether to use learnable affine parameters (gamma and beta).
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured affine setting.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::model::layer::BatchNorm2dBuilder;
    /// let mut bn_builder = BatchNorm2dBuilder::default();
    /// bn_builder.affine(true);
    /// ```
    pub fn affine(&mut self, affine: bool) -> &mut Self {
        self.affine = affine;
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
    /// ```no_run
    /// use nove::model::layer::BatchNorm2dBuilder;
    /// use nove::tensor::Device;
    /// let mut bn_builder = BatchNorm2dBuilder::default();
    /// bn_builder.device(Device::cpu());
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
    /// ```no_run
    /// use nove::model::layer::BatchNorm2dBuilder;
    /// use nove::tensor::DType;
    /// let mut bn_builder = BatchNorm2dBuilder::default();
    /// bn_builder.dtype(DType::F32);
    /// ```
    pub fn dtype(&mut self, dtype: DType) -> &mut Self {
        self.dtype = dtype;
        self
    }

    /// Build the 2D batch normalization layer.
    ///
    /// # Returns
    /// * `Ok(BatchNorm2d)` - The built 2D batch normalization layer.
    /// * `Err(ModelError)` - The error when building the 2D batch normalization layer.
    ///
    /// # Examples
    /// ```
    /// use nove::model::layer::BatchNorm2dBuilder;
    /// let mut bn_builder = BatchNorm2dBuilder::default();
    /// bn_builder.num_features(64);
    /// let bn = bn_builder.build().unwrap();
    /// ```
    pub fn build(&self) -> Result<BatchNorm2d, ModelError> {
        let num_features = self.num_features.ok_or(ModelError::MissingArgument(
            "num_features in BatchNorm2dBuilder".to_string(),
        ))?;

        if num_features == 0 {
            return Err(ModelError::InvalidArgument(
                "num_features in BatchNorm2dBuilder must be greater than 0".to_string(),
            ));
        }

        if self.epsilon <= 0.0 {
            return Err(ModelError::InvalidArgument(
                "epsilon in BatchNorm2dBuilder must be greater than 0".to_string(),
            ));
        }

        if self.momentum < 0.0 || self.momentum > 1.0 {
            return Err(ModelError::InvalidArgument(
                "momentum in BatchNorm2dBuilder must be between 0 and 1".to_string(),
            ));
        }

        let id = ID.fetch_add(1, Ordering::Relaxed);

        let gamma = Tensor::ones(
            &Shape::from_dims(&[num_features]),
            &self.dtype,
            &self.device,
            self.affine,
        )?
        .to_dtype(&self.dtype)?
        .require_name(&format!("batch_norm2d.{}.gamma", id))?;

        let beta = Tensor::zeros(
            &Shape::from_dims(&[num_features]),
            &self.dtype,
            &self.device,
            self.affine,
        )?
        .to_dtype(&self.dtype)?
        .require_name(&format!("batch_norm2d.{}.beta", id))?;

        let running_mean = Tensor::zeros(
            &Shape::from_dims(&[num_features]),
            &self.dtype,
            &self.device,
            false,
        )?
        .to_dtype(&self.dtype)?
        .require_name(&format!("batch_norm2d.{}.running_mean", id))?;

        let running_var = Tensor::ones(
            &Shape::from_dims(&[num_features]),
            &self.dtype,
            &self.device,
            false,
        )?
        .to_dtype(&self.dtype)?
        .require_name(&format!("batch_norm2d.{}.running_var", id))?;

        Ok(BatchNorm2d {
            gamma,
            beta,
            running_mean,
            running_var,
            num_features,
            epsilon: self.epsilon,
            momentum: self.momentum,
            id,
            affine: self.affine,
        })
    }
}
