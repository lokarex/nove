use nove_tensor::Tensor;

use crate::{Optimizer, OptimizerError};

struct AdamParam {
    param: Tensor,
    m: Tensor,
    v: Tensor,
}

/// Adam (Adaptive Moment Estimation) optimizer.
///
/// <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" id="MathJax-script" async></script>
///
/// Adam is a stochastic gradient descent method that computes adaptive learning rates
/// for each parameter based on estimates of first and second moments of the gradients.
///
/// The parameter update is performed as follows:
///
/// $$ m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t $$
///
/// $$ v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2 $$
///
/// $$ \hat{m}_t = \frac{m_t}{1 - \beta_1^t} $$
///
/// $$ \hat{v}_t = \frac{v_t}{1 - \beta_2^t} $$
///
/// $$ \theta_t = \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} $$
///
/// Where:
/// - θ_t is the parameter at time step t
/// - g_t is the gradient at time step t
/// - m_t is the first moment estimate (exponential moving average of gradients)
/// - v_t is the second moment estimate (exponential moving average of squared gradients)
/// - m̂_t is the bias-corrected first moment estimate
/// - v̂_t is the bias-corrected second moment estimate
/// - α is the learning rate
/// - β_1 is the exponential decay rate for the first moment
/// - β_2 is the exponential decay rate for the second moment
/// - ε is a small constant for numerical stability
/// - t is the current time step
///
/// # Notes
/// * The `Adam` optimizer is now only created by the `AdamBuilder`.
///
/// # Fields
/// * `params` - The list of parameters to optimize.
/// * `learning_rate` - The learning rate (step size).
/// * `beta1` - The exponential decay rate for the first moment.
/// * `beta2` - The exponential decay rate for the second moment.
/// * `epsilon` - A small constant for numerical stability.
/// * `t` - The current time step.
///
/// # Examples
/// ```
/// use nove::optimizer::AdamBuilder;
/// use nove::tensor::{Device, Tensor};
///
/// let device = Device::cpu();
/// let param1 = Tensor::from_data(vec![1.0, 2.0, 3.0], &device, true).unwrap();
/// let param2 = Tensor::from_data(vec![4.0, 5.0, 6.0], &device, true).unwrap();
///
/// let adam = AdamBuilder::default()
///     .params(vec![param1, param2])  // Required
///     .learning_rate(0.001)          // Required
///     .beta1(0.9)                    // Optional, default is 0.9
///     .beta2(0.999)                  // Optional, default is 0.999
///     .epsilon(1e-8)                 // Optional, default is 1e-8
///     .build()
///     .unwrap();
/// ```
pub struct Adam {
    params: Vec<AdamParam>,
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    t: usize,
}

/// The builder for the Adam optimizer.
///
/// # Notes
/// * The `AdamBuilder` implements the `Default` trait, so you can
///   use `AdamBuilder::default()` to create a builder with default values.
///
/// # Required Arguments
/// * `params` - The list of parameters to optimize.
/// * `learning_rate` - The learning rate (step size).
///
/// # Optional Arguments
/// * `beta1` - The exponential decay rate for the first moment. Default is `0.9`.
/// * `beta2` - The exponential decay rate for the second moment. Default is `0.999`.
/// * `epsilon` - A small constant for numerical stability. Default is `1e-8`.
///
/// # Fields
/// * `params` - The list of parameters to optimize.
/// * `learning_rate` - The learning rate (step size).
/// * `beta1` - The exponential decay rate for the first moment.
/// * `beta2` - The exponential decay rate for the second moment.
/// * `epsilon` - A small constant for numerical stability.
///
/// # Examples
/// ```
/// use nove::optimizer::AdamBuilder;
/// use nove::tensor::{Device, Tensor};
///
/// let device = Device::cpu();
/// let param1 = Tensor::from_data(vec![1.0, 2.0, 3.0], &device, true).unwrap();
/// let param2 = Tensor::from_data(vec![4.0, 5.0, 6.0], &device, true).unwrap();
///
/// let adam = AdamBuilder::default()
///     .params(vec![param1, param2])  // Required
///     .learning_rate(0.001)          // Required
///     .beta1(0.9)                    // Optional, default is 0.9
///     .beta2(0.999)                  // Optional, default is 0.999
///     .epsilon(1e-8)                 // Optional, default is 1e-8
///     .build()
///     .unwrap();
/// ```
pub struct AdamBuilder {
    params: Option<Vec<Tensor>>,
    learning_rate: Option<f64>,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
}

impl Default for AdamBuilder {
    fn default() -> Self {
        Self {
            params: None,
            learning_rate: None,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
}

impl AdamBuilder {
    /// Configure the parameters to optimize.
    ///
    /// # Arguments
    /// * `params` - The list of parameters to optimize.
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured parameters.
    ///
    /// # Examples
    /// ```
    /// use nove::optimizer::AdamBuilder;
    /// use nove::tensor::{Device, Tensor};
    ///
    /// let device = Device::cpu();
    /// let param = Tensor::from_data(vec![1.0, 2.0, 3.0], &device, true).unwrap();
    /// let mut builder = AdamBuilder::default();
    /// builder.params(vec![param]);
    /// ```
    pub fn params(&mut self, params: Vec<Tensor>) -> &mut Self {
        self.params = Some(params);
        self
    }

    /// Configure the learning rate.
    ///
    /// # Arguments
    /// * `learning_rate` - The learning rate (step size).
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured learning rate.
    ///
    /// # Examples
    /// ```
    /// use nove::optimizer::AdamBuilder;
    /// let mut builder = AdamBuilder::default();
    /// builder.learning_rate(0.001);
    /// ```
    pub fn learning_rate(&mut self, learning_rate: f64) -> &mut Self {
        self.learning_rate = Some(learning_rate);
        self
    }

    /// Configure the exponential decay rate for the first moment.
    ///
    /// # Arguments
    /// * `beta1` - The exponential decay rate for the first moment.
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured beta1.
    ///
    /// # Examples
    /// ```
    /// use nove::optimizer::AdamBuilder;
    /// let mut builder = AdamBuilder::default();
    /// builder.beta1(0.9);
    /// ```
    pub fn beta1(&mut self, beta1: f64) -> &mut Self {
        self.beta1 = beta1;
        self
    }

    /// Configure the exponential decay rate for the second moment.
    ///
    /// # Arguments
    /// * `beta2` - The exponential decay rate for the second moment.
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured beta2.
    ///
    /// # Examples
    /// ```
    /// use nove::optimizer::AdamBuilder;
    /// let mut builder = AdamBuilder::default();
    /// builder.beta2(0.999);
    /// ```
    pub fn beta2(&mut self, beta2: f64) -> &mut Self {
        self.beta2 = beta2;
        self
    }

    /// Configure the epsilon for numerical stability.
    ///
    /// # Arguments
    /// * `epsilon` - A small constant for numerical stability.
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured epsilon.
    ///
    /// # Examples
    /// ```
    /// use nove::optimizer::AdamBuilder;
    /// let mut builder = AdamBuilder::default();
    /// builder.epsilon(1e-8);
    /// ```
    pub fn epsilon(&mut self, epsilon: f64) -> &mut Self {
        self.epsilon = epsilon;
        self
    }

    /// Build the Adam optimizer.
    ///
    /// # Returns
    /// * `Ok(Adam)` - The built Adam optimizer.
    /// * `Err(OptimizerError)` - The error when building the Adam optimizer.
    ///
    /// # Errors
    /// * `OptimizerError::MissingArgument` - If `params` or `learning_rate` is not set.
    /// * `OptimizerError::InvalidArgument` - If `learning_rate`, `beta1`, `beta2`, or `epsilon` is invalid.
    ///
    /// # Examples
    /// ```
    /// use nove::optimizer::AdamBuilder;
    /// use nove::tensor::{Device, Tensor};
    ///
    /// let device = Device::cpu();
    /// let param = Tensor::from_data(vec![1.0, 2.0, 3.0], &device, true).unwrap();
    ///
    /// let mut builder = AdamBuilder::default();
    /// builder.params(vec![param]);
    /// builder.learning_rate(0.001);
    /// let adam = builder.build().unwrap();
    /// ```
    pub fn build(&self) -> Result<Adam, OptimizerError> {
        let params = self
            .params
            .as_ref()
            .ok_or(OptimizerError::MissingArgument(
                "params in AdamBuilder".to_string(),
            ))?
            .iter()
            .map(|param| param.copy())
            .collect::<Vec<_>>();

        let learning_rate = self.learning_rate.ok_or_else(|| {
            OptimizerError::MissingArgument("learning_rate in AdamBuilder".to_string())
        })?;

        if learning_rate <= 0.0 {
            return Err(OptimizerError::InvalidArgument(
                "learning_rate must be positive".to_string(),
            ));
        }

        if self.beta1 < 0.0 || self.beta1 >= 1.0 {
            return Err(OptimizerError::InvalidArgument(
                "beta1 must be in [0, 1)".to_string(),
            ));
        }

        if self.beta2 < 0.0 || self.beta2 >= 1.0 {
            return Err(OptimizerError::InvalidArgument(
                "beta2 must be in [0, 1)".to_string(),
            ));
        }

        if self.epsilon <= 0.0 {
            return Err(OptimizerError::InvalidArgument(
                "epsilon must be positive".to_string(),
            ));
        }

        if params.is_empty() {
            return Ok(Adam {
                params: vec![],
                learning_rate,
                beta1: self.beta1,
                beta2: self.beta2,
                epsilon: self.epsilon,
                t: 0,
            });
        }

        let device = params[0].device()?;
        let dtype = params[0].dtype()?;

        let adam_params = params
            .into_iter()
            .map(|param| {
                let shape = param.shape()?;
                let m = Tensor::zeros(&shape, &dtype, &device, false)?;
                let v = Tensor::zeros(&shape, &dtype, &device, false)?;
                Ok(AdamParam { param, m, v })
            })
            .collect::<Result<Vec<_>, OptimizerError>>()?;

        Ok(Adam {
            params: adam_params,
            learning_rate,
            beta1: self.beta1,
            beta2: self.beta2,
            epsilon: self.epsilon,
            t: 0,
        })
    }
}

impl Optimizer for Adam {
    type StepOutput = ();

    fn step(&mut self) -> Result<Self::StepOutput, OptimizerError> {
        self.t += 1;

        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        for adam_param in &mut self.params {
            if !adam_param.param.grad_enabled()? {
                continue;
            }

            let grad = adam_param.param.grad()?.ok_or(OptimizerError::OtherError(
                "Adam: parameter gradient is None".to_string(),
            ))?;

            let m_update = grad.affine(1.0 - self.beta1, 0.0)?;
            let m_scaled = adam_param.m.affine(self.beta1, 0.0)?;
            let new_m = m_scaled.add(&m_update)?;
            adam_param.m.update_from_tensor(&new_m.detach()?)?;

            let grad_sq = grad.mul(&grad)?;
            let v_update = grad_sq.affine(1.0 - self.beta2, 0.0)?;
            let v_scaled = adam_param.v.affine(self.beta2, 0.0)?;
            let new_v = v_scaled.add(&v_update)?;
            adam_param.v.update_from_tensor(&new_v.detach()?)?;

            let m_hat = adam_param.m.affine(1.0 / bias_correction1, 0.0)?;
            let v_hat = adam_param.v.affine(1.0 / bias_correction2, 0.0)?;
            let v_hat_sqrt = v_hat.sqrt()?;
            let denom = v_hat_sqrt.affine(1.0, self.epsilon)?;
            let update = m_hat.div(&denom)?.affine(-self.learning_rate, 0.0)?;
            let new_param = adam_param.param.add(&update)?;
            adam_param.param.update_from_tensor(&new_param.detach()?)?;
        }

        Ok(())
    }

    fn zero_grad(&mut self) -> Result<(), OptimizerError> {
        for adam_param in &mut self.params {
            adam_param.param.zero_grad()?;
        }
        Ok(())
    }
}
