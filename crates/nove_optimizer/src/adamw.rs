use nove_tensor::Tensor;

use crate::{Optimizer, OptimizerError};

struct AdamWParam {
    param: Tensor,
    m: Tensor,
    v: Tensor,
}

/// AdamW (Adam with Decoupled Weight Decay) optimizer.
///
/// <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" id="MathJax-script" async></script>
///
/// AdamW is a variant of Adam that decouples weight decay from the gradient-based update.
/// This decoupling provides better generalization performance and is the recommended way
/// to apply weight decay in Adam-like optimizers.
///
/// The key difference between AdamW and Adam is how weight decay is applied:
/// - **Adam**: Weight decay is applied to the gradient (L2 regularization)
/// - **AdamW**: Weight decay is applied directly to the weights (decoupled)
///
/// The update rules are as follows:
///
/// 1. **First moment update** (exponential moving average of gradients):
/// $$ m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t $$
///
/// 2. **Second moment update** (exponential moving average of squared gradients):
/// $$ v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2 $$
///
/// 3. **Bias correction**:
/// $$ \hat{m}_t = \frac{m_t}{1 - \beta_1^t} $$
///
/// $$ \hat{v}_t = \frac{v_t}{1 - \beta_2^t} $$
///
/// 4. **Parameter update**:
///
/// **Without weight decay** (λ = 0):
/// $$ u_t = \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} $$
///
/// $$ \theta_t = \theta_{t-1} - \alpha \cdot u_t $$
///
/// **With weight decay** (λ > 0):
/// $$ u_t = \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} $$
///
/// $$ \theta_t = \theta_{t-1} - \alpha \cdot u_t - \alpha \lambda \theta_{t-1} $$
///
/// Or equivalently:
/// $$ \theta_t = (1 - \alpha \lambda) \theta_{t-1} - \alpha \cdot u_t $$
///
/// Where:
/// - θ_t is the parameter at time step t
/// - g_t is the gradient at time step t (NOT modified by weight decay)
/// - m_t is the first moment estimate (exponential moving average of gradients)
/// - v_t is the second moment estimate (exponential moving average of squared gradients)
/// - m̂_t is the bias-corrected first moment estimate
/// - v̂_t is the bias-corrected second moment estimate
/// - u_t is the adaptive update direction (note: in code we directly compute the learning-rate-scaled update -α·u_t)
/// - α is the learning rate
/// - β_1 is the exponential decay rate for the first moment
/// - β_2 is the exponential decay rate for the second moment
/// - ε is a small constant for numerical stability
/// - λ is the weight decay coefficient (applied directly to weights, not to gradients)
/// - t is the current time step
///
/// # Key Difference from Adam
/// In Adam with L2 regularization (weight decay), the weight decay term is added to the gradient:
/// $$ g_t' = g_t + \lambda \cdot \theta_{t-1} $$
///
/// This causes the weight decay to be scaled by the adaptive learning rate,
/// which can lead to suboptimal regularization.
///
/// In AdamW, weight decay is applied directly to the weights, independent of the
/// adaptive learning rate, which provides more consistent and effective regularization.
///
/// # Notes
/// * The `AdamW` optimizer is created by the `AdamWBuilder`.
/// * Use `AdamWBuilder::new(params, learning_rate)` to create a builder.
///
/// # Fields
/// * `params` - The list of parameters to optimize.
/// * `learning_rate` - The learning rate (step size).
/// * `beta1` - The exponential decay rate for the first moment.
/// * `beta2` - The exponential decay rate for the second moment.
/// * `epsilon` - A small constant for numerical stability.
/// * `weight_decay` - The weight decay coefficient (0.0 means no weight decay).
/// * `t` - The current time step.
///
/// # Examples
/// ```no_run
/// use nove::optimizer::AdamWBuilder;
/// use nove::tensor::{Device, Tensor};
///
/// let device = Device::cpu();
/// let param1 = Tensor::from_data(vec![1.0, 2.0, 3.0], &device, true).unwrap();
/// let param2 = Tensor::from_data(vec![4.0, 5.0, 6.0], &device, true).unwrap();
///
/// let adamw = AdamWBuilder::new(vec![param1, param2], 0.001)
///     .beta1(0.9)                    // Optional, default is 0.9
///     .beta2(0.999)                  // Optional, default is 0.999
///     .epsilon(1e-8)                 // Optional, default is 1e-8
///     .weight_decay(0.01)            // Optional, default is 0.0 (no weight decay)
///     .build()
///     .unwrap();
/// ```
pub struct AdamW {
    params: Vec<AdamWParam>,
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    weight_decay: f64,
    t: usize,
}

/// The builder for the AdamW optimizer.
///
/// # Required Arguments
/// * `params` - The list of parameters to optimize.
/// * `learning_rate` - The learning rate (step size).
///
/// # Optional Arguments
/// * `beta1` - The exponential decay rate for the first moment. Default is `0.9`.
/// * `beta2` - The exponential decay rate for the second moment. Default is `0.999`.
/// * `epsilon` - A small constant for numerical stability. Default is `1e-8`.
/// * `weight_decay` - The weight decay coefficient. Default is `0.0` (no weight decay).
///
/// # Fields
/// * `params` - The list of parameters to optimize.
/// * `learning_rate` - The learning rate (step size).
/// * `beta1` - The exponential decay rate for the first moment.
/// * `beta2` - The exponential decay rate for the second moment.
/// * `epsilon` - A small constant for numerical stability.
///
/// # Examples
/// ```no_run
/// use nove::optimizer::AdamWBuilder;
/// use nove::tensor::{Device, Tensor};
///
/// let device = Device::cpu();
/// let param1 = Tensor::from_data(vec![1.0, 2.0, 3.0], &device, true).unwrap();
/// let param2 = Tensor::from_data(vec![4.0, 5.0, 6.0], &device, true).unwrap();
///
/// let adamw = AdamWBuilder::new(vec![param1, param2], 0.001)
///     .beta1(0.9)                    // Optional, default is 0.9
///     .beta2(0.999)                  // Optional, default is 0.999
///     .epsilon(1e-8)                 // Optional, default is 1e-8
///     .weight_decay(0.01)            // Optional, default is 0.0 (no weight decay)
///     .build()
///     .unwrap();
/// ```
pub struct AdamWBuilder {
    params: Vec<Tensor>,
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    weight_decay: f64,
}

impl AdamWBuilder {
    /// Create a new AdamWBuilder with the required parameters and learning rate.
    ///
    /// # Arguments
    /// * `params` - The list of parameters to optimize.
    /// * `learning_rate` - The learning rate (step size).
    ///
    /// # Returns
    /// * `Self` - A new AdamWBuilder instance.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::optimizer::AdamWBuilder;
    /// use nove::tensor::{Device, Tensor};
    ///
    /// let device = Device::cpu();
    /// let param1 = Tensor::from_data(vec![1.0, 2.0, 3.0], &device, true).unwrap();
    /// let param2 = Tensor::from_data(vec![4.0, 5.0, 6.0], &device, true).unwrap();
    ///
    /// let adamw = AdamWBuilder::new(vec![param1, param2], 0.001)
    ///     .beta1(0.9)
    ///     .beta2(0.999)
    ///     .epsilon(1e-8)
    ///     .weight_decay(0.01)
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn new(params: Vec<Tensor>, learning_rate: f64) -> Self {
        Self {
            params,
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
        }
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
    /// ```no_run
    /// use nove::optimizer::AdamWBuilder;
    /// use nove::tensor::{Device, Tensor};
    ///
    /// let device = Device::cpu();
    /// let param = Tensor::from_data(vec![1.0, 2.0, 3.0], &device, true).unwrap();
    /// let mut builder = AdamWBuilder::new(vec![param], 0.001);
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
    /// ```no_run
    /// use nove::optimizer::AdamWBuilder;
    /// use nove::tensor::{Device, Tensor};
    ///
    /// let device = Device::cpu();
    /// let param = Tensor::from_data(vec![1.0, 2.0, 3.0], &device, true).unwrap();
    /// let mut builder = AdamWBuilder::new(vec![param], 0.001);
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
    /// ```no_run
    /// use nove::optimizer::AdamWBuilder;
    /// use nove::tensor::{Device, Tensor};
    ///
    /// let device = Device::cpu();
    /// let param = Tensor::from_data(vec![1.0, 2.0, 3.0], &device, true).unwrap();
    /// let mut builder = AdamWBuilder::new(vec![param], 0.001);
    /// builder.epsilon(1e-8);
    /// ```
    pub fn epsilon(&mut self, epsilon: f64) -> &mut Self {
        self.epsilon = epsilon;
        self
    }

    /// Configure the weight decay coefficient.
    ///
    /// Note: In AdamW, weight decay is applied directly to the weights (decoupled),
    /// not to the gradients as in Adam with L2 regularization.
    ///
    /// # Arguments
    /// * `weight_decay` - The weight decay coefficient.
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured weight decay.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::optimizer::AdamWBuilder;
    /// use nove::tensor::{Device, Tensor};
    ///
    /// let device = Device::cpu();
    /// let param = Tensor::from_data(vec![1.0, 2.0, 3.0], &device, true).unwrap();
    /// let mut builder = AdamWBuilder::new(vec![param], 0.001);
    /// builder.weight_decay(0.01);
    /// ```
    pub fn weight_decay(&mut self, weight_decay: f64) -> &mut Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Build the AdamW optimizer.
    ///
    /// # Returns
    /// * `Ok(AdamW)` - The built AdamW optimizer.
    /// * `Err(OptimizerError)` - The error when building the AdamW optimizer.
    ///
    /// # Errors
    /// * `OptimizerError::InvalidArgument` - If `learning_rate`, `beta1`, `beta2`, `epsilon`, or `weight_decay` is invalid.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::optimizer::AdamWBuilder;
    /// use nove::tensor::{Device, Tensor};
    ///
    /// let device = Device::cpu();
    /// let param = Tensor::from_data(vec![1.0, 2.0, 3.0], &device, true).unwrap();
    ///
    /// let adamw = AdamWBuilder::new(vec![param], 0.001)
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn build(&self) -> Result<AdamW, OptimizerError> {
        let params = self
            .params
            .iter()
            .map(|param| param.copy())
            .collect::<Vec<_>>();

        let learning_rate = self.learning_rate;

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

        if self.weight_decay < 0.0 {
            return Err(OptimizerError::InvalidArgument(
                "weight_decay must be non-negative".to_string(),
            ));
        }

        if params.is_empty() {
            return Ok(AdamW {
                params: vec![],
                learning_rate,
                beta1: self.beta1,
                beta2: self.beta2,
                epsilon: self.epsilon,
                weight_decay: self.weight_decay,
                t: 0,
            });
        }

        let device = params[0].device()?;
        let dtype = params[0].dtype()?;

        let adamw_params = params
            .into_iter()
            .map(|param| {
                let shape = param.shape()?;
                let m = Tensor::zeros(&shape, &dtype, &device, false)?;
                let v = Tensor::zeros(&shape, &dtype, &device, false)?;
                Ok(AdamWParam { param, m, v })
            })
            .collect::<Result<Vec<_>, OptimizerError>>()?;

        Ok(AdamW {
            params: adamw_params,
            learning_rate,
            beta1: self.beta1,
            beta2: self.beta2,
            epsilon: self.epsilon,
            weight_decay: self.weight_decay,
            t: 0,
        })
    }
}

impl Optimizer for AdamW {
    type StepOutput = ();

    fn step(&mut self) -> Result<Self::StepOutput, OptimizerError> {
        self.t += 1;

        let bias_correction1 = 1.0 - self.beta1.powf(self.t as f64);
        let bias_correction2 = 1.0 - self.beta2.powf(self.t as f64);

        for adamw_param in &mut self.params {
            if !adamw_param.param.grad_enabled()? {
                continue;
            }

            let grad = adamw_param.param.grad()?.ok_or(OptimizerError::OtherError(
                "AdamW: parameter gradient is None".to_string(),
            ))?;

            // Update first moment (m): m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
            // Note: We use the original gradient, NOT modified by weight decay
            // This is the key difference from Adam with L2 regularization
            let m_update = grad.affine(1.0 - self.beta1, 0.0)?;
            let m_scaled = adamw_param.m.affine(self.beta1, 0.0)?;
            let new_m = m_scaled.add(&m_update)?;
            adamw_param.m.update_from_tensor(&new_m.detach()?)?;

            // Update second moment (v): v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
            // Note: We use the original gradient, NOT modified by weight decay
            let grad_sq = grad.mul(&grad)?;
            let v_update = grad_sq.affine(1.0 - self.beta2, 0.0)?;
            let v_scaled = adamw_param.v.affine(self.beta2, 0.0)?;
            let new_v = v_scaled.add(&v_update)?;
            adamw_param.v.update_from_tensor(&new_v.detach()?)?;

            // Compute bias-corrected estimates
            let m_hat = adamw_param.m.affine(1.0 / bias_correction1, 0.0)?;
            let v_hat = adamw_param.v.affine(1.0 / bias_correction2, 0.0)?;

            // Compute the Adam update: -lr * m_hat / (sqrt(v_hat) + epsilon)
            let v_hat_sqrt = v_hat.sqrt()?;
            let denom = v_hat_sqrt.affine(1.0, self.epsilon)?;
            let adam_update = m_hat.div(&denom)?.affine(-self.learning_rate, 0.0)?;

            // Apply decoupled weight decay: theta = theta - lr * lambda * theta
            // This is the key difference from Adam:
            // - Adam: weight decay is added to gradient (g' = g + lambda * theta)
            // - AdamW: weight decay is applied directly to weights
            // The weight decay is applied AFTER the Adam update, making it
            // independent of the adaptive learning rate
            if self.weight_decay > 0.0 {
                // theta_new = theta_old * (1 - lr * lambda) + adam_update
                // This is equivalent to: theta_new = theta_old - lr * lambda * theta_old + adam_update
                let decay_factor = 1.0 - self.learning_rate * self.weight_decay;
                let param_decayed = adamw_param.param.affine(decay_factor, 0.0)?;
                let new_param = param_decayed.add(&adam_update)?;
                adamw_param.param.update_from_tensor(&new_param.detach()?)?;
            } else {
                // No weight decay: just apply the Adam update
                let new_param = adamw_param.param.add(&adam_update)?;
                adamw_param.param.update_from_tensor(&new_param.detach()?)?;
            }
        }

        Ok(())
    }

    fn zero_grad(&mut self) -> Result<(), OptimizerError> {
        for adamw_param in &mut self.params {
            adamw_param.param.zero_grad()?;
        }
        Ok(())
    }
}
