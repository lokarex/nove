use nove_tensor::Tensor;

use crate::{Optimizer, OptimizerError};

struct RmspropParam {
    param: Tensor,
    square_avg: Tensor,
    momentum_buffer: Option<Tensor>,
}

/// RMSprop (Root Mean Square Propagation) optimizer.
///
/// <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" id="MathJax-script" async></script>
///
/// RMSprop is an adaptive learning rate optimization algorithm that maintains
/// a moving average of squared gradients to scale the learning rate adaptively.
///
/// The update rules, with optional momentum, are as follows:
///
/// 1. **Update squared gradient average**:
/// $$ E[g_t^2] = \rho \cdot E[g_{t-1}^2] + (1 - \rho) \cdot g_t^2 $$
///
/// 2. **Compute the adaptive learning rate term**:
/// $$ \text{adaptive\_term} = \frac{g_t}{\sqrt{E[g_t^2]} + \epsilon} $$
///
/// 3. **With momentum** (if momentum > 0):
/// $$ v_t = \mu \cdot v_{t-1} + \text{adaptive\_term} $$
/// $$ \theta_t = \theta_{t-1} - \alpha \cdot v_t $$
///
/// 4. **Without momentum** (if momentum = 0):
/// $$ \theta_t = \theta_{t-1} - \alpha \cdot \text{adaptive\_term} $$
///
/// Where:
/// - θ_t is the parameter at time step t
/// - g_t is the gradient at time step t
/// - \(E[g_t^2]\) is the moving average of squared gradients
/// - v_t is the momentum buffer (only when momentum > 0)
/// - α is the learning rate
/// - ρ is the decay rate (also called alpha or gamma)
/// - μ is the momentum factor (0.0 means no momentum)
/// - ε is a small constant for numerical stability
/// - t is the current time step
///
/// # Notes
/// * The `Rmsprop` optimizer is created by the `RmspropBuilder`.
/// * Use `RmspropBuilder::new(params, learning_rate)` to create a builder.
///
/// # Fields
/// * `params` - The list of parameters to optimize.
/// * `learning_rate` - The learning rate (step size).
/// * `rho` - The decay rate for the moving average of squared gradients.
/// * `momentum` - The momentum factor (0.0 means no momentum).
/// * `epsilon` - A small constant for numerical stability.
/// * `weight_decay` - The weight decay coefficient (0.0 means no weight decay).
///
/// # Examples
/// ```no_run
/// use nove::optimizer::RmspropBuilder;
/// use nove::tensor::{Device, Tensor};
///
/// let device = Device::cpu();
/// let param1 = Tensor::from_data(vec![1.0, 2.0, 3.0], &device, true).unwrap();
/// let param2 = Tensor::from_data(vec![4.0, 5.0, 6.0], &device, true).unwrap();
///
/// let rmsprop = RmspropBuilder::new(vec![param1, param2], 0.001)
///     .rho(0.99)                      // Optional, default is 0.99
///     .momentum(0.0)                  // Optional, default is 0.0 (no momentum)
///     .epsilon(1e-8)                  // Optional, default is 1e-8
///     .weight_decay(0.0001)           // Optional, default is 0.0 (no weight decay)
///     .build()
///     .unwrap();
/// ```
pub struct Rmsprop {
    params: Vec<RmspropParam>,
    learning_rate: f64,
    rho: f64,
    momentum: f64,
    epsilon: f64,
    weight_decay: f64,
}

/// The builder for the RMSprop optimizer.
///
/// # Required Arguments
/// * `params` - The list of parameters to optimize.
/// * `learning_rate` - The learning rate (step size).
///
/// # Optional Arguments
/// * `rho` - The decay rate for the moving average of squared gradients. Default is `0.99`.
/// * `momentum` - The momentum factor. Default is `0.0` (no momentum).
/// * `epsilon` - A small constant for numerical stability. Default is `1e-8`.
/// * `weight_decay` - The weight decay coefficient. Default is `0.0` (no weight decay).
///
/// # Fields
/// * `params` - The list of parameters to optimize.
/// * `learning_rate` - The learning rate (step size).
/// * `rho` - The decay rate for the moving average of squared gradients.
/// * `momentum` - The momentum factor.
/// * `epsilon` - A small constant for numerical stability.
///
/// # Examples
/// ```no_run
/// use nove::optimizer::RmspropBuilder;
/// use nove::tensor::{Device, Tensor};
///
/// let device = Device::cpu();
/// let param1 = Tensor::from_data(vec![1.0, 2.0, 3.0], &device, true).unwrap();
/// let param2 = Tensor::from_data(vec![4.0, 5.0, 6.0], &device, true).unwrap();
///
/// let rmsprop = RmspropBuilder::new(vec![param1, param2], 0.001)
///     .rho(0.99)
///     .momentum(0.0)
///     .epsilon(1e-8)
///     .weight_decay(0.0001)
///     .build()
///     .unwrap();
/// ```
pub struct RmspropBuilder {
    params: Vec<Tensor>,
    learning_rate: f64,
    rho: f64,
    momentum: f64,
    epsilon: f64,
    weight_decay: f64,
}

impl RmspropBuilder {
    /// Create a new RmspropBuilder with the required parameters and learning rate.
    ///
    /// # Arguments
    /// * `params` - The list of parameters to optimize.
    /// * `learning_rate` - The learning rate (step size).
    ///
    /// # Returns
    /// * `Self` - A new RmspropBuilder instance.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::optimizer::RmspropBuilder;
    /// use nove::tensor::{Device, Tensor};
    ///
    /// let device = Device::cpu();
    /// let param1 = Tensor::from_data(vec![1.0, 2.0, 3.0], &device, true).unwrap();
    /// let param2 = Tensor::from_data(vec![4.0, 5.0, 6.0], &device, true).unwrap();
    ///
    /// let rmsprop = RmspropBuilder::new(vec![param1, param2], 0.001)
    ///     .rho(0.99)
    ///     .momentum(0.0)
    ///     .epsilon(1e-8)
    ///     .weight_decay(0.0001)
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn new(params: Vec<Tensor>, learning_rate: f64) -> Self {
        Self {
            params,
            learning_rate,
            rho: 0.99,
            momentum: 0.0,
            epsilon: 1e-8,
            weight_decay: 0.0,
        }
    }

    /// Configure the decay rate for the moving average of squared gradients.
    ///
    /// # Arguments
    /// * `rho` - The decay rate for the moving average of squared gradients.
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured rho.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::optimizer::RmspropBuilder;
    /// use nove::tensor::{Device, Tensor};
    ///
    /// let device = Device::cpu();
    /// let param = Tensor::from_data(vec![1.0, 2.0, 3.0], &device, true).unwrap();
    /// let mut builder = RmspropBuilder::new(vec![param], 0.001);
    /// builder.rho(0.99);
    /// ```
    pub fn rho(&mut self, rho: f64) -> &mut Self {
        self.rho = rho;
        self
    }

    /// Configure the momentum factor.
    ///
    /// # Arguments
    /// * `momentum` - The momentum factor.
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured momentum.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::optimizer::RmspropBuilder;
    /// use nove::tensor::{Device, Tensor};
    ///
    /// let device = Device::cpu();
    /// let param = Tensor::from_data(vec![1.0, 2.0, 3.0], &device, true).unwrap();
    /// let mut builder = RmspropBuilder::new(vec![param], 0.001);
    /// builder.momentum(0.9);
    /// ```
    pub fn momentum(&mut self, momentum: f64) -> &mut Self {
        self.momentum = momentum;
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
    /// use nove::optimizer::RmspropBuilder;
    /// use nove::tensor::{Device, Tensor};
    ///
    /// let device = Device::cpu();
    /// let param = Tensor::from_data(vec![1.0, 2.0, 3.0], &device, true).unwrap();
    /// let mut builder = RmspropBuilder::new(vec![param], 0.001);
    /// builder.epsilon(1e-8);
    /// ```
    pub fn epsilon(&mut self, epsilon: f64) -> &mut Self {
        self.epsilon = epsilon;
        self
    }

    /// Configure the weight decay coefficient.
    ///
    /// # Arguments
    /// * `weight_decay` - The weight decay coefficient.
    ///
    /// # Returns
    /// * `&mut Self` - The builder with the configured weight decay.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::optimizer::RmspropBuilder;
    /// use nove::tensor::{Device, Tensor};
    ///
    /// let device = Device::cpu();
    /// let param = Tensor::from_data(vec![1.0, 2.0, 3.0], &device, true).unwrap();
    /// let mut builder = RmspropBuilder::new(vec![param], 0.001);
    /// builder.weight_decay(0.0001);
    /// ```
    pub fn weight_decay(&mut self, weight_decay: f64) -> &mut Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Build the RMSprop optimizer.
    ///
    /// # Returns
    /// * `Ok(Rmsprop)` - The built RMSprop optimizer.
    /// * `Err(OptimizerError)` - The error when building the RMSprop optimizer.
    ///
    /// # Errors
    /// * `OptimizerError::InvalidArgument` - If `learning_rate`, `rho`, `momentum`, `epsilon`, or `weight_decay` is invalid.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::optimizer::RmspropBuilder;
    /// use nove::tensor::{Device, Tensor};
    ///
    /// let device = Device::cpu();
    /// let param = Tensor::from_data(vec![1.0, 2.0, 3.0], &device, true).unwrap();
    ///
    /// let rmsprop = RmspropBuilder::new(vec![param], 0.001)
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn build(&self) -> Result<Rmsprop, OptimizerError> {
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

        if self.rho < 0.0 || self.rho > 1.0 {
            return Err(OptimizerError::InvalidArgument(
                "rho must be in [0, 1]".to_string(),
            ));
        }

        if self.momentum < 0.0 {
            return Err(OptimizerError::InvalidArgument(
                "momentum must be non-negative".to_string(),
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
            return Ok(Rmsprop {
                params: vec![],
                learning_rate,
                rho: self.rho,
                momentum: self.momentum,
                epsilon: self.epsilon,
                weight_decay: self.weight_decay,
            });
        }

        let device = params[0].device()?;
        let dtype = params[0].dtype()?;

        let rmsprop_params = params
            .into_iter()
            .map(|param| {
                let shape = param.shape()?;
                let square_avg = Tensor::zeros(&shape, &dtype, &device, false)?;
                let momentum_buffer = if self.momentum > 0.0 {
                    Some(Tensor::zeros(&shape, &dtype, &device, false)?)
                } else {
                    None
                };
                Ok(RmspropParam {
                    param,
                    square_avg,
                    momentum_buffer,
                })
            })
            .collect::<Result<Vec<_>, OptimizerError>>()?;

        Ok(Rmsprop {
            params: rmsprop_params,
            learning_rate,
            rho: self.rho,
            momentum: self.momentum,
            epsilon: self.epsilon,
            weight_decay: self.weight_decay,
        })
    }
}

impl Optimizer for Rmsprop {
    type StepOutput = ();

    fn step(&mut self) -> Result<Self::StepOutput, OptimizerError> {
        for rmsprop_param in &mut self.params {
            if !rmsprop_param.param.grad_enabled()? {
                continue;
            }

            let grad = rmsprop_param.param.grad()?.ok_or(OptimizerError::OtherError(
                "Rmsprop: parameter gradient is None".to_string(),
            ))?;

            // Apply weight decay if specified
            let grad_with_decay = if self.weight_decay > 0.0 {
                let param_with_decay = rmsprop_param.param.affine(self.weight_decay, 0.0)?;
                grad.add(&param_with_decay)?
            } else {
                grad
            };

            // Update squared gradient average: E[g^2]_t = rho * E[g^2]_{t-1} + (1 - rho) * g_t^2
            let grad_sq = grad_with_decay.mul(&grad_with_decay)?;
            let square_avg_scaled = rmsprop_param.square_avg.affine(self.rho, 0.0)?;
            let update_scaled = grad_sq.affine(1.0 - self.rho, 0.0)?;
            let new_square_avg = square_avg_scaled.add(&update_scaled)?;
            rmsprop_param
                .square_avg
                .update_from_tensor(&new_square_avg.detach()?)?;

            // Compute adaptive term: g_t / (sqrt(E[g^2]_t) + epsilon)
            let square_avg_sqrt = rmsprop_param.square_avg.sqrt()?;
            let denom = square_avg_sqrt.affine(1.0, self.epsilon)?;
            let adaptive_term = grad_with_decay.div(&denom)?;

            if self.momentum > 0.0 {
                // Update momentum buffer: v_t = momentum * v_{t-1} + adaptive_term
                let momentum_buffer = rmsprop_param
                    .momentum_buffer
                    .as_mut()
                    .ok_or(OptimizerError::OtherError(
                        "Rmsprop: momentum_buffer is None but momentum > 0".to_string(),
                    ))?;

                let momentum_scaled = momentum_buffer.affine(self.momentum, 0.0)?;
                let new_momentum = momentum_scaled.add(&adaptive_term)?;
                momentum_buffer.update_from_tensor(&new_momentum.detach()?)?;

                // Update parameter: theta_t = theta_{t-1} - learning_rate * v_t
                let update_scaled_by_lr = momentum_buffer.affine(self.learning_rate, 0.0)?;
                let new_param = rmsprop_param.param.sub(&update_scaled_by_lr)?;
                rmsprop_param.param.update_from_tensor(&new_param.detach()?)?;
            } else {
                // Update parameter directly: theta_t = theta_{t-1} - learning_rate * adaptive_term
                let update_scaled_by_lr = adaptive_term.affine(self.learning_rate, 0.0)?;
                let new_param = rmsprop_param.param.sub(&update_scaled_by_lr)?;
                rmsprop_param.param.update_from_tensor(&new_param.detach()?)?;
            }
        }

        Ok(())
    }

    fn zero_grad(&mut self) -> Result<(), OptimizerError> {
        for rmsprop_param in &mut self.params {
            rmsprop_param.param.zero_grad()?;
        }
        Ok(())
    }
}