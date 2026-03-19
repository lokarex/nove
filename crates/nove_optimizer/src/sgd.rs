use nove_tensor::Tensor;

use crate::{Optimizer, OptimizerError};

struct SgdParam {
    param: Tensor,
    velocity: Option<Tensor>,
}

/// Stochastic Gradient Descent (SGD) optimizer with optional momentum and weight decay.
///
/// <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" id="MathJax-script" async></script>
///
/// SGD is a simple yet effective optimization algorithm that updates parameters
/// in the direction of the negative gradient, scaled by the learning rate.
///
/// The update rules for different combinations of momentum and weight decay are:
///
/// 1. **No momentum, no weight decay** (β = 0, λ = 0):
/// $$ \theta_t = \theta_{t-1} - \alpha \cdot g_t $$
///
/// 2. **With momentum, no weight decay** (β > 0, λ = 0):
/// $$ v_t = \beta \cdot v_{t-1} + g_t $$
/// $$ \theta_t = \theta_{t-1} - \alpha \cdot v_t $$
///
/// 3. **No momentum, with weight decay** (β = 0, λ > 0):
/// $$ \theta_t = \theta_{t-1} - \alpha \cdot (g_t + \lambda \cdot \theta_{t-1}) $$
///
/// 4. **With momentum and weight decay** (β > 0, λ > 0):
/// $$ v_t = \beta \cdot v_{t-1} + (g_t + \lambda \cdot \theta_{t-1}) $$
/// $$ \theta_t = \theta_{t-1} - \alpha \cdot v_t $$
///
/// Where:
/// - θ_t is the parameter at time step t
/// - g_t is the gradient at time step t
/// - α is the learning rate
/// - β is the momentum factor
/// - λ is the weight decay coefficient
/// - v_t is the velocity (only used when momentum > 0)
///
/// # Notes
/// * The `Sgd` optimizer is created by the `SgdBuilder`.
/// * Use `SgdBuilder::new(params, learning_rate)` to create a builder.
///
/// # Fields
/// * `params` - The list of parameters to optimize with their velocities.
/// * `learning_rate` - The learning rate (step size).
/// * `momentum` - The momentum factor (0.0 means no momentum).
/// * `weight_decay` - The weight decay coefficient (0.0 means no weight decay).
///
/// # Examples
/// ```no_run
/// use nove::optimizer::SgdBuilder;
/// use nove::tensor::{Device, Tensor};
///
/// let device = Device::cpu();
/// let param1 = Tensor::from_data(vec![1.0, 2.0, 3.0], &device, true).unwrap();
/// let param2 = Tensor::from_data(vec![4.0, 5.0, 6.0], &device, true).unwrap();
///
/// let sgd = SgdBuilder::new(vec![param1, param2], 0.01)
///     .momentum(0.9)                    // Optional, default is 0.0 (no momentum)
///     .weight_decay(0.0001)             // Optional, default is 0.0 (no weight decay)
///     .build()
///     .unwrap();
/// ```
pub struct Sgd {
    params: Vec<SgdParam>,
    learning_rate: f64,
    momentum: f64,
    weight_decay: f64,
}

/// The builder for the SGD optimizer.
///
/// # Required Arguments
/// * `params` - The list of parameters to optimize.
/// * `learning_rate` - The learning rate (step size).
///
/// # Optional Arguments
/// * `momentum` - The momentum factor. Default is `0.0` (no momentum).
/// * `weight_decay` - The weight decay coefficient. Default is `0.0` (no weight decay).
///
/// # Fields
/// * `params` - The list of parameters to optimize.
/// * `learning_rate` - The learning rate (step size).
/// * `momentum` - The momentum factor.
/// * `weight_decay` - The weight decay coefficient.
///
/// # Examples
/// ```no_run
/// use nove::optimizer::SgdBuilder;
/// use nove::tensor::{Device, Tensor};
///
/// let device = Device::cpu();
/// let param1 = Tensor::from_data(vec![1.0, 2.0, 3.0], &device, true).unwrap();
/// let param2 = Tensor::from_data(vec![4.0, 5.0, 6.0], &device, true).unwrap();
///
/// let sgd = SgdBuilder::new(vec![param1, param2], 0.01)
///     .momentum(0.9)
///     .weight_decay(0.0001)
///     .build()
///     .unwrap();
/// ```
pub struct SgdBuilder {
    params: Vec<Tensor>,
    learning_rate: f64,
    momentum: f64,
    weight_decay: f64,
}

impl SgdBuilder {
    /// Create a new SgdBuilder with the required parameters and learning rate.
    ///
    /// # Arguments
    /// * `params` - The list of parameters to optimize.
    /// * `learning_rate` - The learning rate (step size).
    ///
    /// # Returns
    /// * `Self` - A new SgdBuilder instance.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::optimizer::SgdBuilder;
    /// use nove::tensor::{Device, Tensor};
    ///
    /// let device = Device::cpu();
    /// let param1 = Tensor::from_data(vec![1.0, 2.0, 3.0], &device, true).unwrap();
    /// let param2 = Tensor::from_data(vec![4.0, 5.0, 6.0], &device, true).unwrap();
    ///
    /// let sgd = SgdBuilder::new(vec![param1, param2], 0.01)
    ///     .momentum(0.9)
    ///     .weight_decay(0.0001)
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn new(params: Vec<Tensor>, learning_rate: f64) -> Self {
        Self {
            params,
            learning_rate,
            momentum: 0.0,
            weight_decay: 0.0,
        }
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
    /// use nove::optimizer::SgdBuilder;
    /// use nove::tensor::{Device, Tensor};
    ///
    /// let device = Device::cpu();
    /// let param = Tensor::from_data(vec![1.0, 2.0, 3.0], &device, true).unwrap();
    /// let mut builder = SgdBuilder::new(vec![param], 0.01);
    /// builder.momentum(0.9);
    /// ```
    pub fn momentum(&mut self, momentum: f64) -> &mut Self {
        self.momentum = momentum;
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
    /// use nove::optimizer::SgdBuilder;
    /// use nove::tensor::{Device, Tensor};
    ///
    /// let device = Device::cpu();
    /// let param = Tensor::from_data(vec![1.0, 2.0, 3.0], &device, true).unwrap();
    /// let mut builder = SgdBuilder::new(vec![param], 0.01);
    /// builder.weight_decay(0.0001);
    /// ```
    pub fn weight_decay(&mut self, weight_decay: f64) -> &mut Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Build the SGD optimizer.
    ///
    /// # Returns
    /// * `Ok(Sgd)` - The built SGD optimizer.
    /// * `Err(OptimizerError)` - The error when building the SGD optimizer.
    ///
    /// # Errors
    /// * `OptimizerError::InvalidArgument` - If `learning_rate`, `momentum`, or `weight_decay` is invalid.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::optimizer::SgdBuilder;
    /// use nove::tensor::{Device, Tensor};
    ///
    /// let device = Device::cpu();
    /// let param = Tensor::from_data(vec![1.0, 2.0, 3.0], &device, true).unwrap();
    ///
    /// let sgd = SgdBuilder::new(vec![param], 0.01)
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn build(&self) -> Result<Sgd, OptimizerError> {
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

        if self.momentum < 0.0 {
            return Err(OptimizerError::InvalidArgument(
                "momentum must be non-negative".to_string(),
            ));
        }

        if self.weight_decay < 0.0 {
            return Err(OptimizerError::InvalidArgument(
                "weight_decay must be non-negative".to_string(),
            ));
        }

        if params.is_empty() {
            return Ok(Sgd {
                params: vec![],
                learning_rate,
                momentum: self.momentum,
                weight_decay: self.weight_decay,
            });
        }

        let device = params[0].device()?;
        let dtype = params[0].dtype()?;

        let sgd_params = params
            .into_iter()
            .map(|param| {
                let velocity = if self.momentum > 0.0 {
                    let shape = param.shape()?;
                    Some(Tensor::zeros(&shape, &dtype, &device, false)?)
                } else {
                    None
                };
                Ok(SgdParam { param, velocity })
            })
            .collect::<Result<Vec<_>, OptimizerError>>()?;

        Ok(Sgd {
            params: sgd_params,
            learning_rate,
            momentum: self.momentum,
            weight_decay: self.weight_decay,
        })
    }
}

impl Optimizer for Sgd {
    type StepOutput = ();

    fn step(&mut self) -> Result<Self::StepOutput, OptimizerError> {
        for sgd_param in &mut self.params {
            if !sgd_param.param.grad_enabled()? {
                continue;
            }

            let grad = sgd_param.param.grad()?.ok_or(OptimizerError::OtherError(
                "Sgd: parameter gradient is None".to_string(),
            ))?;

            // Apply weight decay if specified
            let grad_with_decay = if self.weight_decay > 0.0 {
                let param_with_decay = sgd_param.param.affine(self.weight_decay, 0.0)?;
                grad.add(&param_with_decay)?
            } else {
                grad
            };

            if self.momentum > 0.0 {
                // Update velocity: v_t = momentum * v_{t-1} + grad_with_decay
                let velocity = sgd_param
                    .velocity
                    .as_mut()
                    .ok_or(OptimizerError::OtherError(
                        "Sgd: velocity is None but momentum > 0".to_string(),
                    ))?;

                let velocity_scaled = velocity.affine(self.momentum, 0.0)?;
                let new_velocity = velocity_scaled.add(&grad_with_decay)?;
                velocity.update_from_tensor(&new_velocity.detach()?)?;

                // Update parameter: param_t = param_{t-1} - learning_rate * v_t
                let velocity_scaled_by_lr = velocity.affine(self.learning_rate, 0.0)?;
                let new_param = sgd_param.param.sub(&velocity_scaled_by_lr)?;
                sgd_param.param.update_from_tensor(&new_param.detach()?)?;
            } else {
                // Update parameter directly: param_t = param_{t-1} - learning_rate * grad_with_decay
                let scaled_grad = grad_with_decay.affine(self.learning_rate, 0.0)?;
                let new_param = sgd_param.param.sub(&scaled_grad)?;
                sgd_param.param.update_from_tensor(&new_param.detach()?)?;
            }
        }

        Ok(())
    }

    fn zero_grad(&mut self) -> Result<(), OptimizerError> {
        for sgd_param in &mut self.params {
            sgd_param.param.zero_grad()?;
        }
        Ok(())
    }
}
