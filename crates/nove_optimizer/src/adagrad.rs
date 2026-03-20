use nove_tensor::Tensor;

use crate::{Optimizer, OptimizerError};

struct AdagradParam {
    param: Tensor,
    sum_square_grad: Tensor,
}

/// Adagrad (Adaptive Gradient Algorithm) optimizer.
///
/// <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" id="MathJax-script" async></script>
///
/// Adagrad is an adaptive learning rate optimization algorithm that adapts the
/// learning rate to each parameter based on the historical gradient information.
///
/// The update rules are as follows:
///
/// 1. **Accumulate squared gradients**:
/// $$ G_t = G_{t-1} + g_t^2 $$
///
/// 2. **Compute parameter update**:
/// $$ \theta_t = \theta_{t-1} - \alpha \cdot \frac{g_t}{\sqrt{G_t} + \epsilon} $$
///
/// Where:
/// - θ_t is the parameter at time step t
/// - g_t is the gradient at time step t
/// - G_t is the sum of squared gradients up to time t
/// - α is the learning rate
/// - ε is a small constant for numerical stability
/// - t is the current time step
///
/// # Characteristics
/// * The learning rate for each parameter decreases as more updates are performed,
///   because G_t grows monotonically.
/// * This makes Adagrad suitable for sparse data where infrequent parameters
///   get larger updates.
/// * The main drawback is that the learning rate can become infinitesimally small
///   over time, effectively stopping learning.
///
/// # Notes
/// * The `Adagrad` optimizer is created by the `AdagradBuilder`.
/// * Use `AdagradBuilder::new(params, learning_rate)` to create a builder.
///
/// # Fields
/// * `params` - The list of parameters to optimize.
/// * `learning_rate` - The learning rate (step size).
/// * `epsilon` - A small constant for numerical stability.
/// * `weight_decay` - The weight decay coefficient (0.0 means no weight decay).
///
/// # Examples
/// ```no_run
/// use nove::optimizer::AdagradBuilder;
/// use nove::tensor::{Device, Tensor};
///
/// let device = Device::cpu();
/// let param1 = Tensor::from_data(vec![1.0, 2.0, 3.0], &device, true).unwrap();
/// let param2 = Tensor::from_data(vec![4.0, 5.0, 6.0], &device, true).unwrap();
///
/// let adagrad = AdagradBuilder::new(vec![param1, param2], 0.01)
///     .epsilon(1e-8)                  // Optional, default is 1e-8
///     .weight_decay(0.0001)           // Optional, default is 0.0 (no weight decay)
///     .build()
///     .unwrap();
/// ```
pub struct Adagrad {
    params: Vec<AdagradParam>,
    learning_rate: f64,
    epsilon: f64,
    weight_decay: f64,
}

/// The builder for the Adagrad optimizer.
///
/// # Required Arguments
/// * `params` - The list of parameters to optimize.
/// * `learning_rate` - The learning rate (step size).
///
/// # Optional Arguments
/// * `epsilon` - A small constant for numerical stability. Default is `1e-8`.
/// * `weight_decay` - The weight decay coefficient. Default is `0.0` (no weight decay).
///
/// # Fields
/// * `params` - The list of parameters to optimize.
/// * `learning_rate` - The learning rate (step size).
/// * `epsilon` - A small constant for numerical stability.
///
/// # Examples
/// ```no_run
/// use nove::optimizer::AdagradBuilder;
/// use nove::tensor::{Device, Tensor};
///
/// let device = Device::cpu();
/// let param1 = Tensor::from_data(vec![1.0, 2.0, 3.0], &device, true).unwrap();
/// let param2 = Tensor::from_data(vec![4.0, 5.0, 6.0], &device, true).unwrap();
///
/// let adagrad = AdagradBuilder::new(vec![param1, param2], 0.01)
///     .epsilon(1e-8)
///     .weight_decay(0.0001)
///     .build()
///     .unwrap();
/// ```
pub struct AdagradBuilder {
    params: Vec<Tensor>,
    learning_rate: f64,
    epsilon: f64,
    weight_decay: f64,
}

impl AdagradBuilder {
    /// Create a new AdagradBuilder with the required parameters and learning rate.
    ///
    /// # Arguments
    /// * `params` - The list of parameters to optimize.
    /// * `learning_rate` - The learning rate (step size).
    ///
    /// # Returns
    /// * `Self` - A new AdagradBuilder instance.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::optimizer::AdagradBuilder;
    /// use nove::tensor::{Device, Tensor};
    ///
    /// let device = Device::cpu();
    /// let param1 = Tensor::from_data(vec![1.0, 2.0, 3.0], &device, true).unwrap();
    /// let param2 = Tensor::from_data(vec![4.0, 5.0, 6.0], &device, true).unwrap();
    ///
    /// let adagrad = AdagradBuilder::new(vec![param1, param2], 0.01)
    ///     .epsilon(1e-8)
    ///     .weight_decay(0.0001)
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn new(params: Vec<Tensor>, learning_rate: f64) -> Self {
        Self {
            params,
            learning_rate,
            epsilon: 1e-8,
            weight_decay: 0.0,
        }
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
    /// use nove::optimizer::AdagradBuilder;
    /// use nove::tensor::{Device, Tensor};
    ///
    /// let device = Device::cpu();
    /// let param = Tensor::from_data(vec![1.0, 2.0, 3.0], &device, true).unwrap();
    /// let mut builder = AdagradBuilder::new(vec![param], 0.01);
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
    /// use nove::optimizer::AdagradBuilder;
    /// use nove::tensor::{Device, Tensor};
    ///
    /// let device = Device::cpu();
    /// let param = Tensor::from_data(vec![1.0, 2.0, 3.0], &device, true).unwrap();
    /// let mut builder = AdagradBuilder::new(vec![param], 0.01);
    /// builder.weight_decay(0.0001);
    /// ```
    pub fn weight_decay(&mut self, weight_decay: f64) -> &mut Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Build the Adagrad optimizer.
    ///
    /// # Returns
    /// * `Ok(Adagrad)` - The built Adagrad optimizer.
    /// * `Err(OptimizerError)` - The error when building the Adagrad optimizer.
    ///
    /// # Errors
    /// * `OptimizerError::InvalidArgument` - If `learning_rate`, `epsilon`, or `weight_decay` is invalid.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::optimizer::AdagradBuilder;
    /// use nove::tensor::{Device, Tensor};
    ///
    /// let device = Device::cpu();
    /// let param = Tensor::from_data(vec![1.0, 2.0, 3.0], &device, true).unwrap();
    ///
    /// let adagrad = AdagradBuilder::new(vec![param], 0.01)
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn build(&self) -> Result<Adagrad, OptimizerError> {
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
            return Ok(Adagrad {
                params: vec![],
                learning_rate,
                epsilon: self.epsilon,
                weight_decay: self.weight_decay,
            });
        }

        let device = params[0].device()?;
        let dtype = params[0].dtype()?;

        let adagrad_params = params
            .into_iter()
            .map(|param| {
                let shape = param.shape()?;
                let sum_square_grad = Tensor::zeros(&shape, &dtype, &device, false)?;
                Ok(AdagradParam {
                    param,
                    sum_square_grad,
                })
            })
            .collect::<Result<Vec<_>, OptimizerError>>()?;

        Ok(Adagrad {
            params: adagrad_params,
            learning_rate,
            epsilon: self.epsilon,
            weight_decay: self.weight_decay,
        })
    }
}

impl Optimizer for Adagrad {
    type StepOutput = ();

    fn step(&mut self) -> Result<Self::StepOutput, OptimizerError> {
        for adagrad_param in &mut self.params {
            if !adagrad_param.param.grad_enabled()? {
                continue;
            }

            let grad = adagrad_param
                .param
                .grad()?
                .ok_or(OptimizerError::OtherError(
                    "Adagrad: parameter gradient is None".to_string(),
                ))?;

            // Apply weight decay if specified
            let grad_with_decay = if self.weight_decay > 0.0 {
                let param_with_decay = adagrad_param.param.affine(self.weight_decay, 0.0)?;
                grad.add(&param_with_decay)?
            } else {
                grad
            };

            // Accumulate squared gradients: G_t = G_{t-1} + g_t^2
            let grad_sq = grad_with_decay.mul(&grad_with_decay)?;
            let new_sum_square_grad = adagrad_param.sum_square_grad.add(&grad_sq)?;
            adagrad_param
                .sum_square_grad
                .update_from_tensor(&new_sum_square_grad.detach()?)?;

            // Compute parameter update: theta_t = theta_{t-1} - lr * g_t / (sqrt(G_t) + epsilon)
            let sum_square_grad_sqrt = adagrad_param.sum_square_grad.sqrt()?;
            let denom = sum_square_grad_sqrt.affine(1.0, self.epsilon)?;
            let adaptive_term = grad_with_decay.div(&denom)?;
            let update_scaled_by_lr = adaptive_term.affine(self.learning_rate, 0.0)?;
            let new_param = adagrad_param.param.sub(&update_scaled_by_lr)?;
            adagrad_param
                .param
                .update_from_tensor(&new_param.detach()?)?;
        }

        Ok(())
    }

    fn zero_grad(&mut self) -> Result<(), OptimizerError> {
        for adagrad_param in &mut self.params {
            adagrad_param.param.zero_grad()?;
        }
        Ok(())
    }
}
