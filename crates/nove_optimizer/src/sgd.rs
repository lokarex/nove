use nove_tensor::Tensor;

use crate::{Optimizer, OptimizerError};

/// Stochastic Gradient Descent (SGD) optimizer.
///
/// <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" id="MathJax-script" async></script>
///
/// SGD is a simple yet effective optimization algorithm that updates parameters
/// in the direction of the negative gradient, scaled by the learning rate.
///
/// The parameter update is performed as follows:
///
/// $$ \theta_t = \theta_{t-1} - \alpha \cdot g_t $$
///
/// Where:
/// - θ_t is the parameter at time step t
/// - g_t is the gradient at time step t
/// - α is the learning rate
///
/// # Fields
/// * `params` - The list of parameters to optimize.
/// * `learning_rate` - The learning rate (step size).
///
/// # Examples
/// ```
/// use nove::optimizer::Sgd;
/// use nove::tensor::{Device, Tensor};
///
/// let device = Device::cpu();
/// let param1 = Tensor::from_data(vec![1.0, 2.0, 3.0], &device, true).unwrap();
/// let param2 = Tensor::from_data(vec![4.0, 5.0, 6.0], &device, true).unwrap();
///
/// let sgd = Sgd::new(vec![param1, param2], 0.01);
/// ```
pub struct Sgd {
    params: Vec<Tensor>,
    learning_rate: f64,
}

impl Sgd {
    /// Create a new SGD optimizer.
    ///
    /// # Arguments
    /// * `params` - The list of parameters to optimize.
    /// * `learning_rate` - The learning rate (step size).
    ///
    /// # Returns
    /// * `Sgd` - The SGD optimizer.
    ///
    /// # Examples
    /// ```
    /// use nove::optimizer::Sgd;
    /// use nove::tensor::{Device, Tensor};
    ///
    /// let device = Device::cpu();
    /// let param = Tensor::from_data(vec![1.0, 2.0, 3.0], &device, true).unwrap();
    /// let sgd = Sgd::new(vec![param], 0.01);
    /// ```
    pub fn new(params: Vec<Tensor>, learning_rate: f64) -> Self {
        Self {
            params,
            learning_rate,
        }
    }
}

impl Optimizer for Sgd {
    type StepOutput = ();

    fn step(&mut self) -> Result<Self::StepOutput, OptimizerError> {
        for param in &mut self.params {
            param.update_from_tensor(
                &param.sub(
                    &param
                        .grad()?
                        .ok_or(OptimizerError::OtherError(
                            "Sgd: parameter gradient is None".to_string(),
                        ))?
                        .affine(self.learning_rate, 0f64)?,
                )?,
            )?;
            param.clear_grad()?;
        }
        Ok(())
    }

    fn zero_grad(&mut self) -> Result<(), OptimizerError> {
        for param in &mut self.params {
            param.zero_grad()?;
        }
        Ok(())
    }
}
