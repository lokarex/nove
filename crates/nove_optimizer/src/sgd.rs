use nove_tensor::Tensor;

use crate::{Optimizer, OptimizerError};

pub struct Sgd {
    params: Vec<Tensor>,
    learning_rate: f64,
}

impl Sgd {
    pub fn new(params: Vec<Tensor>, learning_rate: f64) -> Self {
        Self {
            params,
            learning_rate,
        }
    }

    pub fn zero_grad(&mut self) -> Result<(), OptimizerError> {
        for param in &mut self.params {
            param.zero_grad()?;
        }
        Ok(())
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
                        .affine(-self.learning_rate, 0f64)?,
                )?,
            )?;
        }
        Ok(())
    }
}
