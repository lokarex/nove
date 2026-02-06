use nove_tensor::Tensor;

use crate::{LossFn, LossFnError};

pub struct NllLossFn;

impl NllLossFn {
    pub fn new() -> Self {
        Self
    }
}

impl LossFn for NllLossFn {
    type Input = (Tensor, Tensor);
    type Output = Tensor;

    fn loss(&self, input: Self::Input) -> Result<Self::Output, LossFnError> {
        let (input, target) = input;

        match (input.num_dim()?, target.num_dim()?) {
            (2, 1) => {
                let batch_size = input.shape()?.dims()[0];
                Ok(input
                    .gather(&target.unsqueeze(1)?, 1)?
                    .sum(None)?
                    .affine(-1f64 / batch_size as f64, 0f64)?)
            }
            _ => Err(LossFnError::OtherError(
                "NllLossFn: input and target must be 2D and 1D respectively".to_string(),
            )),
        }
    }
}
