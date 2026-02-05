use nove_tensor::Tensor;

use crate::{LossFn, LossFnError};

pub struct CrossEntropyLossFn;

impl CrossEntropyLossFn {
    pub fn new() -> Self {
        Self
    }
}

impl LossFn for CrossEntropyLossFn {
    type Input = (Tensor, Tensor);
    type Output = Tensor;

    fn loss(&self, _input: Self::Input) -> Result<Self::Output, LossFnError> {
        todo!()
    }
}
