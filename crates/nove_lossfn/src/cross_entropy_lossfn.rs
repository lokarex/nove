use nove_tensor::Tensor;

use crate::{LossFn, LossFnError, NllLossFn, common::log_softmax};

pub struct CrossEntropyLossFn {
    nll_lossfn: NllLossFn,
}

impl CrossEntropyLossFn {
    pub fn new() -> Self {
        Self {
            nll_lossfn: NllLossFn::new(),
        }
    }
}

impl LossFn for CrossEntropyLossFn {
    type Input = (Tensor, Tensor);
    type Output = Tensor;

    fn loss(&self, input: Self::Input) -> Result<Self::Output, LossFnError> {
        let (input, target) = input;

        self.nll_lossfn.loss((log_softmax(&input, 1)?, target))
    }
}
