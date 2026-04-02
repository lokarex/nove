use nove_tensor::Tensor;

use crate::{LossFn, LossFnError, NllLoss, common::log_softmax};

/// Cross Entropy Loss function.
///
/// <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" id="MathJax-script" async></script>
///
/// The Cross Entropy loss combines LogSoftmax and Negative Log Likelihood (NLL) loss
/// into a single loss function. It is commonly used for multi-class classification tasks.
///
/// The loss is computed by applying LogSoftmax followed by NLL loss:
///
/// $$ \text{CrossEntropyLoss}(x, y) = \text{NLL}(\text{LogSoftmax}(x), y) $$
///
/// Where:
/// - x is the input tensor of shape (batch_size, num_classes) containing raw logits
/// - y is the target tensor of shape (batch_size) containing class indices
/// - NLL is the Negative Log Likelihood loss function
/// - LogSoftmax is the log-softmax function applied along the class dimension
///
/// # Notes:
/// * The input is a `(input, target)` tuple
///   - `input`: A 2D tensor of shape (batch_size, num_classes) containing raw logits
///   - `target`: A 1D tensor of shape (batch_size) containing real class indices
/// * The output is A scalar tensor representing the average cross entropy loss over the batch.
#[derive(Debug, Clone)]
pub struct CrossEntropyLoss {
    nll_lossfn: NllLoss,
}

impl CrossEntropyLoss {
    pub fn new() -> Self {
        Self {
            nll_lossfn: NllLoss::new(),
        }
    }
}

impl Default for CrossEntropyLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl LossFn for CrossEntropyLoss {
    type Input = (Tensor, Tensor);
    type Output = Tensor;

    fn loss(&self, input: Self::Input) -> Result<Self::Output, LossFnError> {
        let (input, target) = input;

        self.nll_lossfn.loss((log_softmax(&input, 1)?, target))
    }
}
