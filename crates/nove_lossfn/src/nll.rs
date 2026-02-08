use nove_tensor::Tensor;

use crate::{LossFn, LossFnError};

/// Negative Log Likelihood Loss function.
///
/// <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" id="MathJax-script" async></script>
///
/// The Negative Log Likelihood (NLL) loss measures the performance of a classification model
/// whose output is log-probabilities. The loss encourages the model to assign higher
/// probability to the correct class.
///
/// Given a batch of N samples, the NLL loss is defined as:
///
/// $$ \text{NLL}(x, y) = -\frac{1}{N} \sum_{i=1}^{N} x_i\[y_i\] $$
///
/// Where:
/// - N is the batch size
/// - x_i is the log-probability output for the i-th sample (typically from log_softmax)
/// - y_i is the target class index for the i-th sample
/// - x_i\[y_i\] is the log-probability at the target class position
///
/// # Note:
/// * The input is a `(input, target)` tuple
///   - `input`: A 2D tensor of shape (batch_size, num_classes) containing log-probabilities
///   - `target`: A 1D tensor of shape (batch_size) containing real class indices
/// * The output is A scalar tensor representing the average NLL loss over the batch.
pub struct Nll;

impl Nll {
    pub fn new() -> Self {
        Self
    }
}

impl LossFn for Nll {
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
                "Nll: input and target must be 2D and 1D respectively".to_string(),
            )),
        }
    }
}
