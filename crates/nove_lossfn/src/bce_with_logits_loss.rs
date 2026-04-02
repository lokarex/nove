use nove_tensor::Tensor;

use crate::{LossFn, LossFnError};

/// Binary Cross Entropy with Logits Loss function.
///
/// <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" id="MathJax-script" async></script>
///
/// The Binary Cross Entropy with Logits (BCEWithLogits) loss combines a sigmoid layer
/// and the binary cross entropy loss in a single numerically stable function.
///
/// For a batch of N samples, the BCEWithLogits loss is defined as:
///
/// $$ \text{BCEWithLogits}(x, y) = \frac{1}{N} \sum_{i=1}^{N} \left[ \max(x_i, 0) - x_i y_i + \log(1 + \exp(-|x_i|)) \right] $$
///
/// This formulation is mathematically equivalent to applying sigmoid followed by BCE loss:
///
/// $$ \text{BCEWithLogits}(x, y) = \text{BCE}(\sigma(x), y) $$
/// $$ \sigma(x) = \frac{1}{1 + \exp(-x)} $$
///
/// But provides better numerical stability by avoiding extreme values in the sigmoid function.
///
/// Where:
/// - N is the total number of elements in the batch
/// - x_i is the input logit for the i-th element
/// - y_i is the true label for the i-th element (should be in [0, 1])
///
/// # Notes:
/// * The input is a `(input, target)` tuple
///   - `input`: A tensor of any shape containing logits
///   - `target`: A tensor of the same shape as input containing true binary labels
/// * The output is A scalar tensor representing the average BCEWithLogits loss over all elements.
#[derive(Debug, Clone)]
pub struct BCEWithLogitsLoss;

impl BCEWithLogitsLoss {
    pub fn new() -> Self {
        Self
    }
}

impl Default for BCEWithLogitsLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl LossFn for BCEWithLogitsLoss {
    type Input = (Tensor, Tensor);
    type Output = Tensor;

    fn loss(&self, input: Self::Input) -> Result<Self::Output, LossFnError> {
        let (input, target) = input;

        if input.shape()? != target.shape()? {
            return Err(LossFnError::OtherError(
                "BCEWithLogitsLoss: input and target must have the same shape".to_string(),
            ));
        }

        let max_zero = input.clip(0.0, f64::INFINITY)?;
        let term1 = max_zero.sub(&input.mul(&target)?)?;

        let abs_input = input.abs()?;
        let neg_abs_input = abs_input.affine(-1.0, 0.0)?;
        let exp_neg_abs = neg_abs_input.exp()?;
        let one = Tensor::from_scalar(1.0, &exp_neg_abs.device()?, false)?;
        let log_term = exp_neg_abs.add(&one)?.log()?;

        let loss_per_element = term1.add(&log_term)?;
        Ok(loss_per_element.mean(None)?)
    }
}
