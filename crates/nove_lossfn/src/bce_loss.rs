use nove_tensor::Tensor;

use crate::{LossFn, LossFnError};

/// Binary Cross Entropy Loss function.
///
/// <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" id="MathJax-script" async></script>
///
/// The Binary Cross Entropy (BCE) loss measures the performance of a binary classification model.
/// It computes the cross entropy between the predicted probabilities and the true binary labels.
///
/// For a batch of N samples, the BCE loss is defined as:
///
/// $$ \text{BCE}(x, y) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(x_i) + (1 - y_i) \log(1 - x_i) \right] $$
///
/// Where:
/// - N is the total number of elements in the batch
/// - x_i is the predicted probability for the i-th element (should be in [0, 1])
/// - y_i is the true label for the i-th element (should be in [0, 1])
///
/// # Notes:
/// * The input is a `(input, target)` tuple
///   - `input`: A tensor of any shape containing predicted probabilities
///   - `target`: A tensor of the same shape as input containing true binary labels
/// * The output is A scalar tensor representing the average BCE loss over all elements.
/// * For numerical stability, the input is clamped to [ε, 1-ε] where ε is a small positive value.
#[derive(Debug, Clone)]
pub struct BCELoss {
    epsilon: f64,
}

impl BCELoss {
    pub fn new(epsilon: f64) -> Self {
        Self { epsilon }
    }
}

impl Default for BCELoss {
    fn default() -> Self {
        Self::new(1e-8)
    }
}

impl LossFn for BCELoss {
    type Input = (Tensor, Tensor);
    type Output = Tensor;

    fn loss(&self, input: Self::Input) -> Result<Self::Output, LossFnError> {
        let (input, target) = input;

        if input.shape()? != target.shape()? {
            return Err(LossFnError::OtherError(
                "BCELoss: input and target must have the same shape".to_string(),
            ));
        }

        let clamped_input = input.clip(self.epsilon, 1.0 - self.epsilon)?;

        let term1 = target.mul(&clamped_input.log()?)?;
        let one_target = Tensor::from_scalar(1.0, &target.device()?, false)?;
        let term2 = Tensor::sub(&one_target, &target)?.mul(
            &Tensor::sub(
                &Tensor::from_scalar(1.0, &clamped_input.device()?, false)?,
                &clamped_input,
            )?
            .log()?,
        )?;

        let loss_per_element = term1.add(&term2)?.affine(-1.0, 0.0)?;
        Ok(loss_per_element.mean(None)?)
    }
}
