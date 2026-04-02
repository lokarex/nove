use nove_tensor::Tensor;

use crate::{LossFn, LossFnError};

/// Mean Squared Error Loss function.
///
/// <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" id="MathJax-script" async></script>
///
/// The Mean Squared Error (MSE) loss measures the average squared difference
/// between predicted values and target values. It is commonly used for regression tasks.
///
/// For a batch of N samples, the MSE loss is defined as:
///
/// $$ \text{MSE}(x, y) = \frac{1}{N} \sum_{i=1}^{N} (x_i - y_i)^2 $$
///
/// Where:
/// - N is the total number of elements in the batch
/// - x_i is the predicted value for the i-th element
/// - y_i is the target value for the i-th element
///
/// # Notes:
/// * The input is a `(input, target)` tuple
///   - `input`: A tensor of any shape containing predicted values
///   - `target`: A tensor of the same shape as input containing target values
/// * The output is A scalar tensor representing the average MSE loss over all elements.
#[derive(Debug, Clone)]
pub struct MSELoss;

impl MSELoss {
    pub fn new() -> Self {
        Self
    }
}

impl Default for MSELoss {
    fn default() -> Self {
        Self::new()
    }
}

impl LossFn for MSELoss {
    type Input = (Tensor, Tensor);
    type Output = Tensor;

    fn loss(&self, input: Self::Input) -> Result<Self::Output, LossFnError> {
        let (input, target) = input;

        if input.shape()? != target.shape()? {
            return Err(LossFnError::OtherError(
                "MSELoss: input and target must have the same shape".to_string(),
            ));
        }

        let diff = input.sub(&target)?;
        let squared = diff.mul(&diff)?;
        Ok(squared.mean(None)?)
    }
}
