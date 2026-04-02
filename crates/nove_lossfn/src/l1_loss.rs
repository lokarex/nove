use nove_tensor::Tensor;

use crate::{LossFn, LossFnError};

/// L1 Loss (Mean Absolute Error) function.
///
/// <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" id="MathJax-script" async></script>
///
/// The L1 loss measures the average absolute difference between predicted values
/// and target values. It is also known as Mean Absolute Error (MAE).
///
/// For a batch of N samples, the L1 loss is defined as:
///
/// $$ \text{L1}(x, y) = \frac{1}{N} \sum_{i=1}^{N} |x_i - y_i| $$
///
/// Where:
/// - N is the total number of elements in the batch
/// - x_i is the predicted value for the i-th element
/// - y_i is the target value for the i-th element
/// - |·| denotes the absolute value
///
/// # Notes:
/// * The input is a `(input, target)` tuple
///   - `input`: A tensor of any shape containing predicted values
///   - `target`: A tensor of the same shape as input containing target values
/// * The output is A scalar tensor representing the average L1 loss over all elements.
#[derive(Debug, Clone)]
pub struct L1Loss;

impl L1Loss {
    pub fn new() -> Self {
        Self
    }
}

impl Default for L1Loss {
    fn default() -> Self {
        Self::new()
    }
}

impl LossFn for L1Loss {
    type Input = (Tensor, Tensor);
    type Output = Tensor;

    fn loss(&self, input: Self::Input) -> Result<Self::Output, LossFnError> {
        let (input, target) = input;

        if input.shape()? != target.shape()? {
            return Err(LossFnError::OtherError(
                "L1Loss: input and target must have the same shape".to_string(),
            ));
        }

        let diff = input.sub(&target)?;
        let abs_diff = diff.abs()?;
        Ok(abs_diff.mean(None)?)
    }
}
