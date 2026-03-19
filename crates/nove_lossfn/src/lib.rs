use nove_tensor::TensorError;
use thiserror::Error;

mod cross_entropy_loss;
pub use cross_entropy_loss::CrossEntropyLoss;
mod nll_loss;
pub use nll_loss::NllLoss;
mod bce_loss;
pub use bce_loss::BCELoss;
mod bce_with_logits_loss;
pub use bce_with_logits_loss::BCEWithLogitsLoss;
mod mse_loss;
pub use mse_loss::MSELoss;
mod l1_loss;
pub use l1_loss::L1Loss;

pub mod common;

#[derive(Debug, Error)]
pub enum LossFnError {
    /// Tensor errors from the `nove_tensor` crate.
    #[error(transparent)]
    TensorError(#[from] TensorError),

    /// Other errors.
    #[error("{0}")]
    OtherError(String),
}

pub trait LossFn {
    type Input;
    type Output;

    /// Computes the loss.
    ///
    /// # Arguments
    /// * `input` - The input to the loss function.
    ///
    /// # Returns
    /// * `Ok(Self::Output)` - The output of the loss function.
    /// * `Err(LossFnError)` - The error when computing the loss.
    fn loss(&self, input: Self::Input) -> Result<Self::Output, LossFnError>;
}
