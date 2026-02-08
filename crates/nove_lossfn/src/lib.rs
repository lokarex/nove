use nove_tensor::TensorError;
use thiserror::Error;

mod cross_entropy;
pub use cross_entropy::CrossEntropy;
mod nll;
pub use nll::Nll;

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
