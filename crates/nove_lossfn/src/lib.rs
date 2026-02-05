use thiserror::Error;

mod cross_entropy_lossfn;
pub use cross_entropy_lossfn::CrossEntropyLossFn;

#[derive(Debug, Clone, Error)]
pub enum LossFnError {
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
