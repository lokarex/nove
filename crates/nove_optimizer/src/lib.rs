use nove_tensor::TensorError;
use thiserror::Error;

mod sgd;
pub use sgd::{Sgd, SgdBuilder};

mod adam;
pub use adam::{Adam, AdamBuilder};

mod adamw;
pub use adamw::{AdamW, AdamWBuilder};

mod rmsprop;
pub use rmsprop::{Rmsprop, RmspropBuilder};

mod adagrad;
pub use adagrad::{Adagrad, AdagradBuilder};

#[derive(Debug, Error)]
pub enum OptimizerError {
    /// Tensor errors from the `nove_tensor` crate.
    #[error(transparent)]
    TensorError(#[from] TensorError),

    /// Missing argument error.
    #[error("Missing argument: {0}")]
    MissingArgument(String),

    /// Invalid argument error.
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    /// Other errors.
    #[error("{0}")]
    OtherError(String),
}

pub trait Optimizer {
    type StepOutput;

    /// Performs an optimization step.
    ///
    /// # Returns
    /// * `Ok(Self::StepOutput)` - The output of the optimization step.
    /// * `Err(OptimizerError)` - The error when performing the optimization step.
    fn step(&mut self) -> Result<Self::StepOutput, OptimizerError>;

    /// Zeros the gradients of all parameters.
    ///
    /// # Returns
    /// * `Ok(())` - The gradients of all parameters are zeroed.
    /// * `Err(OptimizerError)` - The error when zeroing the gradients.
    fn zero_grad(&mut self) -> Result<(), OptimizerError>;
}
