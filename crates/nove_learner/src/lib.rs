use nove_dataloader::DataloaderError;
use nove_lossfn::LossFnError;
use nove_metric::AnyMetric;
use nove_model::ModelError;
use nove_optimizer::OptimizerError;
use nove_tensor::TensorError;
use thiserror::Error;

pub mod common;

#[derive(Error, Debug)]
pub enum LearnerError {
    /// Tensor errors from the `nove_tensor` crate.
    #[error(transparent)]
    TensorError(#[from] TensorError),

    /// Dataloader errors from the `nove_dataloader` crate.
    #[error(transparent)]
    DataloaderError(#[from] DataloaderError),

    /// Optimizer errors from the `nove_optimizer` crate.
    #[error(transparent)]
    OptimizerError(#[from] OptimizerError),

    /// Model errors from the `nove_model` crate.
    #[error(transparent)]
    ModelError(#[from] ModelError),

    /// Lossfn errors from the `nove_lossfn` crate.
    #[error(transparent)]
    LossfnError(#[from] LossFnError),

    /// Metric errors from the `nove_metric` crate.
    #[error(transparent)]
    MetricError(#[from] nove_metric::MetricError),

    /// Missing argument.
    #[error("Missing argument: {0}")]
    MissingArgument(String),

    /// Invalid argument.
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    /// Invalid path.
    #[error("Invalid path: {0}")]
    InvalidPath(String),

    /// Other errors.
    #[error("Other errors: {0}")]
    OtherError(String),
}

pub trait Learner {
    fn train(&mut self) -> Result<(), LearnerError>;
    fn validate(&mut self) -> Result<&[AnyMetric], LearnerError>;
    fn test(&mut self) -> Result<&[AnyMetric], LearnerError>;
}
