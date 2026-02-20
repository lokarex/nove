use nove_lossfn::LossFnError;
use nove_tensor::{Tensor, TensorError};
use std::fmt::Display;
use std::sync::PoisonError;
use thiserror::Error;

mod loss;
pub use loss::LossMetric;

mod acc;
pub use acc::AccuracyMetric;

mod cpu;
pub use cpu::CpuFrequencyMetric;
pub use cpu::CpuUsageMetric;

#[derive(Debug, Error)]
pub enum MetricError {
    /// Tensor errors from the `nove_tensor` crate.
    #[error(transparent)]
    TensorError(#[from] TensorError),

    /// Loss function errors from the `nove_lossfn` crate.
    #[error(transparent)]
    LossFnError(#[from] LossFnError),

    /// Lock poisoned error.
    #[error("Lock poisoned: {0}")]
    LockPoisoned(String),

    /// IO error.
    #[error(transparent)]
    IoError(#[from] std::io::Error),

    #[error("RwLock poisoned: {0}")]
    RwLockPoisoned(String),
}

impl<T> From<PoisonError<T>> for MetricError {
    fn from(err: PoisonError<T>) -> Self {
        MetricError::LockPoisoned(err.to_string())
    }
}

#[derive(Debug, Clone)]
pub enum MetricValue {
    Scalar(f64),
    Vector(Vec<f64>),
}

impl Display for MetricValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetricValue::Scalar(v) => write!(f, "{}", v),
            MetricValue::Vector(v) => write!(f, "{:?}", v),
        }
    }
}

impl PartialEq for MetricValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (MetricValue::Scalar(a), MetricValue::Scalar(b)) => a == b,
            (MetricValue::Vector(a), MetricValue::Vector(b)) => a == b,
            _ => false,
        }
    }
}

pub trait Metric {
    /// Returns the name of the metric.
    ///
    /// # Returns
    /// * `Ok(String)` - The name of the metric.
    /// * `Err(MetricError)` - If an error occurs while getting the name.
    fn name(&self) -> Result<String, MetricError>;
}

pub trait EvaluationMetric: Metric {
    /// Evaluates the metric on the given output and target tensors.
    ///
    /// # Arguments
    /// * `output` - The output tensor.
    /// * `target` - The target tensor.
    ///
    /// # Returns
    /// * `Ok(MetricValue)` - The evaluated metric value.
    /// * `Err(MetricError)` - If an error occurs while evaluating the metric.
    fn evaluate(&self, output: &Tensor, target: &Tensor) -> Result<MetricValue, MetricError>;
}

pub trait ResourceMetric: Metric {
    /// Samples the metric value.
    ///
    /// # Returns
    /// * `Ok(MetricValue)` - The sampled metric value.
    /// * `Err(MetricError)` - If an error occurs while sampling the metric.
    fn sample(&self) -> Result<MetricValue, MetricError>;
}
