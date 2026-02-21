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

    #[error("Vector length mismatch: expected {expected}, got {actual}")]
    VectorLengthMismatch { expected: usize, actual: usize },

    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
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

impl MetricValue {
    pub fn add(&self, rhs: &Self) -> Result<MetricValue, MetricError> {
        match (self, rhs) {
            (MetricValue::Scalar(a), MetricValue::Scalar(b)) => Ok(MetricValue::Scalar(a + b)),
            (MetricValue::Vector(a), MetricValue::Vector(b)) => {
                if a.len() != b.len() {
                    return Err(MetricError::VectorLengthMismatch {
                        expected: a.len(),
                        actual: b.len(),
                    });
                }
                let result: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
                Ok(MetricValue::Vector(result))
            }
            (MetricValue::Scalar(s), MetricValue::Vector(v)) => {
                Ok(MetricValue::Vector(v.iter().map(|x| s + x).collect()))
            }
            (MetricValue::Vector(v), MetricValue::Scalar(s)) => {
                Ok(MetricValue::Vector(v.iter().map(|x| x + s).collect()))
            }
        }
    }

    pub fn div(&self, rhs: &Self) -> Result<Self, MetricError> {
        match (self, rhs) {
            (MetricValue::Scalar(a), MetricValue::Scalar(b)) => Ok(MetricValue::Scalar(a / b)),
            (MetricValue::Vector(a), MetricValue::Vector(b)) => {
                if a.len() != b.len() {
                    return Err(MetricError::VectorLengthMismatch {
                        expected: a.len(),
                        actual: b.len(),
                    });
                }
                let result: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x / y).collect();
                Ok(MetricValue::Vector(result))
            }
            (MetricValue::Scalar(s), MetricValue::Vector(v)) => {
                Ok(MetricValue::Vector(v.iter().map(|x| s / x).collect()))
            }
            (MetricValue::Vector(v), MetricValue::Scalar(s)) => {
                Ok(MetricValue::Vector(v.iter().map(|x| x / s).collect()))
            }
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

    /// Returns the value of the metric.
    ///
    /// # Returns
    /// * `Ok(MetricValue)` - The value of the metric.
    /// * `Err(MetricError)` - If an error occurs while getting the value.
    fn value(&self) -> Result<MetricValue, MetricError>;

    /// Updates the metric value.
    ///
    /// # Arguments
    /// * `value` - The new metric value.
    ///
    /// # Returns
    /// * `Ok(())` - If the metric value is updated successfully.
    /// * `Err(MetricError)` - If an error occurs while updating the metric value.
    fn update(&mut self, value: MetricValue) -> Result<(), MetricError>;
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

pub enum AnyMetric {
    Evaluation(Box<dyn EvaluationMetric>),
    Resource(Box<dyn ResourceMetric>),
}

impl AnyMetric {
    pub fn name(&self) -> Result<String, MetricError> {
        match self {
            AnyMetric::Evaluation(m) => m.name(),
            AnyMetric::Resource(m) => m.name(),
        }
    }

    pub fn value(&self) -> Result<MetricValue, MetricError> {
        match self {
            AnyMetric::Evaluation(m) => m.value(),
            AnyMetric::Resource(m) => m.value(),
        }
    }

    pub fn update(&mut self, value: MetricValue) -> Result<(), MetricError> {
        match self {
            AnyMetric::Evaluation(m) => m.update(value),
            AnyMetric::Resource(m) => m.update(value),
        }
    }

    pub fn clear(&mut self) -> Result<(), MetricError> {
        self.update(MetricValue::Scalar(0.0))
    }

    pub fn sample(&self) -> Result<MetricValue, MetricError> {
        match self {
            AnyMetric::Evaluation(_) => Err(MetricError::InvalidOperation(
                "Cannot sample EvaluationMetric".to_string(),
            )),
            AnyMetric::Resource(m) => m.sample(),
        }
    }

    pub fn evaluate(&self, output: &Tensor, target: &Tensor) -> Result<MetricValue, MetricError> {
        match self {
            AnyMetric::Evaluation(m) => m.evaluate(output, target),
            AnyMetric::Resource(_) => Err(MetricError::InvalidOperation(
                "Cannot evaluate ResourceMetric".to_string(),
            )),
        }
    }

    pub fn is_evaluation(&self) -> bool {
        matches!(self, AnyMetric::Evaluation(_))
    }

    pub fn is_resource(&self) -> bool {
        matches!(self, AnyMetric::Resource(_))
    }
}
