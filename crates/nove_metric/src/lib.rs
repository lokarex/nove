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
            MetricValue::Scalar(v) => {
                if let Some(precision) = f.precision() {
                    write!(f, "{:.1$}", v, precision)
                } else {
                    write!(f, "{}", v)
                }
            }
            MetricValue::Vector(v) => {
                if let Some(precision) = f.precision() {
                    let formatted: Vec<String> =
                        v.iter().map(|x| format!("{:.1$}", x, precision)).collect();
                    write!(f, "{:?}", formatted)
                } else {
                    write!(f, "{:?}", v)
                }
            }
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

    /// Clears the metric value.
    ///
    /// # Returns
    /// * `Ok(())` - If the metric value is cleared successfully.
    /// * `Err(MetricError)` - If an error occurs while clearing the metric value.
    fn clear(&mut self) -> Result<(), MetricError> {
        self.update(MetricValue::Scalar(0.0))
    }
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
    fn evaluate(&mut self, output: &Tensor, target: &Tensor) -> Result<(), MetricError>;
}

pub trait ResourceMetric: Metric {
    /// Samples the metric value.
    ///
    /// # Returns
    /// * `Ok(MetricValue)` - The sampled metric value.
    /// * `Err(MetricError)` - If an error occurs while sampling the metric.
    fn sample(&mut self) -> Result<(), MetricError>;
}

macro_rules! define_any_metric {
    (
        evaluation: [$($eval_name:ident($eval_type:ty)),* $(,)?],
        resource: [$($res_name:ident($res_type:ty)),* $(,)?],
    ) => {
        #[derive(Debug, Clone)]
        pub enum AnyMetric {
            $($eval_name($eval_type),)*
            $($res_name($res_type),)*
        }

        impl AnyMetric {
            pub fn is_evaluation(&self) -> bool {
                match self {
                    $(AnyMetric::$eval_name(_) => true,)*
                    $(AnyMetric::$res_name(_) => false,)*
                }
            }

            pub fn is_resource(&self) -> bool {
                match self {
                    $(AnyMetric::$eval_name(_) => false,)*
                    $(AnyMetric::$res_name(_) => true,)*
                }
            }
        }

        impl Metric for AnyMetric {
            fn name(&self) -> Result<String, MetricError> {
                match self {
                    $(AnyMetric::$eval_name(m) => m.name(),)*
                    $(AnyMetric::$res_name(m) => m.name(),)*
                }
            }

            fn value(&self) -> Result<MetricValue, MetricError> {
                match self {
                    $(AnyMetric::$eval_name(m) => m.value(),)*
                    $(AnyMetric::$res_name(m) => m.value(),)*
                }
            }

            fn update(&mut self, value: MetricValue) -> Result<(), MetricError> {
                match self {
                    $(AnyMetric::$eval_name(m) => m.update(value),)*
                    $(AnyMetric::$res_name(m) => m.update(value),)*
                }
            }
        }

        impl EvaluationMetric for AnyMetric {
            fn evaluate(&mut self, output: &Tensor, target: &Tensor) -> Result<(), MetricError> {
                match self {
                    $(AnyMetric::$eval_name(m) => m.evaluate(output, target),)*
                    $(AnyMetric::$res_name(_) => Err(MetricError::InvalidOperation(
                        concat!("Cannot evaluate ", stringify!($res_name)).to_string(),
                    )),)*
                }
            }
        }

        impl ResourceMetric for AnyMetric {
            fn sample(&mut self) -> Result<(), MetricError> {
                match self {
                    $(AnyMetric::$eval_name(_) => Err(MetricError::InvalidOperation(
                        concat!("Cannot sample ", stringify!($eval_name)).to_string(),
                    )),)*
                    $(AnyMetric::$res_name(m) => m.sample(),)*
                }
            }
        }
    };
}

define_any_metric! {
    evaluation: [
        AccuracyMetric(AccuracyMetric),
        LossMetric(LossMetric),
    ],
    resource: [
        CpuUsageMetric(CpuUsageMetric),
        CpuFrequencyMetric(CpuFrequencyMetric),
    ],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_value_display_with_precision() {
        let value = MetricValue::Scalar(3.141592653589793);

        assert_eq!(format!("{:.4}", value), "3.1416");
        assert_eq!(format!("{:.2}", value), "3.14");
        assert_eq!(format!("{:.0}", value), "3");
    }

    #[test]
    fn test_metric_value_display_without_precision() {
        let value = MetricValue::Scalar(3.141592653589793);

        assert_eq!(format!("{}", value), "3.141592653589793");
    }

    #[test]
    fn test_metric_value_display_vector_without_precision() {
        let value = MetricValue::Vector(vec![1.0, 2.0, 3.0]);

        assert_eq!(format!("{}", value), "[1.0, 2.0, 3.0]");
    }

    #[test]
    fn test_metric_value_display_vector_with_precision() {
        let value = MetricValue::Vector(vec![1.234567, 2.345678, 3.456789]);

        assert_eq!(format!("{:.2}", value), "[\"1.23\", \"2.35\", \"3.46\"]");
        assert_eq!(
            format!("{:.4}", value),
            "[\"1.2346\", \"2.3457\", \"3.4568\"]"
        );
    }
}
