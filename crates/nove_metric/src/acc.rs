use nove_tensor::{DType, Tensor};

use crate::{EvaluationMetric, Metric, MetricError, MetricValue};

/// Accuracy metric.
///
/// # Notes
/// * It should be used for binary/multi-class classification tasks.
/// * It requires the output of the model to have the shape `(batch_size, num_classes)` and
///   the target with dtype `u32` to have the shape `(batch_size,)`, which contains the real class indices (not one-hot encoding).
///
/// # Fields
/// * `name` - The name of the metric.
///
/// # Examples
/// ```
/// use nove::tensor::{Device, Tensor};
/// use nove::metric::{MetricValue, AccuracyMetric, EvaluationMetric};
///
/// let device = Device::cpu();
/// let output = Tensor::from_data(vec![
///     vec![0.1f64, 0.2f64, 0.7f64],
///     vec![0.3f64, 0.4f64, 0.3f64],
///     vec![0.6f64, 0.1f64, 0.3f64],
/// ], &device, false).unwrap();
/// let target = Tensor::from_data(vec![2u32, 1u32, 1u32], &device, false).unwrap();
///
/// let metric = AccuracyMetric::new();
/// let accuracy = metric.evaluate(&output, &target).unwrap();
/// assert_eq!(accuracy, MetricValue::Scalar(0.6666666666666666));
/// ```
pub struct AccuracyMetric {
    name: String,
}

impl AccuracyMetric {
    pub fn new() -> Self {
        Self {
            name: "Accuracy".to_string(),
        }
    }
}

impl Metric for AccuracyMetric {
    fn name(&self) -> Result<String, MetricError> {
        Ok(self.name.clone())
    }
}

impl EvaluationMetric for AccuracyMetric {
    fn evaluate(&self, output: &Tensor, target: &Tensor) -> Result<MetricValue, MetricError> {
        let correct = output
            .argmax((1, false))?
            .eq(&target)?
            .to_dtype(&DType::F64)?;
        let accuracy = correct.mean(None)?;
        Ok(MetricValue::Scalar(
            accuracy.to_dtype(&DType::F64)?.to_scalar()?,
        ))
    }
}
