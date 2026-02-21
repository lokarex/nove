use nove_lossfn::LossFn;
use nove_tensor::{DType, Tensor};

use crate::{EvaluationMetric, Metric, MetricError, MetricValue};

/// Loss metric.
///
/// # Notes
/// * Requires a loss function that accepts output and target tensors,
///   and returns a scalar loss tensor.
///
/// # Fields
/// * `name` - The name of the metric.
/// * `lossfn` - The loss function.
/// * `value` - The value of the metric.
///
/// # Examples
/// ```
/// use nove::metric::{Metric, MetricValue, LossMetric, EvaluationMetric};
/// use nove_lossfn::CrossEntropy;
/// use nove_tensor::{Device, Tensor};
///
/// let lossfn = CrossEntropy::new();
/// let metric = LossMetric::new(lossfn);
///
/// let device = Device::cpu();
/// let output = Tensor::from_data(vec![
///     vec![0.1f64, 0.2f64, 0.7f64],
///     vec![0.3f64, 0.4f64, 0.3f64],
///     vec![0.6f64, 0.1f64, 0.3f64],
/// ], &device, false).unwrap();
/// let target = Tensor::from_data(vec![2u32, 1u32, 1u32], &device, false).unwrap();
///
/// let loss = metric.evaluate(&output, &target).unwrap();
/// ```
pub struct LossMetric {
    name: String,
    lossfn: Box<dyn LossFn<Input = (Tensor, Tensor), Output = Tensor>>,
    value: MetricValue,
}

impl LossMetric {
    pub fn new<LF: LossFn<Input = (Tensor, Tensor), Output = Tensor> + 'static>(
        lossfn: LF,
    ) -> Self {
        Self {
            name: stringify!(LF).to_string(),
            lossfn: Box::new(lossfn),
            value: MetricValue::Scalar(0.0),
        }
    }
}

impl Metric for LossMetric {
    fn name(&self) -> Result<String, MetricError> {
        Ok(self.name.clone())
    }

    fn value(&self) -> Result<MetricValue, MetricError> {
        Ok(self.value.clone())
    }

    fn update(&mut self, value: MetricValue) -> Result<(), MetricError> {
        self.value = value;
        Ok(())
    }
}

impl EvaluationMetric for LossMetric {
    fn evaluate(&self, output: &Tensor, target: &Tensor) -> Result<MetricValue, MetricError> {
        let loss = self.lossfn.loss((output.clone(), target.clone()))?;
        Ok(MetricValue::Scalar(
            loss.to_dtype(&DType::F64)?.to_scalar()?,
        ))
    }
}
