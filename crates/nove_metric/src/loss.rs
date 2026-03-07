use std::sync::Arc;

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
/// let mut metric = LossMetric::new(lossfn);
///
/// let device = Device::cpu();
/// let output = Tensor::from_data(vec![
///     vec![0.1f64, 0.2f64, 0.7f64],
///     vec![0.3f64, 0.4f64, 0.3f64],
///     vec![0.6f64, 0.1f64, 0.3f64],
/// ], &device, false).unwrap();
/// let target = Tensor::from_data(vec![2u32, 1u32, 1u32], &device, false).unwrap();
///
/// metric.evaluate(&output, &target).unwrap();
/// let loss = metric.value().unwrap();
/// ```
#[derive(Clone)]
pub struct LossMetric {
    name: String,
    lossfn: Arc<Box<dyn LossFn<Input = (Tensor, Tensor), Output = Tensor>>>,
    lossfn_name: String,
    value: MetricValue,
    total_samples: usize,
    total_loss: f64,
}

impl std::fmt::Debug for LossMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LossMetric")
            .field("lossfn", &self.lossfn_name)
            .field("name", &self.name)
            .field("value", &self.value)
            .finish()
    }
}

impl LossMetric {
    pub fn new<LF: LossFn<Input = (Tensor, Tensor), Output = Tensor> + 'static>(
        lossfn: LF,
    ) -> Self {
        let full_name = std::any::type_name::<LF>();
        let type_name = full_name
            .rsplit("::")
            .next()
            .unwrap_or(full_name)
            .split('<')
            .next()
            .unwrap_or(full_name);
        Self {
            name: type_name.to_string(),
            lossfn: Arc::new(Box::new(lossfn)),
            lossfn_name: stringify!(LF).to_string(),
            value: MetricValue::Scalar(0.0),
            total_samples: 0,
            total_loss: 0.0,
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

    fn clear(&mut self) -> Result<(), MetricError> {
        self.value = MetricValue::Scalar(0.0);
        self.total_samples = 0;
        self.total_loss = 0.0;
        Ok(())
    }
}

impl EvaluationMetric for LossMetric {
    fn evaluate(&mut self, output: &Tensor, target: &Tensor) -> Result<(), MetricError> {
        let loss = self.lossfn.loss((output.clone(), target.clone()))?;

        let batch_size = output.shape()?.dims()[0];
        let batch_loss = loss.to_dtype(&DType::F64)?.to_scalar::<f64>()?;

        self.total_samples += batch_size;
        self.total_loss += batch_loss * batch_size as f64;

        let new_loss = if self.total_samples > 0 {
            self.total_loss / self.total_samples as f64
        } else {
            0.0
        };

        self.value = MetricValue::Scalar(new_loss);

        Ok(())
    }
}
