use nove_dataloader::Dataloader;
use nove_lossfn::LossFn;
use nove_metric::{
    AccuracyMetric, AnyMetric, EvaluationMetric, LossMetric, Metric, MetricValue, ResourceMetric,
};
use nove_model::Model;
use nove_optimizer::Optimizer;
use nove_tensor::Tensor;

use crate::common::{EpochLearner, EpochLearnerBuilder};
use crate::{Learner, LearnerError};

/// Image classification learner, specialized for image classification tasks.
///
/// This learner delegates core functionality to [`EpochLearner`] via predefined closures.
///
/// # Notes
/// * The `ImageClassificationLearner` can only be created by the [`ImageClassificationLearnerBuilder`].
pub struct ImageClassificationLearner<D, M, L, O>
where
    D: Dataloader<Output = (Tensor, Tensor)>,
    M: Model<Input = Tensor, Output = Tensor>,
    L: LossFn<Input = (Tensor, Tensor), Output = Tensor> + Clone,
    O: Optimizer<StepOutput = ()>,
{
    epoch_learner: EpochLearner<D, M, L, O>,
}

impl<D, M, L, O> ImageClassificationLearner<D, M, L, O>
where
    D: Dataloader<Output = (Tensor, Tensor)>,
    M: Model<Input = Tensor, Output = Tensor>,
    L: LossFn<Input = (Tensor, Tensor), Output = Tensor> + Clone,
    O: Optimizer<StepOutput = ()>,
{
    /// Get model reference.
    ///
    /// # Returns
    /// * `&M` - The model reference.
    pub fn model(&self) -> &M {
        self.epoch_learner.model()
    }

    /// Get mutable model reference.
    ///
    /// # Returns
    /// * `&mut M` - The mutable model reference.
    pub fn model_mut(&mut self) -> &mut M {
        self.epoch_learner.model_mut()
    }

    /// Get metrics reference.
    ///
    /// # Returns
    /// * `&[AnyMetric]` - The metrics reference.
    pub fn metrics(&self) -> &[AnyMetric] {
        self.epoch_learner.metrics()
    }

    /// Get mutable metrics reference.
    ///
    /// # Returns
    /// * `&mut Vec<AnyMetric>` - The mutable metrics reference.
    pub fn metrics_mut(&mut self) -> &mut Vec<AnyMetric> {
        self.epoch_learner.metrics_mut()
    }

    /// Get number of epochs.
    ///
    /// # Returns
    /// * `usize` - The number of epochs.
    pub fn num_epochs(&self) -> usize {
        self.epoch_learner.num_epochs()
    }

    /// Get log interval.
    ///
    /// # Returns
    /// * `usize` - The log interval.
    pub fn log_interval(&self) -> usize {
        self.epoch_learner.log_interval()
    }

    /// Get mutable train dataloader reference.
    ///
    /// # Returns
    /// * `Option<&mut D>` - The mutable train dataloader reference if set.
    pub fn train_dataloader_mut(&mut self) -> Option<&mut D> {
        self.epoch_learner.train_dataloader_mut()
    }

    /// Get mutable validation dataloader reference.
    ///
    /// # Returns
    /// * `Option<&mut D>` - The mutable validation dataloader reference if set.
    pub fn validation_dataloader_mut(&mut self) -> Option<&mut D> {
        self.epoch_learner.validation_dataloader_mut()
    }

    /// Get mutable test dataloader reference.
    ///
    /// # Returns
    /// * `Option<&mut D>` - The mutable test dataloader reference if set.
    pub fn test_dataloader_mut(&mut self) -> Option<&mut D> {
        self.epoch_learner.test_dataloader_mut()
    }

    /// Get loss function reference.
    ///
    /// # Returns
    /// * `Option<&L>` - The loss function reference if set.
    pub fn loss_fn(&self) -> Option<&L> {
        self.epoch_learner.loss_fn()
    }

    /// Get mutable optimizer reference.
    ///
    /// # Returns
    /// * `Option<&mut O>` - The mutable optimizer reference if set.
    pub fn optimizer_mut(&mut self) -> Option<&mut O> {
        self.epoch_learner.optimizer_mut()
    }
}

impl<D, M, L, O> Learner for ImageClassificationLearner<D, M, L, O>
where
    D: Dataloader<Output = (Tensor, Tensor)>,
    M: Model<Input = Tensor, Output = Tensor>,
    L: LossFn<Input = (Tensor, Tensor), Output = Tensor> + Clone,
    O: Optimizer<StepOutput = ()>,
{
    /// Run the training loop.
    ///
    /// # Returns
    /// * `Ok(())` - If training completes successfully.
    /// * `Err(LearnerError)` - If training fails.
    fn train(&mut self) -> Result<(), LearnerError> {
        self.epoch_learner.train()
    }

    /// Run validation.
    ///
    /// # Returns
    /// * `Ok(Vec<AnyMetric>)` - The validation metrics.
    /// * `Err(LearnerError)` - If validation fails.
    fn validate(&mut self) -> Result<Vec<AnyMetric>, LearnerError> {
        self.epoch_learner.validate()
    }

    /// Run testing.
    ///
    /// # Returns
    /// * `Ok(Vec<AnyMetric>)` - The test metrics.
    /// * `Err(LearnerError)` - If testing fails.
    fn test(&mut self) -> Result<Vec<AnyMetric>, LearnerError> {
        self.epoch_learner.test()
    }
}

/// Builder for constructing [`ImageClassificationLearner`] instances.
///
/// The builder delegates to [`EpochLearnerBuilder`] internally, setting up
/// predefined closures for image classification tasks.
///
/// # Notes
/// * The `ImageClassificationLearnerBuilder` implements the `Default` trait, so you can use
///   `ImageClassificationLearnerBuilder::default()` to create a builder with default values.
/// * `AccuracyMetric` is already added during [`ImageClassificationLearnerBuilder::new()`].
/// * `LossMetric` is automatically added when [`Self::loss_fn()`] is called.
/// * Argument validation is deferred to the actual method calls (`train`/`validate`/`test`).
///
/// # Required Arguments
/// * `model` - The model to train/evaluate. Required for all operations.
///
/// # Conditional Required Arguments by Method
/// * `train()` requires: `train_dataloader`, `validation_dataloader`, `loss_fn`, `optimizer`, `num_epochs` > 0, `log_interval` > 0, `output_dir`
/// * `validate()` requires: `validation_dataloader`
/// * `test()` requires: `test_dataloader`
///
/// # Optional Arguments
/// * `train_dataloader` - Dataloader for training. Required only for `train()`.
/// * `validation_dataloader` - Dataloader for validation. Required for `train()` and `validate()`.
/// * `test_dataloader` - Dataloader for testing. Required only for `test()`.
/// * `loss_fn` - Loss function. Required only for `train()`.
/// * `optimizer` - Optimizer. Required only for `train()`.
/// * `num_epochs` - Number of training epochs. Default is 10.
/// * `log_interval` - Logging frequency during training. Default is 10.
/// * `metrics` - Custom metrics to track.
/// * `output_dir` - Directory to save learner output.
pub struct ImageClassificationLearnerBuilder<D, M, L, O>
where
    D: Dataloader<Output = (Tensor, Tensor)>,
    M: Model<Input = Tensor, Output = Tensor>,
    L: LossFn<Input = (Tensor, Tensor), Output = Tensor> + Clone,
    O: Optimizer<StepOutput = ()>,
{
    inner: EpochLearnerBuilder<D, M, L, O>,
    loss_fn: Option<L>,
}

impl<D, M, L, O> Default for ImageClassificationLearnerBuilder<D, M, L, O>
where
    D: Dataloader<Output = (Tensor, Tensor)>,
    M: Model<Input = Tensor, Output = Tensor>,
    L: LossFn<Input = (Tensor, Tensor), Output = Tensor> + Clone + 'static,
    O: Optimizer<StepOutput = ()>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<D, M, L, O> ImageClassificationLearnerBuilder<D, M, L, O>
where
    D: Dataloader<Output = (Tensor, Tensor)>,
    M: Model<Input = Tensor, Output = Tensor>,
    L: LossFn<Input = (Tensor, Tensor), Output = Tensor> + Clone + 'static,
    O: Optimizer<StepOutput = ()>,
{
    /// Creates a new `ImageClassificationLearnerBuilder` with default values.
    ///
    /// # Notes
    /// * `AccuracyMetric` is automatically added by default.
    ///
    /// # Returns
    /// * `Self` - A new builder instance with default values.
    pub fn new() -> Self {
        let mut inner = EpochLearnerBuilder::new();
        inner.metric(AnyMetric::AccuracyMetric(AccuracyMetric::new()));
        Self {
            inner,
            loss_fn: None,
        }
    }

    /// Sets the training dataloader.
    ///
    /// # Arguments
    /// * `dataloader` - The training dataloader.
    ///
    /// # Returns
    /// * `&mut Self` - The builder itself.
    pub fn train_dataloader(&mut self, dataloader: D) -> &mut Self {
        self.inner.train_dataloader(dataloader);
        self
    }

    /// Sets the validation dataloader.
    ///
    /// # Arguments
    /// * `dataloader` - The validation dataloader.
    ///
    /// # Returns
    /// * `&mut Self` - The builder itself.
    pub fn validation_dataloader(&mut self, dataloader: D) -> &mut Self {
        self.inner.validation_dataloader(dataloader);
        self
    }

    /// Sets the test dataloader.
    ///
    /// # Arguments
    /// * `dataloader` - The test dataloader.
    ///
    /// # Returns
    /// * `&mut Self` - The builder itself.
    pub fn test_dataloader(&mut self, dataloader: D) -> &mut Self {
        self.inner.test_dataloader(dataloader);
        self
    }

    /// Sets the model.
    ///
    /// # Arguments
    /// * `model` - The model to train/evaluate.
    ///
    /// # Returns
    /// * `&mut Self` - The builder itself.
    pub fn model(&mut self, model: M) -> &mut Self {
        self.inner.model(model);
        self
    }

    /// Sets the loss function.
    ///
    /// # Notes
    /// * If a loss function has been set, [`Self::build()`] also adds a [`LossMetric`]
    ///   to track the loss.
    ///
    /// # Arguments
    /// * `loss_fn` - The loss function.
    ///
    /// # Returns
    /// * `&mut Self` - The builder itself.
    pub fn loss_fn(&mut self, loss_fn: L) -> &mut Self {
        self.loss_fn = Some(loss_fn);
        self
    }

    /// Sets the optimizer.
    ///
    /// # Arguments
    /// * `optimizer` - The optimizer.
    ///
    /// # Returns
    /// * `&mut Self` - The builder itself.
    pub fn optimizer(&mut self, optimizer: O) -> &mut Self {
        self.inner.optimizer(optimizer);
        self
    }

    /// Sets the number of training epochs.
    ///
    /// # Arguments
    /// * `num_epochs` - The number of epochs. Default is 10.
    ///
    /// # Returns
    /// * `&mut Self` - The builder itself.
    pub fn num_epochs(&mut self, num_epochs: usize) -> &mut Self {
        self.inner.num_epochs(num_epochs);
        self
    }

    /// Adds additional metrics to track.
    ///
    /// # Notes
    /// * `AccuracyMetric` is already added during [`ImageClassificationLearnerBuilder::new()`].
    /// * `LossMetric` is automatically added when [`Self::loss_fn()`] is called.
    /// * Do not manually add these two metrics.
    ///
    /// # Arguments
    /// * `metrics` - A vector of metrics to track.
    ///
    /// # Returns
    /// * `&mut Self` - The builder itself.
    pub fn metrics(&mut self, metrics: Vec<AnyMetric>) -> &mut Self {
        self.inner.metrics(metrics);
        self
    }

    /// Adds a single metric to track.
    ///
    /// # Notes
    /// * `AccuracyMetric` is already added during [`ImageClassificationLearnerBuilder::new()`].
    /// * `LossMetric` is automatically added when [`Self::loss_fn()`] is called.
    /// * Do not manually add these two metrics.
    ///
    /// # Arguments
    /// * `metric` - The metric to add.
    ///
    /// # Returns
    /// * `&mut Self` - The builder itself.
    pub fn metric<MT: Into<AnyMetric>>(&mut self, metric: MT) -> &mut Self {
        self.inner.metric(metric);
        self
    }

    /// Sets the log interval.
    ///
    /// # Arguments
    /// * `log_interval` - The number of batches between logging. Default is 10.
    ///
    /// # Returns
    /// * `&mut Self` - The builder itself.
    pub fn log_interval(&mut self, log_interval: usize) -> &mut Self {
        self.inner.log_interval(log_interval);
        self
    }

    /// Sets the output directory.
    ///
    /// # Arguments
    /// * `output_dir` - The directory to save learner output.
    ///
    /// # Returns
    /// * `&mut Self` - The builder itself.
    pub fn output_dir(&mut self, output_dir: &str) -> &mut Self {
        self.inner.output_dir(output_dir);
        self
    }

    /// Build the ImageClassificationLearner.
    ///
    /// This method sets up predefined closures for image classification tasks
    /// and delegates to EpochLearnerBuilder for actual construction.
    ///
    /// # Returns
    /// * `Ok(ImageClassificationLearner)` - The built learner.
    /// * `Err(LearnerError)` - If required arguments are missing or invalid.
    pub fn build(&mut self) -> Result<ImageClassificationLearner<D, M, L, O>, LearnerError> {
        // Predefined training step closure for image classification
        let train_step_fn = move |batch: (Tensor, Tensor),
                                  model: &mut M,
                                  loss_fn: &mut L,
                                  optimizer: &mut O,
                                  metrics: &mut [AnyMetric]|
              -> Result<(), LearnerError> {
            let (inputs, targets) = batch;
            let outputs = model.forward(inputs)?;

            for metric in metrics.iter_mut() {
                if metric.is_evaluation() {
                    metric.evaluate(&outputs, &targets)?
                } else {
                    metric.sample()?
                };
            }

            let loss = loss_fn.loss((outputs, targets))?;
            optimizer.zero_grad()?;
            loss.backward()?;
            drop(loss);
            optimizer.step()?;

            Ok(())
        };

        // Predefined validation/test step closure for image classification
        let validation_step_fn = move |batch: (Tensor, Tensor),
                                       model: &mut M,
                                       metrics: &mut [AnyMetric]|
              -> Result<(), LearnerError> {
            let (inputs, targets) = batch;
            let outputs = model.forward(inputs)?;

            for metric in metrics.iter_mut() {
                if metric.is_evaluation() {
                    metric.evaluate(&outputs, &targets)?
                } else {
                    metric.sample()?
                };
            }

            Ok(())
        };

        if let Some(loss_fn) = &self.loss_fn {
            self.inner.loss_fn(loss_fn.clone());
            self.inner.metric(LossMetric::new(loss_fn.clone()));
        }

        // Predefined best_model_criterion: compares accuracy (first metric)
        let best_model_criterion =
            move |best_metrics: Vec<AnyMetric>, current_metrics: Vec<AnyMetric>| -> bool {
                let best_first = best_metrics
                    .first()
                    .and_then(|m| m.value().ok())
                    .and_then(|v| match v {
                        MetricValue::Scalar(s) => Some(s),
                        _ => None,
                    })
                    .unwrap_or(0.0);
                let current_first = current_metrics
                    .first()
                    .and_then(|m| m.value().ok())
                    .and_then(|v| match v {
                        MetricValue::Scalar(s) => Some(s),
                        _ => None,
                    })
                    .unwrap_or(0.0);
                current_first > best_first
            };

        // Take ownership of inner, set closures, and build
        let inner = std::mem::take(&mut self.inner);
        let epoch_learner = inner
            .train_step_fn(train_step_fn)
            .validation_step_fn(validation_step_fn)
            .test_step_fn(validation_step_fn)
            .best_model_criterion(best_model_criterion)
            .build()?;

        Ok(ImageClassificationLearner { epoch_learner })
    }
}
