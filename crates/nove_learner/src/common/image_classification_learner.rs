use std::path::PathBuf;

use nove_dataloader::Dataloader;
use nove_lossfn::LossFn;
use nove_metric::{
    AccuracyMetric, AnyMetric, EvaluationMetric, LossMetric, Metric, MetricValue, ResourceMetric,
};
use nove_model::Model;
use nove_optimizer::Optimizer;
use nove_tensor::Tensor;

use crate::{Learner, LearnerError};

/// Image classification learner, specialized for image classification tasks.
///
/// # Notes
/// * The `ImageClassificationLearner` can only be created by the [`ImageClassificationLearnerBuilder`].
/// * Argument validation is deferred to the actual method calls (`train`/`validate`/`test`).
///
/// # Fields
/// * `train_dataloader` - Dataloader for training data.
/// * `validate_dataloader` - Dataloader for validation data.
/// * `test_dataloader` - Dataloader for test data.
/// * `model` - The model to train/evaluate.
/// * `lossfn` - Loss function for training.
/// * `optimizer` - Optimizer for training.
/// * `epoch` - Number of training epochs.
/// * `metrics` - Metrics to track during training/evaluation.
/// * `log_interval` - Number of batches between logging.
/// * `model_name` - Name of the model for checkpointing.
/// * `result_dir` - Directory to save checkpoints.
/// * `best_accuracy` - Best validation accuracy achieved during training.
pub struct ImageClassificationLearner<D, M, L, O>
where
    D: Dataloader<Output = (Tensor, Tensor)>,
    M: Model<Input = (Tensor, bool), Output = Tensor>,
    L: LossFn<Input = (Tensor, Tensor), Output = Tensor> + Clone,
    O: Optimizer<StepOutput = ()>,
{
    train_dataloader: Option<D>,
    validate_dataloader: Option<D>,
    test_dataloader: Option<D>,
    model: M,
    lossfn: Option<L>,
    optimizer: Option<O>,
    epoch: usize,
    metrics: Vec<AnyMetric>,
    log_interval: usize,
    model_name: String,
    result_dir: PathBuf,
    best_accuracy: f64,
}

impl<D, M, L, O> ImageClassificationLearner<D, M, L, O>
where
    D: Dataloader<Output = (Tensor, Tensor)>,
    M: Model<Input = (Tensor, bool), Output = Tensor>,
    L: LossFn<Input = (Tensor, Tensor), Output = Tensor> + Clone,
    O: Optimizer<StepOutput = ()>,
{
    fn ensure_train_ready(&self) -> Result<(), LearnerError> {
        self.train_dataloader
            .as_ref()
            .ok_or(LearnerError::MissingArgument(
                "train_dataloader is required for training".to_string(),
            ))?;
        self.validate_dataloader
            .as_ref()
            .ok_or(LearnerError::MissingArgument(
                "validate_dataloader is required for training".to_string(),
            ))?;
        self.lossfn.as_ref().ok_or(LearnerError::MissingArgument(
            "lossfn is required for training".to_string(),
        ))?;
        self.optimizer
            .as_ref()
            .ok_or(LearnerError::MissingArgument(
                "optimizer is required for training".to_string(),
            ))?;
        if self.epoch == 0 {
            return Err(LearnerError::InvalidArgument(
                "epoch must be greater than 0 for training".to_string(),
            ));
        }
        if self.log_interval == 0 {
            return Err(LearnerError::InvalidArgument(
                "log_interval must be greater than 0 for training".to_string(),
            ));
        }
        Ok(())
    }

    fn ensure_validate_ready(&self) -> Result<(), LearnerError> {
        self.validate_dataloader
            .as_ref()
            .ok_or(LearnerError::MissingArgument(
                "validate_dataloader is required for validation".to_string(),
            ))?;
        Ok(())
    }

    fn ensure_test_ready(&self) -> Result<(), LearnerError> {
        self.test_dataloader
            .as_ref()
            .ok_or(LearnerError::MissingArgument(
                "test_dataloader is required for testing".to_string(),
            ))?;
        Ok(())
    }
}

impl<D, M, L, O> Learner for ImageClassificationLearner<D, M, L, O>
where
    D: Dataloader<Output = (Tensor, Tensor)>,
    M: Model<Input = (Tensor, bool), Output = Tensor>,
    L: LossFn<Input = (Tensor, Tensor), Output = Tensor> + Clone,
    O: Optimizer<StepOutput = ()>,
{
    fn train(&mut self) -> Result<(), LearnerError> {
        self.ensure_train_ready()?;

        let train_dataloader = self.train_dataloader.as_mut().unwrap();
        let validate_dataloader = self.validate_dataloader.as_mut().unwrap();
        let lossfn = self.lossfn.as_ref().unwrap();
        let optimizer = self.optimizer.as_mut().unwrap();

        for epoch in 0..self.epoch {
            let mut batch_count: usize = 0;

            loop {
                let (inputs, targets) = match train_dataloader.next()? {
                    Some((inputs, targets)) => (inputs, targets),
                    None => break,
                };
                let outputs = self.model.forward((inputs, true))?;

                for metric in self.metrics.iter_mut() {
                    if metric.is_evaluation() {
                        metric.evaluate(&outputs, &targets)?
                    } else {
                        metric.sample()?
                    };
                }

                let loss = lossfn.loss((outputs, targets))?;

                optimizer.zero_grad()?;
                loss.backward()?;
                drop(loss);

                optimizer.step()?;

                batch_count += 1;
                if batch_count == self.log_interval {
                    print!("Train: Epoch {}/{}", epoch + 1, self.epoch,);
                    for metric in self.metrics.iter_mut() {
                        print!(", {}= {:.3}", metric.name()?, metric.value()?);
                        metric.clear()?;
                    }
                    println!();
                    batch_count = 0;
                }
            }
            train_dataloader.reset()?;

            print!("Validate: ");
            let metrics_len = self.metrics.len();
            loop {
                let (inputs, targets) = match validate_dataloader.next()? {
                    Some((inputs, targets)) => (inputs, targets),
                    None => break,
                };
                let outputs = self.model.forward((inputs, false))?;
                for metric in self.metrics.iter_mut() {
                    if metric.is_evaluation() {
                        metric.evaluate(&outputs, &targets)?
                    } else {
                        metric.sample()?
                    };
                }
            }
            validate_dataloader.reset()?;
            for (i, metric) in self.metrics.iter_mut().enumerate() {
                print!("{}= {:.3}", metric.name()?, metric.value()?);
                if i != metrics_len - 1 {
                    print!(", ");
                }
            }
            println!();

            self.model.save(
                self.result_dir
                    .join(format!("{}_{}.safetensors", self.model_name, epoch + 1))
                    .to_str()
                    .ok_or(LearnerError::InvalidPath(
                        "result_dir in ImageClassificationLearnerBuilder".to_string(),
                    ))?,
            )?;

            if let AnyMetric::AccuracyMetric(accuracy) = &mut self.metrics[0]
                && let MetricValue::Scalar(accuracy) = accuracy.value()?
                && accuracy > self.best_accuracy
            {
                self.best_accuracy = accuracy;

                self.model.save(
                    self.result_dir
                        .join(format!("{}_best.safetensors", self.model_name))
                        .to_str()
                        .ok_or(LearnerError::InvalidPath(
                            "result_dir in ImageClassificationLearnerBuilder".to_string(),
                        ))?,
                )?;
            }

            for metric in self.metrics.iter_mut() {
                metric.clear()?;
            }
        }

        Ok(())
    }

    fn validate(&mut self) -> Result<Vec<AnyMetric>, LearnerError> {
        self.ensure_validate_ready()?;

        let validate_dataloader = self.validate_dataloader.as_mut().unwrap();

        loop {
            let (inputs, targets) = match validate_dataloader.next()? {
                Some((inputs, targets)) => (inputs, targets),
                None => break,
            };
            let outputs = self.model.forward((inputs, false))?;

            for metric in self.metrics.iter_mut() {
                if metric.is_evaluation() {
                    metric.evaluate(&outputs, &targets)?
                } else {
                    metric.sample()?
                };
            }
        }
        validate_dataloader.reset()?;

        let metrics = self.metrics.clone();
        for metric in self.metrics.iter_mut() {
            metric.clear()?;
        }

        Ok(metrics)
    }

    fn test(&mut self) -> Result<Vec<AnyMetric>, LearnerError> {
        self.ensure_test_ready()?;

        let test_dataloader = self.test_dataloader.as_mut().unwrap();

        loop {
            let (inputs, targets) = match test_dataloader.next()? {
                Some((inputs, targets)) => (inputs, targets),
                None => break,
            };
            let outputs = self.model.forward((inputs, false))?;

            for metric in self.metrics.iter_mut() {
                if metric.is_evaluation() {
                    metric.evaluate(&outputs, &targets)?
                } else {
                    metric.sample()?
                };
            }
        }
        test_dataloader.reset()?;

        let metrics = self.metrics.clone();
        for metric in self.metrics.iter_mut() {
            metric.clear()?;
        }

        Ok(metrics)
    }
}

/// Builder for constructing [`ImageClassificationLearner`] instances.
///
/// # Notes
/// * The `ImageClassificationLearnerBuilder` implements the `Default` trait, so you can
///   use `ImageClassificationLearnerBuilder::default()` to create a builder with default values.
/// * Argument validation is deferred to the actual method calls (`train`/`validate`/`test`).
///
/// # Required Arguments
/// * `model` - The model to train/evaluate. Required for all operations.
///
/// # Conditional Required Arguments by Method
/// * `train()` requires: `train_dataloader`, `validate_dataloader`, `lossfn`, `optimizer`, `epoch` > 0, `log_interval` > 0
/// * `validate()` requires: `validate_dataloader`
/// * `test()` requires: `test_dataloader`
///
/// # Optional Arguments
/// * `train_dataloader` - Dataloader for training. Required only for `train()`.
/// * `validate_dataloader` - Dataloader for validation. Required for `train()` and `validate()`.
/// * `test_dataloader` - Dataloader for testing. Required only for `test()`.
/// * `lossfn` - Loss function. Required only for `train()`.
/// * `optimizer` - Optimizer. Required only for `train()`.
/// * `epoch` - Number of training epochs. Default is 10. Must be > 0 for `train()`.
/// * `log_interval` - Logging frequency during training. Default is 10. Must be > 0 for `train()`.
/// * `metrics` - Custom metrics to track. Default includes `AccuracyMetric`.
/// * `result_dir` - Directory to save checkpoints. Default is current directory.
///
/// # Fields
/// * `train_dataloader` - Dataloader for training data.
/// * `validate_dataloader` - Dataloader for validation data.
/// * `test_dataloader` - Dataloader for test data.
/// * `model` - The model to train/evaluate.
/// * `lossfn` - Loss function for training.
/// * `optimizer` - Optimizer for training.
/// * `epoch` - Number of training epochs.
/// * `metrics` - Metrics to track.
/// * `log_interval` - Number of batches between logging.
/// * `model_name` - Name of the model for checkpointing.
/// * `result_dir` - Directory to save checkpoints.
pub struct ImageClassificationLearnerBuilder<D, M, L, O>
where
    D: Dataloader<Output = (Tensor, Tensor)>,
    M: Model<Input = (Tensor, bool), Output = Tensor>,
    L: LossFn<Input = (Tensor, Tensor), Output = Tensor> + Clone,
    O: Optimizer<StepOutput = ()>,
{
    train_dataloader: Option<D>,
    validate_dataloader: Option<D>,
    test_dataloader: Option<D>,
    model: Option<M>,
    lossfn: Option<L>,
    optimizer: Option<O>,
    epoch: usize,
    metrics: Vec<AnyMetric>,
    log_interval: usize,
    model_name: Option<String>,
    result_dir: Option<String>,
}

impl<D, M, L, O> Default for ImageClassificationLearnerBuilder<D, M, L, O>
where
    D: Dataloader<Output = (Tensor, Tensor)>,
    M: Model<Input = (Tensor, bool), Output = Tensor>,
    L: LossFn<Input = (Tensor, Tensor), Output = Tensor> + Clone,
    O: Optimizer<StepOutput = ()>,
{
    fn default() -> Self {
        Self {
            train_dataloader: None,
            validate_dataloader: None,
            test_dataloader: None,
            model: None,
            lossfn: None,
            optimizer: None,
            epoch: 10,
            metrics: vec![],
            log_interval: 10,
            model_name: None,
            result_dir: None,
        }
    }
}

impl<D, M, L, O> ImageClassificationLearnerBuilder<D, M, L, O>
where
    D: Dataloader<Output = (Tensor, Tensor)>,
    M: Model<Input = (Tensor, bool), Output = Tensor>,
    L: LossFn<Input = (Tensor, Tensor), Output = Tensor> + Clone,
    O: Optimizer<StepOutput = ()>,
{
    pub fn train_dataloader(&mut self, dataloader: D) -> &mut Self {
        self.train_dataloader = Some(dataloader);
        self
    }

    pub fn validate_dataloader(&mut self, dataloader: D) -> &mut Self {
        self.validate_dataloader = Some(dataloader);
        self
    }

    pub fn test_dataloader(&mut self, dataloader: D) -> &mut Self {
        self.test_dataloader = Some(dataloader);
        self
    }

    pub fn model(&mut self, model: M) -> &mut Self {
        self.model = Some(model);
        let full_name = std::any::type_name::<M>();
        let type_name = full_name
            .rsplit("::")
            .next()
            .unwrap_or("model")
            .split('<')
            .next()
            .unwrap_or("model");
        self.model_name = Some(type_name.to_string());
        self
    }

    pub fn lossfn(&mut self, lossfn: L) -> &mut Self {
        self.lossfn = Some(lossfn);
        self
    }

    pub fn optimizer(&mut self, optimizer: O) -> &mut Self {
        self.optimizer = Some(optimizer);
        self
    }

    pub fn epoch(&mut self, epoch: usize) -> &mut Self {
        self.epoch = epoch;
        self
    }

    pub fn metrics(&mut self, metrics: Vec<AnyMetric>) -> &mut Self {
        self.metrics.extend(metrics);
        self
    }

    pub fn metric<MT: Into<AnyMetric>>(&mut self, metric: MT) -> &mut Self {
        self.metrics.push(metric.into());
        self
    }

    pub fn log_interval(&mut self, log_interval: usize) -> &mut Self {
        self.log_interval = log_interval;
        self
    }

    pub fn result_dir(&mut self, result_dir: &str) -> &mut Self {
        self.result_dir = Some(result_dir.to_string());
        self
    }

    pub fn build(&mut self) -> Result<ImageClassificationLearner<D, M, L, O>, LearnerError> {
        let model = self.model.take().ok_or(LearnerError::MissingArgument(
            "model in ImageClassificationLearnerBuilder is required".to_string(),
        ))?;
        let model_name = self.model_name.take().ok_or(LearnerError::MissingArgument(
            "model_name in ImageClassificationLearnerBuilder (set by model() method)".to_string(),
        ))?;

        self.metrics
            .insert(0, AnyMetric::AccuracyMetric(AccuracyMetric::new()));

        if self.lossfn.is_some() {
            self.metrics.insert(
                1,
                AnyMetric::LossMetric(LossMetric::new(nove_lossfn::CrossEntropyLoss::new())),
            );
        }

        let result_dir = if let Some(result_dir) = self.result_dir.take() {
            std::fs::create_dir_all(result_dir.clone()).map_err(|e| {
                LearnerError::InvalidPath(format!(
                    "Failed to create result_dir {:?}: {}",
                    result_dir, e
                ))
            })?;
            PathBuf::from(result_dir)
        } else {
            PathBuf::from(".")
        };

        Ok(ImageClassificationLearner {
            train_dataloader: self.train_dataloader.take(),
            validate_dataloader: self.validate_dataloader.take(),
            test_dataloader: self.test_dataloader.take(),
            model,
            lossfn: self.lossfn.take(),
            optimizer: self.optimizer.take(),
            epoch: self.epoch,
            metrics: std::mem::take(&mut self.metrics),
            log_interval: self.log_interval,
            model_name,
            result_dir,
            best_accuracy: 0.0,
        })
    }
}
