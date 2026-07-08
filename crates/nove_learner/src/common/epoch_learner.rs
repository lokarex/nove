use std::path::PathBuf;

use nove_dataloader::Dataloader;
use nove_lossfn::LossFn;
use nove_metric::{AnyMetric, Metric, MetricValue};
use nove_model::Model;
use nove_optimizer::Optimizer;

use crate::{Learner, LearnerError};

/// A customizable training learner that delegates per-batch behavior to step
/// closures.
///
/// # Notes
/// * The `EpochLearner` can only be created by the [`EpochLearnerBuilder`].
/// * Argument validation is deferred to the actual method calls (`train`/`validate`/`test`).
/// * Users provide step closures (`train_step_fn`, `validation_step_fn`, `test_step_fn`)
///   to control the training, validation, and testing loops.
#[allow(clippy::type_complexity)]
pub struct EpochLearner<D, M, L, O>
where
    D: Dataloader,
    M: Model,
    L: LossFn,
    O: Optimizer,
{
    // Data
    train_dataloader: Option<D>,
    validation_dataloader: Option<D>,
    test_dataloader: Option<D>,

    // Model, loss function, and optimizer
    model: M,
    loss_fn: Option<L>,
    optimizer: Option<O>,

    // Step closures
    train_step_fn: Option<
        Box<
            dyn for<'a> FnMut(
                    D::Output,
                    &'a mut M,
                    &'a mut L,
                    &'a mut O,
                    &'a mut [AnyMetric],
                ) -> Result<(), LearnerError>
                + Send
                + 'static,
        >,
    >,
    validation_step_fn: Option<
        Box<
            dyn for<'a> FnMut(D::Output, &'a mut M, &'a mut [AnyMetric]) -> Result<(), LearnerError>
                + Send
                + 'static,
        >,
    >,
    test_step_fn: Option<
        Box<
            dyn for<'a> FnMut(D::Output, &'a mut M, &'a mut [AnyMetric]) -> Result<(), LearnerError>
                + Send
                + 'static,
        >,
    >,

    // Configuration
    num_epochs: usize,
    log_interval: usize,
    metrics: Vec<AnyMetric>,

    // Checkpoint
    model_name: Option<String>,
    output_dir: Option<PathBuf>,
    save_every_epoch: bool,
    is_best_model: Option<Box<dyn Fn(Vec<AnyMetric>, Vec<AnyMetric>) -> bool + Send + 'static>>,
    best_model_metrics: Option<Vec<AnyMetric>>,
}

impl<D, M, L, O> EpochLearner<D, M, L, O>
where
    D: Dataloader,
    M: Model,
    L: LossFn,
    O: Optimizer,
{
    /// Ensure training is ready.
    ///
    /// # Returns
    /// * `Ok(())` - If training is ready.
    /// * `Err(LearnerError)` - If required arguments are missing or invalid.
    pub fn ensure_train_ready(&self) -> Result<(), LearnerError> {
        self.train_dataloader
            .as_ref()
            .ok_or(LearnerError::MissingArgument(
                "train_dataloader is required for training".to_string(),
            ))?;
        self.validation_dataloader
            .as_ref()
            .ok_or(LearnerError::MissingArgument(
                "validation_dataloader is required for training".to_string(),
            ))?;
        self.loss_fn.as_ref().ok_or(LearnerError::MissingArgument(
            "loss_fn is required for training".to_string(),
        ))?;
        self.optimizer
            .as_ref()
            .ok_or(LearnerError::MissingArgument(
                "optimizer is required for training".to_string(),
            ))?;
        if self.num_epochs == 0 {
            return Err(LearnerError::InvalidArgument(
                "num_epochs must be greater than 0 for training".to_string(),
            ));
        }
        if self.log_interval == 0 {
            return Err(LearnerError::InvalidArgument(
                "log_interval must be greater than 0 for training".to_string(),
            ));
        }
        if self.train_step_fn.is_none() {
            return Err(LearnerError::MissingArgument(
                "train_step_fn is required for training".to_string(),
            ));
        }
        Ok(())
    }

    /// Ensure validation is ready.
    ///
    /// # Returns
    /// * `Ok(())` - If validation is ready.
    /// * `Err(LearnerError)` - If required arguments are missing or invalid.
    pub fn ensure_validation_ready(&self) -> Result<(), LearnerError> {
        self.validation_dataloader
            .as_ref()
            .ok_or(LearnerError::MissingArgument(
                "validation_dataloader is required for validation".to_string(),
            ))?;
        if self.validation_step_fn.is_none() {
            return Err(LearnerError::MissingArgument(
                "validation_step_fn is required for validation".to_string(),
            ));
        }
        Ok(())
    }

    /// Ensure test is ready.
    ///
    /// # Returns
    /// * `Ok(())` - If test is ready.
    /// * `Err(LearnerError)` - If required arguments are missing or invalid.
    pub fn ensure_test_ready(&self) -> Result<(), LearnerError> {
        self.test_dataloader
            .as_ref()
            .ok_or(LearnerError::MissingArgument(
                "test_dataloader is required for testing".to_string(),
            ))?;
        if self.test_step_fn.is_none() {
            return Err(LearnerError::MissingArgument(
                "test_step_fn is required for testing".to_string(),
            ));
        }
        Ok(())
    }

    /// Get model reference.
    ///
    /// # Returns
    /// * `&M` - The model reference.
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Get mutable model reference.
    ///
    /// # Returns
    /// * `&mut M` - The mutable model reference.
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }

    /// Get metrics reference.
    ///
    /// # Returns
    /// * `&[AnyMetric]` - The metrics reference.
    pub fn metrics(&self) -> &[AnyMetric] {
        &self.metrics
    }

    /// Get mutable metrics reference.
    ///
    /// # Returns
    /// * `&mut Vec<AnyMetric>` - The mutable metrics reference.
    pub fn metrics_mut(&mut self) -> &mut Vec<AnyMetric> {
        &mut self.metrics
    }

    /// Get number of epochs.
    ///
    /// # Returns
    /// * `usize` - The number of epochs.
    pub fn num_epochs(&self) -> usize {
        self.num_epochs
    }

    /// Get log interval.
    ///
    /// # Returns
    /// * `usize` - The log interval.
    pub fn log_interval(&self) -> usize {
        self.log_interval
    }

    /// Get mutable train dataloader reference.
    ///
    /// # Returns
    /// * `Option<&mut D>` - The mutable train dataloader reference if set.
    pub fn train_dataloader_mut(&mut self) -> Option<&mut D> {
        self.train_dataloader.as_mut()
    }

    /// Get mutable validation dataloader reference.
    ///
    /// # Returns
    /// * `Option<&mut D>` - The mutable validation dataloader reference if set.
    pub fn validation_dataloader_mut(&mut self) -> Option<&mut D> {
        self.validation_dataloader.as_mut()
    }

    /// Get mutable test dataloader reference.
    ///
    /// # Returns
    /// * `Option<&mut D>` - The mutable test dataloader reference if set.
    pub fn test_dataloader_mut(&mut self) -> Option<&mut D> {
        self.test_dataloader.as_mut()
    }

    /// Get loss function reference.
    ///
    /// # Returns
    /// * `Option<&L>` - The loss function reference if set.
    pub fn loss_fn(&self) -> Option<&L> {
        self.loss_fn.as_ref()
    }

    /// Get mutable optimizer reference.
    ///
    /// # Returns
    /// * `Option<&mut O>` - The mutable optimizer reference if set.
    pub fn optimizer_mut(&mut self) -> Option<&mut O> {
        self.optimizer.as_mut()
    }

    /// Save checkpoint.
    ///
    /// # Arguments
    /// * `epoch_index` - The zero-based epoch index.
    ///
    /// # Returns
    /// * `Ok(())` - If successful.
    /// * `Err(LearnerError)` - If saving fails.
    pub fn save_checkpoint(&mut self, epoch_index: usize) -> Result<(), LearnerError> {
        let model_name = self
            .model_name
            .as_ref()
            .ok_or(LearnerError::MissingArgument(
                "model_name is required for saving checkpoint".to_string(),
            ))?;
        let output_dir = self
            .output_dir
            .as_ref()
            .ok_or(LearnerError::MissingArgument(
                "output_dir is required for saving checkpoint".to_string(),
            ))?;

        let path = output_dir.join(format!("{}_{}.safetensors", model_name, epoch_index + 1));
        let path_str = path.to_str().ok_or(LearnerError::InvalidPath(
            "output_dir in EpochLearner".to_string(),
        ))?;
        self.model.save(path_str)?;
        Ok(())
    }

    /// Save best model checkpoint.
    ///
    /// # Returns
    /// * `Ok(())` - If successful.
    /// * `Err(LearnerError)` - If saving fails.
    pub fn save_best_checkpoint(&mut self) -> Result<(), LearnerError> {
        let model_name = self
            .model_name
            .as_ref()
            .ok_or(LearnerError::MissingArgument(
                "model_name is required for saving best checkpoint".to_string(),
            ))?;
        let output_dir = self
            .output_dir
            .as_ref()
            .ok_or(LearnerError::MissingArgument(
                "output_dir is required for saving best checkpoint".to_string(),
            ))?;

        let path = output_dir.join(format!("{}_best.safetensors", model_name));
        let path_str = path.to_str().ok_or(LearnerError::InvalidPath(
            "output_dir in EpochLearner".to_string(),
        ))?;
        self.model.save(path_str)?;
        Ok(())
    }

    fn check_and_update_best_model(&mut self) -> Result<bool, LearnerError> {
        if let Some(is_best_model_fn) = &self.is_best_model {
            let current_metrics = self.metrics.clone();
            let is_best = is_best_model_fn(
                self.best_model_metrics.clone().unwrap_or_default(),
                current_metrics.clone(),
            );
            if is_best {
                self.best_model_metrics = Some(current_metrics);
            }
            Ok(is_best)
        } else {
            Ok(false)
        }
    }

    /// Print metrics.
    fn print_metrics(&self, prefix: &str, epoch_index: usize, batch_count: usize) {
        print!("{}: Epoch {}/{}", prefix, epoch_index + 1, self.num_epochs);
        print!(", batch {}", batch_count);
        for metric in self.metrics.iter() {
            let name = metric.name().unwrap_or_else(|_| "unknown".to_string());
            let value = match metric.value() {
                Ok(MetricValue::Scalar(v)) => v,
                _ => 0.0,
            };
            print!(", {}= {:.3}", name, value);
        }
        println!();
    }

    /// Execute training step.
    fn execute_train_step(&mut self, batch: D::Output) -> Result<(), LearnerError> {
        let train_step_fn = self.train_step_fn.as_mut().unwrap();
        train_step_fn(
            batch,
            &mut self.model,
            self.loss_fn.as_mut().unwrap(),
            self.optimizer.as_mut().unwrap(),
            &mut self.metrics,
        )?;
        Ok(())
    }

    /// Execute validation step.
    fn execute_validation_step(&mut self, batch: D::Output) -> Result<(), LearnerError> {
        let validation_step_fn = self.validation_step_fn.as_mut().unwrap();
        validation_step_fn(batch, &mut self.model, &mut self.metrics)?;
        Ok(())
    }

    /// Execute test step.
    fn execute_test_step(&mut self, batch: D::Output) -> Result<(), LearnerError> {
        let test_step_fn = self.test_step_fn.as_mut().unwrap();
        test_step_fn(batch, &mut self.model, &mut self.metrics)?;
        Ok(())
    }
}

impl<D, M, L, O> Learner for EpochLearner<D, M, L, O>
where
    D: Dataloader,
    M: Model,
    L: LossFn,
    O: Optimizer,
{
    /// Run the training loop.
    ///
    /// # Returns
    /// * `Ok(())` - If training completes successfully.
    /// * `Err(LearnerError)` - If training fails.
    fn train(&mut self) -> Result<(), LearnerError> {
        self.ensure_train_ready()?;
        self.model.train(true)?;

        let log_interval = self.log_interval;
        let num_epochs = self.num_epochs;

        for epoch_index in 0..num_epochs {
            // Training phase - take dataloader out to avoid borrow conflicts
            {
                let mut dataloader = self.train_dataloader.take().unwrap();
                let mut batch_count: usize = 0;

                loop {
                    let batch = match dataloader.next()? {
                        Some(b) => b,
                        None => break,
                    };

                    self.execute_train_step(batch)?;
                    batch_count += 1;

                    // Log output
                    if batch_count.is_multiple_of(log_interval) {
                        self.print_metrics("Train", epoch_index, batch_count);
                        // Clear metrics for next interval
                        for metric in self.metrics.iter_mut() {
                            metric.clear().ok();
                        }
                    }
                }
                dataloader.reset()?;
                self.train_dataloader = Some(dataloader);
            }

            // Validation phase
            {
                self.model.eval()?;

                let mut validation_dataloader = self.validation_dataloader.take().unwrap();
                loop {
                    let batch = match validation_dataloader.next()? {
                        Some(b) => b,
                        None => break,
                    };
                    self.execute_validation_step(batch)?;
                }
                validation_dataloader.reset()?;
                self.validation_dataloader = Some(validation_dataloader);

                self.print_metrics("Validate", epoch_index, 0);

                if self.check_and_update_best_model()? {
                    self.save_best_checkpoint()?;
                }

                for metric in self.metrics.iter_mut() {
                    metric.clear().ok();
                }
            }

            if self.save_every_epoch {
                self.save_checkpoint(epoch_index)?;
            }
        }

        Ok(())
    }

    /// Run validation.
    ///
    /// # Returns
    /// * `Ok(Vec<AnyMetric>)` - The validation metrics.
    /// * `Err(LearnerError)` - If validation fails.
    fn validate(&mut self) -> Result<Vec<AnyMetric>, LearnerError> {
        self.ensure_validation_ready()?;
        self.model.eval()?;

        let mut validation_dataloader = self.validation_dataloader.take().unwrap();
        loop {
            let batch = match validation_dataloader.next()? {
                Some(b) => b,
                None => break,
            };
            self.execute_validation_step(batch)?;
        }
        validation_dataloader.reset()?;
        self.validation_dataloader = Some(validation_dataloader);

        let metrics = self.metrics.clone();
        for metric in self.metrics.iter_mut() {
            metric.clear().ok();
        }

        Ok(metrics)
    }

    /// Run testing.
    ///
    /// # Returns
    /// * `Ok(Vec<AnyMetric>)` - The test metrics.
    /// * `Err(LearnerError)` - If testing fails.
    fn test(&mut self) -> Result<Vec<AnyMetric>, LearnerError> {
        self.ensure_test_ready()?;
        self.model.eval()?;

        let mut test_dataloader = self.test_dataloader.take().unwrap();
        loop {
            let batch = match test_dataloader.next()? {
                Some(b) => b,
                None => break,
            };
            self.execute_test_step(batch)?;
        }
        test_dataloader.reset()?;
        self.test_dataloader = Some(test_dataloader);

        let metrics = self.metrics.clone();
        for metric in self.metrics.iter_mut() {
            metric.clear().ok();
        }

        Ok(metrics)
    }
}

/// Builder for constructing [`EpochLearner`] instances.
///
/// # Notes
/// * The `EpochLearnerBuilder` implements the `Default` trait, so you can use
///   `EpochLearnerBuilder::default()` to create a builder with default values.
/// * Argument validation is deferred to the actual method calls (`train`/`validate`/`test`).
///
/// # Required Arguments
/// * `model` - The model to train/evaluate. Required for all operations.
///
/// # Conditional Required Arguments by Method
/// * `train()` requires: `train_dataloader`, `validation_dataloader`, `loss_fn`, `optimizer`, `num_epochs` > 0, `log_interval` > 0, `train_step_fn`, `validation_step_fn`, `output_dir`
/// * `validate()` requires: `validation_dataloader` and `validation_step_fn`
/// * `test()` requires: `test_dataloader` and `test_step_fn`
///
/// # Optional Arguments
/// * `train_dataloader` - Dataloader for training. Required only for `train()`.
/// * `validation_dataloader` - Dataloader for validation. Required for `train()` and `validate()`.
/// * `test_dataloader` - Dataloader for testing. Required only for `test()`.
/// * `loss_fn` - Loss function. Required only for `train()`.
/// * `optimizer` - Optimizer. Required only for `train()`.
/// * `num_epochs` - Number of training epochs. Default is 10. Must be > 0 for `train()`.
/// * `log_interval` - Logging frequency during training. Default is 10. Must be > 0 for `train()`.
/// * `metrics` - Custom metrics to track.
/// * `output_dir` - Directory to save learner output.
/// * `model_name` - Name of the model for checkpointing.
/// * `save_every_epoch` - Whether to save checkpoint every epoch. Default is true.
/// * `is_best_model` - Function to determine best model.
/// * `train_step_fn` - Step closures for training.
/// * `validation_step_fn` - Step closures for validation.
/// * `test_step_fn` - Step closure for testing.
#[allow(clippy::type_complexity)]
pub struct EpochLearnerBuilder<D, M, L, O>
where
    D: Dataloader,
    M: Model,
    L: LossFn,
    O: Optimizer,
{
    train_dataloader: Option<D>,
    validation_dataloader: Option<D>,
    test_dataloader: Option<D>,
    model: Option<M>,
    loss_fn: Option<L>,
    optimizer: Option<O>,

    // Step closures
    train_step_fn: Option<
        Box<
            dyn for<'a> FnMut(
                    D::Output,
                    &'a mut M,
                    &'a mut L,
                    &'a mut O,
                    &'a mut [AnyMetric],
                ) -> Result<(), LearnerError>
                + Send
                + 'static,
        >,
    >,
    validation_step_fn: Option<
        Box<
            dyn for<'a> FnMut(D::Output, &'a mut M, &'a mut [AnyMetric]) -> Result<(), LearnerError>
                + Send
                + 'static,
        >,
    >,
    test_step_fn: Option<
        Box<
            dyn for<'a> FnMut(D::Output, &'a mut M, &'a mut [AnyMetric]) -> Result<(), LearnerError>
                + Send
                + 'static,
        >,
    >,

    // Configuration
    num_epochs: usize,
    log_interval: usize,
    metrics: Vec<AnyMetric>,
    model_name: Option<String>,
    output_dir: Option<PathBuf>,
    save_every_epoch: bool,
    is_best_model: Option<Box<dyn Fn(Vec<AnyMetric>, Vec<AnyMetric>) -> bool + Send + 'static>>,
}

impl<D, M, L, O> Default for EpochLearnerBuilder<D, M, L, O>
where
    D: Dataloader,
    M: Model,
    L: LossFn,
    O: Optimizer,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<D, M, L, O> EpochLearnerBuilder<D, M, L, O>
where
    D: Dataloader,
    M: Model,
    L: LossFn,
    O: Optimizer,
{
    pub fn new() -> Self {
        Self {
            train_dataloader: None,
            validation_dataloader: None,
            test_dataloader: None,
            model: None,
            loss_fn: None,
            optimizer: None,
            train_step_fn: None,
            validation_step_fn: None,
            test_step_fn: None,
            num_epochs: 10,
            log_interval: 10,
            metrics: Vec::new(),
            model_name: None,
            output_dir: None,
            save_every_epoch: true,
            is_best_model: None,
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
        self.train_dataloader = Some(dataloader);
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
        self.validation_dataloader = Some(dataloader);
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
        self.test_dataloader = Some(dataloader);
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
        let full_name = std::any::type_name::<M>();
        let type_name = full_name
            .rsplit("::")
            .next()
            .unwrap_or("model")
            .split('<')
            .next()
            .unwrap_or("model");
        self.model_name = Some(type_name.to_string());
        self.model = Some(model);
        self
    }

    /// Sets the loss function.
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
        self.optimizer = Some(optimizer);
        self
    }

    /// Sets the training step function.
    ///
    /// # Arguments
    /// * `f` - The training step function closure with signature `FnMut(D::Output, &mut M, &mut L, &mut O, &mut [AnyMetric]) -> Result<(), LearnerError>`.
    ///
    /// # Returns
    /// * `Self` - The builder itself.
    pub fn train_step_fn<StepFn>(mut self, f: StepFn) -> Self
    where
        StepFn: for<'a> FnMut(
                D::Output,
                &'a mut M,
                &'a mut L,
                &'a mut O,
                &'a mut [AnyMetric],
            ) -> Result<(), LearnerError>
            + Send
            + 'static,
    {
        self.train_step_fn = Some(Box::new(f));
        self
    }

    /// Sets the validation step function.
    ///
    /// # Arguments
    /// * `f` - The validation step function closure with signature `FnMut(D::Output, &mut M, &mut [AnyMetric]) -> Result<(), LearnerError>`.
    ///
    /// # Returns
    /// * `Self` - The builder itself.
    pub fn validation_step_fn<StepFn>(mut self, f: StepFn) -> Self
    where
        StepFn: for<'a> FnMut(D::Output, &'a mut M, &'a mut [AnyMetric]) -> Result<(), LearnerError>
            + Send
            + 'static,
    {
        self.validation_step_fn = Some(Box::new(f));
        self
    }

    /// Sets the test step function.
    ///
    /// # Arguments
    /// * `f` - The test step function closure with signature `FnMut(D::Output, &mut M, &mut [AnyMetric]) -> Result<(), LearnerError>`.
    ///
    /// # Returns
    /// * `Self` - The builder itself.
    pub fn test_step_fn<StepFn>(mut self, f: StepFn) -> Self
    where
        StepFn: for<'a> FnMut(D::Output, &'a mut M, &'a mut [AnyMetric]) -> Result<(), LearnerError>
            + Send
            + 'static,
    {
        self.test_step_fn = Some(Box::new(f));
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
        self.num_epochs = num_epochs;
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
        self.log_interval = log_interval;
        self
    }

    /// Sets the metrics to track.
    ///
    /// # Arguments
    /// * `metrics` - A vector of metrics to track.
    ///
    /// # Returns
    /// * `&mut Self` - The builder itself.
    pub fn metrics(&mut self, metrics: Vec<AnyMetric>) -> &mut Self {
        self.metrics.extend(metrics);
        self
    }

    /// Adds a single metric to track.
    ///
    /// # Arguments
    /// * `metric` - The metric to add.
    ///
    /// # Returns
    /// * `&mut Self` - The builder itself.
    pub fn metric<MT: Into<AnyMetric>>(&mut self, metric: MT) -> &mut Self {
        self.metrics.push(metric.into());
        self
    }

    /// Sets the model name.
    ///
    /// # Arguments
    /// * `name` - The name of the model for checkpointing.
    ///
    /// # Returns
    /// * `&mut Self` - The builder itself.
    pub fn model_name(&mut self, name: &str) -> &mut Self {
        self.model_name = Some(name.to_string());
        self
    }

    /// Sets the output directory.
    ///
    /// # Arguments
    /// * `dir` - The directory to save learner output.
    ///
    /// # Returns
    /// * `&mut Self` - The builder itself.
    pub fn output_dir(&mut self, dir: &str) -> &mut Self {
        self.output_dir = Some(PathBuf::from(dir));
        self
    }

    /// Sets whether to save checkpoint every epoch.
    ///
    /// # Arguments
    /// * `save_every_epoch` - Whether to save checkpoint every epoch. Default is true.
    ///
    /// # Returns
    /// * `&mut Self` - The builder itself.
    pub fn save_every_epoch(&mut self, save_every_epoch: bool) -> &mut Self {
        self.save_every_epoch = save_every_epoch;
        self
    }

    /// Sets the best model determination function.
    ///
    /// # Arguments
    /// * `f` - The function with signature `Fn(Vec<AnyMetric>, Vec<AnyMetric>) -> bool`.
    ///
    /// # Returns
    /// * `Self` - The builder itself.
    pub fn is_best_model<PredicateFn>(mut self, f: PredicateFn) -> Self
    where
        PredicateFn: Fn(Vec<AnyMetric>, Vec<AnyMetric>) -> bool + Send + 'static,
    {
        self.is_best_model = Some(Box::new(f));
        self
    }

    /// Builds the EpochLearner.
    ///
    /// # Returns
    /// * `Ok(EpochLearner<D, M, L, O>)` - The built learner.
    /// * `Err(LearnerError)` - If required arguments are missing or invalid.
    pub fn build(&mut self) -> Result<EpochLearner<D, M, L, O>, LearnerError> {
        let model = self.model.take().ok_or(LearnerError::MissingArgument(
            "model in EpochLearnerBuilder is required".to_string(),
        ))?;

        let output_dir = if let Some(output_dir) = self.output_dir.take() {
            std::fs::create_dir_all(output_dir.clone()).map_err(|e| {
                LearnerError::InvalidPath(format!(
                    "Failed to create output_dir {:?}: {}",
                    output_dir, e
                ))
            })?;
            Some(output_dir)
        } else {
            None
        };

        Ok(EpochLearner {
            train_dataloader: self.train_dataloader.take(),
            validation_dataloader: self.validation_dataloader.take(),
            test_dataloader: self.test_dataloader.take(),
            model,
            loss_fn: self.loss_fn.take(),
            optimizer: self.optimizer.take(),
            train_step_fn: self.train_step_fn.take(),
            validation_step_fn: self.validation_step_fn.take(),
            test_step_fn: self.test_step_fn.take(),
            num_epochs: self.num_epochs,
            log_interval: self.log_interval,
            metrics: std::mem::take(&mut self.metrics),
            model_name: self.model_name.take(),
            output_dir,
            save_every_epoch: self.save_every_epoch,
            is_best_model: self.is_best_model.take(),
            best_model_metrics: None,
        })
    }
}
