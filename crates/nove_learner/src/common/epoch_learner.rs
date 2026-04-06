use std::path::PathBuf;

use nove_dataloader::Dataloader;
use nove_lossfn::LossFn;
use nove_metric::{AnyMetric, Metric, MetricValue};
use nove_model::Model;
use nove_optimizer::Optimizer;
use nove_tensor::Tensor;

use crate::{Learner, LearnerError};

/// A highly customizable training learner with closures for forward, backward,
/// metrics, and optimizer steps.
///
/// # Notes
/// * The `EpochLearner` can only be created by the [`EpochLearnerBuilder`].
/// * Argument validation is deferred to the actual method calls (`train`/`validate`/`test`).
/// * Users can choose between fine-grained closures (`forward_fn`, `metrics_fn`, etc.)
///   or combined closures (`train_step_fn`, `eval_step_fn`) to control the training loop.
///
/// # Fields
/// * `train_dataloader` - Dataloader for training data.
/// * `validate_dataloader` - Dataloader for validation data.
/// * `test_dataloader` - Dataloader for test data.
/// * `model` - The model to train/evaluate.
/// * `lossfn` - Loss function for training.
/// * `optimizer` - Optimizer for training.
/// * `forward_fn` - Fine-grained forward function closure.
/// * `metrics_fn` - Fine-grained metrics function closure.
/// * `loss_fn` - Fine-grained loss function closure.
/// * `backward_fn` - Fine-grained backward function closure.
/// * `optimizer_step_fn` - Fine-grained optimizer step function closure.
/// * `train_step_fn` - Combined training step function closure.
/// * `eval_step_fn` - Combined evaluation step function closure.
/// * `epoch` - Number of training epochs.
/// * `log_interval` - Number of batches between logging.
/// * `metrics` - Metrics to track during training/evaluation.
/// * `model_name` - Name of the model for checkpointing.
/// * `result_dir` - Directory to save checkpoints.
/// * `save_every_epoch` - Whether to save checkpoint every epoch.
/// * `is_best_model` - Function to determine if current model is the best.
/// * `best_model_metrics` - Best validation metrics achieved during training.
/// * `is_best_model_current_epoch` - Whether the best model is from the current epoch.
pub struct EpochLearner<D, M, L, O>
where
    D: Dataloader,
    M: Model,
    L: LossFn,
    O: Optimizer,
{
    // Data
    train_dataloader: Option<D>,
    validate_dataloader: Option<D>,
    test_dataloader: Option<D>,

    // Model, loss function, and optimizer
    model: M,
    lossfn: Option<L>,
    optimizer: Option<O>,

    // Fine-grained closures (optional, takes precedence over combined closures)
    forward_fn: Option<
        Box<
            dyn for<'a> FnMut(&'a mut M, &D::Output) -> Result<M::Output, LearnerError>
                + Send
                + 'static,
        >,
    >,
    metrics_fn: Option<
        Box<
            dyn for<'a> FnMut(
                    &M::Output,
                    &D::Output,
                    &'a mut [AnyMetric],
                ) -> Result<(), LearnerError>
                + Send
                + 'static,
        >,
    >,
    loss_fn: Option<
        Box<
            dyn for<'a> FnMut(&'a mut L, &M::Output, &D::Output) -> Result<Tensor, LearnerError>
                + Send
                + 'static,
        >,
    >,
    backward_fn: Option<
        Box<dyn for<'a> FnMut(&'a mut O, Tensor) -> Result<(), LearnerError> + Send + 'static>,
    >,
    optimizer_step_fn: Option<
        Box<dyn for<'a> FnMut(&'a mut O) -> Result<O::StepOutput, LearnerError> + Send + 'static>,
    >,

    // Combined closures (optional, used when fine-grained closures are not set)
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
    eval_step_fn: Option<
        Box<
            dyn for<'a> FnMut(D::Output, &'a mut M, &'a mut [AnyMetric]) -> Result<(), LearnerError>
                + Send
                + 'static,
        >,
    >,

    // Configuration
    epoch: usize,
    log_interval: usize,
    metrics: Vec<AnyMetric>,

    // Checkpoint
    model_name: Option<String>,
    result_dir: Option<PathBuf>,
    save_every_epoch: bool,
    is_best_model: Option<Box<dyn Fn(Vec<AnyMetric>, Vec<AnyMetric>) -> bool + Send + 'static>>,
    best_model_metrics: Option<Vec<AnyMetric>>,
    is_best_model_current_epoch: bool,
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
        // Check if we have training closures
        let has_fine_grained = self.forward_fn.is_some()
            && self.loss_fn.is_some()
            && self.backward_fn.is_some()
            && self.optimizer_step_fn.is_some();
        let has_combined = self.train_step_fn.is_some();

        if !has_fine_grained && !has_combined {
            return Err(LearnerError::MissingArgument(
                "Either train_step_fn or all of (forward_fn, loss_fn, backward_fn, optimizer_step_fn) must be set"
                    .to_string(),
            ));
        }
        Ok(())
    }

    /// Ensure validation is ready.
    ///
    /// # Returns
    /// * `Ok(())` - If validation is ready.
    /// * `Err(LearnerError)` - If required arguments are missing or invalid.
    pub fn ensure_validate_ready(&self) -> Result<(), LearnerError> {
        self.validate_dataloader
            .as_ref()
            .ok_or(LearnerError::MissingArgument(
                "validate_dataloader is required for validation".to_string(),
            ))?;
        let has_fine_grained = self.forward_fn.is_some() && self.metrics_fn.is_some();
        let has_combined = self.eval_step_fn.is_some();

        if !has_fine_grained && !has_combined {
            return Err(LearnerError::MissingArgument(
                "Either eval_step_fn or (forward_fn, metrics_fn) must be set".to_string(),
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
        let has_fine_grained = self.forward_fn.is_some() && self.metrics_fn.is_some();
        let has_combined = self.eval_step_fn.is_some();

        if !has_fine_grained && !has_combined {
            return Err(LearnerError::MissingArgument(
                "Either eval_step_fn or (forward_fn, metrics_fn) must be set".to_string(),
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

    /// Get epoch count.
    ///
    /// # Returns
    /// * `usize` - The number of epochs.
    pub fn epoch_count(&self) -> usize {
        self.epoch
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

    /// Get mutable validate dataloader reference.
    ///
    /// # Returns
    /// * `Option<&mut D>` - The mutable validate dataloader reference if set.
    pub fn validate_dataloader_mut(&mut self) -> Option<&mut D> {
        self.validate_dataloader.as_mut()
    }

    /// Get mutable test dataloader reference.
    ///
    /// # Returns
    /// * `Option<&mut D>` - The mutable test dataloader reference if set.
    pub fn test_dataloader_mut(&mut self) -> Option<&mut D> {
        self.test_dataloader.as_mut()
    }

    /// Get lossfn reference.
    ///
    /// # Returns
    /// * `Option<&L>` - The lossfn reference if set.
    pub fn lossfn(&self) -> Option<&L> {
        self.lossfn.as_ref()
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
    /// * `epoch` - The epoch number.
    ///
    /// # Returns
    /// * `Ok(())` - If successful.
    /// * `Err(LearnerError)` - If saving fails.
    pub fn save_checkpoint(&mut self, epoch: usize) -> Result<(), LearnerError> {
        let model_name = self
            .model_name
            .as_ref()
            .ok_or(LearnerError::MissingArgument(
                "model_name is required for saving checkpoint".to_string(),
            ))?;
        let result_dir = self
            .result_dir
            .as_ref()
            .ok_or(LearnerError::MissingArgument(
                "result_dir is required for saving checkpoint".to_string(),
            ))?;

        let path = result_dir.join(format!("{}_{}.safetensors", model_name, epoch + 1));
        let path_str = path.to_str().ok_or(LearnerError::InvalidPath(
            "result_dir in EpochLearner".to_string(),
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
        let result_dir = self
            .result_dir
            .as_ref()
            .ok_or(LearnerError::MissingArgument(
                "result_dir is required for saving best checkpoint".to_string(),
            ))?;

        let path = result_dir.join(format!("{}_best.safetensors", model_name));
        let path_str = path.to_str().ok_or(LearnerError::InvalidPath(
            "result_dir in EpochLearner".to_string(),
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
            self.is_best_model_current_epoch = is_best;
            Ok(is_best)
        } else {
            Ok(false)
        }
    }

    /// Print metrics.
    fn print_metrics(&self, prefix: &str, epoch: usize, batch_count: usize) {
        print!("{}: Epoch {}/{}", prefix, epoch + 1, self.epoch);
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

    /// Execute training step using fine-grained closures.
    fn execute_train_step_fine_grained(&mut self, batch: &D::Output) -> Result<(), LearnerError> {
        let forward_fn = self.forward_fn.as_mut().unwrap();
        let output = forward_fn(&mut self.model, batch)?;

        if let Some(ref mut metrics_fn) = self.metrics_fn {
            metrics_fn(&output, batch, &mut self.metrics)?;
        }

        let loss_fn = self.loss_fn.as_mut().unwrap();
        let loss = loss_fn(self.lossfn.as_mut().unwrap(), &output, batch)?;

        let backward_fn = self.backward_fn.as_mut().unwrap();
        backward_fn(self.optimizer.as_mut().unwrap(), loss)?;

        let optimizer_step_fn = self.optimizer_step_fn.as_mut().unwrap();
        optimizer_step_fn(self.optimizer.as_mut().unwrap())?;

        Ok(())
    }

    /// Execute training step using combined closure.
    fn execute_train_step_combined(&mut self, batch: D::Output) -> Result<(), LearnerError> {
        let train_step_fn = self.train_step_fn.as_mut().unwrap();
        train_step_fn(
            batch,
            &mut self.model,
            self.lossfn.as_mut().unwrap(),
            self.optimizer.as_mut().unwrap(),
            &mut self.metrics,
        )?;
        Ok(())
    }

    /// Execute evaluation step using fine-grained closures.
    fn execute_eval_step_fine_grained(&mut self, batch: &D::Output) -> Result<(), LearnerError> {
        let forward_fn = self.forward_fn.as_mut().unwrap();
        let output = forward_fn(&mut self.model, batch)?;

        if let Some(ref mut metrics_fn) = self.metrics_fn {
            metrics_fn(&output, batch, &mut self.metrics)?;
        }
        Ok(())
    }

    /// Execute evaluation step using combined closure.
    fn execute_eval_step_combined(&mut self, batch: D::Output) -> Result<(), LearnerError> {
        let eval_step_fn = self.eval_step_fn.as_mut().unwrap();
        eval_step_fn(batch, &mut self.model, &mut self.metrics)?;
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

        let use_fine_grained = self.forward_fn.is_some();
        let log_interval = self.log_interval;
        let total_epochs = self.epoch;

        for epoch in 0..total_epochs {
            // Training phase - take dataloader out to avoid borrow conflicts
            {
                let mut dataloader = self.train_dataloader.take().unwrap();
                let mut batch_count: usize = 0;

                loop {
                    let batch = match dataloader.next()? {
                        Some(b) => b,
                        None => break,
                    };

                    // Execute training step
                    if use_fine_grained {
                        self.execute_train_step_fine_grained(&batch)?;
                    } else {
                        self.execute_train_step_combined(batch)?;
                    }
                    batch_count += 1;

                    // Log output
                    if batch_count % log_interval == 0 {
                        self.print_metrics("Train", epoch, batch_count);
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

                let mut validate_dataloader = self.validate_dataloader.take().unwrap();
                loop {
                    let batch = match validate_dataloader.next()? {
                        Some(b) => b,
                        None => break,
                    };
                    if use_fine_grained {
                        self.execute_eval_step_fine_grained(&batch)?;
                    } else {
                        self.execute_eval_step_combined(batch)?;
                    }
                }
                validate_dataloader.reset()?;
                self.validate_dataloader = Some(validate_dataloader);

                self.print_metrics("Validate", epoch, 0);
                for metric in self.metrics.iter_mut() {
                    metric.clear().ok();
                }
            }

            if self.save_every_epoch {
                self.save_checkpoint(epoch)?;
            }

            if self.check_and_update_best_model()? {
                self.save_best_checkpoint()?;
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
        self.ensure_validate_ready()?;
        self.model.eval()?;

        let use_fine_grained = self.forward_fn.is_some();

        let mut validate_dataloader = self.validate_dataloader.take().unwrap();
        loop {
            let batch = match validate_dataloader.next()? {
                Some(b) => b,
                None => break,
            };
            if use_fine_grained {
                self.execute_eval_step_fine_grained(&batch)?;
            } else {
                self.execute_eval_step_combined(batch)?;
            }
        }
        validate_dataloader.reset()?;
        self.validate_dataloader = Some(validate_dataloader);

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

        let use_fine_grained = self.forward_fn.is_some();

        let mut test_dataloader = self.test_dataloader.take().unwrap();
        loop {
            let batch = match test_dataloader.next()? {
                Some(b) => b,
                None => break,
            };
            if use_fine_grained {
                self.execute_eval_step_fine_grained(&batch)?;
            } else {
                self.execute_eval_step_combined(batch)?;
            }
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
/// * `train()` requires: `train_dataloader`, `validate_dataloader`, `lossfn`, `optimizer`, `epoch` > 0, `log_interval` > 0, and either `train_step_fn` or all fine-grained closures (`forward_fn`, `loss_fn`, `backward_fn`, `optimizer_step_fn`)
/// * `validate()` requires: `validate_dataloader`, and either `eval_step_fn` or (`forward_fn`, `metrics_fn`)
/// * `test()` requires: `test_dataloader`, and either `eval_step_fn` or (`forward_fn`, `metrics_fn`)
///
/// # Optional Arguments
/// * `train_dataloader` - Dataloader for training. Required only for `train()`.
/// * `validate_dataloader` - Dataloader for validation. Required for `train()` and `validate()`.
/// * `test_dataloader` - Dataloader for testing. Required only for `test()`.
/// * `lossfn` - Loss function. Required only for `train()`.
/// * `optimizer` - Optimizer. Required only for `train()`.
/// * `epoch` - Number of training epochs. Default is 10. Must be > 0 for `train()`.
/// * `log_interval` - Logging frequency during training. Default is 10. Must be > 0 for `train()`.
/// * `metrics` - Custom metrics to track.
/// * `result_dir` - Directory to save checkpoints.
/// * `model_name` - Name of the model for checkpointing.
/// * `save_every_epoch` - Whether to save checkpoint every epoch. Default is true.
/// * `is_best_model` - Function to determine best model.
/// * `forward_fn`, `metrics_fn`, `loss_fn`, `backward_fn`, `optimizer_step_fn` - Fine-grained closures.
/// * `train_step_fn`, `eval_step_fn` - Combined closures.
///
/// # Fields
/// * `train_dataloader` - Dataloader for training data.
/// * `validate_dataloader` - Dataloader for validation data.
/// * `test_dataloader` - Dataloader for testing data.
/// * `model` - The model to train/evaluate.
/// * `lossfn` - Loss function for training.
/// * `optimizer` - Optimizer for training.
/// * `forward_fn` - Fine-grained forward function closure.
/// * `metrics_fn` - Fine-grained metrics function closure.
/// * `loss_fn` - Fine-grained loss function closure.
/// * `backward_fn` - Fine-grained backward function closure.
/// * `optimizer_step_fn` - Fine-grained optimizer step function closure.
/// * `train_step_fn` - Combined training step function closure.
/// * `eval_step_fn` - Combined evaluation step function closure.
/// * `epoch` - Number of training epochs.
/// * `log_interval` - Number of batches between logging.
/// * `metrics` - Metrics to track.
/// * `model_name` - Name of the model for checkpointing.
/// * `result_dir` - Directory to save checkpoints.
/// * `save_every_epoch` - Whether to save checkpoint every epoch.
/// * `is_best_model` - Function to determine best model.
pub struct EpochLearnerBuilder<D, M, L, O>
where
    D: Dataloader,
    M: Model,
    L: LossFn,
    O: Optimizer,
{
    train_dataloader: Option<D>,
    validate_dataloader: Option<D>,
    test_dataloader: Option<D>,
    model: Option<M>,
    lossfn: Option<L>,
    optimizer: Option<O>,

    // Fine-grained closures
    forward_fn: Option<
        Box<
            dyn for<'a> FnMut(&'a mut M, &D::Output) -> Result<M::Output, LearnerError>
                + Send
                + 'static,
        >,
    >,
    metrics_fn: Option<
        Box<
            dyn for<'a> FnMut(
                    &M::Output,
                    &D::Output,
                    &'a mut [AnyMetric],
                ) -> Result<(), LearnerError>
                + Send
                + 'static,
        >,
    >,
    loss_fn: Option<
        Box<
            dyn for<'a> FnMut(&'a mut L, &M::Output, &D::Output) -> Result<Tensor, LearnerError>
                + Send
                + 'static,
        >,
    >,
    backward_fn: Option<
        Box<dyn for<'a> FnMut(&'a mut O, Tensor) -> Result<(), LearnerError> + Send + 'static>,
    >,
    optimizer_step_fn: Option<
        Box<dyn for<'a> FnMut(&'a mut O) -> Result<O::StepOutput, LearnerError> + Send + 'static>,
    >,

    // Combined closures
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
    eval_step_fn: Option<
        Box<
            dyn for<'a> FnMut(D::Output, &'a mut M, &'a mut [AnyMetric]) -> Result<(), LearnerError>
                + Send
                + 'static,
        >,
    >,

    // Configuration
    epoch: usize,
    log_interval: usize,
    metrics: Vec<AnyMetric>,
    model_name: Option<String>,
    result_dir: Option<PathBuf>,
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
            validate_dataloader: None,
            test_dataloader: None,
            model: None,
            lossfn: None,
            optimizer: None,
            forward_fn: None,
            metrics_fn: None,
            loss_fn: None,
            backward_fn: None,
            optimizer_step_fn: None,
            train_step_fn: None,
            eval_step_fn: None,
            epoch: 10,
            log_interval: 10,
            metrics: Vec::new(),
            model_name: None,
            result_dir: None,
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
    pub fn validate_dataloader(&mut self, dataloader: D) -> &mut Self {
        self.validate_dataloader = Some(dataloader);
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
    /// * `lossfn` - The loss function.
    ///
    /// # Returns
    /// * `&mut Self` - The builder itself.
    pub fn lossfn(&mut self, lossfn: L) -> &mut Self {
        self.lossfn = Some(lossfn);
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

    /// Sets the forward function for fine-grained control.
    ///
    /// # Arguments
    /// * `f` - The forward function closure with signature `FnMut(&mut M, &D::Output) -> Result<M::Output, LearnerError>`.
    ///
    /// # Returns
    /// * `Self` - The builder itself.
    pub fn forward_fn<Fn>(mut self, f: Fn) -> Self
    where
        Fn: for<'a> FnMut(&'a mut M, &D::Output) -> Result<M::Output, LearnerError>
            + Send
            + 'static,
    {
        self.forward_fn = Some(Box::new(f));
        self
    }

    /// Sets the metrics function for fine-grained control.
    ///
    /// # Arguments
    /// * `f` - The metrics function closure with signature `FnMut(&M::Output, &D::Output, &mut [AnyMetric]) -> Result<(), LearnerError>`.
    ///
    /// # Returns
    /// * `Self` - The builder itself.
    pub fn metrics_fn<Fn>(mut self, f: Fn) -> Self
    where
        Fn: for<'a> FnMut(&M::Output, &D::Output, &'a mut [AnyMetric]) -> Result<(), LearnerError>
            + Send
            + 'static,
    {
        self.metrics_fn = Some(Box::new(f));
        self
    }

    /// Sets the loss function for fine-grained control.
    ///
    /// # Arguments
    /// * `f` - The loss function closure with signature `FnMut(&mut L, &M::Output, &D::Output) -> Result<Tensor, LearnerError>`.
    ///
    /// # Returns
    /// * `Self` - The builder itself.
    pub fn loss_fn<Fn>(mut self, f: Fn) -> Self
    where
        Fn: for<'a> FnMut(&'a mut L, &M::Output, &D::Output) -> Result<Tensor, LearnerError>
            + Send
            + 'static,
    {
        self.loss_fn = Some(Box::new(f));
        self
    }

    /// Sets the backward function for fine-grained control.
    ///
    /// # Arguments
    /// * `f` - The backward function closure with signature `FnMut(&mut O, Tensor) -> Result<(), LearnerError>`.
    ///
    /// # Returns
    /// * `Self` - The builder itself.
    pub fn backward_fn<Fn>(mut self, f: Fn) -> Self
    where
        Fn: for<'a> FnMut(&'a mut O, Tensor) -> Result<(), LearnerError> + Send + 'static,
    {
        self.backward_fn = Some(Box::new(f));
        self
    }

    /// Sets the optimizer step function for fine-grained control.
    ///
    /// # Arguments
    /// * `f` - The optimizer step function closure with signature `FnMut(&mut O) -> Result<O::StepOutput, LearnerError>`.
    ///
    /// # Returns
    /// * `Self` - The builder itself.
    pub fn optimizer_step_fn<Fn>(mut self, f: Fn) -> Self
    where
        Fn: for<'a> FnMut(&'a mut O) -> Result<O::StepOutput, LearnerError> + Send + 'static,
    {
        self.optimizer_step_fn = Some(Box::new(f));
        self
    }

    /// Sets the combined training step function.
    ///
    /// # Arguments
    /// * `f` - The training step function closure with signature `FnMut(D::Output, &mut M, &mut L, &mut O, &mut [AnyMetric]) -> Result<(), LearnerError>`.
    ///
    /// # Returns
    /// * `Self` - The builder itself.
    pub fn train_step_fn<Fn>(mut self, f: Fn) -> Self
    where
        Fn: for<'a> FnMut(
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

    /// Sets the combined evaluation step function.
    ///
    /// # Arguments
    /// * `f` - The evaluation step function closure with signature `FnMut(D::Output, &mut M, &mut [AnyMetric]) -> Result<(), LearnerError>`.
    ///
    /// # Returns
    /// * `Self` - The builder itself.
    pub fn eval_step_fn<Fn>(mut self, f: Fn) -> Self
    where
        Fn: for<'a> FnMut(D::Output, &'a mut M, &'a mut [AnyMetric]) -> Result<(), LearnerError>
            + Send
            + 'static,
    {
        self.eval_step_fn = Some(Box::new(f));
        self
    }

    /// Sets the number of training epochs.
    ///
    /// # Arguments
    /// * `epoch` - The number of epochs. Default is 10.
    ///
    /// # Returns
    /// * `&mut Self` - The builder itself.
    pub fn epoch(&mut self, epoch: usize) -> &mut Self {
        self.epoch = epoch;
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

    /// Sets the result directory.
    ///
    /// # Arguments
    /// * `dir` - The directory to save checkpoints.
    ///
    /// # Returns
    /// * `&mut Self` - The builder itself.
    pub fn result_dir(&mut self, dir: &str) -> &mut Self {
        self.result_dir = Some(PathBuf::from(dir));
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
    pub fn is_best_model<F>(mut self, f: F) -> Self
    where
        F: Fn(Vec<AnyMetric>, Vec<AnyMetric>) -> bool + Send + 'static,
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

        let lossfn = self.lossfn.take().ok_or(LearnerError::MissingArgument(
            "lossfn in EpochLearnerBuilder is required".to_string(),
        ))?;

        let optimizer = self.optimizer.take().ok_or(LearnerError::MissingArgument(
            "optimizer in EpochLearnerBuilder is required".to_string(),
        ))?;

        let result_dir = if let Some(result_dir) = self.result_dir.take() {
            std::fs::create_dir_all(result_dir.clone()).map_err(|e| {
                LearnerError::InvalidPath(format!(
                    "Failed to create result_dir {:?}: {}",
                    result_dir, e
                ))
            })?;
            Some(result_dir)
        } else {
            None
        };

        Ok(EpochLearner {
            train_dataloader: self.train_dataloader.take(),
            validate_dataloader: self.validate_dataloader.take(),
            test_dataloader: self.test_dataloader.take(),
            model,
            lossfn: Some(lossfn),
            optimizer: Some(optimizer),
            forward_fn: self.forward_fn.take(),
            metrics_fn: self.metrics_fn.take(),
            loss_fn: self.loss_fn.take(),
            backward_fn: self.backward_fn.take(),
            optimizer_step_fn: self.optimizer_step_fn.take(),
            train_step_fn: self.train_step_fn.take(),
            eval_step_fn: self.eval_step_fn.take(),
            epoch: self.epoch,
            log_interval: self.log_interval,
            metrics: std::mem::take(&mut self.metrics),
            model_name: self.model_name.take(),
            result_dir,
            save_every_epoch: self.save_every_epoch,
            is_best_model: self.is_best_model.take(),
            best_model_metrics: None,
            is_best_model_current_epoch: false,
        })
    }
}
