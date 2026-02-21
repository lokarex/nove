use std::path::PathBuf;

use nove_dataloader::Dataloader;
use nove_lossfn::LossFn;
use nove_metric::{AnyMetric, MetricValue};
use nove_model::Model;
use nove_optimizer::Optimizer;
use nove_tensor::Tensor;

use crate::{Learner, LearnerError};

pub struct EpochLearner {
    train_dataloader: Box<dyn Dataloader<Output = (Tensor, Tensor)>>,
    validate_dataloader: Box<dyn Dataloader<Output = (Tensor, Tensor)>>,
    test_dataloader: Box<dyn Dataloader<Output = (Tensor, Tensor)>>,
    model: Box<dyn Model<Input = (Tensor, bool), Output = Tensor>>,
    lossfn: Box<dyn LossFn<Input = (Tensor, Tensor), Output = Tensor>>,
    optimizer: Box<dyn Optimizer<StepOutput = ()>>,
    epoch: usize,
    metrics: Vec<AnyMetric>,
    log_interval: usize,
    model_name: &'static str,
    result_dir: PathBuf,
}

impl Learner for EpochLearner {
    fn train(&mut self) -> Result<(), LearnerError> {
        for epoch in 0..self.epoch {
            // Train model
            let mut counter: usize = 0;
            let mut total_loss = 0.0;
            loop {
                let (inputs, targets) = match self.train_dataloader.next()? {
                    Some((inputs, targets)) => (inputs, targets),
                    None => break,
                };
                let outputs = self.model.forward((inputs, false))?;

                // Aggregate metrics
                for metric in self.metrics.iter_mut() {
                    let value = match metric {
                        AnyMetric::Evaluation(m) => m.evaluate(&outputs, &targets)?,
                        AnyMetric::Resource(m) => m.sample()?,
                    };
                    metric.update(metric.value()?.add(&value)?)?;
                }

                let loss = self.lossfn.loss((outputs, targets))?;
                total_loss += loss.to_scalar::<f64>()?;

                self.optimizer.zero_grad()?;
                loss.backward()?;
                self.optimizer.step()?;

                counter += 1;
                if counter == self.log_interval {
                    counter = 0;
                    print!(
                        "Train: Epoch {}/{}, Average Loss= {}",
                        epoch, self.epoch, total_loss
                    );
                    for metric in self.metrics.iter_mut() {
                        print!(
                            ", Average {}= {}",
                            metric.name()?,
                            metric.value()?.div(&MetricValue::Scalar(counter as f64))?
                        );
                        metric.clear()?;
                    }
                    println!();
                }
            }
            self.train_dataloader.reset()?;

            // Validate model
            self.validate()?;
            let metrics_len = self.metrics.len();
            for (i, metric) in self.metrics.iter_mut().enumerate() {
                print!("Validate: ");
                print!(
                    "Average {}= {}",
                    metric.name()?,
                    metric.value()?.div(&MetricValue::Scalar(counter as f64))?
                );
                if i != metrics_len - 1 {
                    print!(", ");
                }
                metric.clear()?;
            }

            // Save model
            self.model.save(
                &self
                    .result_dir
                    .join(self.model_name)
                    .join(format!("_{}.safetensors", epoch))
                    .to_str()
                    .ok_or(LearnerError::InvalidPath(
                        "result_dir in EpochLearnerBuilder".to_string(),
                    ))?,
            )?;
        }
        Ok(())
    }

    fn validate(&mut self) -> Result<&[AnyMetric], LearnerError> {
        let mut counter: usize = 0;

        loop {
            let (inputs, targets) = match self.validate_dataloader.next()? {
                Some((inputs, targets)) => (inputs, targets),
                None => break,
            };
            let outputs = self.model.forward((inputs, true))?;

            // Aggregate metrics
            for metric in self.metrics.iter_mut() {
                let value = match metric {
                    AnyMetric::Evaluation(m) => m.evaluate(&outputs, &targets)?,
                    AnyMetric::Resource(m) => m.sample()?,
                };
                metric.update(metric.value()?.add(&value)?)?;
            }

            counter += 1;
        }
        self.validate_dataloader.reset()?;

        // Compute average metrics
        for metric in self.metrics.iter_mut() {
            metric.update(metric.value()?.div(&MetricValue::Scalar(counter as f64))?)?;
        }

        Ok(&self.metrics)
    }

    fn test(&mut self) -> Result<&[AnyMetric], LearnerError> {
        let mut counter: usize = 0;

        loop {
            let (inputs, targets) = match self.test_dataloader.next()? {
                Some((inputs, targets)) => (inputs, targets),
                None => break,
            };
            let outputs = self.model.forward((inputs, true))?;

            // Aggregate metrics
            for metric in self.metrics.iter_mut() {
                let value = match metric {
                    AnyMetric::Evaluation(m) => m.evaluate(&outputs, &targets)?,
                    AnyMetric::Resource(m) => m.sample()?,
                };
                metric.update(metric.value()?.add(&value)?)?;
            }

            counter += 1;
        }
        self.test_dataloader.reset()?;

        // Compute average metrics
        for metric in self.metrics.iter_mut() {
            metric.update(metric.value()?.div(&MetricValue::Scalar(counter as f64))?)?;
        }

        Ok(&self.metrics)
    }
}

pub struct EpochLearnerBuilder {
    train_dataloader: Option<Box<dyn Dataloader<Output = (Tensor, Tensor)>>>,
    validate_dataloader: Option<Box<dyn Dataloader<Output = (Tensor, Tensor)>>>,
    test_dataloader: Option<Box<dyn Dataloader<Output = (Tensor, Tensor)>>>,
    model: Option<Box<dyn Model<Input = (Tensor, bool), Output = Tensor>>>,
    lossfn: Option<Box<dyn LossFn<Input = (Tensor, Tensor), Output = Tensor>>>,
    optimizer: Option<Box<dyn Optimizer<StepOutput = ()>>>,
    epoch: usize,
    metrics: Vec<AnyMetric>,
    log_interval: usize,
    model_name: Option<&'static str>,
    result_dir: Option<String>,
}

impl Default for EpochLearnerBuilder {
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

impl EpochLearnerBuilder {
    pub fn train_dataloader(
        &mut self,
        dataloader: Box<dyn Dataloader<Output = (Tensor, Tensor)>>,
    ) -> &mut Self {
        self.train_dataloader = Some(dataloader);
        self
    }

    pub fn validate_dataloader(
        &mut self,
        dataloader: Box<dyn Dataloader<Output = (Tensor, Tensor)>>,
    ) -> &mut Self {
        self.validate_dataloader = Some(dataloader);
        self
    }

    pub fn test_dataloader(
        &mut self,
        dataloader: Box<dyn Dataloader<Output = (Tensor, Tensor)>>,
    ) -> &mut Self {
        self.test_dataloader = Some(dataloader);
        self
    }

    pub fn model<M>(&mut self, model: M) -> &mut Self
    where
        M: Model<Input = (Tensor, bool), Output = Tensor> + 'static,
    {
        self.model = Some(Box::new(model));
        self.model_name = Some(stringify!(M));
        self
    }

    pub fn lossfn(
        &mut self,
        lossfn: Box<dyn LossFn<Input = (Tensor, Tensor), Output = Tensor>>,
    ) -> &mut Self {
        self.lossfn = Some(lossfn);
        self
    }

    pub fn optimizer(&mut self, optimizer: Box<dyn Optimizer<StepOutput = ()>>) -> &mut Self {
        self.optimizer = Some(optimizer);
        self
    }

    pub fn epoch(&mut self, epoch: usize) -> &mut Self {
        self.epoch = epoch;
        self
    }

    pub fn metrics(&mut self, metrics: Vec<AnyMetric>) -> &mut Self {
        self.metrics = metrics;
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

    pub fn build(&mut self) -> Result<EpochLearner, LearnerError> {
        let train_dataloader =
            self.train_dataloader
                .take()
                .ok_or(LearnerError::MissingArgument(
                    "train_dataloader in EpochLearnerBuilder".to_string(),
                ))?;
        let validate_dataloader =
            self.validate_dataloader
                .take()
                .ok_or(LearnerError::MissingArgument(
                    "validate_dataloader in EpochLearnerBuilder".to_string(),
                ))?;
        let test_dataloader = self
            .test_dataloader
            .take()
            .ok_or(LearnerError::MissingArgument(
                "test_dataloader in EpochLearnerBuilder".to_string(),
            ))?;
        let result_dir = self.result_dir.take().ok_or(LearnerError::MissingArgument(
            "result_dir in EpochLearnerBuilder".to_string(),
        ))?;
        let model = self.model.take().ok_or(LearnerError::MissingArgument(
            "model in EpochLearnerBuilder".to_string(),
        ))?;
        let model_name = self.model_name.take().ok_or(LearnerError::MissingArgument(
            "model_name in EpochLearnerBuilder".to_string(),
        ))?;
        let lossfn = self.lossfn.take().ok_or(LearnerError::MissingArgument(
            "lossfn in EpochLearnerBuilder".to_string(),
        ))?;
        let optimizer = self.optimizer.take().ok_or(LearnerError::MissingArgument(
            "optimizer in EpochLearnerBuilder".to_string(),
        ))?;
        if self.epoch == 0 {
            return Err(LearnerError::InvalidArgument(
                "epoch in EpochLearnerBuilder must be greater than 0".to_string(),
            ));
        }
        if self.log_interval == 0 {
            return Err(LearnerError::InvalidArgument(
                "log_interval in EpochLearnerBuilder must be greater than 0".to_string(),
            ));
        }

        Ok(EpochLearner {
            train_dataloader,
            validate_dataloader,
            test_dataloader,
            model,
            lossfn,
            optimizer,
            epoch: self.epoch,
            metrics: std::mem::take(&mut self.metrics),
            log_interval: self.log_interval,
            model_name,
            result_dir: PathBuf::from(result_dir),
        })
    }
}
