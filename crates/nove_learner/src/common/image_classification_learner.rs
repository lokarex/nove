use std::path::PathBuf;

use nove_dataloader::Dataloader;
use nove_lossfn::LossFn;
use nove_metric::{
    AccuracyMetric, AnyMetric, EvaluationMetric, LossMetric, Metric, ResourceMetric,
};
use nove_model::Model;
use nove_optimizer::Optimizer;
use nove_tensor::Tensor;

use crate::{Learner, LearnerError};

/// Image classification learner, specialized for image classification tasks.
pub struct ImageClassificationLearner<D, M, L, O>
where
    D: Dataloader<Output = (Tensor, Tensor)>,
    M: Model<Input = (Tensor, bool), Output = Tensor>,
    L: LossFn<Input = (Tensor, Tensor), Output = Tensor> + Clone,
    O: Optimizer<StepOutput = ()>,
{
    train_dataloader: D,
    validate_dataloader: D,
    test_dataloader: D,
    model: M,
    lossfn: L,
    optimizer: O,
    epoch: usize,
    metrics: Vec<AnyMetric>,
    log_interval: usize,
    model_name: String,
    result_dir: PathBuf,
}

impl<D, M, L, O> Learner for ImageClassificationLearner<D, M, L, O>
where
    D: Dataloader<Output = (Tensor, Tensor)>,
    M: Model<Input = (Tensor, bool), Output = Tensor>,
    L: LossFn<Input = (Tensor, Tensor), Output = Tensor> + Clone,
    O: Optimizer<StepOutput = ()>,
{
    fn train(&mut self) -> Result<(), LearnerError> {
        for epoch in 0..self.epoch {
            let mut batch_count: usize = 0;

            loop {
                let (inputs, targets) = match self.train_dataloader.next()? {
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

                let loss = self.lossfn.loss((outputs, targets))?;

                self.optimizer.zero_grad()?;
                loss.backward()?;
                drop(loss);

                self.optimizer.step()?;

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
            self.train_dataloader.reset()?;

            print!("Validate: ");
            let metrics_len = self.metrics.len();
            for (i, metric) in self.validate()?.iter_mut().enumerate() {
                print!("{}= {:.3}", metric.name()?, metric.value()?);
                if i != metrics_len - 1 {
                    print!(", ");
                }
                metric.clear()?;
            }
            println!();

            self.model.save(
                &self
                    .result_dir
                    .join(format!("{}_{}.safetensors", self.model_name, epoch + 1))
                    .to_str()
                    .ok_or(LearnerError::InvalidPath(
                        "result_dir in ImageClassificationLearnerBuilder".to_string(),
                    ))?,
            )?;
        }

        self.test()?;

        Ok(())
    }

    fn validate(&mut self) -> Result<Vec<AnyMetric>, LearnerError> {
        loop {
            let (inputs, targets) = match self.validate_dataloader.next()? {
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
        }
        self.validate_dataloader.reset()?;

        let metrics = self.metrics.clone();
        for metric in self.metrics.iter_mut() {
            metric.clear()?;
        }

        Ok(metrics)
    }

    fn test(&mut self) -> Result<Vec<AnyMetric>, LearnerError> {
        loop {
            let (inputs, targets) = match self.test_dataloader.next()? {
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
        }
        self.test_dataloader.reset()?;

        let metrics = self.metrics.clone();
        for metric in self.metrics.iter_mut() {
            metric.clear()?;
        }

        Ok(metrics)
    }
}

/// Builder for constructing [`ImageClassificationLearner`] instances.
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

    pub fn build(&mut self) -> Result<ImageClassificationLearner<D, M, L, O>, LearnerError> {
        let train_dataloader =
            self.train_dataloader
                .take()
                .ok_or(LearnerError::MissingArgument(
                    "train_dataloader in ImageClassificationLearnerBuilder".to_string(),
                ))?;
        let validate_dataloader =
            self.validate_dataloader
                .take()
                .ok_or(LearnerError::MissingArgument(
                    "validate_dataloader in ImageClassificationLearnerBuilder".to_string(),
                ))?;
        let test_dataloader = self
            .test_dataloader
            .take()
            .ok_or(LearnerError::MissingArgument(
                "test_dataloader in ImageClassificationLearnerBuilder".to_string(),
            ))?;
        let result_dir = self.result_dir.take().ok_or(LearnerError::MissingArgument(
            "result_dir in ImageClassificationLearnerBuilder".to_string(),
        ))?;
        let model = self.model.take().ok_or(LearnerError::MissingArgument(
            "model in ImageClassificationLearnerBuilder".to_string(),
        ))?;
        let model_name = self.model_name.take().ok_or(LearnerError::MissingArgument(
            "model_name in ImageClassificationLearnerBuilder".to_string(),
        ))?;
        let lossfn = self.lossfn.take().ok_or(LearnerError::MissingArgument(
            "lossfn in ImageClassificationLearnerBuilder".to_string(),
        ))?;
        let optimizer = self.optimizer.take().ok_or(LearnerError::MissingArgument(
            "optimizer in ImageClassificationLearnerBuilder".to_string(),
        ))?;
        if self.epoch == 0 {
            return Err(LearnerError::InvalidArgument(
                "epoch in ImageClassificationLearnerBuilder must be greater than 0".to_string(),
            ));
        }
        if self.log_interval == 0 {
            return Err(LearnerError::InvalidArgument(
                "log_interval in ImageClassificationLearnerBuilder must be greater than 0"
                    .to_string(),
            ));
        }

        self.metrics
            .insert(0, AnyMetric::AccuracyMetric(AccuracyMetric::new()));
        self.metrics.insert(
            1,
            AnyMetric::LossMetric(LossMetric::new(nove_lossfn::CrossEntropy::new())),
        );

        std::fs::create_dir_all(result_dir.clone()).map_err(|e| {
            LearnerError::InvalidPath(format!(
                "Failed to create result_dir {:?}: {}",
                self.result_dir, e
            ))
        })?;

        Ok(ImageClassificationLearner {
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
