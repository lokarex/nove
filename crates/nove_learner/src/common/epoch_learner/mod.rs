use nove_dataloader::Dataloader;
use nove_lossfn::LossFn;
use nove_metric::Metric;
use nove_model::Model;
use nove_optimizer::Optimizer;
use nove_tensor::Tensor;

pub struct EpochLearner {
    _dataloader: Box<dyn Dataloader<Output = Tensor>>,
    _model: Box<dyn Model<Input = (Tensor, bool), Output = Tensor>>,
    _lossfn: Box<dyn LossFn<Input = (Tensor, Tensor), Output = Tensor>>,
    _optimizer: Box<dyn Optimizer<StepOutput = ()>>,
    _epoch: usize,
    _metrics: Vec<Box<dyn Metric>>,
}

pub struct EpochLearnerBuilder {}
