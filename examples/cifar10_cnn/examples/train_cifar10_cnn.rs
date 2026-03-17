use cifar10_cnn::{dataloader, lossfn, model, optimizer};
use nove::learner::Learner;
use nove::learner::common::ImageClassificationLearnerBuilder;
use nove::metric::{CpuFrequencyMetric, CpuUsageMetric};
use nove::model::Model;
use nove::tensor::Device;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::cuda(0)?;

    println!("Loading CIFAR-10 dataset and creating dataloaders...");
    let batch_size = 128;
    let (train_dataloader, validate_dataloader, _) =
        dataloader("data/cifar10_cnn", batch_size, Some(42), 16, device.clone())?;

    println!("Creating CNN model...");
    let model = model(device.clone())?;

    println!("Creating loss function and optimizer...");
    let lossfn = lossfn();

    let learning_rate = 0.01;
    let params = model.parameters()?;
    let optimizer = optimizer(params, learning_rate)?;

    println!("Creating learner...");
    let epoch = 50;
    let mut learner = ImageClassificationLearnerBuilder::default()
        .train_dataloader(train_dataloader)
        .validate_dataloader(validate_dataloader)
        .model(model)
        .lossfn(lossfn)
        .optimizer(optimizer)
        .epoch(epoch)
        .log_interval(10)
        .result_dir("result/cifar10_cnn")
        .metric(CpuUsageMetric::new(true))
        .metric(CpuFrequencyMetric::new(true))
        .build()?;

    println!("Starting training...");
    learner.train()?;

    println!("Training completed!");
    Ok(())
}
