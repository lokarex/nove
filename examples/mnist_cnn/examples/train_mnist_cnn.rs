use mnist_cnn::{dataloader, lossfn, model, optimizer};
use nove::learner::Learner;
use nove::learner::common::ImageClassificationLearnerBuilder;
use nove::metric::{CpuFrequencyMetric, CpuUsageMetric};
use nove::model::Model;
use nove::tensor::Device;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::cuda(0)?;

    println!("Loading MNIST dataset and creating dataloaders...");
    let batch_size = 64;
    let (train_dataloader, validate_dataloader, _) =
        dataloader("data/mnist_cnn", batch_size, Some(42), 16, device.clone())?;

    println!("Creating CNN model...");
    let model = model(device.clone())?;

    println!("Creating loss function and optimizer...");
    let lossfn = lossfn();

    let learning_rate = 0.01;
    let params = model.parameters()?;
    let optimizer = optimizer(params, learning_rate);

    println!("Creating learner...");
    let mut learner = ImageClassificationLearnerBuilder::default()
        .train_dataloader(train_dataloader)
        .validate_dataloader(validate_dataloader)
        .model(model)
        .lossfn(lossfn)
        .optimizer(optimizer)
        .epoch(5)
        .log_interval(10)
        .result_dir("result/mnist_cnn")
        .metric(CpuUsageMetric::new(true))
        .metric(CpuFrequencyMetric::new(true))
        .build()?;

    println!("Starting training...");
    learner.train()?;

    println!("Training completed!");
    Ok(())
}
