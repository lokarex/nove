use cifar10_cnn::{dataloader, loss_fn, model, optimizer};
use nove::learner::Learner;
use nove::learner::common::ImageClassificationLearnerBuilder;
use nove::metric::{CpuFrequencyMetric, CpuUsageMetric};
use nove::model::Model;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = nove::tensor::Device::default();
    println!("Using device: {:?}", device);

    println!("Loading CIFAR-10 dataset and creating dataloaders...");
    let batch_size = 128;
    let (train_dataloader, validation_dataloader, _) =
        dataloader("data/cifar10_cnn", batch_size, Some(42), 16, device.clone())?;

    println!("Creating CNN model...");
    let model = model(device.clone())?;

    println!("Creating loss function and optimizer...");
    let loss_fn = loss_fn();

    let learning_rate = 0.01;
    let params = model.parameters()?;
    let optimizer = optimizer(params, learning_rate)?;

    println!("Creating learner...");
    let num_epochs = 50;
    let mut learner = ImageClassificationLearnerBuilder::default()
        .train_dataloader(train_dataloader)
        .validation_dataloader(validation_dataloader)
        .model(model)
        .loss_fn(loss_fn)
        .optimizer(optimizer)
        .num_epochs(num_epochs)
        .log_interval(10)
        .output_dir("output/cifar10_cnn")
        .metric(CpuUsageMetric::new(true))
        .metric(CpuFrequencyMetric::new(true))
        .build()?;

    println!("Starting training...");
    learner.train()?;

    println!("Training completed!");
    Ok(())
}
