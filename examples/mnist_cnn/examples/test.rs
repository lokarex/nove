use mnist_cnn::{dataloader, lossfn, model, optimizer};
use nove::{
    learner::{Learner, common::ImageClassificationLearnerBuilder},
    metric::Metric,
    model::Model,
    tensor::Device,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::cuda(0)?;

    println!("Loading MNIST dataset and creating dataloaders...");
    let batch_size = 128;
    let (train_dataloader, validate_dataloader, test_dataloader) =
        dataloader(batch_size, Some(42), device.clone())?;

    println!("Creating CNN model...");
    let mut model = model(device.clone())?;
    println!("Loading pre-trained model...");
    model.load("result/CNN_best.safetensors", &device)?;

    println!("Creating loss function and optimizer...");
    let lossfn = lossfn();

    let learning_rate = 0.01;
    let params = model.parameters()?;
    let optimizer = optimizer(params, learning_rate);

    let mut learner = ImageClassificationLearnerBuilder::default()
        .train_dataloader(train_dataloader)
        .validate_dataloader(validate_dataloader)
        .test_dataloader(test_dataloader)
        .model(model)
        .lossfn(lossfn)
        .optimizer(optimizer)
        .epoch(5)
        .log_interval(10)
        .result_dir("results")
        .build()?;

    println!("Start testing...");
    print!("Test: ");
    let test_metrics = learner.test()?;
    let test_metrics_len = test_metrics.len();
    for (i, metric) in test_metrics.iter().enumerate() {
        print!("{}= {:.3}", metric.name()?, metric.value()?);
        if i != test_metrics_len - 1 {
            print!(", ");
        }
    }
    println!();
    println!("Test completed.");

    Ok(())
}
