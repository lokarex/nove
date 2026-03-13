use mnist_cnn::{dataloader, model};
use nove::lossfn::CrossEntropy;
use nove::optimizer::Sgd;
use nove::{
    learner::{
        Learner,
        common::{ImageClassificationLearner, ImageClassificationLearnerBuilder},
    },
    metric::Metric,
    model::Model,
    tensor::Device,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::cuda(0)?;

    println!("Loading MNIST dataset and creating dataloaders...");
    let batch_size = 64;
    let (_, _, test_dataloader) = dataloader(batch_size, Some(84), 16, device.clone())?;

    println!("Creating CNN model...");
    let mut model = model(device.clone())?;
    println!("Loading pre-trained model...");
    model.load("result/CNN_best.safetensors", &device)?;

    let mut learner: ImageClassificationLearner<_, _, CrossEntropy, Sgd> =
        ImageClassificationLearnerBuilder::default()
            .test_dataloader(test_dataloader)
            .model(model)
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
