use cifar10_cnn::{dataloader, model};
use nove::lossfn::CrossEntropyLoss;
use nove::optimizer::Adam;
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

    println!("Loading CIFAR-10 dataset and creating dataloaders...");
    let batch_size = 64;
    let (_, _, test_dataloader) =
        dataloader("data/cifar10_cnn", batch_size, Some(84), 16, device.clone())?;

    println!("Creating CNN model...");
    let mut model = model(device.clone())?;
    println!("Loading pre-trained model...");
    model.load("result/cifar10_cnn/Cifar10CNN_best.safetensors", &device)?;

    let mut learner: ImageClassificationLearner<_, _, CrossEntropyLoss, Adam> =
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
