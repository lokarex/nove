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
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(any(feature = "candle-cuda", feature = "candle-metal")))]
    let device =
        nove::device::candle::cpu().expect("candle-cpu feature is required for the default device");
    #[cfg(feature = "candle-cuda")]
    let device = nove::device::candle::cuda_if_available(0)
        .expect("candle-cuda feature is required for Candle CUDA device");
    #[cfg(feature = "candle-metal")]
    let device = nove::device::candle::metal_if_available(0)
        .expect("candle-metal feature is required for Candle Metal device");
    println!("Using device: {:?}", device);

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
