use nove::dataloader::DataloaderError;
use nove::dataloader::common::{
    ImageClassificationDataloader, ImageClassificationDataloaderBuilder,
};
use nove::dataset::resource::{Mnist, MnistDataset};
use nove::lossfn::CrossEntropy;
use nove::model::layer::{CNNBuilder, CNNConvBlock, CNNLinearBlock};
use nove::optimizer::Sgd;
use nove::tensor::{DType, Device, Tensor};

pub fn model(device: Device) -> Result<nove::model::layer::CNN, nove::model::ModelError> {
    CNNBuilder::default()
        .conv_block(CNNConvBlock::new(3, 32).use_pool(true))
        .conv_block(CNNConvBlock::new(32, 64).use_pool(true))
        .linear_block(CNNLinearBlock::new(3136, 128).use_relu(true))
        .linear_block(CNNLinearBlock::new(128, 10))
        .device(device)
        .dtype(DType::F32)
        .grad_enabled(true)
        .build()
}

pub fn dataloader(
    batch_size: usize,
    shuffle_seed: Option<usize>,
    device: Device,
) -> Result<
    (
        ImageClassificationDataloader<MnistDataset>,
        ImageClassificationDataloader<MnistDataset>,
        ImageClassificationDataloader<MnistDataset>,
    ),
    DataloaderError,
> {
    let mnist = Mnist::new("data")?;

    let train_dataloader = ImageClassificationDataloaderBuilder::default()
        .dataset(mnist.train()?)
        .batch_size(batch_size)
        .resize(28, 28)
        .device(device.clone())
        .shuffle_seed(shuffle_seed)
        .build()?;

    let validate_dataloader = ImageClassificationDataloaderBuilder::default()
        .dataset(mnist.test()?)
        .batch_size(batch_size)
        .resize(28, 28)
        .device(device.clone())
        .shuffle_seed(None)
        .build()?;

    let test_dataloader = ImageClassificationDataloaderBuilder::default()
        .dataset(mnist.test()?)
        .batch_size(batch_size)
        .resize(28, 28)
        .device(device.clone())
        .shuffle_seed(None)
        .build()?;

    Ok((train_dataloader, validate_dataloader, test_dataloader))
}

pub fn lossfn() -> CrossEntropy {
    CrossEntropy::new()
}

pub fn optimizer(params: Vec<Tensor>, learning_rate: f64) -> Sgd {
    Sgd::new(params, learning_rate)
}
