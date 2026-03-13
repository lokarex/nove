use nove::dataloader::DataloaderError;
use nove::dataloader::common::{
    ImageClassificationDataloader, ImageClassificationDataloaderBuilder, PrefetchDataloader,
    PrefetchDataloaderBuilder,
};
use nove::dataset::resource::{Mnist, MnistDataset};
use nove::lossfn::CrossEntropy;
use nove::model::layer::{CNNBuilder, CNNConvBlock, CNNLinearBlock};
use nove::optimizer::Sgd;
use nove::tensor::{DType, Device, Tensor};

pub fn model(device: Device) -> Result<nove::model::layer::CNN, nove::model::ModelError> {
    let cnn = CNNBuilder::default()
        .conv_block(CNNConvBlock::new(3, 32).use_pool(true))
        .conv_block(CNNConvBlock::new(32, 64).use_pool(true))
        .linear_block(CNNLinearBlock::new(3136, 128).use_relu(true))
        .linear_block(CNNLinearBlock::new(128, 10))
        .device(device)
        .dtype(DType::F32)
        .grad_enabled(true)
        .build()?;
    println!("{}", cnn);
    Ok(cnn)
}

type MnistDataloader = (
    PrefetchDataloader<ImageClassificationDataloader<MnistDataset>>,
    PrefetchDataloader<ImageClassificationDataloader<MnistDataset>>,
    PrefetchDataloader<ImageClassificationDataloader<MnistDataset>>,
);

pub fn dataloader(
    batch_size: usize,
    shuffle_seed: Option<usize>,
    buffer_size: usize,
    device: Device,
) -> Result<MnistDataloader, DataloaderError> {
    let mnist = Mnist::new("data")?;

    let train_dataloader = ImageClassificationDataloaderBuilder::default()
        .dataset(mnist.train()?)
        .batch_size(batch_size)
        .resize(28, 28)
        .device(device.clone())
        .shuffle_seed(shuffle_seed)
        .build()?;
    let train_dataloader = PrefetchDataloaderBuilder::default()
        .dataloader(train_dataloader)
        .buffer_size(buffer_size)
        .build()?;

    let validate_dataloader = ImageClassificationDataloaderBuilder::default()
        .dataset(mnist.test()?)
        .batch_size(batch_size)
        .resize(28, 28)
        .device(device.clone())
        .shuffle_seed(None)
        .build()?;
    let validate_dataloader = PrefetchDataloaderBuilder::default()
        .dataloader(validate_dataloader)
        .buffer_size(buffer_size)
        .build()?;

    let test_dataloader = ImageClassificationDataloaderBuilder::default()
        .dataset(mnist.test()?)
        .batch_size(batch_size)
        .resize(28, 28)
        .device(device.clone())
        .shuffle_seed(None)
        .build()?;
    let test_dataloader = PrefetchDataloaderBuilder::default()
        .dataloader(test_dataloader)
        .buffer_size(buffer_size)
        .build()?;

    Ok((train_dataloader, validate_dataloader, test_dataloader))
}

pub fn lossfn() -> CrossEntropy {
    CrossEntropy::new()
}

pub fn optimizer(params: Vec<Tensor>, learning_rate: f64) -> Sgd {
    Sgd::new(params, learning_rate)
}
