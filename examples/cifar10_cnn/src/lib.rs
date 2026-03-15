use nove::dataloader::DataloaderError;
use nove::dataloader::common::{
    ImageClassificationDataloader, ImageClassificationDataloaderBuilder, PrefetchDataloader,
    PrefetchDataloaderBuilder,
};
use nove::dataset::resource::{Cifar10, Cifar10Dataset};
use nove::lossfn::CrossEntropy;
use nove::model::layer::{CNNBuilder, CNNConvBlock, CNNLinearBlock};
use nove::optimizer::Sgd;
use nove::tensor::{DType, Device, Tensor};

pub fn model(device: Device) -> Result<nove::model::layer::CNN, nove::model::ModelError> {
    let cnn = CNNBuilder::default()
        .conv_block(
            CNNConvBlock::new(3, 32, (3, 3), 1, 1)
                .use_relu()
                .use_max_pool((2, 2), (2, 2)),
        )
        .conv_block(
            CNNConvBlock::new(32, 64, (3, 3), 1, 1)
                .use_relu()
                .use_max_pool((2, 2), (2, 2)),
        )
        .linear_block(CNNLinearBlock::new(4096, 256).use_relu())
        .linear_block(CNNLinearBlock::new(256, 10))
        .device(device)
        .dtype(DType::F32)
        .grad_enabled(true)
        .build()?;
    println!("{}", cnn);
    Ok(cnn)
}

type Cifar10Dataloader = (
    PrefetchDataloader<ImageClassificationDataloader<Cifar10Dataset>>,
    PrefetchDataloader<ImageClassificationDataloader<Cifar10Dataset>>,
    PrefetchDataloader<ImageClassificationDataloader<Cifar10Dataset>>,
);

pub fn dataloader(
    dataset_dir: &str,
    batch_size: usize,
    shuffle_seed: Option<usize>,
    buffer_size: usize,
    device: Device,
) -> Result<Cifar10Dataloader, DataloaderError> {
    let cifar10 = Cifar10::new(dataset_dir)?;

    let normalization_mean = [0.4914, 0.4822, 0.4465];
    let normalization_std = [0.2471, 0.2435, 0.2616];

    let train_dataloader = ImageClassificationDataloaderBuilder::default()
        .dataset(cifar10.train()?)
        .batch_size(batch_size)
        .resize(32, 32)
        .device(device.clone())
        .shuffle_seed(shuffle_seed)
        .use_normalization(normalization_mean, normalization_std)
        .build()?;
    let train_dataloader = PrefetchDataloaderBuilder::default()
        .dataloader(train_dataloader)
        .buffer_size(buffer_size)
        .build()?;

    let validate_dataloader = ImageClassificationDataloaderBuilder::default()
        .dataset(cifar10.test()?)
        .batch_size(batch_size)
        .resize(32, 32)
        .device(device.clone())
        .shuffle_seed(None)
        .use_normalization(normalization_mean, normalization_std)
        .build()?;
    let validate_dataloader = PrefetchDataloaderBuilder::default()
        .dataloader(validate_dataloader)
        .buffer_size(buffer_size)
        .build()?;

    let test_dataloader = ImageClassificationDataloaderBuilder::default()
        .dataset(cifar10.test()?)
        .batch_size(batch_size)
        .resize(32, 32)
        .device(device.clone())
        .shuffle_seed(None)
        .use_normalization(normalization_mean, normalization_std)
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
