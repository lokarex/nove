use nove::dataloader::DataloaderError;
use nove::dataloader::common::{
    ImageClassificationDataloader, ImageClassificationDataloaderBuilder, PrefetchDataloader,
    PrefetchDataloaderBuilder,
};
use nove::dataset::resource::{Cifar10, Cifar10Dataset};
use nove::lossfn::CrossEntropyLoss;
use nove::r#macro::Model;
use nove::model::nn::{
    Activation, Conv2dBlock, Conv2dBlockBuilder, LinearBlock, LinearBlockBuilder, MaxPool2d, Pool2d,
};
use nove::model::{Model, ModelError};
use nove::optimizer::{OptimizerError, Sgd, SgdBuilder};
use nove::tensor::{Device, Tensor};

pub fn model(device: Device) -> Result<Cifar10CNN, ModelError> {
    let cnn = Cifar10CNN::new(device)?;
    println!("{}", cnn);
    Ok(cnn)
}

#[derive(Debug, Clone, Model)]
#[model(input = "Tensor", output = "Tensor")]
pub struct Cifar10CNN {
    conv1: Conv2dBlock,
    conv2: Conv2dBlock,
    linear1: LinearBlock,
    linear2: LinearBlock,
}

impl Cifar10CNN {
    fn new(device: Device) -> Result<Self, ModelError> {
        let conv1 = Conv2dBlockBuilder::new(3, 32, (3, 3), 1, 1)
            .with_batch_norm2d()
            .with_activation(Activation::relu())
            .with_pool2d(Pool2d::MaxPool2d(MaxPool2d::new((2, 2), Some((2, 2)))?))
            .device(device.clone())
            .build()?;
        let conv2 = Conv2dBlockBuilder::new(32, 64, (3, 3), 1, 1)
            .with_batch_norm2d()
            .with_activation(Activation::relu())
            .with_pool2d(Pool2d::MaxPool2d(MaxPool2d::new((2, 2), Some((2, 2)))?))
            .device(device.clone())
            .build()?;
        let linear1 = LinearBlockBuilder::new(4096, 256)
            .with_activation(Activation::relu())
            .device(device.clone())
            .build()?;
        let linear2 = LinearBlockBuilder::new(256, 10)
            .device(device.clone())
            .build()?;

        Ok(Self {
            conv1,
            conv2,
            linear1,
            linear2,
        })
    }

    fn forward(&mut self, input: Tensor) -> Result<Tensor, ModelError> {
        let mut x = self.conv1.forward(input)?;
        x = self.conv2.forward(x)?;

        x = x.flatten(Some(1), None)?;

        x = self.linear1.forward(x)?;
        x = self.linear2.forward(x)?;
        Ok(x)
    }
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

pub fn lossfn() -> CrossEntropyLoss {
    CrossEntropyLoss::new()
}

pub fn optimizer(params: Vec<Tensor>, learning_rate: f64) -> Result<Sgd, OptimizerError> {
    SgdBuilder::new(params, learning_rate).build()
}
