use nove::dataloader::DataloaderError;
use nove::dataloader::common::{
    ImageClassificationDataloader, ImageClassificationDataloaderBuilder, PrefetchDataloader,
    PrefetchDataloaderBuilder,
};
use nove::dataset::resource::{Mnist, MnistDataset};
use nove::lossfn::CrossEntropy;
use nove::r#macro::Model;
use nove::model::layer::{Conv2dBlock, Conv2dBlockBuilder, LinearBlock, LinearBlockBuilder};
use nove::model::{Model, ModelError};
use nove::optimizer::{Sgd, SgdBuilder};
use nove::tensor::{Device, Shape, Tensor};

pub fn model(device: Device) -> Result<MnistCNN, ModelError> {
    let cnn = MnistCNN::new(device)?;
    println!("{}", cnn);
    Ok(cnn)
}

#[derive(Debug, Clone, Model)]
#[model(input = "(Tensor, bool)", output = "Tensor")]
pub struct MnistCNN {
    conv1: Conv2dBlock,
    conv2: Conv2dBlock,
    linear1: LinearBlock,
    linear2: LinearBlock,
}

impl MnistCNN {
    fn new(device: Device) -> Result<Self, ModelError> {
        let conv1 = Conv2dBlockBuilder::new(3, 32, (3, 3), 1, 1)
            .with_relu()
            .with_max_pool((2, 2), (2, 2))
            .device(device.clone())
            .build()?;
        let conv2 = Conv2dBlockBuilder::new(32, 64, (3, 3), 1, 1)
            .with_relu()
            .with_max_pool((2, 2), (2, 2))
            .device(device.clone())
            .build()?;
        let linear1 = LinearBlockBuilder::new(3136, 128)
            .with_relu()
            .device(device.clone())
            .build()?;
        let linear2 = LinearBlockBuilder::new(128, 10)
            .device(device.clone())
            .build()?;
        Ok(Self {
            conv1,
            conv2,
            linear1,
            linear2,
        })
    }

    fn forward(&mut self, input: (Tensor, bool)) -> Result<Tensor, ModelError> {
        let (x, training) = input;
        let mut x = self.conv1.forward((x, training))?;
        x = self.conv2.forward((x, training))?;

        let shape = x.shape()?;
        let batch_size = shape.dims()[0];
        let flattened_size = shape.dims()[1] * shape.dims()[2] * shape.dims()[3];
        x = x.reshape(&Shape::from(&[batch_size, flattened_size]))?;

        x = self.linear1.forward((x, training))?;
        x = self.linear2.forward((x, training))?;
        Ok(x)
    }
}

type MnistDataloader = (
    PrefetchDataloader<ImageClassificationDataloader<MnistDataset>>,
    PrefetchDataloader<ImageClassificationDataloader<MnistDataset>>,
    PrefetchDataloader<ImageClassificationDataloader<MnistDataset>>,
);

pub fn dataloader(
    dataset_dir: &str,
    batch_size: usize,
    shuffle_seed: Option<usize>,
    buffer_size: usize,
    device: Device,
) -> Result<MnistDataloader, DataloaderError> {
    let mnist = Mnist::new(dataset_dir)?;

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

pub fn optimizer(
    params: Vec<Tensor>,
    learning_rate: f64,
) -> Result<Sgd, nove::optimizer::OptimizerError> {
    SgdBuilder::new(params, learning_rate).build()
}
