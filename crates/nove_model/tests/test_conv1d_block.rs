use nove_model::{
    Model,
    layer::{Activation, AvgPool1d, Conv1dBlockBuilder, MaxPool1d, Pool1d},
};
use nove_tensor::{DType, Device, Shape, Tensor};

#[test]
fn test_conv1d_block_basic() {
    let device = Device::cpu();

    // Create input tensor: [batch_size=2, in_channels=3, length=10]
    let input = Tensor::randn(0.0f32, 1.0, &Shape::from(&[2, 3, 10]), &device, false).unwrap();

    // Build conv1d block with ReLU activation
    let mut block = Conv1dBlockBuilder::new(3, 5, 3, 1, 0)
        .with_activation(Activation::relu())
        .device(device.clone())
        .build()
        .unwrap();

    let output = block.forward((input, false)).unwrap(); // inference mode

    // Output shape: [2, 5, out_length] where out_length = (10 - 3) / 1 + 1 = 8
    assert_eq!(output.shape().unwrap().dims(), &[2, 5, 8]);
}

#[test]
fn test_conv1d_block_with_batch_norm1d() {
    let device = Device::cpu();

    let input = Tensor::randn(0.0f32, 1.0, &Shape::from(&[1, 4, 12]), &device, false).unwrap();

    let mut block = Conv1dBlockBuilder::new(4, 8, 3, 1, 1)
        .with_batch_norm1d()
        .device(device.clone())
        .build()
        .unwrap();

    // Training mode
    let output_train = block.forward((input.clone(), true)).unwrap();
    // Inference mode
    let output_infer = block.forward((input, false)).unwrap();

    // Both should have same shape
    assert_eq!(output_train.shape().unwrap().dims(), &[1, 8, 12]); // padding=1 preserves length
    assert_eq!(output_infer.shape().unwrap().dims(), &[1, 8, 12]);
}

#[test]
fn test_conv1d_block_with_pooling() {
    let device = Device::cpu();

    let input = Tensor::randn(0.0f32, 1.0, &Shape::from(&[2, 2, 16]), &device, false).unwrap();

    // Block with max pooling
    let mut block = Conv1dBlockBuilder::new(2, 4, 3, 1, 0)
        .with_pool1d(Pool1d::MaxPool1d(MaxPool1d::new(2, Some(2)).unwrap()))
        .device(device.clone())
        .build()
        .unwrap();

    let output = block.forward((input, false)).unwrap();

    // Convolution output length = (16 - 3) / 1 + 1 = 14
    // Max pool output length = (14 - 2) / 2 + 1 = 7
    assert_eq!(output.shape().unwrap().dims(), &[2, 4, 7]);
}

#[test]
fn test_conv1d_block_with_avg_pooling() {
    let device = Device::cpu();

    let input = Tensor::randn(0.0, 1.0, &Shape::from(&[1, 1, 20]), &device, false)
        .unwrap()
        .to_dtype(&DType::F32)
        .unwrap();

    let mut block = Conv1dBlockBuilder::new(1, 2, 5, 1, 0)
        .with_pool1d(Pool1d::AvgPool1d(AvgPool1d::new(4, Some(4)).unwrap()))
        .device(device.clone())
        .build()
        .unwrap();

    let output = block.forward((input, false)).unwrap();

    // Convolution output length = (20 - 5) / 1 + 1 = 16
    // Avg pool output length = (16 - 4) / 4 + 1 = 4
    assert_eq!(output.shape().unwrap().dims(), &[1, 2, 4]);
}

#[test]
fn test_conv1d_block_with_activation_and_pooling() {
    let device = Device::cpu();

    let input = Tensor::randn(0.0f32, 1.0, &Shape::from(&[2, 3, 14]), &device, false).unwrap();

    let mut block = Conv1dBlockBuilder::new(3, 6, 3, 2, 1)
        .with_activation(Activation::gelu())
        .with_batch_norm1d()
        .with_pool1d(Pool1d::AvgPool1d(AvgPool1d::new(2, Some(2)).unwrap()))
        .device(device.clone())
        .build()
        .unwrap();

    let output = block.forward((input, false)).unwrap();

    // Convolution with padding=1, stride=2: output length = (14 + 2*1 - 3) / 2 + 1 = 7
    // Avg pool kernel=2, stride=2: output length = (7 - 2) / 2 + 1 = 3
    assert_eq!(output.shape().unwrap().dims(), &[2, 6, 3]);
}
