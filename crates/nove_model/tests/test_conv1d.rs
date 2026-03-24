use nove_model::{Model, layer::Conv1dBuilder};
use nove_tensor::{DType, Device, Shape, Tensor};

#[test]
fn test_conv1d_forward() {
    let device = Device::cpu();

    // Create input tensor: [batch_size=2, in_channels=3, length=10]
    let input = Tensor::randn(0.0, 1.0, &Shape::from(&[2, 3, 10]), &device, false)
        .unwrap()
        .to_dtype(&DType::F32)
        .unwrap();

    // Build conv1d layer: in_channels=3, out_channels=5, kernel_size=3
    let conv = Conv1dBuilder::new(3, 5, 3)
        .stride(1)
        .padding(0)
        .device(device.clone())
        .build()
        .unwrap();

    let mut conv_layer = conv;
    let output = conv_layer.forward(input).unwrap();

    // Output shape: [2, 5, out_length]
    // out_length = (10 - 3) / 1 + 1 = 8
    assert_eq!(output.shape().unwrap().dims(), &[2, 5, 8]);
}

#[test]
fn test_conv1d_with_padding() {
    let device = Device::cpu();

    let input = Tensor::randn(0.0, 1.0, &Shape::from(&[1, 1, 7]), &device, false)
        .unwrap()
        .to_dtype(&DType::F32)
        .unwrap();

    let conv = Conv1dBuilder::new(1, 2, 3)
        .stride(1)
        .padding(1)
        .device(device.clone())
        .build()
        .unwrap();

    let mut conv_layer = conv;
    let output = conv_layer.forward(input).unwrap();

    // With padding=1, output length = 7 (same as input)
    assert_eq!(output.shape().unwrap().dims(), &[1, 2, 7]);
}

#[test]
fn test_conv1d_with_stride() {
    let device = Device::cpu();

    let input = Tensor::randn(0.0, 1.0, &Shape::from(&[2, 4, 12]), &device, false)
        .unwrap()
        .to_dtype(&DType::F32)
        .unwrap();

    let conv = Conv1dBuilder::new(4, 8, 3)
        .stride(2)
        .padding(0)
        .device(device.clone())
        .build()
        .unwrap();

    let mut conv_layer = conv;
    let output = conv_layer.forward(input).unwrap();

    // Output length = (12 - 3) / 2 + 1 = 5
    assert_eq!(output.shape().unwrap().dims(), &[2, 8, 5]);
}

#[test]
fn test_conv1d_weight_and_bias() {
    let device = Device::cpu();

    let conv = Conv1dBuilder::new(2, 3, 5)
        .bias_enabled(true)
        .device(device.clone())
        .build()
        .unwrap();

    let weight = conv.weight();
    let bias = conv.bias();

    // Weight shape: [out_channels, in_channels/groups, kernel_size] = [3, 2, 5]
    assert_eq!(weight.shape().unwrap().dims(), &[3, 2, 5]);

    // Bias shape: [out_channels, 1] = [3, 1]
    assert!(bias.is_some());
    let bias_tensor = bias.unwrap();
    assert_eq!(bias_tensor.shape().unwrap().dims(), &[3, 1]);
}
