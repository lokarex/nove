use nove_model::{
    Model,
    layer::{AvgPool1d, MaxPool1d},
};
use nove_tensor::{DType, Device, Shape, Tensor};

#[test]
fn test_max_pool1d_forward() {
    let device = Device::cpu();

    // Create input tensor: [batch_size=1, channels=1, length=10]
    let input = Tensor::randn(0.0, 1.0, &Shape::from(&[1, 1, 10]), &device, false)
        .unwrap()
        .to_dtype(&DType::F32)
        .unwrap();

    // Create max pool layer with kernel_size=2, stride=2
    let mut max_pool = MaxPool1d::new(2, None).unwrap();
    let output = max_pool.forward(input.clone()).unwrap();

    // Check output shape: [1, 1, 5] (10 / 2)
    assert_eq!(output.shape().unwrap().dims(), &[1, 1, 5]);
}

#[test]
fn test_max_pool1d_with_stride() {
    let device = Device::cpu();

    let input = Tensor::randn(0.0, 1.0, &Shape::from(&[2, 3, 8]), &device, false)
        .unwrap()
        .to_dtype(&DType::F32)
        .unwrap();

    // kernel_size=3, stride=1
    let mut max_pool = MaxPool1d::new(3, Some(1)).unwrap();
    let output = max_pool.forward(input).unwrap();

    // Output length = (8 - 3) / 1 + 1 = 6
    assert_eq!(output.shape().unwrap().dims(), &[2, 3, 6]);
}

#[test]
fn test_avg_pool1d_forward() {
    let device = Device::cpu();

    let input = Tensor::randn(0.0, 1.0, &Shape::from(&[1, 2, 12]), &device, false)
        .unwrap()
        .to_dtype(&DType::F32)
        .unwrap();

    let mut avg_pool = AvgPool1d::new(4, None).unwrap();
    let output = avg_pool.forward(input).unwrap();

    // 12 / 4 = 3
    assert_eq!(output.shape().unwrap().dims(), &[1, 2, 3]);
}

#[test]
fn test_avg_pool1d_with_stride() {
    let device = Device::cpu();

    let input = Tensor::randn(0.0, 1.0, &Shape::from(&[2, 1, 10]), &device, false)
        .unwrap()
        .to_dtype(&DType::F32)
        .unwrap();

    // kernel_size=3, stride=2
    let mut avg_pool = AvgPool1d::new(3, Some(2)).unwrap();
    let output = avg_pool.forward(input).unwrap();

    // Output length = (10 - 3) / 2 + 1 = 4
    assert_eq!(output.shape().unwrap().dims(), &[2, 1, 4]);
}
