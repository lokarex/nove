use nove_model::{Model, layer::Conv2dBuilder};
use nove_tensor::{Device, Shape, Tensor};

#[test]
fn test_conv2d() {
    // Create a conv2d layer and a dummy input tensor.
    let mut conv2d = Conv2dBuilder::default()
        .in_channels(3)
        .out_channels(6)
        .kernel_size((3, 3))
        .build()
        .unwrap();
    let input = Tensor::rand(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[7, 3, 10, 10]),
        &Device::cpu(),
        false,
    )
    .unwrap();

    // Test the forward pass.
    let output = conv2d.forward(input).unwrap();
    assert_eq!(output.shape().unwrap(), Shape::from_dims(&[7, 6, 8, 8]));
}
