use nove_model::{Model, layer::LinearBuilder};
use nove_tensor::{Device, Shape, Tensor};

#[test]
fn test_linear() {
    // Create a linear layer and a dummy input tensor.
    let mut linear = LinearBuilder::default()
        .in_features(10)
        .out_features(20)
        .build()
        .unwrap();
    let input = Tensor::rand(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[7, 10]),
        &Device::cpu(),
        true,
    )
    .unwrap();

    // Test the forward pass.
    let output = linear.forward(input).unwrap();
    assert_eq!(output.shape().unwrap(), Shape::from_dims(&[7, 20]));

    // Test the backward pass.
    // Check if the gradient tensors are None before backward pass.
    assert!(linear.weight().grad().unwrap().is_none());
    assert!(linear.bias().unwrap().grad().unwrap().is_none());
    // Perform backward pass.
    output.backward().unwrap();
    // Check if the gradient tensors are Some after backward pass.
    let weight_grad = linear.weight().grad().unwrap().unwrap();
    assert_eq!(weight_grad.shape().unwrap(), Shape::from_dims(&[10, 20]));
    let bias_grad = linear.bias().unwrap().grad().unwrap().unwrap();
    assert_eq!(bias_grad.shape().unwrap(), Shape::from_dims(&[20]));
}
