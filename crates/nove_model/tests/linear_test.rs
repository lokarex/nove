use nove::model::Model;
use nove::model::nn::LinearBuilder;
use nove::tensor::{DType, Device, Shape, Tensor};

#[test]
fn test_linear_builder_creation() {
    let linear = LinearBuilder::new(10, 20).build().unwrap();

    assert_eq!(
        linear.weight().shape().unwrap(),
        Shape::from_dims(&[10, 20])
    );
    assert!(linear.bias().is_some());
    if let Some(bias) = linear.bias() {
        assert_eq!(bias.shape().unwrap(), Shape::from_dims(&[20]));
    }
}

#[test]
fn test_linear_builder_without_bias() {
    let linear = LinearBuilder::new(5, 8)
        .bias_enabled(false)
        .build()
        .unwrap();

    assert_eq!(linear.weight().shape().unwrap(), Shape::from_dims(&[5, 8]));
    assert!(linear.bias().is_none());
}

#[test]
fn test_linear_forward() {
    let mut linear = LinearBuilder::new(3, 4).bias_enabled(true).build().unwrap();

    let input = Tensor::ones(
        &Shape::from_dims(&[2, 3]),
        &DType::F32,
        &Device::cpu(),
        false,
    )
    .unwrap();
    let output = linear.forward(input).unwrap();

    assert_eq!(output.shape().unwrap(), Shape::from_dims(&[2, 4]));
}

#[test]
fn test_linear_parameters() {
    let linear = LinearBuilder::new(6, 8).bias_enabled(true).build().unwrap();

    let params = linear.parameters().unwrap();
    assert_eq!(params.len(), 2);

    assert_eq!(params[0].shape().unwrap(), Shape::from_dims(&[6, 8]));
    assert_eq!(params[1].shape().unwrap(), Shape::from_dims(&[8]));
}

#[test]
fn test_linear_named_parameters() {
    let linear = LinearBuilder::new(4, 6).bias_enabled(true).build().unwrap();

    let named_params = linear.named_parameters().unwrap();
    assert_eq!(named_params.len(), 2);

    let param_names: Vec<String> = named_params.keys().cloned().collect();
    assert!(param_names.iter().any(|name| name.contains("weight")));
    assert!(param_names.iter().any(|name| name.contains("bias")));
}

#[test]
fn test_linear_require_grad() {
    let mut linear = LinearBuilder::new(5, 7)
        .grad_enabled(false)
        .build()
        .unwrap();

    let weight = linear.weight();
    assert!(!weight.grad_enabled().unwrap());

    linear.require_grad(true).unwrap();

    let weight = linear.weight();
    assert!(weight.grad_enabled().unwrap());

    if let Some(bias) = linear.bias() {
        assert!(bias.grad_enabled().unwrap());
    }
}

#[test]
fn test_linear_to_dtype() {
    let mut linear = LinearBuilder::new(4, 6).build().unwrap();

    let weight_dtype = linear.weight().dtype().unwrap();
    assert_eq!(weight_dtype, DType::F32);

    linear.to_dtype(&DType::F64).unwrap();

    let weight_dtype = linear.weight().dtype().unwrap();
    assert_eq!(weight_dtype, DType::F64);
}

#[test]
fn test_linear_builder_method_chaining() {
    let mut builder = LinearBuilder::new(10, 20);
    builder
        .in_features(15)
        .out_features(25)
        .bias_enabled(false)
        .device(Device::cpu())
        .dtype(DType::F32)
        .grad_enabled(true);

    let linear = builder.build().unwrap();

    assert_eq!(
        linear.weight().shape().unwrap(),
        Shape::from_dims(&[15, 25])
    );
    assert!(linear.bias().is_none());
}

#[test]
fn test_linear_display() {
    let linear = LinearBuilder::new(8, 16)
        .bias_enabled(true)
        .build()
        .unwrap();

    let display_str = format!("{}", linear);
    assert!(display_str.contains("linear"));
    assert!(display_str.contains("in_features=8"));
    assert!(display_str.contains("out_features=16"));
    assert!(display_str.contains("bias_enabled=true"));
}

#[test]
fn test_linear_weight_and_bias_accessors() {
    let linear = LinearBuilder::new(7, 9).bias_enabled(true).build().unwrap();

    let weight = linear.weight();
    assert_eq!(weight.shape().unwrap(), Shape::from_dims(&[7, 9]));

    let bias = linear.bias().unwrap();
    assert_eq!(bias.shape().unwrap(), Shape::from_dims(&[9]));
}
