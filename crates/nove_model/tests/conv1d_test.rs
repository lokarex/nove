use nove::model::Model;
use nove::model::nn::Conv1dBuilder;
use nove::tensor::{DType, Device, Shape, Tensor};

#[test]
fn test_conv1d_builder_creation() {
    let conv = Conv1dBuilder::new(3, 16, 5).build().unwrap();

    assert_eq!(conv.in_channels(), 3);
    assert_eq!(conv.out_channels(), 16);
    assert_eq!(conv.kernel_size(), 5);
    assert_eq!(conv.stride(), 1);
    assert_eq!(conv.padding(), 0);
    assert_eq!(conv.dilation(), 1);
    assert_eq!(conv.groups(), 1);
    assert!(conv.bias().is_some());

    let weight_shape = conv.weight().shape().unwrap();
    assert_eq!(weight_shape, Shape::from_dims(&[16, 3, 5]));
}

#[test]
fn test_conv1d_builder_without_bias() {
    let conv = Conv1dBuilder::new(4, 8, 3)
        .bias_enabled(false)
        .build()
        .unwrap();

    assert_eq!(conv.in_channels(), 4);
    assert_eq!(conv.out_channels(), 8);
    assert_eq!(conv.kernel_size(), 3);
    assert!(conv.bias().is_none());
}

#[test]
fn test_conv1d_forward_basic() {
    let mut conv = Conv1dBuilder::new(2, 4, 3)
        .bias_enabled(true)
        .build()
        .unwrap();

    let input = Tensor::ones(
        &Shape::from_dims(&[1, 2, 10]),
        &DType::F32,
        &Device::cpu(),
        false,
    )
    .unwrap();
    let output = conv.forward(input).unwrap();

    assert_eq!(output.shape().unwrap(), Shape::from_dims(&[1, 4, 8]));
}

#[test]
fn test_conv1d_forward_with_padding() {
    let mut conv = Conv1dBuilder::new(3, 6, 3).padding(1).build().unwrap();

    let input = Tensor::ones(
        &Shape::from_dims(&[2, 3, 8]),
        &DType::F32,
        &Device::cpu(),
        false,
    )
    .unwrap();
    let output = conv.forward(input).unwrap();

    assert_eq!(output.shape().unwrap(), Shape::from_dims(&[2, 6, 8]));
}

#[test]
fn test_conv1d_forward_with_stride() {
    let mut conv = Conv1dBuilder::new(2, 4, 3).stride(2).build().unwrap();

    let input = Tensor::ones(
        &Shape::from_dims(&[1, 2, 10]),
        &DType::F32,
        &Device::cpu(),
        false,
    )
    .unwrap();
    let output = conv.forward(input).unwrap();

    assert_eq!(output.shape().unwrap(), Shape::from_dims(&[1, 4, 4]));
}

#[test]
fn test_conv1d_forward_with_dilation() {
    let mut conv = Conv1dBuilder::new(2, 4, 3).dilation(2).build().unwrap();

    let input = Tensor::ones(
        &Shape::from_dims(&[1, 2, 10]),
        &DType::F32,
        &Device::cpu(),
        false,
    )
    .unwrap();
    let output = conv.forward(input).unwrap();

    assert_eq!(output.shape().unwrap(), Shape::from_dims(&[1, 4, 6]));
}

#[test]
fn test_conv1d_forward_with_padding_stride_dilation() {
    let mut conv = Conv1dBuilder::new(3, 6, 3)
        .padding(2)
        .stride(2)
        .dilation(2)
        .build()
        .unwrap();

    let input = Tensor::ones(
        &Shape::from_dims(&[2, 3, 14]),
        &DType::F32,
        &Device::cpu(),
        false,
    )
    .unwrap();
    let output = conv.forward(input).unwrap();

    assert_eq!(output.shape().unwrap(), Shape::from_dims(&[2, 6, 7]));
}

#[test]
fn test_conv1d_parameters() {
    let conv = Conv1dBuilder::new(3, 6, 3)
        .bias_enabled(true)
        .build()
        .unwrap();

    let params = conv.parameters().unwrap();
    assert_eq!(params.len(), 2);

    assert_eq!(params[0].shape().unwrap(), Shape::from_dims(&[6, 3, 3]));
    assert_eq!(params[1].shape().unwrap(), Shape::from_dims(&[6, 1]));
}

#[test]
fn test_conv1d_named_parameters() {
    let conv = Conv1dBuilder::new(2, 4, 3)
        .bias_enabled(true)
        .build()
        .unwrap();

    let named_params = conv.named_parameters().unwrap();
    assert_eq!(named_params.len(), 2);

    let param_names: Vec<String> = named_params.keys().cloned().collect();
    assert!(param_names.iter().any(|name| name.contains("weight")));
    assert!(param_names.iter().any(|name| name.contains("bias")));
}

#[test]
fn test_conv1d_require_grad() {
    let mut conv = Conv1dBuilder::new(3, 5, 3)
        .grad_enabled(false)
        .build()
        .unwrap();

    let weight = conv.weight();
    assert!(!weight.grad_enabled().unwrap());

    conv.require_grad(true).unwrap();

    let weight = conv.weight();
    assert!(weight.grad_enabled().unwrap());

    if let Some(bias) = conv.bias() {
        assert!(bias.grad_enabled().unwrap());
    }
}

#[test]
fn test_conv1d_to_device() {
    let mut conv = Conv1dBuilder::new(3, 5, 3)
        .device(Device::cpu())
        .build()
        .unwrap();

    let weight_device = conv.weight().device().unwrap();
    assert!(weight_device.is_cpu());

    conv.to_device(&Device::cpu()).unwrap();

    let weight_device = conv.weight().device().unwrap();
    assert!(weight_device.is_cpu());
}

#[test]
fn test_conv1d_to_dtype() {
    let mut conv = Conv1dBuilder::new(4, 6, 3)
        .dtype(DType::F32)
        .build()
        .unwrap();

    let weight_dtype = conv.weight().dtype().unwrap();
    assert_eq!(weight_dtype, DType::F32);

    conv.to_dtype(&DType::F32).unwrap();

    let weight_dtype = conv.weight().dtype().unwrap();
    assert_eq!(weight_dtype, DType::F32);
}

#[test]
fn test_conv1d_builder_method_chaining() {
    let mut builder = Conv1dBuilder::new(3, 8, 5);
    builder
        .padding(2)
        .stride(2)
        .dilation(2)
        .groups(1)
        .bias_enabled(false)
        .device(Device::cpu())
        .dtype(DType::F32)
        .grad_enabled(true);

    let conv = builder.build().unwrap();

    assert_eq!(conv.padding(), 2);
    assert_eq!(conv.stride(), 2);
    assert_eq!(conv.dilation(), 2);
    assert_eq!(conv.groups(), 1);
    assert!(conv.bias().is_none());
}

#[test]
fn test_conv1d_display() {
    let conv = Conv1dBuilder::new(8, 16, 3)
        .bias_enabled(true)
        .build()
        .unwrap();

    let display_str = format!("{}", conv);
    assert!(display_str.contains("conv1d"));
    assert!(display_str.contains("in_channels=8"));
    assert!(display_str.contains("out_channels=16"));
    assert!(display_str.contains("kernel_size=3"));
    assert!(display_str.contains("bias_enabled=true"));
}

#[test]
fn test_conv1d_weight_and_bias_accessors() {
    let conv = Conv1dBuilder::new(7, 9, 3)
        .bias_enabled(true)
        .build()
        .unwrap();

    let weight = conv.weight();
    assert_eq!(weight.shape().unwrap(), Shape::from_dims(&[9, 7, 3]));

    let bias = conv.bias().unwrap();
    assert_eq!(bias.shape().unwrap(), Shape::from_dims(&[9, 1]));
}

#[test]
fn test_conv1d_groups() {
    let conv = Conv1dBuilder::new(8, 12, 3).groups(4).build().unwrap();

    assert_eq!(conv.groups(), 4);

    let weight_shape = conv.weight().shape().unwrap();
    assert_eq!(weight_shape, Shape::from_dims(&[12, 2, 3]));
}

#[test]
fn test_conv1d_input_shape_validation() {
    let conv = Conv1dBuilder::new(10, 10, 0);
    let result = conv.build();
    assert!(result.is_err());

    let mut builder = Conv1dBuilder::new(10, 10, 3);
    builder.stride(0);
    let result = builder.build();
    assert!(result.is_err());

    let mut builder = Conv1dBuilder::new(10, 10, 3);
    builder.dilation(0);
    let result = builder.build();
    assert!(result.is_err());

    let mut builder = Conv1dBuilder::new(10, 10, 3);
    builder.groups(0);
    let result = builder.build();
    assert!(result.is_err());
}
