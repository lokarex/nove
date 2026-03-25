use nove::model::Model;
use nove::model::layer::Conv2dBuilder;
use nove::tensor::{DType, Device, Shape, Tensor};

#[test]
fn test_conv2d_builder_creation() {
    let conv = Conv2dBuilder::new(3, 16, (3, 3)).build().unwrap();

    assert_eq!(conv.in_channels(), 3);
    assert_eq!(conv.out_channels(), 16);
    assert_eq!(conv.kernel_size(), (3, 3));
    assert_eq!(conv.stride(), 1);
    assert_eq!(conv.padding(), 0);
    assert_eq!(conv.dilation(), 1);
    assert_eq!(conv.groups(), 1);
    assert!(conv.bias().is_some());

    let weight_shape = conv.weight().shape().unwrap();
    assert_eq!(weight_shape, Shape::from_dims(&[16, 3, 3, 3]));
}

#[test]
fn test_conv2d_builder_without_bias() {
    let conv = Conv2dBuilder::new(4, 8, (3, 3))
        .bias_enabled(false)
        .build()
        .unwrap();

    assert_eq!(conv.in_channels(), 4);
    assert_eq!(conv.out_channels(), 8);
    assert_eq!(conv.kernel_size(), (3, 3));
    assert!(conv.bias().is_none());

    let weight_shape = conv.weight().shape().unwrap();
    assert_eq!(weight_shape, Shape::from_dims(&[8, 4, 3, 3]));
}

#[test]
fn test_conv2d_forward_basic() {
    let mut conv = Conv2dBuilder::new(2, 4, (3, 3))
        .bias_enabled(true)
        .build()
        .unwrap();

    let input = Tensor::ones(
        &Shape::from_dims(&[1, 2, 10, 10]),
        &DType::F32,
        &Device::cpu(),
        false,
    )
    .unwrap();
    let output = conv.forward(input).unwrap();

    assert_eq!(output.shape().unwrap(), Shape::from_dims(&[1, 4, 8, 8]));
}

#[test]
fn test_conv2d_forward_with_padding() {
    let mut conv = Conv2dBuilder::new(3, 6, (3, 3)).padding(1).build().unwrap();

    let input = Tensor::ones(
        &Shape::from_dims(&[2, 3, 8, 8]),
        &DType::F32,
        &Device::cpu(),
        false,
    )
    .unwrap();
    let output = conv.forward(input).unwrap();

    assert_eq!(output.shape().unwrap(), Shape::from_dims(&[2, 6, 8, 8]));
}

#[test]
fn test_conv2d_forward_with_stride() {
    let mut conv = Conv2dBuilder::new(2, 4, (3, 3)).stride(2).build().unwrap();

    let input = Tensor::ones(
        &Shape::from_dims(&[1, 2, 10, 10]),
        &DType::F32,
        &Device::cpu(),
        false,
    )
    .unwrap();
    let output = conv.forward(input).unwrap();

    assert_eq!(output.shape().unwrap(), Shape::from_dims(&[1, 4, 4, 4]));
}

#[test]
fn test_conv2d_forward_with_dilation() {
    let mut conv = Conv2dBuilder::new(2, 4, (3, 3))
        .dilation(2)
        .build()
        .unwrap();

    let input = Tensor::ones(
        &Shape::from_dims(&[1, 2, 10, 10]),
        &DType::F32,
        &Device::cpu(),
        false,
    )
    .unwrap();
    let output = conv.forward(input).unwrap();

    assert_eq!(output.shape().unwrap(), Shape::from_dims(&[1, 4, 6, 6]));
}

#[test]
fn test_conv2d_forward_with_padding_stride_dilation() {
    let mut conv = Conv2dBuilder::new(3, 6, (3, 3))
        .padding(2)
        .stride(2)
        .dilation(2)
        .build()
        .unwrap();

    let input = Tensor::ones(
        &Shape::from_dims(&[2, 3, 14, 14]),
        &DType::F32,
        &Device::cpu(),
        false,
    )
    .unwrap();
    let output = conv.forward(input).unwrap();

    assert_eq!(output.shape().unwrap(), Shape::from_dims(&[2, 6, 7, 7]));
}

#[test]
fn test_conv2d_forward_with_asymmetric_kernel() {
    let mut conv = Conv2dBuilder::new(2, 4, (3, 5)).build().unwrap();

    let input = Tensor::ones(
        &Shape::from_dims(&[1, 2, 10, 12]),
        &DType::F32,
        &Device::cpu(),
        false,
    )
    .unwrap();
    let output = conv.forward(input).unwrap();

    assert_eq!(output.shape().unwrap(), Shape::from_dims(&[1, 4, 8, 8]));
}

#[test]
fn test_conv2d_forward_with_asymmetric_padding() {
    let conv = Conv2dBuilder::new(3, 6, (3, 3)).padding(2).build().unwrap();

    let input = Tensor::ones(
        &Shape::from_dims(&[2, 3, 8, 8]),
        &DType::F32,
        &Device::cpu(),
        false,
    )
    .unwrap();
    let mut conv_mut = conv;
    let output = conv_mut.forward(input).unwrap();

    assert_eq!(output.shape().unwrap(), Shape::from_dims(&[2, 6, 10, 10]));
}

#[test]
fn test_conv2d_parameters() {
    let conv = Conv2dBuilder::new(3, 6, (3, 3))
        .bias_enabled(true)
        .build()
        .unwrap();

    let params = conv.parameters().unwrap();
    assert_eq!(params.len(), 2);

    assert_eq!(params[0].shape().unwrap(), Shape::from_dims(&[6, 3, 3, 3]));
    assert_eq!(params[1].shape().unwrap(), Shape::from_dims(&[6, 1, 1]));
}

#[test]
fn test_conv2d_named_parameters() {
    let conv = Conv2dBuilder::new(2, 4, (3, 3))
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
fn test_conv2d_require_grad() {
    let mut conv = Conv2dBuilder::new(3, 5, (3, 3))
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
fn test_conv2d_to_device() {
    let mut conv = Conv2dBuilder::new(3, 5, (3, 3))
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
fn test_conv2d_to_dtype() {
    let mut conv = Conv2dBuilder::new(4, 6, (3, 3))
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
fn test_conv2d_builder_method_chaining() {
    let mut builder = Conv2dBuilder::new(3, 8, (5, 5));
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

    assert!(conv.bias().is_none());

    let weight_shape = conv.weight().shape().unwrap();
    assert_eq!(weight_shape, Shape::from_dims(&[8, 3, 5, 5]));
}

#[test]
fn test_conv2d_display() {
    let conv = Conv2dBuilder::new(8, 16, (3, 3))
        .bias_enabled(true)
        .build()
        .unwrap();

    let display_str = format!("{}", conv);
    assert!(display_str.contains("conv2d"));
    assert!(display_str.contains("in_channels=8"));
    assert!(display_str.contains("out_channels=16"));
    assert!(display_str.contains("kernel_size=(3, 3)"));
    assert!(display_str.contains("bias_enabled=true"));
}

#[test]
fn test_conv2d_weight_and_bias_accessors() {
    let conv = Conv2dBuilder::new(7, 9, (3, 3))
        .bias_enabled(true)
        .build()
        .unwrap();

    let weight = conv.weight();
    assert_eq!(weight.shape().unwrap(), Shape::from_dims(&[9, 7, 3, 3]));

    let bias = conv.bias().unwrap();
    assert_eq!(bias.shape().unwrap(), Shape::from_dims(&[9, 1, 1]));
}

#[test]
fn test_conv2d_groups() {
    let conv = Conv2dBuilder::new(8, 12, (3, 3)).groups(4).build().unwrap();

    let weight_shape = conv.weight().shape().unwrap();
    assert_eq!(weight_shape, Shape::from_dims(&[12, 2, 3, 3]));
}

#[test]
fn test_conv2d_input_shape_validation() {
    let conv = Conv2dBuilder::new(0, 10, (3, 3));
    let result = conv.build();
    assert!(result.is_err());

    let conv = Conv2dBuilder::new(10, 0, (3, 3));
    let result = conv.build();
    assert!(result.is_err());

    let conv = Conv2dBuilder::new(10, 10, (0, 3));
    let result = conv.build();
    assert!(result.is_err());

    let conv = Conv2dBuilder::new(10, 10, (3, 0));
    let result = conv.build();
    assert!(result.is_err());
}

#[test]
fn test_conv2d_parameter_initialization() {
    let conv = Conv2dBuilder::new(3, 16, (3, 3))
        .bias_enabled(true)
        .build()
        .unwrap();

    let weight = conv.weight();
    let bias = conv.bias().unwrap();

    assert!(weight.grad_enabled().unwrap());
    assert!(bias.grad_enabled().unwrap());

    let weight_values = weight.to_vec::<f32>().unwrap();
    let bias_values = bias.to_vec::<f32>().unwrap();

    assert!(!weight_values.is_empty());
    assert!(!bias_values.is_empty());

    let weight_mean = weight_values.iter().copied().sum::<f32>() / weight_values.len() as f32;
    let bias_mean = bias_values.iter().copied().sum::<f32>() / bias_values.len() as f32;

    assert!(weight_mean.abs() < 1.0);
    assert_eq!(bias_mean, 0.0);
}
