use nove::model::Model;
use nove::model::nn::{Activation, Conv1dBlockBuilder, Pool1d};
use nove::tensor::{DType, Device, Shape, Tensor};

#[test]
fn test_conv1d_block_builder_creation() {
    let block = Conv1dBlockBuilder::new(3, 16, 5, 1, 1)
        .device(Device::cpu())
        .dtype(DType::F32)
        .grad_enabled(true)
        .with_batch_norm1d()
        .with_activation(Activation::relu())
        .with_pool1d(Pool1d::max_pool1d(2, None).unwrap())
        .build()
        .unwrap();

    let params = block.parameters().unwrap();
    assert!(params.len() >= 2);
}

#[test]
fn test_conv1d_block_forward_sequential() {
    let mut block = Conv1dBlockBuilder::new(2, 4, 3, 1, 1)
        .with_batch_norm1d()
        .with_activation(Activation::relu())
        .with_pool1d(Pool1d::max_pool1d(2, None).unwrap())
        .build()
        .unwrap();

    let input = Tensor::ones(
        &Shape::from_dims(&[1, 2, 10]),
        &DType::F32,
        &Device::cpu(),
        false,
    )
    .unwrap();
    let output = block.forward(input).unwrap();

    assert_eq!(output.shape().unwrap(), Shape::from_dims(&[1, 4, 5]));
}

#[test]
fn test_conv1d_block_forward_without_components() {
    let mut block = Conv1dBlockBuilder::new(2, 4, 3, 1, 1).build().unwrap();

    let input = Tensor::ones(
        &Shape::from_dims(&[1, 2, 10]),
        &DType::F32,
        &Device::cpu(),
        false,
    )
    .unwrap();
    let output = block.forward(input).unwrap();

    assert_eq!(output.shape().unwrap(), Shape::from_dims(&[1, 4, 10]));
}

#[test]
fn test_conv1d_block_forward_with_batch_norm_only() {
    let mut block = Conv1dBlockBuilder::new(2, 4, 3, 1, 1)
        .with_batch_norm1d()
        .build()
        .unwrap();

    let input = Tensor::ones(
        &Shape::from_dims(&[1, 2, 10]),
        &DType::F32,
        &Device::cpu(),
        false,
    )
    .unwrap();
    let output = block.forward(input).unwrap();

    assert_eq!(output.shape().unwrap(), Shape::from_dims(&[1, 4, 10]));
}

#[test]
fn test_conv1d_block_forward_with_activation_only() {
    let mut block = Conv1dBlockBuilder::new(2, 4, 3, 1, 1)
        .with_activation(Activation::relu())
        .build()
        .unwrap();

    let input = Tensor::ones(
        &Shape::from_dims(&[1, 2, 10]),
        &DType::F32,
        &Device::cpu(),
        false,
    )
    .unwrap();
    let output = block.forward(input).unwrap();

    assert_eq!(output.shape().unwrap(), Shape::from_dims(&[1, 4, 10]));
}

#[test]
fn test_conv1d_block_forward_with_pool_only() {
    let mut block = Conv1dBlockBuilder::new(2, 4, 3, 1, 1)
        .with_pool1d(Pool1d::max_pool1d(2, None).unwrap())
        .build()
        .unwrap();

    let input = Tensor::ones(
        &Shape::from_dims(&[1, 2, 10]),
        &DType::F32,
        &Device::cpu(),
        false,
    )
    .unwrap();
    let output = block.forward(input).unwrap();

    assert_eq!(output.shape().unwrap(), Shape::from_dims(&[1, 4, 5]));
}

#[test]
fn test_conv1d_block_builder_method_chaining() {
    let mut builder = Conv1dBlockBuilder::new(3, 8, 5, 1, 1);
    builder
        .in_channels(4)
        .out_channels(16)
        .kernel_size(3)
        .stride(2)
        .padding(1);

    let block = builder
        .with_batch_norm1d()
        .with_activation(Activation::silu())
        .with_pool1d(Pool1d::avg_pool1d(2, None).unwrap())
        .build()
        .unwrap();

    let params = block.parameters().unwrap();
    assert!(!params.is_empty());
}

#[test]
fn test_conv1d_block_parameters() {
    let block = Conv1dBlockBuilder::new(3, 6, 3, 1, 1)
        .with_batch_norm1d()
        .build()
        .unwrap();

    let params = block.parameters().unwrap();
    assert!(!params.is_empty());

    let named_params = block.named_parameters().unwrap();
    assert!(!named_params.is_empty());
}

#[test]
fn test_conv1d_block_require_grad() {
    let mut block = Conv1dBlockBuilder::new(3, 5, 3, 1, 1)
        .grad_enabled(false)
        .build()
        .unwrap();

    let params_before = block.parameters().unwrap();
    for param in params_before {
        assert!(!param.grad_enabled().unwrap());
    }

    block.require_grad(true).unwrap();

    let params_after = block.parameters().unwrap();
    for param in params_after {
        assert!(param.grad_enabled().unwrap());
    }
}

#[test]
fn test_conv1d_block_to_device() {
    let mut block = Conv1dBlockBuilder::new(3, 5, 3, 1, 1)
        .device(Device::cpu())
        .build()
        .unwrap();

    let params = block.parameters().unwrap();
    for param in params {
        assert!(param.device().unwrap().is_cpu());
    }

    block.to_device(&Device::cpu()).unwrap();

    let params = block.parameters().unwrap();
    for param in params {
        assert!(param.device().unwrap().is_cpu());
    }
}

#[test]
fn test_conv1d_block_to_dtype() {
    let mut block = Conv1dBlockBuilder::new(4, 6, 3, 1, 1)
        .dtype(DType::F32)
        .build()
        .unwrap();

    let params = block.parameters().unwrap();
    for param in params {
        assert_eq!(param.dtype().unwrap(), DType::F32);
    }

    block.to_dtype(&DType::F32).unwrap();

    let params = block.parameters().unwrap();
    for param in params {
        assert_eq!(param.dtype().unwrap(), DType::F32);
    }
}

#[test]
fn test_conv1d_block_display() {
    let block = Conv1dBlockBuilder::new(8, 16, 3, 1, 1)
        .with_batch_norm1d()
        .with_activation(Activation::relu())
        .build()
        .unwrap();

    let display_str = format!("{}", block);
    assert!(display_str.contains("conv1d_block"));
}

#[test]
fn test_conv1d_block_builder_without_methods() {
    let block = Conv1dBlockBuilder::new(1, 16, 3, 1, 1)
        .with_batch_norm1d()
        .without_batch_norm1d()
        .with_activation(Activation::relu())
        .without_activation()
        .with_pool1d(Pool1d::max_pool1d(2, None).unwrap())
        .without_pool1d()
        .build()
        .unwrap();

    let params = block.parameters().unwrap();
    assert_eq!(params.len(), 2);
}

#[test]
fn test_conv1d_block_invalid_arguments() {
    let result = Conv1dBlockBuilder::new(0, 10, 3, 1, 1).build();
    assert!(result.is_err());

    let result = Conv1dBlockBuilder::new(10, 0, 3, 1, 1).build();
    assert!(result.is_err());

    let result = Conv1dBlockBuilder::new(10, 10, 0, 1, 1).build();
    assert!(result.is_err());

    let builder = Conv1dBlockBuilder::new(10, 10, 3, 0, 1);
    let result = builder.build();
    assert!(result.is_err());
}
