use nove::model::Model;
use nove::model::nn::{Activation, LinearBlockBuilder};
use nove::tensor::{DType, Device, Shape, Tensor};

#[test]
fn test_linear_block_builder_default() {
    let block = LinearBlockBuilder::new(10, 20).build().unwrap();

    let params = block.parameters().unwrap();
    assert_eq!(params.len(), 2);
    assert_eq!(params[0].shape().unwrap(), Shape::from_dims(&[10, 20]));
    assert_eq!(params[1].shape().unwrap(), Shape::from_dims(&[20]));
}

#[test]
fn test_linear_block_builder_with_activation() {
    let block = LinearBlockBuilder::new(10, 20)
        .with_activation(Activation::relu())
        .build()
        .unwrap();

    let params = block.parameters().unwrap();
    assert_eq!(params.len(), 2);
}

#[test]
fn test_linear_block_builder_with_batch_norm() {
    let block = LinearBlockBuilder::new(10, 20)
        .with_batch_norm1d()
        .build()
        .unwrap();

    let params = block.parameters().unwrap();
    assert_eq!(params.len(), 6);
}

#[test]
fn test_linear_block_builder_with_dropout() {
    let block = LinearBlockBuilder::new(10, 20)
        .with_dropout(0.5)
        .build()
        .unwrap();

    let params = block.parameters().unwrap();
    assert_eq!(params.len(), 2);
}

#[test]
fn test_linear_block_builder_with_all_components() {
    let block = LinearBlockBuilder::new(10, 20)
        .with_activation(Activation::gelu())
        .with_batch_norm1d()
        .with_dropout(0.3)
        .bias_enabled(true)
        .device(Device::cpu())
        .dtype(DType::F32)
        .grad_enabled(true)
        .build()
        .unwrap();

    let params = block.parameters().unwrap();
    assert_eq!(params.len(), 6);
}

#[test]
fn test_linear_block_forward_default() {
    let mut block = LinearBlockBuilder::new(3, 4).build().unwrap();

    let input = Tensor::ones(
        &Shape::from_dims(&[2, 3]),
        &DType::F32,
        &Device::cpu(),
        false,
    )
    .unwrap();
    let output = block.forward(input).unwrap();

    assert_eq!(output.shape().unwrap(), Shape::from_dims(&[2, 4]));
}

#[test]
fn test_linear_block_forward_with_activation() {
    let mut block = LinearBlockBuilder::new(3, 4)
        .with_activation(Activation::relu())
        .build()
        .unwrap();

    let input_data = vec![-1.0f32, 0.0, 1.0, 2.0, -2.0, 0.5];
    let input = Tensor::from_data(input_data, &Device::cpu(), false)
        .unwrap()
        .reshape(&Shape::from_dims(&[2, 3]))
        .unwrap();

    let output = block.forward(input).unwrap();
    let output_vec = output.to_vec::<f32>().unwrap();

    assert_eq!(output.shape().unwrap(), Shape::from_dims(&[2, 4]));
    for &value in &output_vec {
        assert!(value >= 0.0);
    }
}

#[test]
fn test_linear_block_forward_with_batch_norm_training() {
    let mut block = LinearBlockBuilder::new(3, 4)
        .with_batch_norm1d()
        .build()
        .unwrap();

    let input = Tensor::ones(
        &Shape::from_dims(&[2, 3]),
        &DType::F32,
        &Device::cpu(),
        false,
    )
    .unwrap();

    let output_training = block.forward(input.clone()).unwrap();
    let output_inference = block.forward(input).unwrap();

    assert_eq!(output_training.shape().unwrap(), Shape::from_dims(&[2, 4]));
    assert_eq!(output_inference.shape().unwrap(), Shape::from_dims(&[2, 4]));
}

#[test]
fn test_linear_block_forward_with_dropout_training_vs_inference() {
    let mut block = LinearBlockBuilder::new(3, 4)
        .with_dropout(0.5)
        .build()
        .unwrap();

    let input = Tensor::ones(
        &Shape::from_dims(&[2, 3]),
        &DType::F32,
        &Device::cpu(),
        false,
    )
    .unwrap();

    let output_training = block.forward(input.clone()).unwrap();
    let output_inference = block.forward(input).unwrap();

    assert_eq!(output_training.shape().unwrap(), Shape::from_dims(&[2, 4]));
    assert_eq!(output_inference.shape().unwrap(), Shape::from_dims(&[2, 4]));

    let training_values = output_training.to_vec::<f32>().unwrap();
    let inference_values = output_inference.to_vec::<f32>().unwrap();

    assert_ne!(training_values, inference_values);
}

#[test]
fn test_linear_block_forward_complete_chain() {
    let mut block = LinearBlockBuilder::new(5, 8)
        .with_activation(Activation::gelu())
        .with_batch_norm1d()
        .with_dropout(0.2)
        .build()
        .unwrap();

    let input = Tensor::rand(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[4, 5]),
        &Device::cpu(),
        false,
    )
    .unwrap();

    let output_training = block.forward(input.clone()).unwrap();
    let output_inference = block.forward(input).unwrap();

    assert_eq!(output_training.shape().unwrap(), Shape::from_dims(&[4, 8]));
    assert_eq!(output_inference.shape().unwrap(), Shape::from_dims(&[4, 8]));
}

#[test]
fn test_linear_block_builder_configuration_methods() {
    let block = LinearBlockBuilder::new(10, 20)
        .with_activation(Activation::silu())
        .with_batch_norm1d()
        .with_dropout(0.4)
        .bias_enabled(false)
        .grad_enabled(false)
        .build()
        .unwrap();

    let params = block.parameters().unwrap();
    assert_eq!(params.len(), 5);
}

#[test]
fn test_linear_block_builder_without_methods() {
    let block = LinearBlockBuilder::new(10, 20)
        .with_activation(Activation::tanh())
        .with_batch_norm1d()
        .with_dropout(0.3)
        .without_activation()
        .without_batch_norm1d()
        .without_dropout()
        .build()
        .unwrap();

    let params = block.parameters().unwrap();
    assert_eq!(params.len(), 2);
}

#[test]
fn test_linear_block_parameters() {
    let block = LinearBlockBuilder::new(6, 9)
        .with_activation(Activation::sigmoid())
        .with_batch_norm1d()
        .with_dropout(0.1)
        .build()
        .unwrap();

    let params = block.parameters().unwrap();
    assert_eq!(params.len(), 6);

    let named_params = block.named_parameters().unwrap();
    assert_eq!(named_params.len(), 6);
}

#[test]
fn test_linear_block_require_grad() {
    let mut block = LinearBlockBuilder::new(4, 7)
        .with_batch_norm1d()
        .grad_enabled(false)
        .build()
        .unwrap();

    let params = block.parameters().unwrap();
    assert_eq!(params.len(), 6);

    let linear_weight = params.get(0).unwrap();
    assert!(!linear_weight.grad_enabled().unwrap());

    block.require_grad(true).unwrap();

    let params = block.parameters().unwrap();
    assert_eq!(params.len(), 6);

    let linear_weight = params.get(0).unwrap();
    assert!(linear_weight.grad_enabled().unwrap());
}

#[test]
fn test_linear_block_to_device() {
    let mut block = LinearBlockBuilder::new(5, 8)
        .with_activation(Activation::relu())
        .build()
        .unwrap();

    block.to_device(&Device::cpu()).unwrap();

    let params = block.parameters().unwrap();
    for param in params {
        assert!(param.device().unwrap().is_cpu());
    }
}

#[test]
fn test_linear_block_to_dtype() {
    let mut block = LinearBlockBuilder::new(5, 8)
        .with_activation(Activation::gelu())
        .dtype(DType::F32)
        .build()
        .unwrap();

    block.to_dtype(&DType::F32).unwrap();

    let params = block.parameters().unwrap();
    for param in params {
        assert_eq!(param.dtype().unwrap(), DType::F32);
    }
}

#[test]
fn test_linear_block_display() {
    let block = LinearBlockBuilder::new(8, 16)
        .with_activation(Activation::relu())
        .with_batch_norm1d()
        .build()
        .unwrap();

    let display_str = format!("{}", block);
    assert!(display_str.contains("linear_bloc"));
    assert!(display_str.contains("linear"));
}
