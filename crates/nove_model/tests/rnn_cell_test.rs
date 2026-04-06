use nove::model::Model;
use nove::model::nn::Activation;
use nove::model::nn::RnnCellBuilder;
use nove::tensor::{DType, Device, Shape, Tensor};

#[test]
fn test_rnn_cell_builder_creation() {
    let rnn_cell = RnnCellBuilder::new(10, 20)
        .activation(Activation::tanh())
        .bias_enabled(true)
        .device(Device::cpu())
        .dtype(DType::F32)
        .grad_enabled(true)
        .build()
        .unwrap();

    assert_eq!(rnn_cell.input_size(), 10);
    assert_eq!(rnn_cell.hidden_size(), 20);
    assert!(rnn_cell.activation().to_string().starts_with("tanh"));

    let weight_ih_shape = rnn_cell.weight_ih().shape().unwrap();
    let weight_hh_shape = rnn_cell.weight_hh().shape().unwrap();

    assert_eq!(weight_ih_shape.dims(), [20, 10]);
    assert_eq!(weight_hh_shape.dims(), [20, 20]);
}

#[test]
fn test_rnn_cell_builder_without_bias() {
    let rnn_cell = RnnCellBuilder::new(5, 8)
        .bias_enabled(false)
        .build()
        .unwrap();

    assert_eq!(rnn_cell.input_size(), 5);
    assert_eq!(rnn_cell.hidden_size(), 8);
}

#[test]
fn test_rnn_cell_builder_method_chaining() {
    let mut builder = RnnCellBuilder::new(10, 20);
    builder
        .input_size(15)
        .hidden_size(25)
        .activation(Activation::relu())
        .bias_enabled(false)
        .device(Device::cpu())
        .dtype(DType::F32)
        .grad_enabled(true);

    let rnn_cell = builder.build().unwrap();

    assert_eq!(rnn_cell.input_size(), 15);
    assert_eq!(rnn_cell.hidden_size(), 25);
    assert!(rnn_cell.activation().to_string().starts_with("relu"));
}

#[test]
fn test_rnn_cell_builder_invalid_input_size() {
    let result = RnnCellBuilder::new(0, 20).build();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("input_size"));
}

#[test]
fn test_rnn_cell_builder_invalid_hidden_size() {
    let result = RnnCellBuilder::new(10, 0).build();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("hidden_size"));
}

#[test]
fn test_rnn_cell_single_time_step_forward() {
    let mut rnn_cell = RnnCellBuilder::new(3, 5)
        .bias_enabled(true)
        .build()
        .unwrap();

    let batch_size = 2;
    let input = Tensor::randn(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[batch_size, 3]),
        &Device::cpu(),
        false,
    )
    .unwrap();

    let hidden_state = Tensor::zeros(
        &Shape::from_dims(&[batch_size, 5]),
        &DType::F32,
        &Device::cpu(),
        false,
    )
    .unwrap();

    let new_hidden_state = rnn_cell.forward((input, hidden_state)).unwrap();

    let new_hidden_shape = new_hidden_state.shape().unwrap();
    assert_eq!(new_hidden_shape.dims(), [batch_size, 5]);
}

#[test]
fn test_rnn_cell_forward_with_different_activations() {
    let activations = vec![Activation::tanh(), Activation::relu()];

    for activation in activations {
        let mut rnn_cell = RnnCellBuilder::new(4, 6)
            .activation(activation.clone())
            .bias_enabled(true)
            .build()
            .unwrap();

        let batch_size = 3;
        let input = Tensor::randn(
            0.0f32,
            1.0f32,
            &Shape::from_dims(&[batch_size, 4]),
            &Device::cpu(),
            false,
        )
        .unwrap();

        let hidden_state = Tensor::zeros(
            &Shape::from_dims(&[batch_size, 6]),
            &DType::F32,
            &Device::cpu(),
            false,
        )
        .unwrap();

        let new_hidden_state = rnn_cell.forward((input, hidden_state)).unwrap();

        let new_hidden_shape = new_hidden_state.shape().unwrap();
        assert_eq!(new_hidden_shape.dims(), [batch_size, 6]);
        assert_eq!(rnn_cell.activation().to_string(), activation.to_string());
    }
}

#[test]
fn test_rnn_cell_hidden_state_update() {
    let mut rnn_cell = RnnCellBuilder::new(8, 12)
        .bias_enabled(true)
        .build()
        .unwrap();

    let batch_size = 4;

    let input1 = Tensor::randn(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[batch_size, 8]),
        &Device::cpu(),
        false,
    )
    .unwrap();

    let initial_hidden_state = Tensor::zeros(
        &Shape::from_dims(&[batch_size, 12]),
        &DType::F32,
        &Device::cpu(),
        false,
    )
    .unwrap();

    let hidden_state1 = rnn_cell.forward((input1, initial_hidden_state)).unwrap();

    let input2 = Tensor::randn(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[batch_size, 8]),
        &Device::cpu(),
        false,
    )
    .unwrap();

    let hidden_state2 = rnn_cell.forward((input2, hidden_state1.clone())).unwrap();

    let hidden_shape1 = hidden_state1.shape().unwrap();
    let hidden_shape2 = hidden_state2.shape().unwrap();

    assert_eq!(hidden_shape1.dims(), [batch_size, 12]);
    assert_eq!(hidden_shape2.dims(), [batch_size, 12]);
}

#[test]
fn test_rnn_cell_forward_invalid_input_dimensions() {
    let mut rnn_cell = RnnCellBuilder::new(10, 20).build().unwrap();

    let input_1d = Tensor::randn(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[10]),
        &Device::cpu(),
        false,
    )
    .unwrap();

    let hidden_state = Tensor::randn(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[2, 20]),
        &Device::cpu(),
        false,
    )
    .unwrap();

    let result = rnn_cell.forward((input_1d, hidden_state));
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("2 dimensions"));
}

#[test]
fn test_rnn_cell_forward_invalid_hidden_dimensions() {
    let mut rnn_cell = RnnCellBuilder::new(10, 20).build().unwrap();

    let input = Tensor::randn(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[2, 10]),
        &Device::cpu(),
        false,
    )
    .unwrap();

    let hidden_state_1d = Tensor::randn(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[20]),
        &Device::cpu(),
        false,
    )
    .unwrap();

    let result = rnn_cell.forward((input, hidden_state_1d));
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("2 dimensions"));
}

#[test]
fn test_rnn_cell_forward_invalid_input_size() {
    let mut rnn_cell = RnnCellBuilder::new(10, 20).build().unwrap();

    let input = Tensor::randn(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[2, 8]),
        &Device::cpu(),
        false,
    )
    .unwrap();

    let hidden_state = Tensor::randn(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[2, 20]),
        &Device::cpu(),
        false,
    )
    .unwrap();

    let result = rnn_cell.forward((input, hidden_state));
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("input size"));
}

#[test]
fn test_rnn_cell_forward_invalid_hidden_size() {
    let mut rnn_cell = RnnCellBuilder::new(10, 20).build().unwrap();

    let input = Tensor::randn(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[2, 10]),
        &Device::cpu(),
        false,
    )
    .unwrap();

    let hidden_state = Tensor::randn(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[2, 15]),
        &Device::cpu(),
        false,
    )
    .unwrap();

    let result = rnn_cell.forward((input, hidden_state));
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("hidden size"));
}

#[test]
fn test_rnn_cell_forward_batch_size_mismatch() {
    let mut rnn_cell = RnnCellBuilder::new(10, 20).build().unwrap();

    let input = Tensor::randn(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[2, 10]),
        &Device::cpu(),
        false,
    )
    .unwrap();

    let hidden_state = Tensor::randn(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[3, 20]),
        &Device::cpu(),
        false,
    )
    .unwrap();

    let result = rnn_cell.forward((input, hidden_state));
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("batch size"));
}

#[test]
fn test_rnn_cell_parameters() {
    let rnn_cell = RnnCellBuilder::new(10, 20)
        .bias_enabled(true)
        .build()
        .unwrap();

    let params = rnn_cell.parameters().unwrap();
    assert_eq!(params.len(), 4);
}

#[test]
fn test_rnn_cell_named_parameters() {
    let rnn_cell = RnnCellBuilder::new(8, 16)
        .bias_enabled(true)
        .build()
        .unwrap();

    let named_params = rnn_cell.named_parameters().unwrap();
    assert_eq!(named_params.len(), 4);
}

#[test]
fn test_rnn_cell_require_grad() {
    let mut rnn_cell = RnnCellBuilder::new(10, 20)
        .grad_enabled(false)
        .build()
        .unwrap();

    let params = rnn_cell.parameters().unwrap();
    for param in params {
        assert!(!param.grad_enabled().unwrap());
    }

    rnn_cell.require_grad(true).unwrap();

    let params = rnn_cell.parameters().unwrap();
    for param in params {
        assert!(param.grad_enabled().unwrap());
    }
}

#[test]
fn test_rnn_cell_to_device() {
    let rnn_cell = RnnCellBuilder::new(10, 20)
        .device(Device::cpu())
        .build()
        .unwrap();

    let weight_ih_device = rnn_cell.weight_ih().device().unwrap();
    assert!(weight_ih_device.is_cpu());

    let weight_hh_device = rnn_cell.weight_hh().device().unwrap();
    assert!(weight_hh_device.is_cpu());
}

#[test]
fn test_rnn_cell_to_dtype() {
    let rnn_cell = RnnCellBuilder::new(10, 20)
        .dtype(DType::F32)
        .build()
        .unwrap();

    let weight_ih_dtype = rnn_cell.weight_ih().dtype().unwrap();
    assert_eq!(weight_ih_dtype, DType::F32);

    let weight_hh_dtype = rnn_cell.weight_hh().dtype().unwrap();
    assert_eq!(weight_hh_dtype, DType::F32);
}
