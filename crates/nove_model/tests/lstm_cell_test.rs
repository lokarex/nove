use nove::model::Model;
use nove::model::nn::LstmCellBuilder;
use nove::tensor::{DType, Device, Shape, Tensor};

#[test]
fn test_lstm_cell_builder_creation() {
    let lstm_cell = LstmCellBuilder::new(10, 20)
        .bias_enabled(true)
        .device(Device::cpu())
        .dtype(DType::F32)
        .grad_enabled(true)
        .build()
        .unwrap();

    assert_eq!(lstm_cell.input_size(), 10);
    assert_eq!(lstm_cell.hidden_size(), 20);

    let weight_ih_shape = lstm_cell.weight_ih().shape().unwrap();
    let weight_hh_shape = lstm_cell.weight_hh().shape().unwrap();

    assert_eq!(weight_ih_shape.dims(), [80, 10]);
    assert_eq!(weight_hh_shape.dims(), [80, 20]);
}

#[test]
fn test_lstm_cell_builder_without_bias() {
    let lstm_cell = LstmCellBuilder::new(5, 8)
        .bias_enabled(false)
        .build()
        .unwrap();

    assert_eq!(lstm_cell.input_size(), 5);
    assert_eq!(lstm_cell.hidden_size(), 8);

    assert!(lstm_cell.bias_ih().is_none());
    assert!(lstm_cell.bias_hh().is_none());
}

#[test]
fn test_lstm_cell_builder_method_chaining() {
    let mut builder = LstmCellBuilder::new(10, 20);
    builder
        .input_size(15)
        .hidden_size(25)
        .bias_enabled(true)
        .device(Device::cpu())
        .dtype(DType::F32)
        .grad_enabled(true);

    let lstm_cell = builder.build().unwrap();

    assert_eq!(lstm_cell.input_size(), 15);
    assert_eq!(lstm_cell.hidden_size(), 25);
}

#[test]
fn test_lstm_cell_builder_invalid_input_size() {
    let result = LstmCellBuilder::new(0, 20).build();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("input_size"));
}

#[test]
fn test_lstm_cell_builder_invalid_hidden_size() {
    let result = LstmCellBuilder::new(10, 0).build();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("hidden_size"));
}

#[test]
fn test_lstm_cell_single_time_step_forward() {
    let mut lstm_cell = LstmCellBuilder::new(3, 5)
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

    let cell_state = Tensor::zeros(
        &Shape::from_dims(&[batch_size, 5]),
        &DType::F32,
        &Device::cpu(),
        false,
    )
    .unwrap();

    let (new_hidden_state, new_cell_state) = lstm_cell
        .forward((input, (hidden_state, cell_state)))
        .unwrap();

    let new_hidden_shape = new_hidden_state.shape().unwrap();
    let new_cell_shape = new_cell_state.shape().unwrap();

    assert_eq!(new_hidden_shape.dims(), [batch_size, 5]);
    assert_eq!(new_cell_shape.dims(), [batch_size, 5]);
}

#[test]
fn test_lstm_cell_hidden_state_and_cell_state_update() {
    let mut lstm_cell = LstmCellBuilder::new(8, 12)
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

    let initial_cell_state = Tensor::zeros(
        &Shape::from_dims(&[batch_size, 12]),
        &DType::F32,
        &Device::cpu(),
        false,
    )
    .unwrap();

    let (hidden_state1, cell_state1) = lstm_cell
        .forward((
            input1,
            (initial_hidden_state.clone(), initial_cell_state.clone()),
        ))
        .unwrap();

    let input2 = Tensor::randn(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[batch_size, 8]),
        &Device::cpu(),
        false,
    )
    .unwrap();

    let (hidden_state2, cell_state2) = lstm_cell
        .forward((input2, (hidden_state1.clone(), cell_state1.clone())))
        .unwrap();

    let hidden_shape1 = hidden_state1.shape().unwrap();
    let hidden_shape2 = hidden_state2.shape().unwrap();
    let cell_shape1 = cell_state1.shape().unwrap();
    let cell_shape2 = cell_state2.shape().unwrap();

    assert_eq!(hidden_shape1.dims(), [batch_size, 12]);
    assert_eq!(hidden_shape2.dims(), [batch_size, 12]);
    assert_eq!(cell_shape1.dims(), [batch_size, 12]);
    assert_eq!(cell_shape2.dims(), [batch_size, 12]);
}

#[test]
fn test_lstm_cell_gate_mechanisms() {
    let mut lstm_cell = LstmCellBuilder::new(4, 4)
        .bias_enabled(true)
        .build()
        .unwrap();

    let batch_size = 1;
    let input = Tensor::ones(
        &Shape::from_dims(&[batch_size, 4]),
        &DType::F32,
        &Device::cpu(),
        false,
    )
    .unwrap();

    let hidden_state = Tensor::zeros(
        &Shape::from_dims(&[batch_size, 4]),
        &DType::F32,
        &Device::cpu(),
        false,
    )
    .unwrap();

    let cell_state = Tensor::zeros(
        &Shape::from_dims(&[batch_size, 4]),
        &DType::F32,
        &Device::cpu(),
        false,
    )
    .unwrap();

    let (new_hidden_state, new_cell_state) = lstm_cell
        .forward((input, (hidden_state, cell_state)))
        .unwrap();

    let hidden_data = new_hidden_state.to_vec::<f32>().unwrap();
    let cell_data = new_cell_state.to_vec::<f32>().unwrap();

    for i in 0..4 {
        assert!(hidden_data[i].abs() <= 1.0);
        assert!(cell_data[i].abs() <= 1.0);
    }
}

#[test]
fn test_lstm_cell_parameters() {
    let lstm_cell = LstmCellBuilder::new(10, 20)
        .bias_enabled(true)
        .build()
        .unwrap();

    let params = lstm_cell.parameters().unwrap();
    assert_eq!(params.len(), 4);
}

#[test]
fn test_lstm_cell_parameters_without_bias() {
    let lstm_cell = LstmCellBuilder::new(10, 20)
        .bias_enabled(false)
        .build()
        .unwrap();

    let params = lstm_cell.parameters().unwrap();
    assert_eq!(params.len(), 2);
}

#[test]
fn test_lstm_cell_named_parameters() {
    let lstm_cell = LstmCellBuilder::new(8, 16)
        .bias_enabled(true)
        .build()
        .unwrap();

    let named_params = lstm_cell.named_parameters().unwrap();
    assert_eq!(named_params.len(), 4);
}

#[test]
fn test_lstm_cell_require_grad() {
    let mut lstm_cell = LstmCellBuilder::new(10, 20)
        .grad_enabled(false)
        .build()
        .unwrap();

    let params = lstm_cell.parameters().unwrap();
    for param in params {
        assert!(!param.grad_enabled().unwrap());
    }

    lstm_cell.require_grad(true).unwrap();

    let params = lstm_cell.parameters().unwrap();
    for param in params {
        assert!(param.grad_enabled().unwrap());
    }
}

#[test]
fn test_lstm_cell_to_device() {
    let lstm_cell = LstmCellBuilder::new(10, 20)
        .device(Device::cpu())
        .build()
        .unwrap();

    let weight_ih_device = lstm_cell.weight_ih().device().unwrap();
    assert!(weight_ih_device.is_cpu());

    let weight_hh_device = lstm_cell.weight_hh().device().unwrap();
    assert!(weight_hh_device.is_cpu());
}

#[test]
fn test_lstm_cell_to_dtype() {
    let lstm_cell = LstmCellBuilder::new(10, 20)
        .dtype(DType::F32)
        .build()
        .unwrap();

    let weight_ih_dtype = lstm_cell.weight_ih().dtype().unwrap();
    assert_eq!(weight_ih_dtype, DType::F32);

    let weight_hh_dtype = lstm_cell.weight_hh().dtype().unwrap();
    assert_eq!(weight_hh_dtype, DType::F32);
}

#[test]
fn test_lstm_cell_forward_invalid_input_dimensions() {
    let mut lstm_cell = LstmCellBuilder::new(10, 20).build().unwrap();

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

    let cell_state = Tensor::randn(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[2, 20]),
        &Device::cpu(),
        false,
    )
    .unwrap();

    let result = lstm_cell.forward((input_1d, (hidden_state, cell_state)));
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("2 dimensions"));
}

#[test]
fn test_lstm_cell_forward_invalid_hidden_dimensions() {
    let mut lstm_cell = LstmCellBuilder::new(10, 20).build().unwrap();

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

    let cell_state = Tensor::randn(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[2, 20]),
        &Device::cpu(),
        false,
    )
    .unwrap();

    let result = lstm_cell.forward((input, (hidden_state_1d, cell_state)));
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("2 dimensions"));
}

#[test]
fn test_lstm_cell_forward_invalid_input_size() {
    let mut lstm_cell = LstmCellBuilder::new(10, 20).build().unwrap();

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

    let cell_state = Tensor::randn(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[2, 20]),
        &Device::cpu(),
        false,
    )
    .unwrap();

    let result = lstm_cell.forward((input, (hidden_state, cell_state)));
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("input size"));
}

#[test]
fn test_lstm_cell_forward_invalid_hidden_size() {
    let mut lstm_cell = LstmCellBuilder::new(10, 20).build().unwrap();

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

    let cell_state = Tensor::randn(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[2, 15]),
        &Device::cpu(),
        false,
    )
    .unwrap();

    let result = lstm_cell.forward((input, (hidden_state, cell_state)));
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("hidden size"));
}

#[test]
fn test_lstm_cell_forward_batch_size_mismatch() {
    let mut lstm_cell = LstmCellBuilder::new(10, 20).build().unwrap();

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

    let cell_state = Tensor::randn(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[3, 20]),
        &Device::cpu(),
        false,
    )
    .unwrap();

    let result = lstm_cell.forward((input, (hidden_state, cell_state)));
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("batch size"));
}

#[test]
fn test_lstm_cell_display() {
    let lstm_cell = LstmCellBuilder::new(10, 20)
        .bias_enabled(true)
        .build()
        .unwrap();

    let display_str = lstm_cell.to_string();
    assert!(display_str.contains("lstm_cell"));
    assert!(display_str.contains("input_size=10"));
    assert!(display_str.contains("hidden_size=20"));
}
