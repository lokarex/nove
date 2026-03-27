use nove::model::layer::GruCellBuilder;
use nove::model::Model;
use nove::tensor::{Device, DType, Shape, Tensor};

#[test]
fn test_gru_cell_builder_creation() {
    let gru_cell = GruCellBuilder::new(10, 20)
        .bias_enabled(true)
        .device(Device::cpu())
        .dtype(DType::F32)
        .grad_enabled(true)
        .build()
        .unwrap();

    assert_eq!(gru_cell.input_size(), 10);
    assert_eq!(gru_cell.hidden_size(), 20);

    let weight_ih_shape = gru_cell.weight_ih().shape().unwrap();
    let weight_hh_shape = gru_cell.weight_hh().shape().unwrap();

    assert_eq!(weight_ih_shape.dims(), [60, 10]); // 3 * hidden_size, input_size
    assert_eq!(weight_hh_shape.dims(), [60, 20]); // 3 * hidden_size, hidden_size
}

#[test]
fn test_gru_cell_builder_without_bias() {
    let gru_cell = GruCellBuilder::new(5, 8)
        .bias_enabled(false)
        .build()
        .unwrap();

    assert_eq!(gru_cell.input_size(), 5);
    assert_eq!(gru_cell.hidden_size(), 8);

    assert!(gru_cell.bias_ih().is_none());
    assert!(gru_cell.bias_hh().is_none());
}

#[test]
fn test_gru_cell_builder_method_chaining() {
    let mut builder = GruCellBuilder::new(10, 20);
    builder
        .input_size(15)
        .hidden_size(25)
        .bias_enabled(true)
        .device(Device::cpu())
        .dtype(DType::F32)
        .grad_enabled(true);

    let gru_cell = builder.build().unwrap();

    assert_eq!(gru_cell.input_size(), 15);
    assert_eq!(gru_cell.hidden_size(), 25);
}

#[test]
fn test_gru_cell_builder_invalid_input_size() {
    let result = GruCellBuilder::new(0, 20).build();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("input_size"));
}

#[test]
fn test_gru_cell_builder_invalid_hidden_size() {
    let result = GruCellBuilder::new(10, 0).build();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("hidden_size"));
}

#[test]
fn test_gru_cell_single_time_step_forward() {
    let mut gru_cell = GruCellBuilder::new(3, 5)
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
    ).unwrap();

    let hidden_state = Tensor::zeros(
        &Shape::from_dims(&[batch_size, 5]),
        &DType::F32,
        &Device::cpu(),
        false,
    ).unwrap();

    let new_hidden_state = gru_cell.forward((input, hidden_state)).unwrap();

    let new_hidden_shape = new_hidden_state.shape().unwrap();
    assert_eq!(new_hidden_shape.dims(), [batch_size, 5]);
}

#[test]
fn test_gru_cell_hidden_state_update() {
    let mut gru_cell = GruCellBuilder::new(8, 12)
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
    ).unwrap();

    let initial_hidden_state = Tensor::zeros(
        &Shape::from_dims(&[batch_size, 12]),
        &DType::F32,
        &Device::cpu(),
        false,
    ).unwrap();

    let hidden_state1 = gru_cell.forward((input1, initial_hidden_state.clone())).unwrap();

    let input2 = Tensor::randn(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[batch_size, 8]),
        &Device::cpu(),
        false,
    ).unwrap();

    let hidden_state2 = gru_cell.forward((input2, hidden_state1.clone())).unwrap();

    let hidden_shape1 = hidden_state1.shape().unwrap();
    let hidden_shape2 = hidden_state2.shape().unwrap();

    assert_eq!(hidden_shape1.dims(), [batch_size, 12]);
    assert_eq!(hidden_shape2.dims(), [batch_size, 12]);
}

#[test]
fn test_gru_cell_forward_invalid_input_dimensions() {
    let mut gru_cell = GruCellBuilder::new(10, 20)
        .build()
        .unwrap();

    let input_1d = Tensor::randn(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[10]),
        &Device::cpu(),
        false,
    ).unwrap();

    let hidden_state = Tensor::randn(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[2, 20]),
        &Device::cpu(),
        false,
    ).unwrap();

    let result = gru_cell.forward((input_1d, hidden_state));
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("2 dimensions"));
}

#[test]
fn test_gru_cell_forward_invalid_hidden_dimensions() {
    let mut gru_cell = GruCellBuilder::new(10, 20)
        .build()
        .unwrap();

    let input = Tensor::randn(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[2, 10]),
        &Device::cpu(),
        false,
    ).unwrap();

    let hidden_state_1d = Tensor::randn(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[20]),
        &Device::cpu(),
        false,
    ).unwrap();

    let result = gru_cell.forward((input, hidden_state_1d));
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("2 dimensions"));
}

#[test]
fn test_gru_cell_forward_invalid_input_size() {
    let mut gru_cell = GruCellBuilder::new(10, 20)
        .build()
        .unwrap();

    let input = Tensor::randn(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[2, 8]),
        &Device::cpu(),
        false,
    ).unwrap();

    let hidden_state = Tensor::randn(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[2, 20]),
        &Device::cpu(),
        false,
    ).unwrap();

    let result = gru_cell.forward((input, hidden_state));
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("input size"));
}

#[test]
fn test_gru_cell_forward_invalid_hidden_size() {
    let mut gru_cell = GruCellBuilder::new(10, 20)
        .build()
        .unwrap();

    let input = Tensor::randn(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[2, 10]),
        &Device::cpu(),
        false,
    ).unwrap();

    let hidden_state = Tensor::randn(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[2, 15]),
        &Device::cpu(),
        false,
    ).unwrap();

    let result = gru_cell.forward((input, hidden_state));
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("hidden size"));
}

#[test]
fn test_gru_cell_forward_batch_size_mismatch() {
    let mut gru_cell = GruCellBuilder::new(10, 20)
        .build()
        .unwrap();

    let input = Tensor::randn(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[2, 10]),
        &Device::cpu(),
        false,
    ).unwrap();

    let hidden_state = Tensor::randn(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[3, 20]),
        &Device::cpu(),
        false,
    ).unwrap();

    let result = gru_cell.forward((input, hidden_state));
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("batch size"));
}

#[test]
fn test_gru_cell_parameters() {
    let gru_cell = GruCellBuilder::new(10, 20)
        .bias_enabled(true)
        .build()
        .unwrap();

    let params = gru_cell.parameters().unwrap();
    assert_eq!(params.len(), 4);
}

#[test]
fn test_gru_cell_parameters_without_bias() {
    let gru_cell = GruCellBuilder::new(10, 20)
        .bias_enabled(false)
        .build()
        .unwrap();

    let params = gru_cell.parameters().unwrap();
    assert_eq!(params.len(), 2);
}

#[test]
fn test_gru_cell_named_parameters() {
    let gru_cell = GruCellBuilder::new(8, 16)
        .bias_enabled(true)
        .build()
        .unwrap();

    let named_params = gru_cell.named_parameters().unwrap();
    assert_eq!(named_params.len(), 4);
}

#[test]
fn test_gru_cell_require_grad() {
    let mut gru_cell = GruCellBuilder::new(10, 20)
        .grad_enabled(false)
        .build()
        .unwrap();

    let params = gru_cell.parameters().unwrap();
    for param in params {
        assert!(!param.grad_enabled().unwrap());
    }

    gru_cell.require_grad(true).unwrap();

    let params = gru_cell.parameters().unwrap();
    for param in params {
        assert!(param.grad_enabled().unwrap());
    }
}

#[test]
fn test_gru_cell_to_device() {
    let gru_cell = GruCellBuilder::new(10, 20)
        .device(Device::cpu())
        .build()
        .unwrap();

    let weight_ih_device = gru_cell.weight_ih().device().unwrap();
    assert!(weight_ih_device.is_cpu());

    let weight_hh_device = gru_cell.weight_hh().device().unwrap();
    assert!(weight_hh_device.is_cpu());
}

#[test]
fn test_gru_cell_to_dtype() {
    let gru_cell = GruCellBuilder::new(10, 20)
        .dtype(DType::F32)
        .build()
        .unwrap();

    let weight_ih_dtype = gru_cell.weight_ih().dtype().unwrap();
    assert_eq!(weight_ih_dtype, DType::F32);

    let weight_hh_dtype = gru_cell.weight_hh().dtype().unwrap();
    assert_eq!(weight_hh_dtype, DType::F32);
}

#[test]
fn test_gru_cell_display() {
    let gru_cell = GruCellBuilder::new(10, 20)
        .bias_enabled(true)
        .build()
        .unwrap();

    let display_str = gru_cell.to_string();
    assert!(display_str.contains("GruCell"));
    assert!(display_str.contains("input_size: 10"));
    assert!(display_str.contains("hidden_size: 20"));
}