use nove::model::{Model, layer::LstmCellBuilder};
use nove::tensor::{DType, Device, Shape, Tensor};

#[test]
fn test_lstm_cell_builder_success() {
    // Test building with minimal required parameters
    let lstm_cell = LstmCellBuilder::new(10, 20).build().unwrap();

    assert_eq!(lstm_cell.input_size(), 10);
    assert_eq!(lstm_cell.hidden_size(), 20);

    // Test building with all parameters customized
    let lstm_cell = LstmCellBuilder::new(5, 8)
        .bias_enabled(false)
        .device(Device::cpu())
        .dtype(DType::F32)
        .grad_enabled(true)
        .build()
        .unwrap();

    assert_eq!(lstm_cell.input_size(), 5);
    assert_eq!(lstm_cell.hidden_size(), 8);
    assert!(lstm_cell.bias_ih().is_none());
    assert!(lstm_cell.bias_hh().is_none());
}

#[test]
fn test_lstm_cell_builder_failure() {
    // Test with zero input size
    let result = LstmCellBuilder::new(0, 20).build();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("input_size"));

    // Test with zero hidden size
    let result = LstmCellBuilder::new(10, 0).build();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("hidden_size"));
}

#[test]
fn test_lstm_cell_forward_shape() {
    // Test forward pass with different configurations
    let mut lstm_cell = LstmCellBuilder::new(10, 20).build().unwrap();

    // Create input, hidden state, and cell state tensors
    let batch_size = 4;
    let input = Tensor::rand(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[batch_size, 10]),
        &Device::cpu(),
        false,
    )
    .unwrap();

    let hidden = Tensor::rand(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[batch_size, 20]),
        &Device::cpu(),
        false,
    )
    .unwrap();

    let cell = Tensor::rand(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[batch_size, 20]),
        &Device::cpu(),
        false,
    )
    .unwrap();

    // Test forward pass
    let (new_hidden, new_cell) = lstm_cell.forward((input, (hidden, cell))).unwrap();

    // Check output shapes match expected [batch_size, hidden_size]
    assert_eq!(
        new_hidden.shape().unwrap(),
        Shape::from_dims(&[batch_size, 20])
    );
    assert_eq!(
        new_cell.shape().unwrap(),
        Shape::from_dims(&[batch_size, 20])
    );
}

#[test]
fn test_lstm_cell_forward_with_bias() {
    // Test forward pass with bias enabled (default)
    let mut lstm_cell = LstmCellBuilder::new(5, 8)
        .bias_enabled(true)
        .build()
        .unwrap();

    let input = Tensor::rand(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[3, 5]),
        &Device::cpu(),
        false,
    )
    .unwrap();

    let hidden = Tensor::rand(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[3, 8]),
        &Device::cpu(),
        false,
    )
    .unwrap();

    let cell = Tensor::rand(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[3, 8]),
        &Device::cpu(),
        false,
    )
    .unwrap();

    let (new_hidden, new_cell) = lstm_cell.forward((input, (hidden, cell))).unwrap();
    assert_eq!(new_hidden.shape().unwrap(), Shape::from_dims(&[3, 8]));
    assert_eq!(new_cell.shape().unwrap(), Shape::from_dims(&[3, 8]));
}

#[test]
fn test_lstm_cell_forward_without_bias() {
    // Test forward pass with bias disabled
    let mut lstm_cell = LstmCellBuilder::new(5, 8)
        .bias_enabled(false)
        .build()
        .unwrap();

    let input = Tensor::rand(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[2, 5]),
        &Device::cpu(),
        false,
    )
    .unwrap();

    let hidden = Tensor::rand(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[2, 8]),
        &Device::cpu(),
        false,
    )
    .unwrap();

    let cell = Tensor::rand(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[2, 8]),
        &Device::cpu(),
        false,
    )
    .unwrap();

    let (new_hidden, new_cell) = lstm_cell.forward((input, (hidden, cell))).unwrap();
    assert_eq!(new_hidden.shape().unwrap(), Shape::from_dims(&[2, 8]));
    assert_eq!(new_cell.shape().unwrap(), Shape::from_dims(&[2, 8]));
}

#[test]
fn test_lstm_cell_parameters() {
    let lstm_cell = LstmCellBuilder::new(6, 4).build().unwrap();

    let params = lstm_cell.parameters().unwrap();
    // With bias enabled: weight_ih, weight_hh, bias_ih, bias_hh
    assert_eq!(params.len(), 4);

    let named_params = lstm_cell.named_parameters().unwrap();
    assert_eq!(named_params.len(), 4);
    assert!(named_params.contains_key(&format!("lstm_cell.{}.weight_ih", lstm_cell.id())));
    assert!(named_params.contains_key(&format!("lstm_cell.{}.weight_hh", lstm_cell.id())));
    assert!(named_params.contains_key(&format!("lstm_cell.{}.bias_ih", lstm_cell.id())));
    assert!(named_params.contains_key(&format!("lstm_cell.{}.bias_hh", lstm_cell.id())));
}

#[test]
fn test_lstm_cell_parameters_without_bias() {
    let lstm_cell = LstmCellBuilder::new(6, 4)
        .bias_enabled(false)
        .build()
        .unwrap();

    let params = lstm_cell.parameters().unwrap();
    // Without bias: only weight_ih and weight_hh
    assert_eq!(params.len(), 2);

    let named_params = lstm_cell.named_parameters().unwrap();
    assert_eq!(named_params.len(), 2);
    assert!(named_params.contains_key(&format!("lstm_cell.{}.weight_ih", lstm_cell.id())));
    assert!(named_params.contains_key(&format!("lstm_cell.{}.weight_hh", lstm_cell.id())));
    assert!(!named_params.contains_key(&format!("lstm_cell.{}.bias_ih", lstm_cell.id())));
    assert!(!named_params.contains_key(&format!("lstm_cell.{}.bias_hh", lstm_cell.id())));
}

#[test]
fn test_lstm_cell_require_grad() {
    let mut lstm_cell = LstmCellBuilder::new(3, 2)
        .grad_enabled(false)
        .build()
        .unwrap();

    // Initially grad should be disabled
    let _weight_ih = lstm_cell.weight_ih();
    // Note: There's no direct API to check if gradient is enabled, but we can trust the builder

    // Enable gradient
    lstm_cell.require_grad(true).unwrap();
    let _weight_ih_with_grad = lstm_cell.weight_ih();
    // Again, no direct API to verify, but we can test that the method doesn't error
}

#[test]
fn test_lstm_cell_to_device() {
    let mut lstm_cell = LstmCellBuilder::new(4, 3)
        .device(Device::cpu())
        .build()
        .unwrap();

    // Test that to_device doesn't error (device may be the same)
    lstm_cell.to_device(&Device::cpu()).unwrap();
}

#[test]
fn test_lstm_cell_to_dtype() {
    let mut lstm_cell = LstmCellBuilder::new(4, 3)
        .dtype(DType::F32)
        .build()
        .unwrap();

    // Test dtype conversion
    lstm_cell.to_dtype(&DType::F32).unwrap();
    // Note: F64 may not be supported on all devices, but F32 should always work
}

#[test]
fn test_lstm_cell_display() {
    let lstm_cell = LstmCellBuilder::new(7, 5).build().unwrap();
    let display_str = format!("{}", lstm_cell);

    assert!(display_str.starts_with(&format!("lstm_cell.{}", lstm_cell.id())));
    assert!(display_str.contains("input_size=7"));
    assert!(display_str.contains("hidden_size=5"));
    assert!(display_str.contains("bias_enabled=true"));
}

#[test]
fn test_lstm_cell_display_without_bias() {
    let lstm_cell = LstmCellBuilder::new(7, 5)
        .bias_enabled(false)
        .build()
        .unwrap();
    let display_str = format!("{}", lstm_cell);

    assert!(display_str.starts_with(&format!("lstm_cell.{}", lstm_cell.id())));
    assert!(display_str.contains("input_size=7"));
    assert!(display_str.contains("hidden_size=5"));
    assert!(display_str.contains("bias_enabled=false"));
}
