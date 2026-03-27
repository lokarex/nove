use nove::model::Model;
use nove::model::layer::GruBuilder;
use nove::tensor::{DType, Device, Shape, Tensor};

#[test]
fn test_gru_builder_creation() {
    let gru = GruBuilder::new(10, 20)
        .num_layers(2)
        .bias(true)
        .batch_first(false)
        .dropout(0.0)
        .bidirectional(false)
        .device(Device::cpu())
        .dtype(DType::F32)
        .grad_enabled(true)
        .build()
        .unwrap();

    assert_eq!(gru.input_size(), 10);
    assert_eq!(gru.hidden_size(), 20);
    assert_eq!(gru.num_layers(), 2);
    assert_eq!(gru.bias(), true);
    assert_eq!(gru.batch_first(), false);
    assert_eq!(gru.dropout_rate(), 0.0);
    assert_eq!(gru.bidirectional(), false);
    assert_eq!(gru.num_directions(), 1);
}

#[test]
fn test_gru_builder_without_bias() {
    let gru = GruBuilder::new(5, 8).bias(false).build().unwrap();

    assert_eq!(gru.input_size(), 5);
    assert_eq!(gru.hidden_size(), 8);
    assert_eq!(gru.bias(), false);
}

#[test]
fn test_gru_builder_batch_first() {
    let gru = GruBuilder::new(10, 20).batch_first(true).build().unwrap();

    assert_eq!(gru.input_size(), 10);
    assert_eq!(gru.hidden_size(), 20);
    assert_eq!(gru.batch_first(), true);
}

#[test]
fn test_gru_builder_multilayer_bidirectional_dropout() {
    let gru = GruBuilder::new(16, 32)
        .num_layers(3)
        .bidirectional(true)
        .dropout(0.5)
        .build()
        .unwrap();

    assert_eq!(gru.input_size(), 16);
    assert_eq!(gru.hidden_size(), 32);
    assert_eq!(gru.num_layers(), 3);
    assert_eq!(gru.bidirectional(), true);
    assert_eq!(gru.num_directions(), 2);
    assert_eq!(gru.dropout_rate(), 0.5);
}

#[test]
fn test_gru_builder_invalid_input_size() {
    let result = GruBuilder::new(0, 20).build();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("input_size"));
}

#[test]
fn test_gru_builder_invalid_hidden_size() {
    let result = GruBuilder::new(10, 0).build();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("hidden_size"));
}

#[test]
fn test_gru_builder_invalid_num_layers() {
    let result = GruBuilder::new(10, 20).num_layers(0).build();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("num_layers"));
}

#[test]
fn test_gru_builder_invalid_dropout() {
    let result = GruBuilder::new(10, 20).dropout(1.5).build();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("dropout"));
}

#[test]
fn test_gru_forward_batch_last() {
    let mut gru = GruBuilder::new(8, 16)
        .num_layers(2)
        .bidirectional(false)
        .dropout(0.0)
        .batch_first(false)
        .build()
        .unwrap();

    let seq_len = 5;
    let batch_size = 3;
    let input = Tensor::randn(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[seq_len, batch_size, 8]),
        &Device::cpu(),
        false,
    )
    .unwrap();

    let (output, hidden_states) = gru.forward(input).unwrap();

    let output_shape = output.shape().unwrap();
    let hidden_shape = hidden_states.shape().unwrap();

    assert_eq!(output_shape.dims(), [seq_len, batch_size, 16]);
    assert_eq!(hidden_shape.dims(), [2, batch_size, 16]);
}

#[test]
fn test_gru_forward_batch_first() {
    let mut gru = GruBuilder::new(12, 24)
        .num_layers(1)
        .bidirectional(false)
        .dropout(0.0)
        .batch_first(true)
        .build()
        .unwrap();

    let batch_size = 4;
    let seq_len = 6;
    let input = Tensor::randn(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[batch_size, seq_len, 12]),
        &Device::cpu(),
        false,
    )
    .unwrap();

    let (output, hidden_states) = gru.forward(input).unwrap();

    let output_shape = output.shape().unwrap();
    let hidden_shape = hidden_states.shape().unwrap();

    assert_eq!(output_shape.dims(), [batch_size, seq_len, 24]);
    assert_eq!(hidden_shape.dims(), [1, batch_size, 24]);
}

#[test]
fn test_gru_forward_bidirectional() {
    let mut gru = GruBuilder::new(10, 20)
        .num_layers(1)
        .bidirectional(true)
        .dropout(0.0)
        .batch_first(false)
        .build()
        .unwrap();

    let seq_len = 7;
    let batch_size = 2;
    let input = Tensor::randn(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[seq_len, batch_size, 10]),
        &Device::cpu(),
        false,
    )
    .unwrap();

    let (output, hidden_states) = gru.forward(input).unwrap();

    let output_shape = output.shape().unwrap();
    let hidden_shape = hidden_states.shape().unwrap();

    assert_eq!(output_shape.dims(), [seq_len, batch_size, 40]);
    assert_eq!(hidden_shape.dims(), [2, batch_size, 20]);
}

#[test]
fn test_gru_forward_multilayer_bidirectional_dropout() {
    let mut gru = GruBuilder::new(16, 32)
        .num_layers(3)
        .bidirectional(true)
        .dropout(0.3)
        .batch_first(false)
        .build()
        .unwrap();

    let seq_len = 8;
    let batch_size = 2;
    let input = Tensor::randn(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[seq_len, batch_size, 16]),
        &Device::cpu(),
        false,
    )
    .unwrap();

    let (output, hidden_states) = gru.forward(input).unwrap();

    let output_shape = output.shape().unwrap();
    let hidden_shape = hidden_states.shape().unwrap();

    assert_eq!(output_shape.dims(), [seq_len, batch_size, 64]);
    assert_eq!(hidden_shape.dims(), [6, batch_size, 32]);
}

#[test]
fn test_gru_forward_single_layer_single_direction() {
    let mut gru = GruBuilder::new(8, 16)
        .num_layers(1)
        .bidirectional(false)
        .dropout(0.0)
        .batch_first(false)
        .build()
        .unwrap();

    let seq_len = 4;
    let batch_size = 2;
    let input = Tensor::randn(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[seq_len, batch_size, 8]),
        &Device::cpu(),
        false,
    )
    .unwrap();

    let (output, hidden_states) = gru.forward(input).unwrap();

    let output_shape = output.shape().unwrap();
    let hidden_shape = hidden_states.shape().unwrap();

    assert_eq!(output_shape.dims(), [seq_len, batch_size, 16]);
    assert_eq!(hidden_shape.dims(), [1, batch_size, 16]);
}

#[test]
fn test_gru_forward_multiple_layers_hidden_state_passing() {
    let mut gru = GruBuilder::new(10, 20)
        .num_layers(3)
        .bidirectional(false)
        .dropout(0.0)
        .batch_first(false)
        .build()
        .unwrap();

    let seq_len = 5;
    let batch_size = 2;
    let input = Tensor::randn(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[seq_len, batch_size, 10]),
        &Device::cpu(),
        false,
    )
    .unwrap();

    let (output, hidden_states) = gru.forward(input).unwrap();

    let output_shape = output.shape().unwrap();
    let hidden_shape = hidden_states.shape().unwrap();

    assert_eq!(output_shape.dims(), [seq_len, batch_size, 20]);
    assert_eq!(hidden_shape.dims(), [3, batch_size, 20]);
}

#[test]
fn test_gru_forward_with_dropout_between_layers() {
    let mut gru = GruBuilder::new(8, 16)
        .num_layers(3)
        .bidirectional(false)
        .dropout(0.5)
        .batch_first(false)
        .build()
        .unwrap();

    let seq_len = 4;
    let batch_size = 2;
    let input = Tensor::randn(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[seq_len, batch_size, 8]),
        &Device::cpu(),
        false,
    )
    .unwrap();

    let (output, hidden_states) = gru.forward(input).unwrap();

    let output_shape = output.shape().unwrap();
    let hidden_shape = hidden_states.shape().unwrap();

    assert_eq!(output_shape.dims(), [seq_len, batch_size, 16]);
    assert_eq!(hidden_shape.dims(), [3, batch_size, 16]);
}

#[test]
fn test_gru_parameters() {
    let gru = GruBuilder::new(10, 20)
        .num_layers(2)
        .bidirectional(true)
        .build()
        .unwrap();

    let params = gru.parameters().unwrap();
    assert!(!params.is_empty());
}

#[test]
fn test_gru_named_parameters() {
    let gru = GruBuilder::new(8, 16)
        .num_layers(1)
        .bidirectional(false)
        .build()
        .unwrap();

    let named_params = gru.named_parameters().unwrap();
    assert!(!named_params.is_empty());
}

#[test]
fn test_gru_require_grad() {
    let mut gru = GruBuilder::new(10, 20).grad_enabled(false).build().unwrap();

    let params = gru.parameters().unwrap();
    for param in params {
        assert!(!param.grad_enabled().unwrap());
    }

    gru.require_grad(true).unwrap();

    let params = gru.parameters().unwrap();
    for param in params {
        assert!(param.grad_enabled().unwrap());
    }
}

#[test]
fn test_gru_to_device() {
    let gru = GruBuilder::new(10, 20)
        .device(Device::cpu())
        .build()
        .unwrap();

    let params = gru.parameters().unwrap();
    for param in params {
        assert_eq!(param.device().unwrap(), Device::cpu());
    }
}

#[test]
fn test_gru_to_dtype() {
    let gru = GruBuilder::new(10, 20).dtype(DType::F32).build().unwrap();

    let params = gru.parameters().unwrap();
    for param in params {
        assert_eq!(param.dtype().unwrap(), DType::F32);
    }
}

#[test]
fn test_gru_forward_invalid_input_dimensions() {
    let mut gru = GruBuilder::new(10, 20).build().unwrap();

    let input = Tensor::randn(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[5, 10]),
        &Device::cpu(),
        false,
    )
    .unwrap();

    let result = gru.forward(input);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("3D input"));
}

#[test]
fn test_gru_forward_invalid_input_size() {
    let mut gru = GruBuilder::new(10, 20).build().unwrap();

    let input = Tensor::randn(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[5, 3, 8]),
        &Device::cpu(),
        false,
    )
    .unwrap();

    let result = gru.forward(input);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("input size"));
}

#[test]
fn test_gru_parameters_across_multiple_layers() {
    let gru = GruBuilder::new(10, 20)
        .num_layers(3)
        .bidirectional(true)
        .bias(true)
        .build()
        .unwrap();

    let params = gru.parameters().unwrap();
    let named_params = gru.named_parameters().unwrap();

    assert!(!params.is_empty());
    assert!(!named_params.is_empty());
}

#[test]
fn test_gru_dropout_only_between_layers() {
    let mut gru_single_layer = GruBuilder::new(10, 20)
        .num_layers(1)
        .dropout(0.5)
        .build()
        .unwrap();

    let mut gru_multi_layer = GruBuilder::new(10, 20)
        .num_layers(3)
        .dropout(0.5)
        .build()
        .unwrap();

    let seq_len = 4;
    let batch_size = 2;
    let input = Tensor::randn(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[seq_len, batch_size, 10]),
        &Device::cpu(),
        false,
    )
    .unwrap();

    let (output_single, _) = gru_single_layer.forward(input.clone()).unwrap();
    let (output_multi, _) = gru_multi_layer.forward(input).unwrap();

    let output_single_shape = output_single.shape().unwrap();
    let output_multi_shape = output_multi.shape().unwrap();

    assert_eq!(output_single_shape.dims(), [seq_len, batch_size, 20]);
    assert_eq!(output_multi_shape.dims(), [seq_len, batch_size, 20]);
}
