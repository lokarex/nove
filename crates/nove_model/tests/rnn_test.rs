use nove::model::Model;
use nove::model::nn::Activation;
use nove::model::nn::RnnBuilder;
use nove::tensor::{DType, Device, Shape, Tensor};

#[test]
fn test_rnn_builder_creation() {
    let rnn = RnnBuilder::new(10, 20)
        .num_layers(2)
        .nonlinearity(Activation::tanh())
        .bias(true)
        .batch_first(false)
        .dropout(0.0)
        .bidirectional(false)
        .device(Device::cpu())
        .dtype(DType::F32)
        .grad_enabled(true)
        .build()
        .unwrap();

    assert_eq!(rnn.input_size(), 10);
    assert_eq!(rnn.hidden_size(), 20);
    assert_eq!(rnn.num_layers(), 2);
    assert!(rnn.nonlinearity().to_string().starts_with("tanh"));
    assert_eq!(rnn.bias(), true);
    assert_eq!(rnn.batch_first(), false);
    assert_eq!(rnn.dropout_rate(), 0.0);
    assert_eq!(rnn.bidirectional(), false);
    assert_eq!(rnn.num_directions(), 1);
}

#[test]
fn test_rnn_builder_without_bias() {
    let rnn = RnnBuilder::new(5, 8).bias(false).build().unwrap();

    assert_eq!(rnn.input_size(), 5);
    assert_eq!(rnn.hidden_size(), 8);
    assert_eq!(rnn.bias(), false);
}

#[test]
fn test_rnn_builder_batch_first() {
    let rnn = RnnBuilder::new(10, 20).batch_first(true).build().unwrap();

    assert_eq!(rnn.input_size(), 10);
    assert_eq!(rnn.hidden_size(), 20);
    assert_eq!(rnn.batch_first(), true);
}

#[test]
fn test_rnn_builder_multilayer_bidirectional_dropout() {
    let rnn = RnnBuilder::new(16, 32)
        .num_layers(3)
        .bidirectional(true)
        .dropout(0.5)
        .nonlinearity(Activation::relu())
        .build()
        .unwrap();

    assert_eq!(rnn.input_size(), 16);
    assert_eq!(rnn.hidden_size(), 32);
    assert_eq!(rnn.num_layers(), 3);
    assert_eq!(rnn.bidirectional(), true);
    assert_eq!(rnn.num_directions(), 2);
    assert_eq!(rnn.dropout_rate(), 0.5);
    assert!(rnn.nonlinearity().to_string().starts_with("relu"));
}

#[test]
fn test_rnn_builder_invalid_input_size() {
    let result = RnnBuilder::new(0, 20).build();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("input_size"));
}

#[test]
fn test_rnn_builder_invalid_hidden_size() {
    let result = RnnBuilder::new(10, 0).build();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("hidden_size"));
}

#[test]
fn test_rnn_builder_invalid_num_layers() {
    let result = RnnBuilder::new(10, 20).num_layers(0).build();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("num_layers"));
}

#[test]
fn test_rnn_builder_invalid_dropout() {
    let result = RnnBuilder::new(10, 20).dropout(1.5).build();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("dropout"));
}

#[test]
fn test_rnn_forward_batch_last() {
    let mut rnn = RnnBuilder::new(8, 16)
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

    let (output, hidden_states) = rnn.forward(input).unwrap();

    let output_shape = output.shape().unwrap();
    let hidden_shape = hidden_states.shape().unwrap();

    assert_eq!(output_shape.dims(), [seq_len, batch_size, 16]);
    assert_eq!(hidden_shape.dims(), [2, batch_size, 16]);
}

#[test]
fn test_rnn_forward_batch_first() {
    let mut rnn = RnnBuilder::new(12, 24)
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

    let (output, hidden_states) = rnn.forward(input).unwrap();

    let output_shape = output.shape().unwrap();
    let hidden_shape = hidden_states.shape().unwrap();

    assert_eq!(output_shape.dims(), [batch_size, seq_len, 24]);
    assert_eq!(hidden_shape.dims(), [1, batch_size, 24]);
}

#[test]
fn test_rnn_forward_bidirectional() {
    let mut rnn = RnnBuilder::new(10, 20)
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

    let (output, hidden_states) = rnn.forward(input).unwrap();

    let output_shape = output.shape().unwrap();
    let hidden_shape = hidden_states.shape().unwrap();

    assert_eq!(output_shape.dims(), [seq_len, batch_size, 40]);
    assert_eq!(hidden_shape.dims(), [2, batch_size, 20]);
}

#[test]
fn test_rnn_forward_multilayer_bidirectional_dropout() {
    let mut rnn = RnnBuilder::new(16, 32)
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

    let (output, hidden_states) = rnn.forward(input).unwrap();

    let output_shape = output.shape().unwrap();
    let hidden_shape = hidden_states.shape().unwrap();

    assert_eq!(output_shape.dims(), [seq_len, batch_size, 64]);
    assert_eq!(hidden_shape.dims(), [6, batch_size, 32]);
}

#[test]
fn test_rnn_forward_with_different_activations() {
    let activations = vec![Activation::tanh(), Activation::relu()];

    for activation in activations {
        let mut rnn = RnnBuilder::new(8, 16)
            .num_layers(1)
            .bidirectional(false)
            .dropout(0.0)
            .nonlinearity(activation.clone())
            .batch_first(false)
            .build()
            .unwrap();

        let seq_len = 3;
        let batch_size = 2;
        let input = Tensor::randn(
            0.0f32,
            1.0f32,
            &Shape::from_dims(&[seq_len, batch_size, 8]),
            &Device::cpu(),
            false,
        )
        .unwrap();

        let (output, hidden_states) = rnn.forward(input).unwrap();

        let output_shape = output.shape().unwrap();
        let hidden_shape = hidden_states.shape().unwrap();

        assert_eq!(output_shape.dims(), [seq_len, batch_size, 16]);
        assert_eq!(hidden_shape.dims(), [1, batch_size, 16]);
        assert_eq!(rnn.nonlinearity().to_string(), activation.to_string());
    }
}

#[test]
fn test_rnn_parameters() {
    let rnn = RnnBuilder::new(10, 20)
        .num_layers(2)
        .bidirectional(true)
        .build()
        .unwrap();

    let params = rnn.parameters().unwrap();
    assert!(!params.is_empty());
}

#[test]
fn test_rnn_named_parameters() {
    let rnn = RnnBuilder::new(8, 16)
        .num_layers(1)
        .bidirectional(false)
        .build()
        .unwrap();

    let named_params = rnn.named_parameters().unwrap();
    assert!(!named_params.is_empty());
}

#[test]
fn test_rnn_require_grad() {
    let mut rnn = RnnBuilder::new(10, 20).grad_enabled(false).build().unwrap();

    let params = rnn.parameters().unwrap();
    for param in params {
        assert!(!param.grad_enabled().unwrap());
    }

    rnn.require_grad(true).unwrap();

    let params = rnn.parameters().unwrap();
    for param in params {
        assert!(param.grad_enabled().unwrap());
    }
}

#[test]
fn test_rnn_to_device() {
    let rnn = RnnBuilder::new(10, 20)
        .device(Device::cpu())
        .build()
        .unwrap();

    let params = rnn.parameters().unwrap();
    for param in params {
        assert_eq!(param.device().unwrap(), Device::cpu());
    }
}

#[test]
fn test_rnn_to_dtype() {
    let rnn = RnnBuilder::new(10, 20).dtype(DType::F32).build().unwrap();

    let params = rnn.parameters().unwrap();
    for param in params {
        assert_eq!(param.dtype().unwrap(), DType::F32);
    }
}

#[test]
fn test_rnn_forward_invalid_input_dimensions() {
    let mut rnn = RnnBuilder::new(10, 20).build().unwrap();

    let input = Tensor::randn(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[5, 10]),
        &Device::cpu(),
        false,
    )
    .unwrap();

    let result = rnn.forward(input);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("3D input"));
}

#[test]
fn test_rnn_forward_invalid_input_size() {
    let mut rnn = RnnBuilder::new(10, 20).build().unwrap();

    let input = Tensor::randn(
        0.0f32,
        1.0f32,
        &Shape::from_dims(&[5, 3, 8]),
        &Device::cpu(),
        false,
    )
    .unwrap();

    let result = rnn.forward(input);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("input size"));
}
