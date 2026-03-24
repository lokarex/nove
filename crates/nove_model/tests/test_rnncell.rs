use nove::model::{
    Model,
    layer::{Activation, RnnCellBuilder},
};
use nove::tensor::{DType, Device, Shape, Tensor};

#[test]
fn test_rnn_cell_builder_success() {
    // Test building with minimal required parameters
    let rnn_cell = RnnCellBuilder::new(10, 20).build().unwrap();

    assert_eq!(rnn_cell.input_size(), 10);
    assert_eq!(rnn_cell.hidden_size(), 20);

    // Test building with all parameters customized
    let rnn_cell = RnnCellBuilder::new(5, 8)
        .activation(Activation::relu())
        .bias_enabled(false)
        .device(Device::cpu())
        .dtype(DType::F32)
        .grad_enabled(true)
        .build()
        .unwrap();

    assert_eq!(rnn_cell.input_size(), 5);
    assert_eq!(rnn_cell.hidden_size(), 8);
    assert!(rnn_cell.bias_ih().is_none());
    assert!(rnn_cell.bias_hh().is_none());
}

#[test]
fn test_rnn_cell_builder_failure() {
    // Test with zero input size
    let result = RnnCellBuilder::new(0, 20).build();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("input_size"));

    // Test with zero hidden size
    let result = RnnCellBuilder::new(10, 0).build();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("hidden_size"));
}

#[test]
fn test_rnn_cell_forward_shape() {
    // Test forward pass with different configurations
    let mut rnn_cell = RnnCellBuilder::new(10, 20).build().unwrap();

    // Create input and hidden state tensors
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

    // Test forward pass
    let output = rnn_cell.forward((input, hidden)).unwrap();

    // Check output shape matches expected [batch_size, hidden_size]
    assert_eq!(output.shape().unwrap(), Shape::from_dims(&[batch_size, 20]));
}

#[test]
fn test_rnn_cell_forward_with_bias() {
    // Test forward pass with bias enabled (default)
    let mut rnn_cell = RnnCellBuilder::new(5, 8)
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

    let output = rnn_cell.forward((input, hidden)).unwrap();
    assert_eq!(output.shape().unwrap(), Shape::from_dims(&[3, 8]));
}

#[test]
fn test_rnn_cell_forward_without_bias() {
    // Test forward pass with bias disabled
    let mut rnn_cell = RnnCellBuilder::new(5, 8)
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

    let output = rnn_cell.forward((input, hidden)).unwrap();
    assert_eq!(output.shape().unwrap(), Shape::from_dims(&[2, 8]));
}

#[test]
fn test_rnn_cell_forward_different_activations() {
    // Test forward pass with different activation functions
    let activations = [
        Activation::tanh(),
        Activation::relu(),
        Activation::sigmoid(),
    ];

    for activation in activations {
        let mut rnn_cell = RnnCellBuilder::new(7, 4)
            .activation(activation)
            .build()
            .unwrap();

        let input = Tensor::rand(
            0.0f32,
            1.0f32,
            &Shape::from_dims(&[5, 7]),
            &Device::cpu(),
            false,
        )
        .unwrap();

        let hidden = Tensor::rand(
            0.0f32,
            1.0f32,
            &Shape::from_dims(&[5, 4]),
            &Device::cpu(),
            false,
        )
        .unwrap();

        let output = rnn_cell.forward((input, hidden)).unwrap();
        assert_eq!(output.shape().unwrap(), Shape::from_dims(&[5, 4]));
    }
}
