use nove::model::Model;
use nove::model::layer::Dropout;
use nove::tensor::{DType, Device, Shape, Tensor};

#[test]
fn test_dropout_creation() {
    let dropout = Dropout::new(0.5).unwrap();
    assert_eq!(dropout.to_string().contains("probability=0.5"), true);
}

#[test]
fn test_dropout_creation_invalid_probability() {
    assert!(Dropout::new(-0.1).is_err());
    assert!(Dropout::new(1.0).is_err());
    assert!(Dropout::new(1.5).is_err());
}

#[test]
fn test_dropout_training_vs_inference_mode() {
    let mut dropout = Dropout::new(0.5).unwrap();
    let device = Device::cpu();

    let input = Tensor::ones(&Shape::from_dims(&[4, 3]), &DType::F32, &device, false).unwrap();

    let training_output = dropout.forward((input.copy(), true)).unwrap();
    let inference_output = dropout.forward((input.copy(), false)).unwrap();

    assert_ne!(
        training_output.to_vec::<f32>().unwrap(),
        input.to_vec::<f32>().unwrap()
    );
    assert_eq!(
        inference_output.to_vec::<f32>().unwrap(),
        input.to_vec::<f32>().unwrap()
    );
}

#[test]
fn test_dropout_probability_zero() {
    let mut dropout = Dropout::new(0.0).unwrap();
    let device = Device::cpu();

    let input = Tensor::ones(&Shape::from_dims(&[2, 2]), &DType::F32, &device, false).unwrap();
    let training_output = dropout.forward((input.copy(), true)).unwrap();

    assert_eq!(
        training_output.to_vec::<f32>().unwrap(),
        input.to_vec::<f32>().unwrap()
    );
}

#[test]
fn test_dropout_probability_high() {
    let mut dropout = Dropout::new(0.9).unwrap();
    let device = Device::cpu();

    let input = Tensor::ones(&Shape::from_dims(&[10]), &DType::F32, &device, false).unwrap();
    let training_output = dropout.forward((input, true)).unwrap();

    let output_values = training_output.to_vec::<f32>().unwrap();
    let zero_count = output_values.iter().filter(|&&x| x == 0.0).count();
    assert!(zero_count > 0);
}

#[test]
fn test_dropout_scaling() {
    let mut dropout = Dropout::new(0.5).unwrap();
    let device = Device::cpu();

    let input_data = [1.0f32, 2.0, 3.0, 4.0];
    let input = Tensor::from_data(&input_data, &device, false).unwrap();
    let training_output = dropout.forward((input.copy(), true)).unwrap();

    let output_values = training_output.to_vec::<f32>().unwrap();
    let input_values = input.to_vec::<f32>().unwrap();

    for (i, &value) in output_values.iter().enumerate() {
        if value != 0.0 {
            assert!((value - 2.0 * input_values[i]).abs() < 1e-6);
        }
    }
}

#[test]
fn test_dropout_require_grad() {
    let mut dropout = Dropout::new(0.3).unwrap();
    assert!(dropout.require_grad(true).is_ok());
    assert!(dropout.require_grad(false).is_ok());
}

#[test]
fn test_dropout_to_device() {
    let mut dropout = Dropout::new(0.4).unwrap();
    assert!(dropout.to_device(&Device::cpu()).is_ok());
}

#[test]
fn test_dropout_to_dtype() {
    let mut dropout = Dropout::new(0.2).unwrap();
    assert!(dropout.to_dtype(&DType::F32).is_ok());
}

#[test]
fn test_dropout_parameters() {
    let dropout = Dropout::new(0.6).unwrap();
    let params = dropout.parameters().unwrap();
    assert_eq!(params.len(), 0);

    let named_params = dropout.named_parameters().unwrap();
    assert_eq!(named_params.len(), 0);
}

#[test]
fn test_dropout_display() {
    let dropout1 = Dropout::new(0.25).unwrap();
    let dropout2 = Dropout::new(0.75).unwrap();

    let display1 = dropout1.to_string();
    let display2 = dropout2.to_string();

    assert!(display1.contains("dropout"));
    assert!(display1.contains("probability=0.25"));
    assert!(display2.contains("dropout"));
    assert!(display2.contains("probability=0.75"));
    assert_ne!(display1, display2);
}
