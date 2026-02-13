use nove::model::Model;
use nove::model::layer::Dropout;
use nove::tensor::{Device, Tensor};

#[test]
fn test_dropout_with_training() {
    let mut dropout = Dropout::new(0.5).unwrap();
    let data = vec![
        0.0f32, 1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32, 9.0f32,
    ];
    let input = Tensor::from_data(data.clone(), &Device::cpu(), false).unwrap();
    let output = dropout.forward((input, true)).unwrap();
    assert_eq!(output.shape().unwrap().dims(), &[10]);

    // Scale factor = 1.0 / (1.0 - dropout_rate) = 1.0 / 0.5 = 2.0
    let scale = 2.0;

    for (input_val, output_val) in data.iter().zip(output.to_vec::<f32>().unwrap().iter()) {
        // If the output value is not zero, it should be scaled by the scale factor.
        if *output_val != 0.0f32 {
            assert_eq!(*input_val * scale, *output_val);
        }
    }
}

#[test]
fn test_dropout_without_training() {
    let mut dropout = Dropout::new(0.5).unwrap();
    let data = vec![
        0.0f32, 1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32, 9.0f32,
    ];
    let input = Tensor::from_data(data.clone(), &Device::cpu(), false).unwrap();
    let output = dropout.forward((input, false)).unwrap();
    assert_eq!(output.shape().unwrap().dims(), &[10]);

    for (input_val, output_val) in data.iter().zip(output.to_vec::<f32>().unwrap().iter()) {
        assert_eq!(*input_val, *output_val);
    }
}
