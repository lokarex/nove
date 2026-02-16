use nove::model::layer::ReLU;
use nove_model::Model;
use nove_tensor::{Device, Tensor};

#[test]
fn test_relu() {
    let mut relu = ReLU::new();
    let input = Tensor::from_data(
        vec![-1.0f32, 2.0f32, -3.0f32, 4.0f32],
        &Device::cpu(),
        false,
    )
    .unwrap();
    let output = relu.forward(input).unwrap();
    assert_eq!(
        output.to_vec::<f32>().unwrap(),
        &[0.0f32, 2.0f32, 0.0f32, 4.0f32]
    );
}
