use nove::model::Model;
use nove::model::layer::Activation;
use nove::tensor::{DType, Device, Shape, Tensor};

#[test]
fn test_activation_relu_forward() {
    let mut relu = Activation::relu();
    let input = Tensor::from_data(vec![-1.0, 2.0, -3.0, 4.0], &Device::cpu(), false).unwrap();
    let output = relu.forward(input).unwrap();
    let output_vec = output.to_vec::<f64>().unwrap();
    assert_eq!(output_vec, vec![0.0, 2.0, 0.0, 4.0]);
}

#[test]
fn test_activation_gelu_forward() {
    let mut gelu = Activation::gelu();
    let input = Tensor::from_data(vec![-1.0, 0.0, 1.0], &Device::cpu(), false).unwrap();
    let output = gelu.forward(input).unwrap();
    let output_vec = output.to_vec::<f64>().unwrap();
    assert_eq!(
        output_vec,
        vec![-0.15880800939172324, 0.0, 0.8411919906082768]
    );
}

#[test]
fn test_activation_silu_forward() {
    let mut silu = Activation::silu();
    let input = Tensor::from_data(vec![-1.0, 0.0, 1.0], &Device::cpu(), false).unwrap();
    let output = silu.forward(input).unwrap();
    let output_vec = output.to_vec::<f64>().unwrap();
    assert_eq!(
        output_vec,
        vec![-0.2689414213699951, 0.0, 0.7310585786300049]
    );
}

#[test]
fn test_activation_sigmoid_forward() {
    let mut sigmoid = Activation::sigmoid();
    let input = Tensor::from_data(vec![-1.0, 0.0, 1.0], &Device::cpu(), false).unwrap();
    let output = sigmoid.forward(input).unwrap();
    let output_vec = output.to_vec::<f64>().unwrap();
    assert_eq!(
        output_vec,
        vec![0.2689414213699951, 0.5, 0.7310585786300049]
    );
}

#[test]
fn test_activation_tanh_forward() {
    let mut tanh = Activation::tanh();
    let input = Tensor::from_data(vec![-1.0, 0.0, 1.0], &Device::cpu(), false).unwrap();
    let output = tanh.forward(input).unwrap();
    let output_vec = output.to_vec::<f64>().unwrap();
    assert_eq!(
        output_vec,
        vec![-0.7615941559557649, 0.0, 0.7615941559557649]
    );
}

#[test]
fn test_activation_relu_require_grad() {
    let mut relu = Activation::relu();
    relu.require_grad(true).unwrap();
    relu.require_grad(false).unwrap();
}

#[test]
fn test_activation_gelu_require_grad() {
    let mut gelu = Activation::gelu();
    gelu.require_grad(true).unwrap();
    gelu.require_grad(false).unwrap();
}

#[test]
fn test_activation_silu_require_grad() {
    let mut silu = Activation::silu();
    silu.require_grad(true).unwrap();
    silu.require_grad(false).unwrap();
}

#[test]
fn test_activation_sigmoid_require_grad() {
    let mut sigmoid = Activation::sigmoid();
    sigmoid.require_grad(true).unwrap();
    sigmoid.require_grad(false).unwrap();
}

#[test]
fn test_activation_tanh_require_grad() {
    let mut tanh = Activation::tanh();
    tanh.require_grad(true).unwrap();
    tanh.require_grad(false).unwrap();
}

#[test]
fn test_activation_relu_different_shapes() {
    let mut relu = Activation::relu();

    let input_1d = Tensor::from_data(vec![-1.0, 2.0, -3.0], &Device::cpu(), false).unwrap();
    let output_1d = relu.forward(input_1d).unwrap();
    assert_eq!(output_1d.shape().unwrap(), Shape::from_dims(&[3]));

    let mut relu2 = Activation::relu();
    let input_2d = Tensor::from_data(vec![-1.0, 2.0, -3.0, 4.0], &Device::cpu(), false)
        .unwrap()
        .reshape(&Shape::from_dims(&[2, 2]))
        .unwrap();
    let output_2d = relu2.forward(input_2d).unwrap();
    assert_eq!(output_2d.shape().unwrap(), Shape::from_dims(&[2, 2]));

    let mut relu3 = Activation::relu();
    let input_3d = Tensor::ones(
        &Shape::from_dims(&[2, 3, 4]),
        &DType::F32,
        &Device::cpu(),
        false,
    )
    .unwrap();
    let output_3d = relu3.forward(input_3d).unwrap();
    assert_eq!(output_3d.shape().unwrap(), Shape::from_dims(&[2, 3, 4]));
}

#[test]
fn test_activation_gelu_different_shapes() {
    let mut gelu = Activation::gelu();

    let input_1d = Tensor::from_data(vec![-1.0, 0.0, 1.0], &Device::cpu(), false).unwrap();
    let output_1d = gelu.forward(input_1d).unwrap();
    assert_eq!(output_1d.shape().unwrap(), Shape::from_dims(&[3]));

    let mut gelu2 = Activation::gelu();
    let input_2d = Tensor::from_data(vec![-1.0, 0.0, 1.0, 2.0], &Device::cpu(), false)
        .unwrap()
        .reshape(&Shape::from_dims(&[2, 2]))
        .unwrap();
    let output_2d = gelu2.forward(input_2d).unwrap();
    assert_eq!(output_2d.shape().unwrap(), Shape::from_dims(&[2, 2]));
}

#[test]
fn test_activation_silu_different_shapes() {
    let mut silu = Activation::silu();

    let input_1d = Tensor::from_data(vec![-1.0, 0.0, 1.0], &Device::cpu(), false).unwrap();
    let output_1d = silu.forward(input_1d).unwrap();
    assert_eq!(output_1d.shape().unwrap(), Shape::from_dims(&[3]));

    let mut silu2 = Activation::silu();
    let input_2d = Tensor::from_data(vec![-1.0, 0.0, 1.0, 2.0], &Device::cpu(), false)
        .unwrap()
        .reshape(&Shape::from_dims(&[2, 2]))
        .unwrap();
    let output_2d = silu2.forward(input_2d).unwrap();
    assert_eq!(output_2d.shape().unwrap(), Shape::from_dims(&[2, 2]));
}

#[test]
fn test_activation_sigmoid_different_shapes() {
    let mut sigmoid = Activation::sigmoid();

    let input_1d = Tensor::from_data(vec![-1.0, 0.0, 1.0], &Device::cpu(), false).unwrap();
    let output_1d = sigmoid.forward(input_1d).unwrap();
    assert_eq!(output_1d.shape().unwrap(), Shape::from_dims(&[3]));

    let mut sigmoid2 = Activation::sigmoid();
    let input_2d = Tensor::from_data(vec![-1.0, 0.0, 1.0, 2.0], &Device::cpu(), false)
        .unwrap()
        .reshape(&Shape::from_dims(&[2, 2]))
        .unwrap();
    let output_2d = sigmoid2.forward(input_2d).unwrap();
    assert_eq!(output_2d.shape().unwrap(), Shape::from_dims(&[2, 2]));
}

#[test]
fn test_activation_tanh_different_shapes() {
    let mut tanh = Activation::tanh();

    let input_1d = Tensor::from_data(vec![-1.0, 0.0, 1.0], &Device::cpu(), false).unwrap();
    let output_1d = tanh.forward(input_1d).unwrap();
    assert_eq!(output_1d.shape().unwrap(), Shape::from_dims(&[3]));

    let mut tanh2 = Activation::tanh();
    let input_2d = Tensor::from_data(vec![-1.0, 0.0, 1.0, 2.0], &Device::cpu(), false)
        .unwrap()
        .reshape(&Shape::from_dims(&[2, 2]))
        .unwrap();
    let output_2d = tanh2.forward(input_2d).unwrap();
    assert_eq!(output_2d.shape().unwrap(), Shape::from_dims(&[2, 2]));
}

#[test]
fn test_activation_parameters() {
    let relu = Activation::relu();
    let params = relu.parameters().unwrap();
    assert_eq!(params.len(), 0);

    let gelu = Activation::gelu();
    let params = gelu.parameters().unwrap();
    assert_eq!(params.len(), 0);

    let silu = Activation::silu();
    let params = silu.parameters().unwrap();
    assert_eq!(params.len(), 0);

    let sigmoid = Activation::sigmoid();
    let params = sigmoid.parameters().unwrap();
    assert_eq!(params.len(), 0);

    let tanh = Activation::tanh();
    let params = tanh.parameters().unwrap();
    assert_eq!(params.len(), 0);
}

#[test]
fn test_activation_named_parameters() {
    let relu = Activation::relu();
    let named_params = relu.named_parameters().unwrap();
    assert_eq!(named_params.len(), 0);

    let gelu = Activation::gelu();
    let named_params = gelu.named_parameters().unwrap();
    assert_eq!(named_params.len(), 0);

    let silu = Activation::silu();
    let named_params = silu.named_parameters().unwrap();
    assert_eq!(named_params.len(), 0);

    let sigmoid = Activation::sigmoid();
    let named_params = sigmoid.named_parameters().unwrap();
    assert_eq!(named_params.len(), 0);

    let tanh = Activation::tanh();
    let named_params = tanh.named_parameters().unwrap();
    assert_eq!(named_params.len(), 0);
}

#[test]
fn test_activation_to_device() {
    let mut relu = Activation::relu();
    relu.to_device(&Device::cpu()).unwrap();

    let mut gelu = Activation::gelu();
    gelu.to_device(&Device::cpu()).unwrap();

    let mut silu = Activation::silu();
    silu.to_device(&Device::cpu()).unwrap();

    let mut sigmoid = Activation::sigmoid();
    sigmoid.to_device(&Device::cpu()).unwrap();

    let mut tanh = Activation::tanh();
    tanh.to_device(&Device::cpu()).unwrap();
}

#[test]
fn test_activation_to_dtype() {
    let mut relu = Activation::relu();
    relu.to_dtype(&DType::F32).unwrap();

    let mut gelu = Activation::gelu();
    gelu.to_dtype(&DType::F32).unwrap();

    let mut silu = Activation::silu();
    silu.to_dtype(&DType::F32).unwrap();

    let mut sigmoid = Activation::sigmoid();
    sigmoid.to_dtype(&DType::F32).unwrap();

    let mut tanh = Activation::tanh();
    tanh.to_dtype(&DType::F32).unwrap();
}

#[test]
fn test_activation_display() {
    let relu = Activation::relu();
    let display_str = format!("{}", relu);
    assert!(display_str.contains("relu"));

    let gelu = Activation::gelu();
    let display_str = format!("{}", gelu);
    assert!(display_str.contains("gelu"));

    let silu = Activation::silu();
    let display_str = format!("{}", silu);
    assert!(display_str.contains("silu"));

    let sigmoid = Activation::sigmoid();
    let display_str = format!("{}", sigmoid);
    assert!(display_str.contains("sigmoid"));

    let tanh = Activation::tanh();
    let display_str = format!("{}", tanh);
    assert!(display_str.contains("tanh"));
}
