use nove::model::Model;
use nove::model::nn::{AvgPool2d, MaxPool2d, Pool2d};
use nove::tensor::{DType, Device, Shape, Tensor};

#[test]
fn test_avg_pool2d_creation() {
    let avg_pool = AvgPool2d::new((2, 2), None).unwrap();

    assert_eq!(avg_pool.kernel_size(), (2, 2));
    assert_eq!(avg_pool.stride(), (2, 2));

    let avg_pool_with_stride = AvgPool2d::new((3, 3), Some((1, 2))).unwrap();
    assert_eq!(avg_pool_with_stride.kernel_size(), (3, 3));
    assert_eq!(avg_pool_with_stride.stride(), (1, 2));
}

#[test]
fn test_max_pool2d_creation() {
    let max_pool = MaxPool2d::new((2, 2), None).unwrap();

    assert_eq!(max_pool.kernel_size(), (2, 2));
    assert_eq!(max_pool.stride(), (2, 2));

    let max_pool_with_stride = MaxPool2d::new((3, 3), Some((2, 1))).unwrap();
    assert_eq!(max_pool_with_stride.kernel_size(), (3, 3));
    assert_eq!(max_pool_with_stride.stride(), (2, 1));
}

#[test]
fn test_avg_pool2d_invalid_arguments() {
    let result = AvgPool2d::new((0, 2), None);
    assert!(result.is_err());

    let result = AvgPool2d::new((2, 0), None);
    assert!(result.is_err());

    let result = AvgPool2d::new((1, 1), Some((0, 1)));
    assert!(result.is_err());
}

#[test]
fn test_max_pool2d_invalid_arguments() {
    let result = MaxPool2d::new((0, 2), None);
    assert!(result.is_err());

    let result = MaxPool2d::new((2, 0), None);
    assert!(result.is_err());

    let result = MaxPool2d::new((1, 1), Some((1, 0)));
    assert!(result.is_err());
}

#[test]
fn test_avg_pool2d_forward_shape() {
    let mut avg_pool = AvgPool2d::new((2, 2), None).unwrap();

    let input = Tensor::ones(
        &Shape::from_dims(&[2, 3, 8, 8]),
        &DType::F32,
        &Device::cpu(),
        false,
    )
    .unwrap();
    let output = avg_pool.forward(input).unwrap();

    assert_eq!(output.shape().unwrap(), Shape::from_dims(&[2, 3, 4, 4]));
}

#[test]
fn test_avg_pool2d_forward_with_asymmetric_kernel_shape() {
    let mut avg_pool = AvgPool2d::new((2, 3), None).unwrap();

    let input = Tensor::ones(
        &Shape::from_dims(&[1, 2, 6, 9]),
        &DType::F32,
        &Device::cpu(),
        false,
    )
    .unwrap();
    let output = avg_pool.forward(input).unwrap();

    assert_eq!(output.shape().unwrap(), Shape::from_dims(&[1, 2, 3, 3]));
}

#[test]
fn test_avg_pool2d_forward_with_asymmetric_stride_shape() {
    let mut avg_pool = AvgPool2d::new((2, 2), Some((1, 2))).unwrap();

    let input = Tensor::ones(
        &Shape::from_dims(&[1, 2, 6, 8]),
        &DType::F32,
        &Device::cpu(),
        false,
    )
    .unwrap();
    let output = avg_pool.forward(input).unwrap();

    assert_eq!(output.shape().unwrap(), Shape::from_dims(&[1, 2, 5, 4]));
}

#[test]
fn test_max_pool2d_forward_shape() {
    let mut max_pool = MaxPool2d::new((2, 2), None).unwrap();

    let input = Tensor::ones(
        &Shape::from_dims(&[2, 3, 8, 8]),
        &DType::F32,
        &Device::cpu(),
        false,
    )
    .unwrap();
    let output = max_pool.forward(input).unwrap();

    assert_eq!(output.shape().unwrap(), Shape::from_dims(&[2, 3, 4, 4]));
}

#[test]
fn test_max_pool2d_forward_with_asymmetric_kernel_shape() {
    let mut max_pool = MaxPool2d::new((3, 2), None).unwrap();

    let input = Tensor::ones(
        &Shape::from_dims(&[1, 2, 9, 6]),
        &DType::F32,
        &Device::cpu(),
        false,
    )
    .unwrap();
    let output = max_pool.forward(input).unwrap();

    assert_eq!(output.shape().unwrap(), Shape::from_dims(&[1, 2, 3, 3]));
}

#[test]
fn test_max_pool2d_forward_with_asymmetric_stride_shape() {
    let mut max_pool = MaxPool2d::new((2, 2), Some((2, 1))).unwrap();

    let input = Tensor::ones(
        &Shape::from_dims(&[1, 2, 8, 6]),
        &DType::F32,
        &Device::cpu(),
        false,
    )
    .unwrap();
    let output = max_pool.forward(input).unwrap();

    assert_eq!(output.shape().unwrap(), Shape::from_dims(&[1, 2, 4, 5]));
}

#[test]
fn test_avg_pool2d_forward_values() {
    let mut avg_pool = AvgPool2d::new((2, 2), None).unwrap();

    let data = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let input = Tensor::from_slice(
        &data,
        &Shape::from_dims(&[1, 1, 4, 4]),
        &Device::cpu(),
        false,
    )
    .unwrap();
    let output = avg_pool.forward(input).unwrap();

    let output_values = output.to_vec::<f32>().unwrap();
    assert_eq!(output_values, vec![3.5, 5.5, 11.5, 13.5]);
}

#[test]
fn test_max_pool2d_forward_values() {
    let mut max_pool = MaxPool2d::new((2, 2), None).unwrap();

    let data = vec![
        1.0f32, 3.0, 2.0, 4.0, 5.0, 8.0, 6.0, 7.0, 9.0, 11.0, 10.0, 12.0, 13.0, 15.0, 14.0, 16.0,
    ];
    let input = Tensor::from_slice(
        &data,
        &Shape::from_dims(&[1, 1, 4, 4]),
        &Device::cpu(),
        false,
    )
    .unwrap();
    let output = max_pool.forward(input).unwrap();

    let output_values = output.to_vec::<f32>().unwrap();
    assert_eq!(output_values, vec![8.0, 7.0, 15.0, 16.0]);
}

#[test]
fn test_pool2d_enum_creation() {
    let avg_pool = Pool2d::avg_pool2d((2, 2), None).unwrap();
    let max_pool = Pool2d::max_pool2d((3, 3), Some((2, 1))).unwrap();

    match avg_pool {
        Pool2d::AvgPool2d(layer) => {
            assert_eq!(layer.kernel_size(), (2, 2));
            assert_eq!(layer.stride(), (2, 2));
        }
        _ => panic!("Expected AvgPool2d"),
    }

    match max_pool {
        Pool2d::MaxPool2d(layer) => {
            assert_eq!(layer.kernel_size(), (3, 3));
            assert_eq!(layer.stride(), (2, 1));
        }
        _ => panic!("Expected MaxPool2d"),
    }
}

#[test]
fn test_pool2d_enum_forward() {
    let mut avg_pool = Pool2d::avg_pool2d((2, 2), None).unwrap();
    let input = Tensor::ones(
        &Shape::from_dims(&[1, 1, 6, 6]),
        &DType::F32,
        &Device::cpu(),
        false,
    )
    .unwrap();
    let output = avg_pool.forward(input).unwrap();

    assert_eq!(output.shape().unwrap(), Shape::from_dims(&[1, 1, 3, 3]));

    let mut max_pool = Pool2d::max_pool2d((3, 2), Some((1, 2))).unwrap();
    let input2 = Tensor::ones(
        &Shape::from_dims(&[1, 1, 9, 8]),
        &DType::F32,
        &Device::cpu(),
        false,
    )
    .unwrap();
    let output2 = max_pool.forward(input2).unwrap();

    assert_eq!(output2.shape().unwrap(), Shape::from_dims(&[1, 1, 7, 4]));
}

#[test]
fn test_pool2d_parameters_empty() {
    let avg_pool2d = AvgPool2d::new((2, 2), None).unwrap();
    let params = avg_pool2d.parameters().unwrap();
    assert_eq!(params.len(), 0);

    let max_pool2d = MaxPool2d::new((2, 2), None).unwrap();
    let params = max_pool2d.parameters().unwrap();
    assert_eq!(params.len(), 0);
}

#[test]
fn test_pool2d_require_grad_noop() {
    let mut avg_pool2d = AvgPool2d::new((2, 2), None).unwrap();
    avg_pool2d.require_grad(true).unwrap();

    let mut max_pool2d = MaxPool2d::new((2, 2), None).unwrap();
    max_pool2d.require_grad(false).unwrap();
}

#[test]
fn test_pool2d_to_device_noop() {
    let mut avg_pool2d = AvgPool2d::new((2, 2), None).unwrap();
    avg_pool2d.to_device(&Device::cpu()).unwrap();

    let mut max_pool2d = MaxPool2d::new((2, 2), None).unwrap();
    max_pool2d.to_device(&Device::cpu()).unwrap();
}

#[test]
fn test_pool2d_to_dtype_noop() {
    let mut avg_pool2d = AvgPool2d::new((2, 2), None).unwrap();
    avg_pool2d.to_dtype(&DType::F32).unwrap();

    let mut max_pool2d = MaxPool2d::new((2, 2), None).unwrap();
    max_pool2d.to_dtype(&DType::F32).unwrap();
}

#[test]
fn test_pool2d_display() {
    let avg_pool2d = AvgPool2d::new((2, 3), None).unwrap();
    let display_str = format!("{}", avg_pool2d);
    assert!(display_str.contains("avgpool2d"));
    assert!(display_str.contains("kernel_size=(2, 3)"));
    assert!(display_str.contains("stride=(2, 3)"));

    let max_pool2d = MaxPool2d::new((3, 2), Some((1, 2))).unwrap();
    let display_str = format!("{}", max_pool2d);
    assert!(display_str.contains("maxpool2d"));
    assert!(display_str.contains("kernel_size=(3, 2)"));
    assert!(display_str.contains("stride=(1, 2)"));
}
