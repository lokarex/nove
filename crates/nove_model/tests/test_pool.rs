use nove::model::layer::{AvgPool2d, MaxPool2d};
use nove_model::Model;
use nove_tensor::{Device, Shape, Tensor};

#[test]
fn test_max_pool2d() {
    let mut max_pool = MaxPool2d::new((2, 2), None).unwrap();
    let input = Tensor::from_data(
        vec![
            1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32, 9.0f32,
            10.0f32, 11.0f32, 12.0f32, 13.0f32, 14.0f32, 15.0f32, 16.0f32,
        ],
        &Device::cpu(),
        false,
    )
    .unwrap()
    .reshape(&Shape::from_dims(&[1, 1, 4, 4]))
    .unwrap();
    let output = max_pool.forward(input).unwrap();
    let result = output.to_vec::<f32>().unwrap();
    assert_eq!(result, &[6.0f32, 8.0f32, 14.0f32, 16.0f32]);
}

#[test]
fn test_max_pool2d_with_stride() {
    let mut max_pool = MaxPool2d::new((2, 2), Some((1, 1))).unwrap();
    let input = Tensor::from_data(
        vec![
            1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32, 9.0f32,
            10.0f32, 11.0f32, 12.0f32, 13.0f32, 14.0f32, 15.0f32, 16.0f32,
        ],
        &Device::cpu(),
        false,
    )
    .unwrap()
    .reshape(&Shape::from_dims(&[1, 1, 4, 4]))
    .unwrap();
    let output = max_pool.forward(input).unwrap();
    let result = output.to_vec::<f32>().unwrap();
    assert_eq!(
        result,
        &[
            6.0f32, 7.0f32, 8.0f32, 10.0f32, 11.0f32, 12.0f32, 14.0f32, 15.0f32, 16.0f32
        ]
    );
}

#[test]
fn test_avg_pool2d() {
    let mut avg_pool = AvgPool2d::new((2, 2), None).unwrap();
    let input = Tensor::from_data(
        vec![
            1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32, 9.0f32,
            10.0f32, 11.0f32, 12.0f32, 13.0f32, 14.0f32, 15.0f32, 16.0f32,
        ],
        &Device::cpu(),
        false,
    )
    .unwrap()
    .reshape(&Shape::from_dims(&[1, 1, 4, 4]))
    .unwrap();
    let output = avg_pool.forward(input).unwrap();
    let result = output.to_vec::<f32>().unwrap();
    assert_eq!(result, &[3.5f32, 5.5f32, 11.5f32, 13.5f32]);
}

#[test]
fn test_avg_pool2d_with_stride() {
    let mut avg_pool = AvgPool2d::new((2, 2), Some((1, 1))).unwrap();
    let input = Tensor::from_data(
        vec![
            1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32, 9.0f32,
            10.0f32, 11.0f32, 12.0f32, 13.0f32, 14.0f32, 15.0f32, 16.0f32,
        ],
        &Device::cpu(),
        false,
    )
    .unwrap()
    .reshape(&Shape::from_dims(&[1, 1, 4, 4]))
    .unwrap();
    let output = avg_pool.forward(input).unwrap();
    let result = output.to_vec::<f32>().unwrap();
    assert_eq!(
        result,
        &[
            3.5f32, 4.5f32, 5.5f32, 7.5f32, 8.5f32, 9.5f32, 11.5f32, 12.5f32, 13.5f32
        ]
    );
}

#[test]
fn test_max_pool2d_invalid_kernel_size() {
    let result = MaxPool2d::new((0, 2), None);
    assert!(result.is_err());
}

#[test]
fn test_avg_pool2d_invalid_kernel_size() {
    let result = AvgPool2d::new((2, 0), None);
    assert!(result.is_err());
}

#[test]
fn test_max_pool2d_invalid_stride() {
    let result = MaxPool2d::new((2, 2), Some((0, 1)));
    assert!(result.is_err());
}

#[test]
fn test_avg_pool2d_invalid_stride() {
    let result = AvgPool2d::new((2, 2), Some((1, 0)));
    assert!(result.is_err());
}

#[test]
fn test_max_pool2d_getters() {
    let max_pool = MaxPool2d::new((3, 3), Some((2, 2))).unwrap();
    assert_eq!(max_pool.kernel_size(), (3, 3));
    assert_eq!(max_pool.stride(), (2, 2));
}

#[test]
fn test_avg_pool2d_getters() {
    let avg_pool = AvgPool2d::new((3, 3), Some((2, 2))).unwrap();
    assert_eq!(avg_pool.kernel_size(), (3, 3));
    assert_eq!(avg_pool.stride(), (2, 2));
}

#[test]
fn test_max_pool2d_default_stride() {
    let max_pool = MaxPool2d::new((2, 2), None).unwrap();
    assert_eq!(max_pool.kernel_size(), (2, 2));
    assert_eq!(max_pool.stride(), (2, 2));
}

#[test]
fn test_avg_pool2d_default_stride() {
    let avg_pool = AvgPool2d::new((2, 2), None).unwrap();
    assert_eq!(avg_pool.kernel_size(), (2, 2));
    assert_eq!(avg_pool.stride(), (2, 2));
}
