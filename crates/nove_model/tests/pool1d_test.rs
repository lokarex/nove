use nove::model::layer::{AvgPool1d, MaxPool1d, Pool1d};
use nove::model::Model;
use nove::tensor::{Device, DType, Shape, Tensor};

#[test]
fn test_avg_pool1d_creation() {
    let avg_pool = AvgPool1d::new(2, None).unwrap();
    
    assert_eq!(avg_pool.kernel_size(), 2);
    assert_eq!(avg_pool.stride(), 2);
    
    let avg_pool_with_stride = AvgPool1d::new(3, Some(1)).unwrap();
    assert_eq!(avg_pool_with_stride.kernel_size(), 3);
    assert_eq!(avg_pool_with_stride.stride(), 1);
}

#[test]
fn test_max_pool1d_creation() {
    let max_pool = MaxPool1d::new(2, None).unwrap();
    
    assert_eq!(max_pool.kernel_size(), 2);
    assert_eq!(max_pool.stride(), 2);
    
    let max_pool_with_stride = MaxPool1d::new(3, Some(2)).unwrap();
    assert_eq!(max_pool_with_stride.kernel_size(), 3);
    assert_eq!(max_pool_with_stride.stride(), 2);
}

#[test]
fn test_avg_pool1d_invalid_arguments() {
    let result = AvgPool1d::new(0, None);
    assert!(result.is_err());
    
    let result = AvgPool1d::new(1, Some(0));
    assert!(result.is_err());
}

#[test]
fn test_max_pool1d_invalid_arguments() {
    let result = MaxPool1d::new(0, None);
    assert!(result.is_err());
    
    let result = MaxPool1d::new(1, Some(0));
    assert!(result.is_err());
}

#[test]
fn test_avg_pool1d_forward_shape() {
    let mut avg_pool = AvgPool1d::new(2, None).unwrap();
    
    let input = Tensor::ones(&Shape::from_dims(&[2, 3, 10]), &DType::F32, &Device::cpu(), false).unwrap();
    let output = avg_pool.forward(input).unwrap();
    
    assert_eq!(output.shape().unwrap(), Shape::from_dims(&[2, 3, 5]));
}

#[test]
fn test_avg_pool1d_forward_with_stride_shape() {
    let mut avg_pool = AvgPool1d::new(3, Some(1)).unwrap();
    
    let input = Tensor::ones(&Shape::from_dims(&[1, 2, 8]), &DType::F32, &Device::cpu(), false).unwrap();
    let output = avg_pool.forward(input).unwrap();
    
    assert_eq!(output.shape().unwrap(), Shape::from_dims(&[1, 2, 6]));
}

#[test]
fn test_max_pool1d_forward_shape() {
    let mut max_pool = MaxPool1d::new(2, None).unwrap();
    
    let input = Tensor::ones(&Shape::from_dims(&[2, 3, 10]), &DType::F32, &Device::cpu(), false).unwrap();
    let output = max_pool.forward(input).unwrap();
    
    assert_eq!(output.shape().unwrap(), Shape::from_dims(&[2, 3, 5]));
}

#[test]
fn test_max_pool1d_forward_with_stride_shape() {
    let mut max_pool = MaxPool1d::new(3, Some(2)).unwrap();
    
    let input = Tensor::ones(&Shape::from_dims(&[1, 2, 9]), &DType::F32, &Device::cpu(), false).unwrap();
    let output = max_pool.forward(input).unwrap();
    
    assert_eq!(output.shape().unwrap(), Shape::from_dims(&[1, 2, 4]));
}

#[test]
fn test_avg_pool1d_forward_values() {
    let mut avg_pool = AvgPool1d::new(2, None).unwrap();
    
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input = Tensor::from_slice(&data, &Shape::from_dims(&[1, 1, 6]), &Device::cpu(), false).unwrap();
    let output = avg_pool.forward(input).unwrap();
    
    let output_values = output.to_vec::<f32>().unwrap();
    assert_eq!(output_values, vec![1.5, 3.5, 5.5]);
}

#[test]
fn test_max_pool1d_forward_values() {
    let mut max_pool = MaxPool1d::new(2, None).unwrap();
    
    let data = vec![1.0f32, 3.0, 2.0, 5.0, 4.0, 6.0];
    let input = Tensor::from_slice(&data, &Shape::from_dims(&[1, 1, 6]), &Device::cpu(), false).unwrap();
    let output = max_pool.forward(input).unwrap();
    
    let output_values = output.to_vec::<f32>().unwrap();
    assert_eq!(output_values, vec![3.0, 5.0, 6.0]);
}

#[test]
fn test_pool1d_enum_creation() {
    let avg_pool = Pool1d::avg_pool1d(2, None).unwrap();
    let max_pool = Pool1d::max_pool1d(3, Some(1)).unwrap();
    
    match avg_pool {
        Pool1d::AvgPool1d(layer) => {
            assert_eq!(layer.kernel_size(), 2);
            assert_eq!(layer.stride(), 2);
        }
        _ => panic!("Expected AvgPool1d"),
    }
    
    match max_pool {
        Pool1d::MaxPool1d(layer) => {
            assert_eq!(layer.kernel_size(), 3);
            assert_eq!(layer.stride(), 1);
        }
        _ => panic!("Expected MaxPool1d"),
    }
}

#[test]
fn test_pool1d_enum_forward() {
    let mut avg_pool = Pool1d::avg_pool1d(2, None).unwrap();
    let input = Tensor::ones(&Shape::from_dims(&[1, 1, 6]), &DType::F32, &Device::cpu(), false).unwrap();
    let output = avg_pool.forward(input).unwrap();
    
    assert_eq!(output.shape().unwrap(), Shape::from_dims(&[1, 1, 3]));
    
    let mut max_pool = Pool1d::max_pool1d(3, Some(1)).unwrap();
    let input2 = Tensor::ones(&Shape::from_dims(&[1, 1, 8]), &DType::F32, &Device::cpu(), false).unwrap();
    let output2 = max_pool.forward(input2).unwrap();
    
    assert_eq!(output2.shape().unwrap(), Shape::from_dims(&[1, 1, 6]));
}

#[test]
fn test_pool1d_parameters_empty() {
    let avg_pool1d = AvgPool1d::new(2, None).unwrap();
    let params = avg_pool1d.parameters().unwrap();
    assert_eq!(params.len(), 0);
    
    let max_pool1d = MaxPool1d::new(2, None).unwrap();
    let params = max_pool1d.parameters().unwrap();
    assert_eq!(params.len(), 0);
}

#[test]
fn test_pool1d_require_grad_noop() {
    let mut avg_pool1d = AvgPool1d::new(2, None).unwrap();
    avg_pool1d.require_grad(true).unwrap();
    
    let mut max_pool1d = MaxPool1d::new(2, None).unwrap();
    max_pool1d.require_grad(false).unwrap();
}

#[test]
fn test_pool1d_to_device_noop() {
    let mut avg_pool1d = AvgPool1d::new(2, None).unwrap();
    avg_pool1d.to_device(&Device::cpu()).unwrap();
    
    let mut max_pool1d = MaxPool1d::new(2, None).unwrap();
    max_pool1d.to_device(&Device::cpu()).unwrap();
}

#[test]
fn test_pool1d_to_dtype_noop() {
    let mut avg_pool1d = AvgPool1d::new(2, None).unwrap();
    avg_pool1d.to_dtype(&DType::F32).unwrap();
    
    let mut max_pool1d = MaxPool1d::new(2, None).unwrap();
    max_pool1d.to_dtype(&DType::F32).unwrap();
}

#[test]
fn test_pool1d_display() {
    let avg_pool1d = AvgPool1d::new(2, None).unwrap();
    let display_str = format!("{}", avg_pool1d);
    assert!(display_str.contains("avgpool1d."));
    assert!(display_str.contains("kernel_size=2"));
    assert!(display_str.contains("stride=2"));
    
    let max_pool1d = MaxPool1d::new(3, Some(1)).unwrap();
    let display_str = format!("{}", max_pool1d);
    assert!(display_str.contains("maxpool1d."));
    assert!(display_str.contains("kernel_size=3"));
    assert!(display_str.contains("stride=1"));
}