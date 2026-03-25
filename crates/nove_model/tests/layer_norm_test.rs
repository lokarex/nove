use nove::model::Model;
use nove::model::layer::LayerNormBuilder;
use nove::tensor::{DType, Device, Shape, Tensor};

#[test]
fn test_layernorm_builder_creation() {
    let layernorm = LayerNormBuilder::new(vec![768])
        .epsilon(1e-5)
        .affine(true)
        .device(Device::cpu())
        .dtype(DType::F32)
        .build()
        .unwrap();

    assert_eq!(layernorm.gamma().shape().unwrap(), Shape::from_dims(&[768]));
    assert_eq!(layernorm.beta().shape().unwrap(), Shape::from_dims(&[768]));
    let display_str = layernorm.to_string();
    assert!(display_str.contains("layer_norm"));
    assert!(display_str.contains("normalized_shape=[768]"));
    assert!(display_str.contains("epsilon="));
    assert!(display_str.contains("affine=true"));
}

#[test]
fn test_layernorm_builder_normalized_shape() {
    let layernorm1 = LayerNormBuilder::new(vec![512]).build().unwrap();
    assert_eq!(
        layernorm1.gamma().shape().unwrap(),
        Shape::from_dims(&[512])
    );

    let layernorm2 = LayerNormBuilder::new(vec![256, 256]).build().unwrap();
    assert_eq!(
        layernorm2.gamma().shape().unwrap(),
        Shape::from_dims(&[256, 256])
    );

    let mut builder = LayerNormBuilder::new(vec![128]);
    builder.normalized_shape(vec![64]);
    let layernorm3 = builder.build().unwrap();
    assert_eq!(layernorm3.gamma().shape().unwrap(), Shape::from_dims(&[64]));
}

#[test]
fn test_layernorm_builder_epsilon() {
    let layernorm1 = LayerNormBuilder::new(vec![256])
        .epsilon(1e-5)
        .build()
        .unwrap();
    let display_str1 = layernorm1.to_string();
    assert!(display_str1.contains("epsilon=0.00001"));

    let layernorm2 = LayerNormBuilder::new(vec![256])
        .epsilon(1e-6)
        .build()
        .unwrap();
    let display_str2 = layernorm2.to_string();
    assert!(display_str2.contains("epsilon=0.000001"));
}

#[test]
fn test_layernorm_builder_affine() {
    let layernorm_with_affine = LayerNormBuilder::new(vec![128])
        .affine(true)
        .build()
        .unwrap();

    let params_with_affine = layernorm_with_affine.parameters().unwrap();
    assert_eq!(params_with_affine.len(), 2);
    assert!(layernorm_with_affine.to_string().contains("affine=true"));

    let layernorm_without_affine = LayerNormBuilder::new(vec![128])
        .affine(false)
        .build()
        .unwrap();

    let params_without_affine = layernorm_without_affine.parameters().unwrap();
    assert_eq!(params_without_affine.len(), 0);
    assert!(
        layernorm_without_affine
            .to_string()
            .contains("affine=false")
    );
}

#[test]
fn test_layernorm_builder_invalid_normalized_shape() {
    let result1 = LayerNormBuilder::new(vec![]).build();
    assert!(result1.is_err());
    assert!(
        result1
            .unwrap_err()
            .to_string()
            .contains("must not be empty")
    );

    let result2 = LayerNormBuilder::new(vec![0, 256]).build();
    assert!(result2.is_err());
    assert!(
        result2
            .unwrap_err()
            .to_string()
            .contains("must be greater than 0")
    );

    let result3 = LayerNormBuilder::new(vec![10, 0]).build();
    assert!(result3.is_err());
    assert!(
        result3
            .unwrap_err()
            .to_string()
            .contains("must be greater than 0")
    );
}

#[test]
fn test_layernorm_builder_invalid_epsilon() {
    let result1 = LayerNormBuilder::new(vec![256]).epsilon(0.0).build();
    assert!(result1.is_err());
    assert!(
        result1
            .unwrap_err()
            .to_string()
            .contains("must be greater than 0")
    );

    let result2 = LayerNormBuilder::new(vec![256]).epsilon(-0.1).build();
    assert!(result2.is_err());
    assert!(
        result2
            .unwrap_err()
            .to_string()
            .contains("must be greater than 0")
    );
}

#[test]
fn test_layernorm_forward_basic() {
    let mut layernorm = LayerNormBuilder::new(vec![4]).affine(true).build().unwrap();

    let input = Tensor::ones(
        &Shape::from_dims(&[2, 3, 4]),
        &DType::F32,
        &Device::cpu(),
        false,
    )
    .unwrap();
    let output = layernorm.forward(input).unwrap();

    assert_eq!(output.shape().unwrap(), Shape::from_dims(&[2, 3, 4]));

    let output_values = output.to_vec::<f32>().unwrap();
    assert!(!output_values.is_empty());
}

#[test]
fn test_layernorm_forward_without_affine() {
    let mut layernorm = LayerNormBuilder::new(vec![3])
        .affine(false)
        .build()
        .unwrap();

    let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input = Tensor::from_data(input_data, &Device::cpu(), false)
        .unwrap()
        .reshape(&Shape::from_dims(&[2, 3]))
        .unwrap();

    let output = layernorm.forward(input).unwrap();
    assert_eq!(output.shape().unwrap(), Shape::from_dims(&[2, 3]));
}

#[test]
fn test_layernorm_forward_invalid_shape() {
    let mut layernorm = LayerNormBuilder::new(vec![5]).build().unwrap();

    let input = Tensor::ones(
        &Shape::from_dims(&[2, 4]),
        &DType::F32,
        &Device::cpu(),
        false,
    )
    .unwrap();
    let result = layernorm.forward(input);
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("expects normalized_shape")
    );
}

#[test]
fn test_layernorm_forward_complex_shape() {
    let mut layernorm = LayerNormBuilder::new(vec![8, 8]).build().unwrap();

    let input = Tensor::ones(
        &Shape::from_dims(&[4, 16, 8, 8]),
        &DType::F32,
        &Device::cpu(),
        false,
    )
    .unwrap();
    let output = layernorm.forward(input).unwrap();
    assert_eq!(output.shape().unwrap(), Shape::from_dims(&[4, 16, 8, 8]));
}

#[test]
fn test_layernorm_gamma_beta_accessors() {
    let layernorm = LayerNormBuilder::new(vec![64])
        .affine(true)
        .build()
        .unwrap();

    let gamma = layernorm.gamma();
    let beta = layernorm.beta();

    assert_eq!(gamma.shape().unwrap(), Shape::from_dims(&[64]));
    assert_eq!(beta.shape().unwrap(), Shape::from_dims(&[64]));

    let gamma_values = gamma.to_vec::<f32>().unwrap();
    let beta_values = beta.to_vec::<f32>().unwrap();

    assert!(gamma_values.iter().all(|&x| (x - 1.0).abs() < 1e-6));
    assert!(beta_values.iter().all(|&x| x.abs() < 1e-6));
}

#[test]
fn test_layernorm_parameters() {
    let layernorm = LayerNormBuilder::new(vec![32])
        .affine(true)
        .build()
        .unwrap();

    let params = layernorm.parameters().unwrap();
    assert_eq!(params.len(), 2);
    assert_eq!(params[0].shape().unwrap(), Shape::from_dims(&[32]));
    assert_eq!(params[1].shape().unwrap(), Shape::from_dims(&[32]));

    let named_params = layernorm.named_parameters().unwrap();
    assert_eq!(named_params.len(), 2);
    let param_names: Vec<String> = named_params.keys().cloned().collect();
    assert!(param_names.iter().any(|name| name.contains("gamma")));
    assert!(param_names.iter().any(|name| name.contains("beta")));
}

#[test]
fn test_layernorm_parameters_without_affine() {
    let layernorm = LayerNormBuilder::new(vec![32])
        .affine(false)
        .build()
        .unwrap();

    let params = layernorm.parameters().unwrap();
    assert_eq!(params.len(), 0);

    let named_params = layernorm.named_parameters().unwrap();
    assert_eq!(named_params.len(), 0);
}

#[test]
fn test_layernorm_require_grad() {
    let mut layernorm = LayerNormBuilder::new(vec![16])
        .affine(true)
        .build()
        .unwrap();

    assert!(layernorm.require_grad(true).is_ok());

    let gamma = layernorm.gamma();
    let beta = layernorm.beta();
    assert!(gamma.grad_enabled().unwrap());
    assert!(beta.grad_enabled().unwrap());

    assert!(layernorm.require_grad(false).is_ok());

    let gamma = layernorm.gamma();
    let beta = layernorm.beta();
    assert!(!gamma.grad_enabled().unwrap());
    assert!(!beta.grad_enabled().unwrap());
}

#[test]
fn test_layernorm_require_grad_without_affine() {
    let mut layernorm = LayerNormBuilder::new(vec![16])
        .affine(false)
        .build()
        .unwrap();

    assert!(layernorm.require_grad(true).is_ok());
    assert!(layernorm.require_grad(false).is_ok());
}

#[test]
fn test_layernorm_to_device() {
    let mut layernorm = LayerNormBuilder::new(vec![8])
        .device(Device::cpu())
        .build()
        .unwrap();

    let gamma_device = layernorm.gamma().device().unwrap();
    assert!(gamma_device.is_cpu());

    assert!(layernorm.to_device(&Device::cpu()).is_ok());

    let gamma_device = layernorm.gamma().device().unwrap();
    assert!(gamma_device.is_cpu());
}

#[test]
fn test_layernorm_to_dtype() {
    let mut layernorm = LayerNormBuilder::new(vec![8])
        .dtype(DType::F32)
        .build()
        .unwrap();

    let gamma_dtype = layernorm.gamma().dtype().unwrap();
    assert_eq!(gamma_dtype, DType::F32);

    assert!(layernorm.to_dtype(&DType::F32).is_ok());

    let gamma_dtype = layernorm.gamma().dtype().unwrap();
    assert_eq!(gamma_dtype, DType::F32);
}

#[test]
fn test_layernorm_builder_method_chaining() {
    let mut builder = LayerNormBuilder::new(vec![256]);
    builder
        .normalized_shape(vec![128])
        .epsilon(1e-6)
        .affine(false)
        .device(Device::cpu())
        .dtype(DType::F32);

    let layernorm = builder.build().unwrap();

    assert_eq!(layernorm.gamma().shape().unwrap(), Shape::from_dims(&[128]));
    assert!(layernorm.to_string().contains("affine=false"));
    assert!(layernorm.to_string().contains("epsilon=0.000001"));
}
