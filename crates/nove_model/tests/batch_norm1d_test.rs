use nove::model::layer::BatchNorm1dBuilder;
use nove::model::Model;
use nove::tensor::{Device, DType, Shape, Tensor};

#[test]
fn test_batch_norm1d_basic_forward() {
    let mut bn = BatchNorm1dBuilder::new(3)
        .epsilon(1e-5)
        .momentum(0.1)
        .affine(true)
        .device(Device::cpu())
        .dtype(DType::F32)
        .build()
        .unwrap();

    let input = Tensor::from_data(
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        &Device::cpu(),
        false,
    )
    .unwrap()
    .reshape(&Shape::from_dims(&[2, 3]))
    .unwrap();

    let output = bn.forward((input, true)).unwrap();
    let output_shape = output.shape().unwrap();
    assert_eq!(output_shape.dims(), &[2, 3]);

    let output_vec = output.to_vec::<f32>().unwrap();
    assert_eq!(output_vec.len(), 6);
}

#[test]
fn test_batch_norm1d_training_vs_inference() {
    let mut bn = BatchNorm1dBuilder::new(3)
        .epsilon(1e-5)
        .momentum(0.1)
        .affine(true)
        .device(Device::cpu())
        .dtype(DType::F32)
        .build()
        .unwrap();

    let input = Tensor::from_data(
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        &Device::cpu(),
        false,
    )
    .unwrap()
    .reshape(&Shape::from_dims(&[2, 3]))
    .unwrap();

    let initial_running_mean = bn.running_mean().to_vec::<f32>().unwrap();
    let initial_running_var = bn.running_var().to_vec::<f32>().unwrap();

    let output_training = bn.forward((input.copy(), true)).unwrap();

    let updated_running_mean = bn.running_mean().to_vec::<f32>().unwrap();
    let updated_running_var = bn.running_var().to_vec::<f32>().unwrap();

    for i in 0..3 {
        assert_ne!(initial_running_mean[i], updated_running_mean[i]);
        assert_ne!(initial_running_var[i], updated_running_var[i]);
    }

    let output_inference = bn.forward((input.copy(), false)).unwrap();
    let output_training_vec = output_training.to_vec::<f32>().unwrap();
    let output_inference_vec = output_inference.to_vec::<f32>().unwrap();

    assert_eq!(output_training_vec.len(), output_inference_vec.len());

    for i in 0..output_training_vec.len() {
        assert!(
            (output_training_vec[i] - output_inference_vec[i]).abs() > 1e-6,
            "Training and inference outputs should differ at index {}: {} vs {}",
            i,
            output_training_vec[i],
            output_inference_vec[i]
        );
    }
}

#[test]
fn test_batch_norm1d_running_statistics_update() {
    let mut bn = BatchNorm1dBuilder::new(2)
        .epsilon(1e-5)
        .momentum(0.1)
        .affine(true)
        .device(Device::cpu())
        .dtype(DType::F32)
        .build()
        .unwrap();

    let input1 = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], &Device::cpu(), false)
        .unwrap()
        .reshape(&Shape::from_dims(&[2, 2]))
        .unwrap();

    let initial_running_mean = bn.running_mean().to_vec::<f32>().unwrap();
    let initial_running_var = bn.running_var().to_vec::<f32>().unwrap();

    bn.forward((input1, true)).unwrap();

    let after_first_running_mean = bn.running_mean().to_vec::<f32>().unwrap();
    let after_first_running_var = bn.running_var().to_vec::<f32>().unwrap();

    for i in 0..2 {
        assert_ne!(initial_running_mean[i], after_first_running_mean[i]);
        assert_ne!(initial_running_var[i], after_first_running_var[i]);
    }

    let input2 = Tensor::from_data(vec![5.0f32, 6.0, 7.0, 8.0], &Device::cpu(), false)
        .unwrap()
        .reshape(&Shape::from_dims(&[2, 2]))
        .unwrap();

    bn.forward((input2, true)).unwrap();

    let after_second_running_mean = bn.running_mean().to_vec::<f32>().unwrap();
    let after_second_running_var = bn.running_var().to_vec::<f32>().unwrap();

    for i in 0..2 {
        assert_ne!(after_first_running_mean[i], after_second_running_mean[i]);
        assert_ne!(after_first_running_var[i], after_second_running_var[i]);
    }

    for i in 0..2 {
        assert!(
            (initial_running_mean[i] - after_second_running_mean[i]).abs() > 1e-6,
            "Running mean should be updated from initial value {} to {}",
            initial_running_mean[i],
            after_second_running_mean[i]
        );
        assert!(
            (initial_running_var[i] - after_second_running_var[i]).abs() > 1e-6,
            "Running var should be updated from initial value {} to {}",
            initial_running_var[i],
            after_second_running_var[i]
        );
    }
}

#[test]
fn test_batch_norm1d_epsilon_effect() {
    let mut bn_small_epsilon = BatchNorm1dBuilder::new(2)
        .epsilon(1e-8)
        .momentum(0.1)
        .affine(true)
        .device(Device::cpu())
        .dtype(DType::F32)
        .build()
        .unwrap();

    let mut bn_large_epsilon = BatchNorm1dBuilder::new(2)
        .epsilon(1e-2)
        .momentum(0.1)
        .affine(true)
        .device(Device::cpu())
        .dtype(DType::F32)
        .build()
        .unwrap();

    let input = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], &Device::cpu(), false)
        .unwrap()
        .reshape(&Shape::from_dims(&[2, 2]))
        .unwrap();

    let output_small = bn_small_epsilon.forward((input.copy(), true)).unwrap();
    let output_large = bn_large_epsilon.forward((input.copy(), true)).unwrap();

    let output_small_vec = output_small.to_vec::<f32>().unwrap();
    let output_large_vec = output_large.to_vec::<f32>().unwrap();

    for i in 0..4 {
        assert!(
            (output_small_vec[i] - output_large_vec[i]).abs() > 1e-6,
            "Different epsilon values should produce different outputs at index {}: {} vs {}",
            i,
            output_small_vec[i],
            output_large_vec[i]
        );
    }
}

#[test]
fn test_batch_norm1d_3d_input() {
    let mut bn = BatchNorm1dBuilder::new(2)
        .epsilon(1e-5)
        .momentum(0.1)
        .affine(true)
        .device(Device::cpu())
        .dtype(DType::F32)
        .build()
        .unwrap();

    let input = Tensor::from_data(
        vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &Device::cpu(),
        false,
    )
    .unwrap()
    .reshape(&Shape::from_dims(&[2, 2, 3]))
    .unwrap();

    let output = bn.forward((input, true)).unwrap();
    let output_shape = output.shape().unwrap();
    assert_eq!(output_shape.dims(), &[2, 2, 3]);

    let output_vec = output.to_vec::<f32>().unwrap();
    assert_eq!(output_vec.len(), 12);
}

#[test]
fn test_batch_norm1d_affine_false() {
    let mut bn = BatchNorm1dBuilder::new(2)
        .epsilon(1e-5)
        .momentum(0.1)
        .affine(false)
        .device(Device::cpu())
        .dtype(DType::F32)
        .build()
        .unwrap();

    let input = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], &Device::cpu(), false)
        .unwrap()
        .reshape(&Shape::from_dims(&[2, 2]))
        .unwrap();

    let output = bn.forward((input, true)).unwrap();
    let output_vec = output.to_vec::<f32>().unwrap();
    assert_eq!(output_vec.len(), 4);

    bn.require_grad(true).unwrap();
    bn.require_grad(false).unwrap();
}

#[test]
fn test_batch_norm1d_parameters() {
    let bn = BatchNorm1dBuilder::new(3)
        .epsilon(1e-5)
        .momentum(0.1)
        .affine(true)
        .device(Device::cpu())
        .dtype(DType::F32)
        .build()
        .unwrap();

    let params = bn.parameters().unwrap();
    assert_eq!(params.len(), 4);

    let named_params = bn.named_parameters().unwrap();
    assert_eq!(named_params.len(), 4);

    for (key, _) in &named_params {
        println!("Key: {}", key);
    }
    assert!(named_params.len() == 4);
}

#[test]
fn test_batch_norm1d_to_device_and_dtype() {
    let mut bn = BatchNorm1dBuilder::new(2)
        .epsilon(1e-5)
        .momentum(0.1)
        .affine(true)
        .device(Device::cpu())
        .dtype(DType::F32)
        .build()
        .unwrap();

    bn.to_device(&Device::cpu()).unwrap();
    bn.to_dtype(&DType::F32).unwrap();
}