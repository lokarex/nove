use nove::tensor::{Device, Shape, Tensor, TensorError};

#[test]
fn test_to_scalar() {
    let device = Device::get_cuda_if_available(0);

    // Test to_scalar with a 1D array.
    let t = Tensor::from_data(&[1.23456f64], &device, false).unwrap();
    assert_eq!(t.to_scalar::<f64>().unwrap(), 1.23456f64);

    // Test to_scalar with a 2D array.
    // It should return an error.
    let t = Tensor::from_data(&[1.23456f64, 2.34567f64], &device, false).unwrap();
    assert!(t.to_scalar::<f64>().is_err());
}

#[test]
fn test_to_vec() {
    let device = Device::get_cuda_if_available(0);

    let t = Tensor::from_data(&[1.0f64, 2.0f64, 3.0f64], &device, false).unwrap();
    assert_eq!(t.to_vec::<f64>().unwrap(), vec![1.0f64, 2.0f64, 3.0f64]);
}

#[test]
fn test_to_device() {
    let device1 = Device::get_cpu();
    let device2 = Device::get_cuda_if_available(0);

    let mut t = Tensor::from_data(&[1.0f64, 2.0f64, 3.0f64], &device1, false).unwrap();
    t.to_device(&device2).unwrap();
}

#[test]
fn test_to_dtype() {
    let device = Device::get_cuda_if_available(0);

    let mut t = Tensor::from_data(&[1.0f64, 2.0f64, 3.0f64], &device, false).unwrap();
    assert_eq!(t.to_vec::<f64>().unwrap(), vec![1.0f64, 2.0f64, 3.0f64]);

    t.to_dtype(candle_core::DType::F32).unwrap();
    assert_eq!(t.to_vec::<f32>().unwrap(), vec![1.0f32, 2.0f32, 3.0f32]);
}

#[test]
fn test_get_dtype() {
    let device = Device::get_cuda_if_available(0);

    let t = Tensor::from_data(&[1.0f64, 2.0f64, 3.0f64], &device, false).unwrap();
    assert_eq!(t.get_dtype().unwrap(), candle_core::DType::F64);
}

#[test]
fn test_get_shape() {
    let device = Device::get_cuda_if_available(0);

    let t = Tensor::from_data(&[1.0f64, 2.0f64, 3.0f64], &device, false).unwrap();
    assert_eq!(t.get_shape().unwrap(), Shape::from(&[3]));

    let t = Tensor::from_data(
        &[[1.0f64, 2.0f64, 3.0f64], [4.0f64, 5.0f64, 6.0f64]],
        &device,
        false,
    )
    .unwrap();
    assert_eq!(t.get_shape().unwrap(), Shape::from(&[2, 3]));
}

#[test]
fn test_get_dim_num() {
    let device = Device::get_cuda_if_available(0);

    let t = Tensor::from_data(&[1.0f64, 2.0f64, 3.0f64], &device, false).unwrap();
    assert_eq!(t.get_dim_num().unwrap(), 1);

    let t = Tensor::from_data(
        &[[1.0f64, 2.0f64, 3.0f64], [4.0f64, 5.0f64, 6.0f64]],
        &device,
        false,
    )
    .unwrap();
    assert_eq!(t.get_dim_num().unwrap(), 2);
}

#[test]
fn test_add() {
    let device = Device::get_cuda_if_available(0);

    // Test add operation with two grad disabled tensors.
    let t1 = Tensor::from_data(&[1.0f64, 2.0f64, 3.0f64], &device, false).unwrap();
    let t2 = Tensor::from_data(&[2.0f64, 3.0f64, 4.0f64], &device, false).unwrap();
    let t3 = t1.add(&t2).unwrap();
    assert_eq!(t3.to_vec::<f64>().unwrap(), vec![3.0f64, 5.0f64, 7.0f64]);

    // Test add operation with one grad disabled tensor and one grad enabled tensor.
    let t1 = Tensor::from_data(&[1.1f64, 2.2f64, 3.3f64], &device, false).unwrap();
    let t2 = Tensor::from_data(&[2.0f64, 3.0f64, 4.0f64], &device, true).unwrap();
    let t3 = t1.add(&t2).unwrap();
    assert_eq!(t3.to_vec::<f64>().unwrap(), vec![3.1f64, 5.2f64, 7.3f64]);

    // Test add operation with two grad enabled tensors.
    let t1 = Tensor::from_data(
        &[[1.1f64, 2.2f64, 3.3f64], [4.4f64, 5.5f64, 6.6f64]],
        &device,
        true,
    )
    .unwrap();
    let t2 = Tensor::from_data(
        &[[2.0f64, 3.0f64, 4.0f64], [5.0f64, 6.0f64, 7.0f64]],
        &device,
        true,
    )
    .unwrap();
    let t3 = t1.add(&t2).unwrap();
    assert_eq!(
        t3.to_vec::<f64>().unwrap(),
        vec![3.1f64, 5.2f64, 7.3f64, 9.4f64, 11.5f64, 13.6f64]
    );
}

#[test]
fn test_stack() {
    let device = Device::get_cuda_if_available(0);

    // Test stack operation with two grad disabled 1D tensors.
    let t1 = Tensor::from_data(&[1.0f64, 2.0f64, 3.0f64], &device, false).unwrap();
    let t2 = Tensor::from_data(&[4.0f64, 5.0f64, 6.0f64], &device, false).unwrap();
    let t3 = Tensor::stack(&[t1, t2], 0).unwrap();
    assert_eq!(
        t3.to_vec::<f64>().unwrap(),
        vec![1.0f64, 2.0f64, 3.0f64, 4.0f64, 5.0f64, 6.0f64]
    );
    assert_eq!(t3.get_shape().unwrap(), Shape::from(&[2, 3]));

    // Test stack operation with one grad enabled tensor and one grad disabled tensor.
    let t1 = Tensor::from_data(&[1.1f64, 2.2f64, 3.3f64], &device, true).unwrap();
    let t2 = Tensor::from_data(&[4.4f64, 5.5f64, 6.6f64], &device, false).unwrap();
    let t3 = Tensor::stack(&[t1, t2], 0).unwrap();
    assert_eq!(
        t3.to_vec::<f64>().unwrap(),
        vec![1.1f64, 2.2f64, 3.3f64, 4.4f64, 5.5f64, 6.6f64]
    );
    assert_eq!(t3.get_shape().unwrap(), Shape::from(&[2, 3]));

    // Test stack operation with two grad enabled 1D tensors.
    let t1 = Tensor::from_data(&[1.1f64, 2.2f64, 3.3f64], &device, true).unwrap();
    let t2 = Tensor::from_data(&[4.4f64, 5.5f64, 6.6f64], &device, true).unwrap();
    let t3 = Tensor::stack(&[t1, t2], 0).unwrap();
    assert_eq!(
        t3.to_vec::<f64>().unwrap(),
        vec![1.1f64, 2.2f64, 3.3f64, 4.4f64, 5.5f64, 6.6f64]
    );
    assert_eq!(t3.get_shape().unwrap(), Shape::from(&[2, 3]));
}

#[test]
fn test_set_grad_enabled() {
    let device = Device::get_cuda_if_available(0);

    let mut t = Tensor::from_data(&[1.0f64, 2.0f64, 3.0f64], &device, false).unwrap();

    // Check if the gradient status is disabled by default
    assert_eq!(t.get_grad_enabled(), false);

    // Check if the gradient status can be enabled
    t.set_grad_enabled(true).unwrap();
    assert_eq!(t.get_grad_enabled(), true);

    // Check if the gradient status can be disabled
    t.set_grad_enabled(false).unwrap();
    assert_eq!(t.get_grad_enabled(), false);

    // Check if the gradient status can't be disabled when it's already disabled
    match t.set_grad_enabled(false) {
        Err(TensorError::AlreadyGradientDisabled) => {}
        _ => panic!("It should return AlreadyGradientDisabled error"),
    }

    // Check if the gradient status can't be enabled when it's already enabled
    t.set_grad_enabled(true).unwrap();
    match t.set_grad_enabled(true) {
        Err(TensorError::AlreadyGradientEnabled) => {}
        _ => panic!("It should return AlreadyGradientEnabled error"),
    }
}

#[test]
fn test_set_grad() {
    let device = Device::get_cuda_if_available(0);

    let mut t = Tensor::from_data(&[1.0f64, 2.0f64, 3.0f64], &device, true).unwrap();

    // Check if the gradient tensor can be set
    let grad = Tensor::from_data(&[0.1f64, 0.2f64, 0.3f64], &device, false).unwrap();
    t.set_grad(grad).unwrap();
    assert_eq!(
        t.get_grad().unwrap().to_vec::<f64>().unwrap(),
        vec![0.1f64, 0.2f64, 0.3f64]
    );
}

#[test]
fn test_backward() {
    let device = Device::get_cuda_if_available(0);

    // Test backward operation with a grad enabled tensor.
    let t1 = Tensor::from_data(&[1.0f64, 2.0f64, 3.0f64], &device, true).unwrap();
    let t2 = Tensor::from_data(&[4.0f64, 5.0f64, 6.0f64], &device, true).unwrap();
    let t3 = t1.add(&t2).unwrap();

    t3.backward().unwrap();
    assert_eq!(
        t1.get_grad().unwrap().to_vec::<f64>().unwrap(),
        vec![1.0f64, 1.0f64, 1.0f64]
    );
    assert_eq!(
        t2.get_grad().unwrap().to_vec::<f64>().unwrap(),
        vec![1.0f64, 1.0f64, 1.0f64]
    );

    // Test gradient accumulation function
    t3.backward().unwrap();
    assert_eq!(
        t1.get_grad().unwrap().to_vec::<f64>().unwrap(),
        vec![2.0f64, 2.0f64, 2.0f64]
    );
    assert_eq!(
        t2.get_grad().unwrap().to_vec::<f64>().unwrap(),
        vec![2.0f64, 2.0f64, 2.0f64]
    );
}
