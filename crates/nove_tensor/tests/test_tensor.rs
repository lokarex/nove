use nove::tensor::{Device, Tensor};

#[test]
fn test_set_grad_enabled() {
    let device = Device::cpu();

    let mut t = Tensor::from_data(&[1.0f64, 2.0f64, 3.0f64], &device, false).unwrap();

    // Check if the gradient status is disabled by default
    assert_eq!(t.grad_enabled().unwrap(), false);

    // Check if the gradient status can be enabled
    t.set_grad_enabled(true).unwrap();
    assert_eq!(t.grad_enabled().unwrap(), true);

    // Check if the gradient status can be disabled
    t.set_grad_enabled(false).unwrap();
    assert_eq!(t.grad_enabled().unwrap(), false);
}

#[test]
fn test_set_grad() {
    let device = Device::cpu();

    let mut t = Tensor::from_data(&[1.0f64, 2.0f64, 3.0f64], &device, true).unwrap();

    // Check if the gradient tensor can be set
    let grad = Tensor::from_data(&[0.1f64, 0.2f64, 0.3f64], &device, false).unwrap();
    t.set_grad(grad).unwrap();
    assert_eq!(
        t.grad().unwrap().to_vec::<f64>().unwrap(),
        vec![0.1f64, 0.2f64, 0.3f64]
    );
}

#[test]
fn test_backward() {
    let device = Device::cpu();

    // Test backward operation with a grad enabled tensor.
    let t1 = Tensor::from_data(&[1.0f64, 2.0f64, 3.0f64], &device, true).unwrap();
    let t2 = Tensor::from_data(&[4.0f64, 5.0f64, 6.0f64], &device, true).unwrap();
    let t3 = Tensor::from_data(&[7.0f64, 8.0f64, 9.0f64], &device, true).unwrap();
    let t4 = t1.add(&t2).unwrap();
    let t5 = t4.add(&t3).unwrap();

    t5.backward().unwrap();
    assert_eq!(
        t1.grad().unwrap().to_vec::<f64>().unwrap(),
        vec![1.0f64, 1.0f64, 1.0f64]
    );
    assert_eq!(
        t2.grad().unwrap().to_vec::<f64>().unwrap(),
        vec![1.0f64, 1.0f64, 1.0f64]
    );
    assert_eq!(
        t3.grad().unwrap().to_vec::<f64>().unwrap(),
        vec![1.0f64, 1.0f64, 1.0f64]
    );

    // Test gradient accumulation function
    t5.backward().unwrap();
    assert_eq!(
        t1.grad().unwrap().to_vec::<f64>().unwrap(),
        vec![2.0f64, 2.0f64, 2.0f64]
    );
    assert_eq!(
        t2.grad().unwrap().to_vec::<f64>().unwrap(),
        vec![2.0f64, 2.0f64, 2.0f64]
    );
    assert_eq!(
        t3.grad().unwrap().to_vec::<f64>().unwrap(),
        vec![2.0f64, 2.0f64, 2.0f64]
    );
}
