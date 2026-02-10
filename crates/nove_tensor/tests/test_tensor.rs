use nove::tensor::{Device, Tensor};

#[test]
fn test_require_grad() {
    let device = Device::cpu();

    let mut t = Tensor::from_data(&[1.0f64, 2.0f64, 3.0f64], &device, false).unwrap();

    // Check if the gradient status is disabled by default
    assert_eq!(t.grad_enabled().unwrap(), false);

    // Check if the gradient status can be enabled
    let mut t = t.require_grad(true).unwrap();
    assert_eq!(t.grad_enabled().unwrap(), true);

    // Check if the gradient status can be disabled
    let t = t.require_grad(false).unwrap();
    assert_eq!(t.grad_enabled().unwrap(), false);
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
        t1.grad().unwrap().unwrap().to_vec::<f64>().unwrap(),
        vec![1.0f64, 1.0f64, 1.0f64]
    );
    assert_eq!(
        t2.grad().unwrap().unwrap().to_vec::<f64>().unwrap(),
        vec![1.0f64, 1.0f64, 1.0f64]
    );
    assert_eq!(
        t3.grad().unwrap().unwrap().to_vec::<f64>().unwrap(),
        vec![1.0f64, 1.0f64, 1.0f64]
    );

    // Test gradient accumulation function
    t5.backward().unwrap();
    assert_eq!(
        t1.grad().unwrap().unwrap().to_vec::<f64>().unwrap(),
        vec![2.0f64, 2.0f64, 2.0f64]
    );
    assert_eq!(
        t2.grad().unwrap().unwrap().to_vec::<f64>().unwrap(),
        vec![2.0f64, 2.0f64, 2.0f64]
    );
    assert_eq!(
        t3.grad().unwrap().unwrap().to_vec::<f64>().unwrap(),
        vec![2.0f64, 2.0f64, 2.0f64]
    );
}
