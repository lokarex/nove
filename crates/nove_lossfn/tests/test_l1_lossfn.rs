use nove_lossfn::{L1Loss, LossFn};
use nove_tensor::{Device, Tensor};

#[test]
fn test_l1_lossfn() {
    let device = Device::cpu();

    // Test case 1: Simple 2-element tensors
    let input = Tensor::from_data(&[1.0, 2.0], &device, false).unwrap();
    let target = Tensor::from_data(&[1.0, 2.0], &device, false).unwrap();

    let lossfn = L1Loss::new();
    let loss = lossfn.loss((input, target)).unwrap();
    let actual = loss.to_scalar::<f64>().unwrap();
    let expected = 0.0;
    assert!(
        (actual - expected).abs() < 1e-6,
        "Expected {}, got {}",
        expected,
        actual
    );

    // Test case 2: Difference of 1 for each element
    let input = Tensor::from_data(&[1.0, 2.0, 3.0], &device, false).unwrap();
    let target = Tensor::from_data(&[2.0, 3.0, 4.0], &device, false).unwrap();
    let loss = lossfn.loss((input, target)).unwrap();
    let actual = loss.to_scalar::<f64>().unwrap();
    let expected = 1.0; // mean(|1|) = 1
    assert!(
        (actual - expected).abs() < 1e-6,
        "Expected {}, got {}",
        expected,
        actual
    );

    // Test case 3: Mixed positive and negative differences
    let input = Tensor::from_data(&[0.5, 1.5, -0.5], &device, false).unwrap();
    let target = Tensor::from_data(&[0.0, 1.0, 0.0], &device, false).unwrap();
    let loss = lossfn.loss((input, target)).unwrap();
    let actual = loss.to_scalar::<f64>().unwrap();
    // (|0.5| + |0.5| + |-0.5|)/3 = (0.5+0.5+0.5)/3 = 1.5/3 = 0.5
    let expected = 0.5;
    assert!(
        (actual - expected).abs() < 1e-6,
        "Expected {}, got {}",
        expected,
        actual
    );

    // Test case 4: 2D tensor
    let input = Tensor::from_data(&[[1.0, 2.0], [3.0, 4.0]], &device, false).unwrap();
    let target = Tensor::from_data(&[[2.0, 3.0], [4.0, 5.0]], &device, false).unwrap();
    let loss = lossfn.loss((input, target)).unwrap();
    let actual = loss.to_scalar::<f64>().unwrap();
    // All differences are 1, absolute is 1, mean over 4 elements = 1
    let expected = 1.0;
    assert!(
        (actual - expected).abs() < 1e-6,
        "Expected {}, got {}",
        expected,
        actual
    );
}

#[test]
fn test_l1_lossfn_shape_mismatch() {
    let device = Device::cpu();
    let input = Tensor::from_data(&[[1.0, 2.0], [3.0, 4.0]], &device, false).unwrap();
    let target = Tensor::from_data(&[1.0, 2.0], &device, false).unwrap();

    let lossfn = L1Loss::new();
    let result = lossfn.loss((input, target));
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("must have the same shape"));
}
