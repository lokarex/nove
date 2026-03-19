use nove_lossfn::{BCELoss, LossFn};
use nove_tensor::{Device, Tensor};

#[test]
fn test_bce_lossfn() {
    let device = Device::cpu();

    // Test case 1: Simple 2-element tensors
    let input = Tensor::from_data(&[0.5, 0.5], &device, false).unwrap();
    let target = Tensor::from_data(&[1.0, 0.0], &device, false).unwrap();

    let lossfn = BCELoss::default();
    let loss = lossfn.loss((input, target)).unwrap();
    let actual = loss.to_scalar::<f64>().unwrap();
    let expected = 0.693147; // -ln(0.5)
    assert!(
        (actual - expected).abs() < 1e-6,
        "Expected {}, got {}",
        expected,
        actual
    );

    // Test case 2: All ones (loss should be 0)
    let input = Tensor::from_data(&[1.0, 1.0, 1.0], &device, false).unwrap();
    let target = Tensor::from_data(&[1.0, 1.0, 1.0], &device, false).unwrap();
    let loss = lossfn.loss((input, target)).unwrap();
    let actual = loss.to_scalar::<f64>().unwrap();
    assert!(actual.abs() < 1e-6, "Expected 0, got {}", actual);

    // Test case 3: All zeros (with epsilon clipping)
    let input = Tensor::from_data(&[0.0, 0.0], &device, false).unwrap();
    let target = Tensor::from_data(&[0.0, 1.0], &device, false).unwrap();
    let loss = lossfn.loss((input, target)).unwrap();
    let actual = loss.to_scalar::<f64>().unwrap();
    // loss = -mean(0*log(epsilon) + 1*log(epsilon) + 0*log(1-epsilon) + 1*log(1-epsilon))? Wait careful:
    // element1: target=0, input=epsilon -> loss = -(0*log(epsilon) + 1*log(1-epsilon)) = -log(1-epsilon) ≈ epsilon
    // element2: target=1, input=epsilon -> loss = -(1*log(epsilon) + 0*log(1-epsilon)) = -log(epsilon) ≈ -ln(1e-8) ≈ 18.420680743952367
    // mean = (epsilon + (-log(epsilon))) / 2 ≈ (1e-8 + 18.420680743952367)/2 ≈ 9.210340376976184
    let epsilon = 1e-8;
    let expected = (-f64::ln(epsilon) - f64::ln(1.0 - epsilon)) / 2.0;
    assert!(
        (actual - expected).abs() < 1e-6,
        "Expected {}, got {}",
        expected,
        actual
    );
}

#[test]
fn test_bce_lossfn_shape_mismatch() {
    let device = Device::cpu();
    let input = Tensor::from_data(&[[0.5, 0.5], [0.5, 0.5]], &device, false).unwrap();
    let target = Tensor::from_data(&[1.0, 0.0], &device, false).unwrap();

    let lossfn = BCELoss::default();
    let result = lossfn.loss((input, target));
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("must have the same shape"));
}
