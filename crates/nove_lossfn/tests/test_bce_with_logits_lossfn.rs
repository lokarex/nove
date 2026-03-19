use nove_lossfn::{BCEWithLogitsLoss, LossFn};
use nove_tensor::{Device, Tensor};

#[test]
fn test_bce_with_logits_lossfn() {
    let device = Device::cpu();

    // Test case 1: Zero logits, equivalent to BCE with p=0.5
    let input = Tensor::from_data(&[0.0, 0.0], &device, false).unwrap();
    let target = Tensor::from_data(&[1.0, 0.0], &device, false).unwrap();

    let lossfn = BCEWithLogitsLoss::new();
    let loss = lossfn.loss((input, target)).unwrap();
    let actual = loss.to_scalar::<f64>().unwrap();
    let expected = 0.693147; // -ln(0.5)
    assert!(
        (actual - expected).abs() < 1e-6,
        "Expected {}, got {}",
        expected,
        actual
    );

    // Test case 2: Large positive logits (sigmoid ≈ 1)
    let input = Tensor::from_data(&[10.0, 10.0], &device, false).unwrap();
    let target = Tensor::from_data(&[1.0, 1.0], &device, false).unwrap();
    let loss = lossfn.loss((input, target)).unwrap();
    let actual = loss.to_scalar::<f64>().unwrap();
    // For large positive logits, loss ≈ max(10,0) - 10*1 + log(1+exp(-10)) ≈ log(1+exp(-10)) ≈ 4.5398899e-05
    assert!(actual.abs() < 1e-4, "Expected near 0, got {}", actual);

    // Test case 3: Large negative logits (sigmoid ≈ 0)
    let input = Tensor::from_data(&[-10.0, -10.0], &device, false).unwrap();
    let target = Tensor::from_data(&[0.0, 0.0], &device, false).unwrap();
    let loss = lossfn.loss((input, target)).unwrap();
    let actual = loss.to_scalar::<f64>().unwrap();
    // max(-10,0)=0, -(-10)*0=0, log(1+exp(-10))≈4.5398899e-05, loss ≈ 4.5398899e-05
    assert!(actual.abs() < 1e-4, "Expected near 0, got {}", actual);

    // Test case 4: Mixed logits
    let input = Tensor::from_data(&[2.0, -2.0, 0.0], &device, false).unwrap();
    let target = Tensor::from_data(&[1.0, 0.0, 0.5], &device, false).unwrap();
    let loss = lossfn.loss((input, target)).unwrap();
    let actual = loss.to_scalar::<f64>().unwrap();
    // Just ensure it computes without panic
    assert!(actual.is_finite());
}

#[test]
fn test_bce_with_logits_lossfn_shape_mismatch() {
    let device = Device::cpu();
    let input = Tensor::from_data(&[[0.5, 0.5], [0.5, 0.5]], &device, false).unwrap();
    let target = Tensor::from_data(&[1.0, 0.0], &device, false).unwrap();

    let lossfn = BCEWithLogitsLoss::new();
    let result = lossfn.loss((input, target));
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("must have the same shape"));
}
