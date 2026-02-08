use nove_lossfn::{CrossEntropy, LossFn};
use nove_tensor::{Device, Tensor};

#[test]
fn test_cross_entropy_lossfn() {
    let device = Device::cpu();

    let input = Tensor::from_data(
        &[
            [2.1, 1.5, 0.3, 3.2, 0.8, 1.9, 2.7, 0.5, 1.2, 2.4],
            [1.8, 2.9, 1.1, 0.7, 2.3, 1.4, 0.9, 3.1, 1.6, 0.4],
            [0.6, 2.2, 3.5, 1.3, 0.9, 2.8, 1.7, 2.0, 0.3, 1.1],
            [2.5, 0.8, 1.9, 2.7, 3.4, 0.5, 1.2, 2.1, 1.8, 0.7],
        ],
        &device,
        false,
    )
    .unwrap();
    let target = Tensor::from_data(&[0i64, 3i64, 7i64, 5i64], &device, false).unwrap();

    let lossfn = CrossEntropy::new();

    let loss = lossfn.loss((input, target)).unwrap();

    let actual = loss.to_scalar::<f64>().unwrap();
    let expected = 3.066291;
    assert!(
        (actual - expected).abs() < 1e-6,
        "Expected {}, got {}",
        expected,
        actual
    );
}
