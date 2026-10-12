use nove_optimizer::{AdamBuilder, Optimizer, SgdBuilder};
use nove_tensor::{DType, Device, Tensor};

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn enabled_cpu_devices() -> Vec<Device> {
    vec![
        #[cfg(feature = "candle-cpu")]
        nove_tensor::device::candle::cpu().unwrap(),
        #[cfg(feature = "native-cpu")]
        nove_tensor::device::native::cpu().unwrap(),
    ]
}

fn assert_values(tensor: &Tensor, expected: &[f32]) -> TestResult {
    let values = tensor.to_vec::<f32>()?;
    assert_eq!(values.len(), expected.len());
    for (value, expected) in values.iter().zip(expected) {
        assert!(
            (value - expected).abs() < 1e-5,
            "expected {expected}, got {value}"
        );
    }
    Ok(())
}

#[test]
fn sgd_updates_parameter_and_momentum_without_changing_metadata() -> TestResult {
    for device in enabled_cpu_devices() {
        let parameter = Tensor::from_data(vec![1.0f32, 2.0], &device, true)?;
        let alias = parameter.copy();
        let id = parameter.graph_id()?;
        let mut optimizer = SgdBuilder::new(vec![parameter.copy()], 0.1)
            .momentum(0.9)
            .build()?;
        for expected in [[0.9, 1.9], [0.71, 1.71]] {
            optimizer.zero_grad()?;
            parameter.sum(None)?.backward()?;
            optimizer.step()?;
            assert_values(&parameter, &expected)?;
            assert_values(&alias, &expected)?;
            assert_eq!(parameter.graph_id()?, id);
            assert_eq!(parameter.shape()?.dims(), &[2]);
            assert_eq!(parameter.dtype()?, DType::F32);
            assert_eq!(parameter.device()?, device);
            assert!(parameter.grad_enabled()?);
        }
    }
    Ok(())
}

#[test]
fn adam_keeps_parameter_aliases_and_gradient_isolation() -> TestResult {
    for device in enabled_cpu_devices() {
        let mut x = Tensor::from_data(vec![1.0f32, 2.0], &device, true)?;
        let y = Tensor::from_data(vec![3.0f32, 4.0], &device, true)?;
        let aliases = [x.copy(), y.copy()];
        let ids = [x.graph_id()?, y.graph_id()?];
        let mut optimizer = AdamBuilder::new(vec![x.copy(), y.copy()], 0.01).build()?;
        for (step, expected) in [[[0.99, 1.99], [2.99, 3.99]], [[0.98, 1.98], [2.98, 3.98]]]
            .into_iter()
            .enumerate()
        {
            x.add(&y)?.sum(None)?.backward()?;
            if step == 0 {
                x.zero_grad()?;
                assert_values(&y.grad()?.unwrap(), &[1.0, 1.0])?;
                x.sum(None)?.backward()?;
            }
            optimizer.step()?;
            for index in 0..2 {
                assert_values(&aliases[index], &expected[index])?;
                assert_eq!(aliases[index].graph_id()?, ids[index]);
                assert_eq!(aliases[index].shape()?.dims(), &[2]);
                assert_eq!(aliases[index].dtype()?, DType::F32);
                assert_eq!(aliases[index].device()?, device);
                assert!(aliases[index].grad_enabled()?);
            }
            optimizer.zero_grad()?;
            assert_values(&x.grad()?.unwrap(), &[0.0, 0.0])?;
            assert_values(&y.grad()?.unwrap(), &[0.0, 0.0])?;
        }
    }
    Ok(())
}
