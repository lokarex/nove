use nove_tensor::{BackendError, DType, Device, Shape, Tensor, TensorError};

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn enabled_cpu_devices() -> Vec<Device> {
    vec![
        #[cfg(feature = "candle-cpu")]
        nove_tensor::device::candle::cpu().unwrap(),
        #[cfg(feature = "native-cpu")]
        nove_tensor::device::native::cpu().unwrap(),
    ]
}

#[derive(Debug, PartialEq)]
struct Snapshot {
    values: Vec<f32>,
    shape: Shape,
    dtype: DType,
    device: Device,
    name: Option<String>,
    requires_grad: bool,
    graph_id: String,
    gradient: Option<(String, Vec<f32>)>,
}

fn snapshot(tensor: &Tensor) -> Result<Snapshot, TensorError> {
    Ok(Snapshot {
        values: tensor.to_vec()?,
        shape: tensor.shape()?,
        dtype: tensor.dtype()?,
        device: tensor.device()?,
        name: tensor.name()?,
        requires_grad: tensor.grad_enabled()?,
        graph_id: tensor.graph_id()?.to_string(),
        gradient: tensor
            .grad()?
            .map(|grad| -> Result<_, TensorError> {
                Ok((grad.graph_id()?.to_string(), grad.to_vec()?))
            })
            .transpose()?,
    })
}

#[test]
fn update_rejects_metadata_changes_without_mutating_target() -> TestResult {
    for device in enabled_cpu_devices() {
        for requires_grad in [false, true] {
            let target = Tensor::from_data(vec![1.0f32, 2.0], &device, requires_grad)?
                .require_name("parameter")?;
            if requires_grad {
                target.sum(None)?.backward()?;
            }
            let before = snapshot(&target)?;
            for shape in [Shape::from_dims(&[1]), Shape::from_dims(&[1, 2])] {
                let source = Tensor::ones(&shape, &DType::F32, &device, false)?;
                assert_eq!(
                    target.update_from_tensor(&source).unwrap_err(),
                    TensorError::ShapeMismatch(Shape::from_dims(&[2]), shape)
                );
                assert_eq!(snapshot(&target)?, before);
            }
            let source = Tensor::from_data(vec![3.0f64, 4.0], &device, false)?;
            assert_eq!(
                target.update_from_tensor(&source).unwrap_err(),
                TensorError::DTypeMismatch(DType::F32, DType::F64)
            );
            assert_eq!(snapshot(&target)?, before);
            for other_device in enabled_cpu_devices() {
                if other_device.backend() == device.backend() {
                    continue;
                }
                // Backend rejection has priority over dtype and shape rejection.
                let source = Tensor::from_data(vec![3.0f64], &other_device, false)?;
                assert!(matches!(target.update_from_tensor(&source).unwrap_err(),
                    TensorError::BackendError(BackendError::BackendMismatch { expected, found })
                    if expected == device.backend() && found == other_device.backend()));
                assert_eq!(snapshot(&target)?, before);
            }
        }
    }
    Ok(())
}

#[test]
fn update_preserves_identity_and_target_settings() -> TestResult {
    for device in enabled_cpu_devices() {
        let target =
            Tensor::from_data(vec![1.0f32, 2.0], &device, true)?.require_name("parameter")?;
        target.sum(None)?.backward()?;
        let alias = target.copy();
        let mut expected = snapshot(&target)?;
        expected.values = vec![3.0, 4.0];
        let source = Tensor::from_data(vec![3.0f32, 4.0], &device.clone(), true)?
            .require_name("replacement")?
            .detach()?;
        target.update_from_tensor(&source)?;
        assert_eq!(snapshot(&target)?, expected);
        assert_eq!(snapshot(&alias)?, expected);
    }
    Ok(())
}

#[test]
fn self_and_alias_updates_do_not_deadlock() -> TestResult {
    let (sender, receiver) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        let result = (|| -> Result<(), TensorError> {
            for device in enabled_cpu_devices() {
                for requires_grad in [false, true] {
                    let x = Tensor::from_data(vec![1.0f32, 2.0], &device, requires_grad)?;
                    let before = snapshot(&x)?;
                    x.update_from_tensor(&x)?;
                    x.update_from_tensor(&x.copy())?;
                    assert_eq!(snapshot(&x)?, before);
                }
            }
            Ok(())
        })();
        sender.send(result).unwrap();
    });
    receiver.recv_timeout(std::time::Duration::from_secs(10))??;
    worker.join().unwrap();
    Ok(())
}

#[test]
fn explicit_conversion_allows_compatible_update() -> TestResult {
    for device in enabled_cpu_devices() {
        let target = Tensor::from_data(vec![1.0f32, 2.0], &device, false)?;
        let source = Tensor::from_data(vec![3.0f64, 4.0], &device, false)?.to_dtype(&DType::F32)?;
        target.update_from_tensor(&source)?;
        assert_eq!(target.to_vec::<f32>()?, vec![3.0, 4.0]);
        for other_device in enabled_cpu_devices() {
            let source =
                Tensor::from_data(vec![5.0f32, 6.0], &other_device, false)?.to_device(&device)?;
            target.update_from_tensor(&source)?;
            assert_eq!(target.to_vec::<f32>()?, vec![5.0, 6.0]);
            assert_eq!(target.device()?, device);
            assert_eq!(target.dtype()?, DType::F32);
            assert_eq!(target.shape()?.dims(), &[2]);
        }
    }
    Ok(())
}

#[cfg(any(feature = "candle-cuda", feature = "candle-metal"))]
fn check_distinct_gpu_handles(first: Device, second: Device) -> TestResult {
    assert_eq!(first, second);
    assert!(
        !first
            .to_candle_device()?
            .same_device(&second.to_candle_device()?)
    );
    let target = Tensor::from_data(vec![1.0f32, 2.0], &first, false)?;
    let before = snapshot(&target)?;
    let source = Tensor::from_data(vec![3.0f32, 4.0], &second, false)?;
    assert!(matches!(
        target.update_from_tensor(&source),
        Err(TensorError::DeviceMismatch(..))
    ));
    assert_eq!(snapshot(&target)?, before);
    let source = Tensor::from_data(vec![5.0f32, 6.0], &first.clone(), false)?;
    target.update_from_tensor(&source)?;
    assert_eq!(target.to_vec::<f32>()?, vec![5.0, 6.0]);
    assert!(
        target
            .device()?
            .to_candle_device()?
            .same_device(&first.to_candle_device()?)
    );
    Ok(())
}

#[cfg(feature = "candle-cuda")]
#[test]
fn gpu_cuda_update_rejects_distinct_actual_handles() -> TestResult {
    let first = match nove_tensor::device::candle::cuda(0) {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping CUDA update test: {error}");
            return Ok(());
        }
    };
    check_distinct_gpu_handles(first, nove_tensor::device::candle::cuda(0)?)
}

#[cfg(feature = "candle-metal")]
#[test]
fn gpu_metal_update_rejects_distinct_actual_handles() -> TestResult {
    let first = match nove_tensor::device::candle::metal(0) {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping Metal update test: {error}");
            return Ok(());
        }
    };
    check_distinct_gpu_handles(first, nove_tensor::device::candle::metal(0)?)
}
