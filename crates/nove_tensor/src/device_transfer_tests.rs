#[cfg(any(feature = "candle-cuda", feature = "candle-metal"))]
use crate::{DType, TensorError};
use crate::{Device, DeviceLocation, Tensor};

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn enabled_cpu_devices() -> Vec<Device> {
    vec![
        #[cfg(feature = "candle-cpu")]
        crate::device::candle::cpu().unwrap(),
        #[cfg(feature = "native-cpu")]
        crate::device::native::cpu().unwrap(),
    ]
}

#[test]
fn device_location_exports_are_the_same_type() {
    for device in enabled_cpu_devices() {
        let location: nove_backend::device::DeviceLocation = device.location();
        let location: nove_backend::DeviceLocation = location;
        let location: DeviceLocation = location;
        let location: nove::tensor::DeviceLocation = location;
        let location: nove::device::DeviceLocation = location;
        assert_eq!(location, device.location());
    }
}

fn check_same_device_transfer(source_device: &Device, target_device: &Device) -> TestResult {
    assert_eq!(source_device, target_device);
    let source =
        Tensor::from_data(vec![1.0f32, 2.0], source_device, true)?.require_name("source")?;
    let moved = source.to_device(target_device)?;
    assert_eq!(moved.graph_id()?, source.graph_id()?);
    assert_eq!(moved.device()?, *target_device);
    assert_actual_device(&source, source_device)?;
    assert_actual_device(&moved, target_device)?;
    let replacement = Tensor::from_data(vec![3.0f32, 4.0], source_device, false)?;
    source.update_from_tensor(&replacement)?;
    assert_eq!(moved.to_vec::<f32>()?, vec![3.0, 4.0]);
    assert_actual_device(&moved, target_device)?;
    assert_eq!(moved.name()?, Some("source".into()));
    assert!(moved.grad_enabled()?);
    Ok(())
}

#[test]
fn cpu_same_device_transfer_shares_tensor_data() -> TestResult {
    for source in enabled_cpu_devices() {
        check_same_device_transfer(&source, &source.clone())?;
        for target in enabled_cpu_devices() {
            if source.backend() == target.backend() {
                check_same_device_transfer(&source, &target)?;
            }
        }
    }
    Ok(())
}

fn assert_actual_device(tensor: &Tensor, device: &Device) -> TestResult {
    assert_eq!(tensor.device()?, *device);
    let reference = crate::backend::BackendStorage::from_data(vec![0.0f32], device)?;
    assert!(tensor.backend_storage()?.same_device(&reference));
    Ok(())
}

#[cfg(feature = "candle")]
fn check_creation_and_clone(device: &Device) -> TestResult {
    let tensor = Tensor::from_data(vec![1.0f32, 2.0], device, true)?;
    assert_actual_device(&tensor, device)?;
    let cloned = tensor.try_clone()?;
    assert_actual_device(&cloned, device)?;
    assert_eq!(cloned.to_vec::<f32>()?, vec![1.0, 2.0]);
    assert!(cloned.grad_enabled()?);
    Ok(())
}

#[cfg(feature = "candle-cpu")]
#[test]
fn candle_cpu_metadata_matches_actual_storage() -> TestResult {
    check_creation_and_clone(&crate::device::candle::cpu()?)
}

#[cfg(feature = "candle-cuda")]
#[test]
#[ignore = "requires a CUDA device"]
fn cuda_same_instance_transfer_shares_tensor_data() -> TestResult {
    let device = crate::device::candle::cuda(0)?;
    check_same_device_transfer(&device, &device.clone())?;
    check_creation_and_clone(&device)
}

#[cfg(feature = "candle-cuda")]
#[test]
#[ignore = "requires a CUDA device"]
fn cuda_transfer_uses_target_instance_and_preserves_metadata() -> TestResult {
    let source_device = crate::device::candle::cuda(0)?;
    let target_device = crate::device::candle::cuda(0)?;
    let source =
        Tensor::from_data(vec![1.0f32, 2.0], &source_device, true)?.require_name("source")?;
    let moved = source.to_device(&target_device)?;
    assert_actual_device(&moved, &target_device)?;
    assert_actual_device(&source, &source_device)?;
    assert_eq!(moved.to_vec::<f32>()?, vec![1.0, 2.0]);
    assert_eq!(moved.shape()?.dims(), &[2]);
    assert_eq!(moved.dtype()?, DType::F32);
    assert_eq!(moved.name()?, source.name()?);
    assert!(moved.grad_enabled()?);
    assert_ne!(moved.graph_id()?, source.graph_id()?);
    let replacement = Tensor::from_data(vec![3.0f32, 4.0], &source_device, false)?;
    source.update_from_tensor(&replacement)?;
    assert_eq!(source.to_vec::<f32>()?, vec![3.0, 4.0]);
    assert_eq!(moved.to_vec::<f32>()?, vec![1.0, 2.0]);

    Ok(())
}

#[cfg(feature = "candle-cuda")]
#[test]
#[ignore = "requires a CUDA device"]
fn cuda_transfer_preserves_large_i64_values_and_independence() -> TestResult {
    let source_device = crate::device::candle::cuda(0)?;
    let target_device = crate::device::candle::cuda(0)?;
    let source = Tensor::from_data(vec![9007199254740993i64], &source_device, false)?;
    let moved = source.to_device(&target_device)?;
    assert_actual_device(&moved, &target_device)?;
    assert_eq!(moved.dtype()?, DType::I64);
    assert_eq!(moved.shape()?.dims(), &[1]);
    assert_eq!(moved.to_vec::<i64>()?, vec![9007199254740993]);
    source.update_from_tensor(&Tensor::from_data(vec![7i64], &source_device, false)?)?;
    assert_eq!(source.to_vec::<i64>()?, vec![7]);
    assert_eq!(moved.to_vec::<i64>()?, vec![9007199254740993]);
    Ok(())
}

#[cfg(feature = "candle-cuda")]
#[test]
#[ignore = "requires a CUDA device"]
fn cuda_transfer_backward_returns_gradient_to_source_instance() -> TestResult {
    let source_device = crate::device::candle::cuda(0)?;
    let target_device = crate::device::candle::cuda(0)?;
    let source = Tensor::from_data(vec![1.0f32, 2.0], &source_device, true)?;
    let moved = source.to_device(&target_device)?;
    assert_actual_device(&moved, &target_device)?;
    moved.sum(None)?.backward()?;
    let grad = source.grad()?.expect("source gradient should be populated");
    assert_actual_device(&grad, &source_device)?;
    assert_eq!(grad.to_vec::<f32>()?, vec![1.0, 1.0]);
    Ok(())
}

#[cfg(feature = "candle-cuda")]
#[test]
#[ignore = "requires a CUDA device"]
fn cuda_update_requires_transfer_to_target_instance() -> TestResult {
    let target_device = crate::device::candle::cuda(0)?;
    let source_device = crate::device::candle::cuda(0)?;
    let target =
        Tensor::from_data(vec![1.0f32, 2.0], &target_device, true)?.require_name("target")?;
    let graph_id = target.graph_id()?;
    let source = Tensor::from_data(vec![3.0f32, 4.0], &source_device, false)?;
    assert!(matches!(
        target.update_from_tensor(&source),
        Err(TensorError::DeviceMismatch(..))
    ));
    assert_eq!(target.to_vec::<f32>()?, vec![1.0, 2.0]);
    assert_eq!(target.graph_id()?, graph_id);
    assert_eq!(target.name()?, Some("target".into()));
    assert!(target.grad_enabled()?);
    assert_actual_device(&target, &target_device)?;
    target.update_from_tensor(&source.to_device(&target_device)?)?;
    assert_eq!(target.to_vec::<f32>()?, vec![3.0, 4.0]);
    assert_actual_device(&target, &target_device)?;
    Ok(())
}

#[cfg(feature = "candle-metal")]
#[test]
#[ignore = "requires a Metal device"]
fn metal_distinct_instance_transfer_preserves_capability_error() -> TestResult {
    let source_device = crate::device::candle::metal(0)?;
    let target_device = crate::device::candle::metal(0)?;
    check_same_device_transfer(&source_device, &source_device.clone())?;
    check_creation_and_clone(&source_device)?;
    let source =
        Tensor::from_data(vec![1.0f32, 2.0], &source_device, true)?.require_name("source")?;
    let graph_id = source.graph_id()?;
    assert!(matches!(
        source.to_device(&target_device),
        Err(TensorError::BackendError(
            crate::BackendError::BackendImplementation {
                backend: crate::BackendKind::Candle,
                ..
            }
        ))
    ));
    assert_eq!(source.to_vec::<f32>()?, vec![1.0, 2.0]);
    assert_eq!(source.graph_id()?, graph_id);
    assert_eq!(source.dtype()?, DType::F32);
    assert_eq!(source.shape()?.dims(), &[2]);
    assert_eq!(source.name()?, Some("source".into()));
    assert!(source.grad_enabled()?);
    assert_actual_device(&source, &source_device)?;
    Ok(())
}
