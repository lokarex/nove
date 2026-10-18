use nove_backend::{DType, Device, Shape, TensorBuffer, TensorPayload, backend::BackendStorage};

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn enabled_cpu_devices() -> Vec<Device> {
    vec![
        #[cfg(feature = "candle-cpu")]
        nove_backend::device::candle::cpu().unwrap(),
        #[cfg(feature = "native-cpu")]
        nove_backend::device::native::cpu().unwrap(),
    ]
}

#[test]
fn clone_preserves_values_and_metadata() -> TestResult {
    for device in enabled_cpu_devices() {
        for buffer in [
            TensorBuffer::U8(vec![1, 2]),
            TensorBuffer::U32(vec![1, 2]),
            TensorBuffer::I64(vec![9_007_199_254_740_993, -9_007_199_254_740_993]),
            TensorBuffer::F32(vec![1.0, 2.0]),
            TensorBuffer::F64(vec![1.0, 2.0]),
        ] {
            let payload = TensorPayload::new(buffer, Shape::from_dims(&[2]))?;
            let source = BackendStorage::from_payload(payload.clone(), &device)?;
            let copy = source.try_clone()?;
            assert_eq!(copy.to_tensor_buffer()?, *payload.buffer());
            assert_eq!(copy.dtype()?, source.dtype()?);
            assert_eq!(copy.shape()?, source.shape()?);
            assert_eq!(copy.backend_kind()?, device.backend());
            assert!(copy.same_device(&source));
        }
        for (values, dims) in [(vec![3.0f32], vec![]), (vec![], vec![2, 0, 3])] {
            let source =
                BackendStorage::from_vec(values.clone(), &Shape::from_dims(&dims), &device)?;
            let copy = source.try_clone()?;
            assert_eq!(copy.to_vec::<f32>()?, values);
            assert_eq!(copy.shape()?.dims(), dims);
            assert!(copy.same_device(&source));
        }
    }
    Ok(())
}

#[test]
fn clone_is_independent_after_zeroing_source() -> TestResult {
    for device in enabled_cpu_devices() {
        let mut source = BackendStorage::from_data(vec![1.0f32, 2.0], &device)?;
        let copy = source.try_clone()?;
        source.zero_set()?;
        assert_eq!(source.to_vec::<f32>()?, vec![0.0, 0.0]);
        assert_eq!(copy.to_vec::<f32>()?, vec![1.0, 2.0]);
        assert_eq!(source.dtype()?, DType::F32);
        assert_eq!(source.shape()?.dims(), &[2]);
    }
    Ok(())
}

#[test]
fn same_device_uses_backend_and_actual_handle() -> TestResult {
    for device in enabled_cpu_devices() {
        let a = BackendStorage::from_data(vec![1.0f32], &device)?;
        let b = BackendStorage::from_data(vec![2.0f32], &device.clone())?;
        assert!(a.same_device(&a));
        assert!(a.same_device(&b));
        assert!(b.same_device(&a));
        for other_device in enabled_cpu_devices() {
            let other = BackendStorage::from_data(vec![1.0f32], &other_device)?;
            assert_eq!(
                a.same_device(&other),
                device.backend() == other_device.backend()
            );
        }
    }
    Ok(())
}

#[cfg(any(feature = "candle-cuda", feature = "candle-metal"))]
fn check_gpu_clone(device: Device) -> TestResult {
    let reference = BackendStorage::from_data(vec![0.0f32], &device)?;
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let mut source = BackendStorage::from_data(vec![1.0f32, 2.0], &device)?.to_dtype(dtype)?;
        let copy = source.try_clone()?;
        source.zero_set()?;
        assert_eq!(copy.dtype()?, dtype);
        assert_eq!(copy.shape()?.dims(), &[2]);
        assert!(copy.same_device(&source));
        assert_eq!(copy.to_dtype(DType::F32)?.to_vec::<f32>()?, vec![1.0, 2.0]);
        assert!(copy.same_device(&reference));
    }
    Ok(())
}

#[cfg(feature = "candle-cuda")]
#[test]
#[ignore = "requires a CUDA device"]
fn gpu_cuda_clone_preserves_values_and_actual_device() -> TestResult {
    check_gpu_clone(nove_backend::device::candle::cuda(0)?)
}

#[cfg(feature = "candle-metal")]
#[test]
#[ignore = "requires a Metal device"]
fn gpu_metal_clone_preserves_values_and_actual_device() -> TestResult {
    check_gpu_clone(nove_backend::device::candle::metal(0)?)
}
