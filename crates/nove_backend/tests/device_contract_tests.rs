#[cfg(any(
    feature = "candle-cpu",
    feature = "native-cpu",
    feature = "candle-cuda",
    feature = "candle-metal"
))]
use nove_backend::{BackendKind, DeviceKind};
use nove_backend::{Device, DeviceError};

#[cfg(any(
    feature = "candle-cpu",
    feature = "native-cpu",
    feature = "candle-cuda",
    feature = "candle-metal"
))]
type TestResult = Result<(), Box<dyn std::error::Error>>;

fn assert_feature_error(result: Result<Device, DeviceError>, feature: &str) {
    let error = result.unwrap_err();
    assert!(
        matches!(&error, DeviceError::BackendFeatureNotEnabled(found) if found == feature),
        "expected feature {feature}, got {error:?}"
    );
}

#[test]
fn disabled_device_constructors_report_required_features() {
    #[cfg(not(feature = "candle-cpu"))]
    assert_feature_error(nove_backend::device::candle::cpu(), "candle-cpu");
    #[cfg(not(feature = "native-cpu"))]
    assert_feature_error(nove_backend::device::native::cpu(), "native-cpu");
    #[cfg(not(feature = "candle-cuda"))]
    {
        assert_feature_error(nove_backend::device::candle::cuda(0), "candle-cuda");
        assert_feature_error(
            nove_backend::device::candle::cuda_if_available(0),
            "candle-cuda",
        );
    }
    #[cfg(not(feature = "candle-metal"))]
    {
        assert_feature_error(nove_backend::device::candle::metal(0), "candle-metal");
        assert_feature_error(
            nove_backend::device::candle::metal_if_available(0),
            "candle-metal",
        );
    }
}

#[cfg(all(
    any(feature = "candle-cpu", feature = "native-cpu"),
    not(any(feature = "candle-cuda", feature = "candle-metal"))
))]
#[test]
fn cpu_default_device_uses_enabled_capability() -> TestResult {
    use nove_backend::backend::BackendStorage;

    let device = Device::default();
    #[cfg(feature = "candle-cpu")]
    let expected_backend = BackendKind::Candle;
    #[cfg(not(feature = "candle-cpu"))]
    let expected_backend = BackendKind::Native;
    assert_eq!(device.backend(), expected_backend);
    assert_eq!(device.location().backend(), expected_backend);
    assert_eq!(device.location().kind(), DeviceKind::Cpu);
    assert_eq!(device.location().index(), 0);
    let storage = BackendStorage::from_data(vec![1.0f32, 2.0], &device)?;
    assert_eq!(storage.backend_kind()?, expected_backend);
    assert_eq!(storage.to_vec::<f32>()?, vec![1.0, 2.0]);
    device.synchronize()?;
    Ok(())
}

#[cfg(all(feature = "native", not(feature = "native-cpu")))]
#[test]
fn native_family_exposes_storage_without_cpu_capability() {
    use nove_backend::backend::{BackendStorage, native::NativeStorage};

    let _: fn(NativeStorage) -> BackendStorage = BackendStorage::Native;
    let _: fn(NativeStorage) -> std::convert::Infallible = |storage| match storage {};
    assert_feature_error(nove_backend::device::native::cpu(), "native-cpu");

    #[cfg(feature = "candle-cpu")]
    {
        let device = nove_backend::device::candle::cpu().unwrap();
        let storage = BackendStorage::from_data(vec![1.0f32, 2.0], &device).unwrap();
        match &storage {
            BackendStorage::Candle(_) => {}
            BackendStorage::Native(storage) => match *storage {},
        }
        assert_eq!(storage.to_vec::<f32>().unwrap(), vec![1.0, 2.0]);
    }
}

#[cfg(feature = "native-cpu")]
#[test]
fn native_cpu_variant_preserves_owned_buffer_and_metadata() -> TestResult {
    use nove_backend::{
        DType, Shape,
        backend::{BackendStorage, native::NativeStorage},
    };
    use nove_native::cpu::{CpuBuffer, CpuDType};

    let data = vec![9_007_199_254_740_993i64, -2];
    let original_ptr = data.as_ptr();
    let device = nove_backend::device::native::cpu()?;
    let storage = BackendStorage::from_vec(data, &Shape::from_dims(&[2]), &device)?;
    assert_eq!(storage.backend_kind()?, BackendKind::Native);
    assert_eq!(storage.dtype()?, DType::I64);
    assert_eq!(storage.shape()?.dims(), &[2]);
    match &storage {
        BackendStorage::Native(NativeStorage::Cpu(storage)) => {
            assert_eq!(storage.dtype(), CpuDType::I64);
            assert_eq!(storage.shape(), &[2]);
            match storage.buffer() {
                CpuBuffer::I64(values) => assert_eq!(values.as_ptr(), original_ptr),
                other => panic!("expected I64 buffer, got {other:?}"),
            }
        }
        #[cfg(feature = "candle")]
        other => panic!("expected Native CPU storage, got {other:?}"),
    }
    assert_eq!(storage.to_vec::<i64>()?, vec![9_007_199_254_740_993, -2]);
    device.synchronize()?;
    Ok(())
}

#[cfg(feature = "native-cpu")]
#[test]
fn native_cpu_factories_select_cpu_storage() -> TestResult {
    use nove_backend::{
        DType, Shape, TensorBuffer, TensorPayload,
        backend::{BackendStorage, native::NativeStorage},
    };

    let device = nove_backend::device::native::cpu()?;
    let shape = Shape::from_dims(&[2]);
    let cases = [
        (
            BackendStorage::from_payload(
                TensorPayload::new(TensorBuffer::F32(vec![2.0f32, 3.0]), shape.clone())?,
                &device,
            )?,
            Some(vec![2.0, 3.0]),
        ),
        (
            BackendStorage::zeros(&shape, DType::F32, &device)?,
            Some(vec![0.0, 0.0]),
        ),
        (
            BackendStorage::ones(&shape, DType::F32, &device)?,
            Some(vec![1.0, 1.0]),
        ),
        (BackendStorage::rand(0.0f32, 1.0, &shape, &device)?, None),
        (
            BackendStorage::randn(2.0f32, 0.0, &shape, &device)?,
            Some(vec![2.0, 2.0]),
        ),
    ];
    for (storage, expected) in cases {
        match &storage {
            BackendStorage::Native(NativeStorage::Cpu(_)) => {}
            #[cfg(feature = "candle")]
            other => panic!("expected Native CPU storage, got {other:?}"),
        }
        assert_eq!(storage.shape()?, shape);
        assert_eq!(storage.dtype()?, DType::F32);
        let values = storage.to_vec::<f32>()?;
        if let Some(expected) = expected {
            assert_eq!(values, expected);
        } else {
            assert!(values.iter().all(|value| (0.0..1.0).contains(value)));
        }
    }
    Ok(())
}

#[cfg(not(any(
    feature = "candle-cpu",
    feature = "candle-cuda",
    feature = "candle-metal",
    feature = "native-cpu"
)))]
#[test]
#[should_panic(expected = "Device::default() requires at least one backend feature")]
fn family_without_device_keeps_default_device_panic() {
    let _ = Device::default();
}

#[cfg(feature = "candle-cuda")]
#[test]
#[ignore = "requires a CUDA device"]
fn cuda_device_selection_and_fallback_respect_features() -> TestResult {
    nove_backend::device::candle::cuda(0)?;
    #[cfg(feature = "native-cpu")]
    assert_eq!(
        nove_backend::device::native::cpu()?.backend(),
        BackendKind::Native
    );
    let device = Device::default();
    assert_eq!(device.backend(), BackendKind::Candle);
    assert_eq!(device.location().kind(), DeviceKind::Cuda);
    assert_eq!(device.location().index(), 0);

    let invalid_index = i32::MAX as usize;
    assert!(nove_backend::device::candle::cuda(invalid_index).is_err());
    let fallback = nove_backend::device::candle::cuda_if_available(invalid_index);
    #[cfg(feature = "candle-cpu")]
    {
        let fallback = fallback?;
        assert_eq!(fallback.backend(), BackendKind::Candle);
        assert_eq!(fallback.location().kind(), DeviceKind::Cpu);
        assert_eq!(fallback.location().index(), 0);
    }
    #[cfg(not(feature = "candle-cpu"))]
    assert_feature_error(fallback, "candle-cpu");
    Ok(())
}

#[cfg(feature = "candle-metal")]
#[test]
#[ignore = "requires a Metal device"]
fn metal_device_selection_uses_enabled_metal() -> TestResult {
    nove_backend::device::candle::metal(0)?;
    #[cfg(not(feature = "candle-cuda"))]
    {
        let device = Device::default();
        assert_eq!(device.backend(), BackendKind::Candle);
        assert_eq!(device.location().kind(), DeviceKind::Metal);
        assert_eq!(device.location().index(), 0);
    }
    let device = nove_backend::device::candle::metal_if_available(0)?;
    assert_eq!(device.backend(), BackendKind::Candle);
    assert_eq!(device.location().kind(), DeviceKind::Metal);
    assert_eq!(device.location().index(), 0);
    Ok(())
}
