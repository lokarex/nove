mod common;

use common::TestDir;
use nove_backend::{BackendError, BackendKind, backend};
#[cfg(any(feature = "candle-cpu", feature = "native-cpu"))]
use nove_backend::{DType, Device, Shape, backend::BackendStorage};
use std::collections::HashMap;
use std::fs;

type TestResult = Result<(), Box<dyn std::error::Error>>;

#[cfg(any(feature = "candle-cpu", feature = "native-cpu"))]
fn named_storages(device: &Device) -> Result<HashMap<String, BackendStorage>, BackendError> {
    Ok(HashMap::from([
        (
            "weights".into(),
            BackendStorage::from_vec(vec![1.0f32, 2.0], &Shape::from_dims(&[1, 2]), device)?,
        ),
        (
            "ids".into(),
            BackendStorage::from_data(
                vec![9_007_199_254_740_993i64, -9_007_199_254_740_993],
                device,
            )?,
        ),
    ]))
}

#[cfg(any(feature = "candle-cpu", feature = "native-cpu"))]
fn assert_named_values(storages: &HashMap<String, BackendStorage>) -> TestResult {
    assert_eq!(storages.len(), 2);
    assert_eq!(storages["weights"].dtype()?, DType::F32);
    assert_eq!(storages["weights"].shape()?.dims(), &[1, 2]);
    assert_eq!(storages["weights"].to_vec::<f32>()?, vec![1.0, 2.0]);
    assert_eq!(storages["ids"].dtype()?, DType::I64);
    assert_eq!(storages["ids"].shape()?.dims(), &[2]);
    assert_eq!(
        storages["ids"].to_vec::<i64>()?,
        vec![9_007_199_254_740_993, -9_007_199_254_740_993]
    );
    Ok(())
}

#[cfg(not(feature = "candle"))]
#[test]
fn native_only_save_is_rejected_before_writing() -> TestResult {
    let dir = TestDir::new()?;
    for name in ["new.safetensors", "existing.safetensors"] {
        let path = dir.file(name);
        let original = (name == "existing.safetensors").then_some(b"unchanged".as_slice());
        if let Some(bytes) = original {
            fs::write(&path, bytes)?;
        }
        let error = backend::save_safetensors(path.to_str().unwrap(), HashMap::new()).unwrap_err();
        assert!(
            matches!(error, BackendError::UnsupportedOperation { backend: BackendKind::Native, operation }
            if operation == "save_safetensors")
        );
        match original {
            Some(bytes) => assert_eq!(fs::read(&path)?, bytes),
            None => assert!(!path.exists()),
        }
    }
    let path = dir.file("missing-parent").join("model.safetensors");
    assert!(
        matches!(backend::save_safetensors(path.to_str().unwrap(), HashMap::new()),
        Err(BackendError::UnsupportedOperation { backend: BackendKind::Native, operation })
        if operation == "save_safetensors")
    );
    assert!(!path.parent().unwrap().exists());
    Ok(())
}

#[cfg(all(feature = "native-cpu", not(feature = "candle")))]
#[test]
fn native_only_save_with_data_is_rejected_without_changing_source() -> TestResult {
    let dir = TestDir::new()?;
    let path = dir.file("native.safetensors");
    fs::write(&path, b"unchanged")?;
    let device = nove_backend::device::native::cpu()?;
    let source = named_storages(&device)?;
    assert!(
        matches!(backend::save_safetensors(path.to_str().unwrap(), source.clone()),
        Err(BackendError::UnsupportedOperation { backend: BackendKind::Native, operation })
        if operation == "save_safetensors")
    );
    assert_named_values(&source)?;
    assert_eq!(fs::read(&path)?, b"unchanged");
    Ok(())
}

#[cfg(feature = "native-cpu")]
#[test]
fn native_load_is_rejected_before_reading() -> TestResult {
    let dir = TestDir::new()?;
    let device = nove_backend::device::native::cpu()?;
    for name in ["absent.safetensors", "malformed.safetensors"] {
        let path = dir.file(name);
        if name == "malformed.safetensors" {
            fs::write(&path, b"unchanged")?;
        }
        assert!(
            matches!(backend::load_safetensors(path.to_str().unwrap(), &device),
            Err(BackendError::UnsupportedOperation { backend: BackendKind::Native, operation })
            if operation == "load_safetensors")
        );
        if name == "malformed.safetensors" {
            assert_eq!(fs::read(&path)?, b"unchanged");
        } else {
            assert!(!path.exists());
        }
    }
    Ok(())
}

#[cfg(feature = "candle")]
#[test]
fn candle_family_can_save_empty_collection_without_a_device() -> TestResult {
    let dir = TestDir::new()?;
    let path = dir.file("empty.safetensors");
    backend::save_safetensors(path.to_str().unwrap(), HashMap::new())?;
    assert!(fs::metadata(&path)?.len() > 0);
    #[cfg(feature = "candle-cpu")]
    assert!(
        backend::load_safetensors(
            path.to_str().unwrap(),
            &nove_backend::device::candle::cpu()?
        )?
        .is_empty()
    );
    Ok(())
}

#[cfg(feature = "candle-cpu")]
#[test]
fn candle_roundtrip_preserves_named_values_and_source() -> TestResult {
    let dir = TestDir::new()?;
    let path = dir.file("named.safetensors");
    let device = nove_backend::device::candle::cpu()?;
    let source = named_storages(&device)?;
    backend::save_safetensors(path.to_str().unwrap(), source.clone())?;
    let loaded = backend::load_safetensors(path.to_str().unwrap(), &device)?;
    assert_named_values(&loaded)?;
    assert_named_values(&source)?;
    for (name, storage) in &source {
        assert_eq!(storage.backend_kind()?, BackendKind::Candle);
        assert!(loaded[name].same_device(storage));
    }
    Ok(())
}

#[cfg(feature = "candle-cpu")]
#[test]
fn candle_roundtrip_preserves_half_dtypes() -> TestResult {
    let dir = TestDir::new()?;
    let path = dir.file("half.safetensors");
    let device = nove_backend::device::candle::cpu()?;
    let mut source = HashMap::new();
    for (name, dtype) in [("f16", DType::F16), ("bf16", DType::BF16)] {
        source.insert(
            name.to_string(),
            BackendStorage::from_data(vec![1.0f32, -2.0], &device)?.to_dtype(dtype)?,
        );
    }
    backend::save_safetensors(path.to_str().unwrap(), source.clone())?;
    let loaded = backend::load_safetensors(path.to_str().unwrap(), &device)?;
    assert_eq!(loaded.len(), 2);
    for (name, dtype) in [("f16", DType::F16), ("bf16", DType::BF16)] {
        assert_eq!(loaded[name].dtype()?, dtype);
        assert_eq!(loaded[name].shape()?.dims(), &[2]);
        assert_eq!(
            loaded[name].to_dtype(DType::F32)?.to_vec::<f32>()?,
            vec![1.0, -2.0]
        );
        assert_eq!(source[name].dtype()?, dtype);
        assert_eq!(
            source[name].to_dtype(DType::F32)?.to_vec::<f32>()?,
            vec![1.0, -2.0]
        );
    }
    Ok(())
}

#[cfg(feature = "candle-cpu")]
#[test]
fn candle_file_errors_keep_implementation_category() -> TestResult {
    let dir = TestDir::new()?;
    let device = nove_backend::device::candle::cpu()?;
    let missing = dir.file("missing.safetensors");
    let malformed = dir.file("malformed.safetensors");
    fs::write(&malformed, b"not safetensors")?;
    for path in [&missing, &malformed] {
        assert!(matches!(
            backend::load_safetensors(path.to_str().unwrap(), &device),
            Err(BackendError::BackendImplementation {
                backend: BackendKind::Candle,
                ..
            })
        ));
    }
    let output = dir.file("missing-parent").join("model.safetensors");
    assert!(matches!(
        backend::save_safetensors(output.to_str().unwrap(), named_storages(&device)?),
        Err(BackendError::BackendImplementation {
            backend: BackendKind::Candle,
            ..
        })
    ));
    assert!(!output.exists());
    assert_eq!(fs::read(&malformed)?, b"not safetensors");
    Ok(())
}

#[cfg(all(feature = "candle-cpu", feature = "native-cpu"))]
#[test]
fn native_and_mixed_storages_save_through_candle() -> TestResult {
    let dir = TestDir::new()?;
    let native = nove_backend::device::native::cpu()?;
    let candle = nove_backend::device::candle::cpu()?;
    for mixed in [false, true] {
        let source = if mixed {
            let mut source = named_storages(&native)?;
            source.insert(
                "weights".into(),
                named_storages(&candle)?.remove("weights").unwrap(),
            );
            source
        } else {
            named_storages(&native)?
        };
        let path = dir.file(if mixed {
            "mixed.safetensors"
        } else {
            "native.safetensors"
        });
        backend::save_safetensors(path.to_str().unwrap(), source.clone())?;
        let loaded = backend::load_safetensors(path.to_str().unwrap(), &candle)?;
        assert_named_values(&loaded)?;
        assert_named_values(&source)?;
        assert_eq!(source["ids"].backend_kind()?, BackendKind::Native);
        assert_eq!(
            source["weights"].backend_kind()?,
            if mixed {
                BackendKind::Candle
            } else {
                BackendKind::Native
            }
        );
        let native_reference = BackendStorage::from_data(vec![0f32], &native)?;
        assert!(source["ids"].same_device(&native_reference));
        for storage in loaded.values() {
            assert_eq!(storage.backend_kind()?, BackendKind::Candle);
        }
    }
    Ok(())
}

#[cfg(all(
    feature = "candle",
    feature = "native-cpu",
    not(feature = "candle-cpu")
))]
#[test]
fn native_save_uses_internal_cpu_without_public_cpu_feature() -> TestResult {
    assert!(matches!(nove_backend::device::candle::cpu(),
        Err(nove_backend::DeviceError::BackendFeatureNotEnabled(feature)) if feature == "candle-cpu"));
    let dir = TestDir::new()?;
    let path = dir.file("internal-cpu.safetensors");
    let device = nove_backend::device::native::cpu()?;
    let source = named_storages(&device)?;
    backend::save_safetensors(path.to_str().unwrap(), source.clone())?;
    assert!(fs::metadata(&path)?.len() > 0);
    assert_named_values(&source)?;
    Ok(())
}
