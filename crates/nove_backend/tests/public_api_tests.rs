use nove_backend as root;
use nove_backend::backend as legacy;
use root::{DType, Device, Shape};
use std::collections::HashMap;

fn enabled_cpu_devices() -> Vec<Device> {
    vec![
        #[cfg(feature = "candle-cpu")]
        root::device::candle::cpu().unwrap(),
        #[cfg(feature = "native-cpu")]
        root::device::native::cpu().unwrap(),
    ]
}

#[test]
fn root_and_backend_paths_remain_usable() {
    let kind: legacy::BackendKind = root::BackendKind::Candle;
    assert_eq!(kind, legacy::BackendKind::Candle);
    let error: root::BackendError = legacy::BackendError::InvalidOperation("invalid input".into());
    assert!(matches!(error, root::BackendError::InvalidOperation(_)));

    let buffer: legacy::TensorBuffer = root::TensorBuffer::F32(vec![1.0, 2.0]);
    let payload: root::TensorPayload =
        legacy::TensorPayload::new(buffer, Shape::from_dims(&[2])).unwrap();
    let legacy_payload: legacy::TensorPayload = payload.clone();
    assert_eq!(legacy_payload.buffer().dtype(), DType::F32);
    assert_eq!(legacy_payload.buffer().len(), 2);
    assert_eq!(legacy_payload.shape().dims(), &[2]);

    assert_eq!(<f32 as root::TensorElement>::dtype(), DType::F32);
    assert_eq!(<f32 as legacy::TensorElement>::dtype(), DType::F32);
    assert_eq!(root::FloatTensorElement::to_f64(1.5f32), 1.5);
    assert_eq!(legacy::FloatTensorElement::to_f64(1.5f32), 1.5);
    let converted: legacy::TensorPayload =
        root::IntoTensorPayload::into_tensor_payload(vec![1.0f32, 2.0]).unwrap();
    let converted_from_legacy: root::TensorPayload =
        legacy::IntoTensorPayload::into_tensor_payload(vec![1.0f32, 2.0]).unwrap();
    assert_eq!(converted, payload);
    assert_eq!(converted_from_legacy, payload);

    for device in enabled_cpu_devices() {
        let storage = legacy::BackendStorage::from_payload(payload.clone(), &device).unwrap();
        assert_eq!(storage.dtype().unwrap(), DType::F32);
        assert_eq!(storage.shape().unwrap().dims(), &[2]);
        assert_eq!(storage.backend_kind().unwrap(), device.backend());
        assert_eq!(storage.to_vec::<f32>().unwrap(), vec![1.0, 2.0]);
    }
}

#[test]
fn enabled_adapter_paths_remain_usable() {
    for device in enabled_cpu_devices() {
        let storage = legacy::BackendStorage::from_data(vec![1.0f32, 2.0], &device).unwrap();
        match &storage {
            #[cfg(feature = "candle")]
            legacy::BackendStorage::Candle(adapter) => {
                let _: &legacy::candle::CandleStorage = adapter;
                assert_eq!(device.backend(), root::BackendKind::Candle);
            }
            #[cfg(feature = "native")]
            legacy::BackendStorage::Native(adapter) => {
                let _: &legacy::native::NativeStorage = adapter;
                assert_eq!(device.backend(), root::BackendKind::Native);
            }
        }
        assert_eq!(storage.to_vec::<f32>().unwrap(), vec![1.0, 2.0]);
    }
}

#[test]
fn serialization_function_signatures_remain_usable() {
    type SaveSafetensors =
        fn(&str, HashMap<String, legacy::BackendStorage>) -> Result<(), root::BackendError>;
    type LoadSafetensors =
        fn(&str, &Device) -> Result<HashMap<String, legacy::BackendStorage>, root::BackendError>;

    let _: SaveSafetensors = legacy::save_safetensors;
    let _: LoadSafetensors = legacy::load_safetensors;
}
