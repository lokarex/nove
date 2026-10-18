use nove_backend::{
    BackendError, DType, Device, IntoTensorPayload, Shape, TensorBuffer, TensorPayload,
    backend::BackendStorage,
};

fn enabled_cpu_devices() -> Vec<Device> {
    vec![
        #[cfg(feature = "candle-cpu")]
        nove_backend::device::candle::cpu().unwrap(),
        #[cfg(feature = "native-cpu")]
        nove_backend::device::native::cpu().unwrap(),
    ]
}

#[test]
fn scalar_into_payload() {
    let p = 42i64.into_tensor_payload().unwrap();
    assert_eq!(p.shape().rank(), 0);
}

#[test]
fn vec_1d_into_payload() {
    let p = vec![1.0f32, 2.0, 3.0].into_tensor_payload().unwrap();
    assert_eq!(p.shape().dims(), &[3]);
    assert_eq!(p.buffer().dtype(), DType::F32);
}

#[test]
fn vec_2d_into_payload() {
    let p = vec![vec![1.0f64; 2]; 2].into_tensor_payload().unwrap();
    assert_eq!(p.shape().dims(), &[2, 2]);
    assert_eq!(p.buffer().dtype(), DType::F64);
}

#[test]
fn vec_3d_into_payload() {
    let p = vec![vec![vec![1u32; 2]; 2]; 2]
        .into_tensor_payload()
        .unwrap();
    assert_eq!(p.shape().dims(), &[2, 2, 2]);
    assert_eq!(p.buffer().dtype(), DType::U32);
}

#[test]
fn ragged_2d_fails() {
    let result = vec![vec![1.0f32; 2], vec![3.0]].into_tensor_payload();
    assert!(result.is_err());
}

#[test]
fn ragged_3d_fails() {
    let result = vec![vec![vec![1u8; 2]; 2], vec![vec![5u8]]].into_tensor_payload();
    assert!(result.is_err());
}

#[test]
fn empty_vec_produces_zero_dim() {
    let p = Vec::<f32>::new().into_tensor_payload().unwrap();
    assert_eq!(p.shape().dims(), &[0]);
}

#[test]
fn slice_into_payload() {
    let p = (&[1.0f32; 3][..]).into_tensor_payload().unwrap();
    assert_eq!(p.shape().dims(), &[3]);
}

#[test]
fn vec_of_slices_2d() {
    let r1 = [1.0f64, 2.0];
    let r2 = [3.0, 4.0];
    let p = vec![&r1[..], &r2[..]].into_tensor_payload().unwrap();
    assert_eq!(p.shape().dims(), &[2, 2]);
}

#[test]
fn fixed_array_2d() {
    let p = (&[[1.0f32, 2.0], [3.0, 4.0]])
        .into_tensor_payload()
        .unwrap();
    assert_eq!(p.shape().dims(), &[2, 2]);
}

#[test]
fn factory_overflow_is_invalid_operation() {
    type Factory = fn(&Shape, &Device) -> Result<BackendStorage, BackendError>;
    let factories: [(&str, Factory); 6] = [
        ("from_vec", |shape, device| {
            BackendStorage::from_vec(Vec::<f32>::new(), shape, device)
        }),
        ("from_slice", |shape, device| {
            BackendStorage::from_slice::<f32>(&[], shape, device)
        }),
        ("zeros", |shape, device| {
            BackendStorage::zeros(shape, DType::F32, device)
        }),
        ("ones", |shape, device| {
            BackendStorage::ones(shape, DType::F32, device)
        }),
        ("rand", |shape, device| {
            BackendStorage::rand(0.0f32, 1.0, shape, device)
        }),
        ("randn", |shape, device| {
            BackendStorage::randn(0.0f32, 1.0, shape, device)
        }),
    ];
    for device in enabled_cpu_devices() {
        for dims in [[usize::MAX, 2], [usize::MAX / 2 + 1, 2]] {
            let shape = Shape::from_dims(&dims);
            for (name, factory) in factories {
                let result = factory(&shape, &device);
                assert!(
                    matches!(result, Err(BackendError::InvalidOperation(_))),
                    "{name} on {:?} should reject {dims:?}: {result:?}",
                    device.backend(),
                );
            }
        }
    }
}

#[test]
fn owned_creation_preserves_all_dtypes() {
    let buffers = [
        TensorBuffer::U8(vec![1, 2]),
        TensorBuffer::U32(vec![1, u32::MAX]),
        TensorBuffer::I64(vec![9_007_199_254_740_993, -9_007_199_254_740_993]),
        TensorBuffer::F32(vec![1.0, -2.0]),
        TensorBuffer::F64(vec![1.25, -2.5]),
    ];
    for device in enabled_cpu_devices() {
        for buffer in &buffers {
            let payload = TensorPayload::new(buffer.clone(), Shape::from_dims(&[2])).unwrap();
            let storage = BackendStorage::from_payload(payload, &device).unwrap();
            assert_eq!(storage.dtype().unwrap(), buffer.dtype());
            assert_eq!(storage.shape().unwrap().dims(), &[2]);
            assert_eq!(storage.backend_kind().unwrap(), device.backend());
            assert_eq!(&storage.to_tensor_buffer().unwrap(), buffer);
        }
    }
}

#[test]
fn factories_preserve_scalar_and_empty_shapes() {
    let cases: &[(&[usize], usize)] = &[(&[], 1), (&[2, 3], 6), (&[2, 0, 3], 0)];
    for device in enabled_cpu_devices() {
        for &(dims, count) in cases {
            let shape = Shape::from_dims(dims);
            let zeros = BackendStorage::zeros(&shape, DType::F32, &device).unwrap();
            let ones = BackendStorage::ones(&shape, DType::F32, &device).unwrap();
            assert_eq!(zeros.to_vec::<f32>().unwrap(), vec![0.0; count]);
            assert_eq!(ones.to_vec::<f32>().unwrap(), vec![1.0; count]);
            let rand = BackendStorage::rand(0.0f32, 1.0, &shape, &device).unwrap();
            let randn = BackendStorage::randn(0.0f32, 1.0, &shape, &device).unwrap();
            for storage in [zeros, ones, rand, randn] {
                assert_eq!(storage.dtype().unwrap(), DType::F32);
                assert_eq!(storage.shape().unwrap().dims(), dims);
                assert_eq!(storage.to_vec::<f32>().unwrap().len(), count);
            }
        }

        let scalar = BackendStorage::from_data(42i64, &device).unwrap();
        assert_eq!(scalar.dtype().unwrap(), DType::I64);
        assert!(scalar.shape().unwrap().dims().is_empty());
        assert_eq!(scalar.to_scalar::<i64>().unwrap(), 42);

        let empty =
            BackendStorage::from_vec(Vec::<i64>::new(), &Shape::from_dims(&[2, 0, 3]), &device)
                .unwrap();
        assert_eq!(empty.dtype().unwrap(), DType::I64);
        assert_eq!(empty.shape().unwrap().dims(), &[2, 0, 3]);
        assert!(empty.to_vec::<i64>().unwrap().is_empty());
    }
}

#[test]
fn borrowed_slice_remains_independent() {
    for device in enabled_cpu_devices() {
        let mut source = [1.0f32, 2.0];
        let storage =
            BackendStorage::from_slice(&source, &Shape::from_dims(&[2]), &device).unwrap();
        source[0] = 99.0;
        assert_eq!(source, [99.0, 2.0]);
        assert_eq!(storage.to_vec::<f32>().unwrap(), vec![1.0, 2.0]);
    }
}

#[test]
fn storage_creation_rejects_mismatched_length() {
    for device in enabled_cpu_devices() {
        let shape = Shape::from_dims(&[2, 2]);
        let from_vec = BackendStorage::from_vec(vec![1.0f32; 3], &shape, &device);
        let from_slice = BackendStorage::from_slice(&[1.0f32; 3], &shape, &device);
        for result in [from_vec, from_slice] {
            assert!(matches!(result, Err(BackendError::InvalidOperation(_))));
        }
    }
}

#[cfg(all(feature = "candle-cpu", feature = "native-cpu"))]
#[test]
fn cross_backend_transfer_preserves_i64() {
    let native = nove_backend::device::native::cpu().unwrap();
    let candle = nove_backend::device::candle::cpu().unwrap();
    let values = vec![9_007_199_254_740_993i64, -9_007_199_254_740_993];
    for (source_device, target_device) in [(&native, &candle), (&candle, &native)] {
        let source =
            BackendStorage::from_vec(values.clone(), &Shape::from_dims(&[2]), source_device)
                .unwrap();
        let transferred = source.to_device(target_device).unwrap();
        assert_eq!(transferred.backend_kind().unwrap(), target_device.backend());
        assert_eq!(transferred.dtype().unwrap(), DType::I64);
        assert_eq!(transferred.shape().unwrap().dims(), &[2]);
        assert_eq!(transferred.to_vec::<i64>().unwrap(), values);
        assert_eq!(source.to_vec::<i64>().unwrap(), values);
    }
}
