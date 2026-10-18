#![cfg(feature = "native-cpu")]

use nove_backend::{BackendError, BackendKind, DType, Shape, backend::BackendStorage, device};

type TestResult = Result<(), Box<dyn std::error::Error>>;

#[test]
fn native_invalid_indexes_are_invalid_operations() -> TestResult {
    let device = device::native::cpu()?;
    let source = BackendStorage::from_data(vec![1.0f32, 2.0], &device)?;
    for index in [-1i64, 2] {
        let indexes = BackendStorage::from_data(vec![index], &device)?;
        assert!(matches!(
            source.gather(&indexes, 0),
            Err(BackendError::InvalidOperation(_))
        ));
        assert_eq!(source.to_vec::<f32>()?, vec![1.0, 2.0]);
        assert_eq!(indexes.to_vec::<i64>()?, vec![index]);
    }
    Ok(())
}

#[test]
fn native_nonunit_squeeze_is_invalid_operation() -> TestResult {
    let device = device::native::cpu()?;
    let source = BackendStorage::from_data(vec![1.0f32, 2.0], &device)?;
    assert!(matches!(
        source.squeeze(0),
        Err(BackendError::InvalidOperation(_))
    ));
    assert_eq!(source.to_vec::<f32>()?, vec![1.0, 2.0]);
    assert_eq!(source.shape()?.dims(), &[2]);
    Ok(())
}

#[test]
fn native_empty_extrema_are_invalid_operations() -> TestResult {
    let device = device::native::cpu()?;
    let source = BackendStorage::from_data(Vec::<f32>::new(), &device)?;
    for result in [source.max_all(), source.min_all()] {
        assert!(matches!(result, Err(BackendError::InvalidOperation(_))));
    }
    assert_eq!(source.to_vec::<f32>()?, Vec::<f32>::new());
    assert_eq!(source.shape()?.dims(), &[0]);
    Ok(())
}

#[test]
fn native_kernel_shape_errors_keep_implementation_category() -> TestResult {
    let device = device::native::cpu()?;
    let source = BackendStorage::ones(&Shape::from_dims(&[1, 2, 3]), DType::F32, &device)?;
    let kernel = BackendStorage::ones(&Shape::from_dims(&[1, 1, 1]), DType::F32, &device)?;
    let error = source.conv1d(&kernel, 0, 1, 1, 1).unwrap_err();
    assert!(
        matches!(error, BackendError::BackendImplementation { backend: BackendKind::Native, message }
        if message.contains("[1, 2, 1]") && message.contains("[1, 1, 1]"))
    );
    assert_eq!(source.to_vec::<f32>()?, vec![1.0; 6]);
    assert_eq!(kernel.to_vec::<f32>()?, vec![1.0]);
    Ok(())
}

#[test]
fn float_gather_indexes_keep_validation_error() -> TestResult {
    let device = device::native::cpu()?;
    let source = BackendStorage::from_data(vec![1.0f32, 2.0], &device)?;
    let indexes = BackendStorage::from_data(vec![0.0f32], &device)?;
    assert!(matches!(
        source.gather(&indexes, 0),
        Err(BackendError::DTypeMismatch {
            operation: "gather",
            input_index: 1,
            expected: DType::I64,
            found: DType::F32,
        })
    ));
    Ok(())
}

#[test]
fn native_batch_matmul_remains_unsupported() -> TestResult {
    let device = device::native::cpu()?;
    let source = BackendStorage::ones(&Shape::from_dims(&[1, 2, 2]), DType::F32, &device)?;
    assert!(matches!(source.broadcast_matmul(&source),
        Err(BackendError::UnsupportedOperation { backend: BackendKind::Native, operation })
        if operation == "broadcast_matmul only supports rank-2 CPU tensors"));
    Ok(())
}
