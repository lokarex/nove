use nove_backend::{BackendError, DType, Shape, TensorBuffer, TensorPayload};

#[test]
fn payload_new_matching_length_succeeds() {
    let p = TensorPayload::new(TensorBuffer::F32(vec![1.0; 4]), Shape::from_dims(&[2, 2])).unwrap();
    assert_eq!(p.shape().dims(), &[2, 2]);
    assert_eq!(p.buffer().dtype(), DType::F32);
}

#[test]
fn payload_new_mismatched_length_fails() {
    let r = TensorPayload::new(TensorBuffer::F32(vec![1.0; 3]), Shape::from_dims(&[2, 2]));
    assert!(matches!(r.unwrap_err(), BackendError::InvalidOperation(_)));
}

#[test]
fn payload_new_zero_elements_succeeds() {
    assert!(TensorPayload::new(TensorBuffer::F32(vec![]), Shape::from_dims(&[0])).is_ok());
}

#[test]
fn payload_scalar_shape() {
    let p = TensorPayload::new(TensorBuffer::I64(vec![42]), Shape::from(())).unwrap();
    assert_eq!(p.shape().rank(), 0);
}

#[test]
fn payload_all_dtypes() {
    let cases = [
        (TensorBuffer::U8(vec![1, 2]), 2),
        (TensorBuffer::U32(vec![1, 2]), 2),
        (TensorBuffer::I64(vec![1, 2]), 2),
        (TensorBuffer::F32(vec![1.0, 2.0]), 2),
        (TensorBuffer::F64(vec![1.0, 2.0]), 2),
    ];
    for (buf, len) in cases {
        let p = TensorPayload::new(buf.clone(), Shape::from_dims(&[len])).unwrap();
        assert_eq!(p.buffer().len(), len);
    }
}

#[test]
fn empty_buffer_is_empty() {
    assert!(TensorBuffer::F64(vec![]).is_empty());
    assert!(!TensorBuffer::F32(vec![1.0]).is_empty());
}

#[test]
fn payload_overflow_is_error() {
    for dims in [[usize::MAX, 2], [usize::MAX / 2 + 1, 2]] {
        let result = TensorPayload::new(TensorBuffer::F32(vec![]), Shape::from_dims(&dims));
        assert!(matches!(result, Err(BackendError::InvalidOperation(_))));
    }
}

#[test]
fn payload_zero_dimension_precedes_overflow() {
    for dims in [[0, usize::MAX, 2], [usize::MAX, 0, 2], [usize::MAX, 2, 0]] {
        let shape = Shape::from_dims(&dims);
        let payload = TensorPayload::new(TensorBuffer::F32(vec![]), shape.clone()).unwrap();
        assert_eq!(payload.shape().dims(), &dims);
        assert!(payload.buffer().is_empty());
        let result = TensorPayload::new(TensorBuffer::F32(vec![1.0]), shape);
        assert!(matches!(result, Err(BackendError::InvalidOperation(_))));
    }
}

#[test]
fn into_parts_preserves_owned_buffer() {
    let data = vec![9_007_199_254_740_993i64];
    let original_data_ptr = data.as_ptr();
    let shape = Shape::from_dims(&[1]);
    let original_shape_ptr = shape.dims().as_ptr();
    let payload = TensorPayload::new(TensorBuffer::I64(data), shape).unwrap();
    let (buffer, shape) = payload.into_parts();
    assert_eq!(shape.dims(), &[1]);
    assert_eq!(shape.dims().as_ptr(), original_shape_ptr);
    match buffer {
        TensorBuffer::I64(values) => {
            assert_eq!(values, vec![9_007_199_254_740_993]);
            assert_eq!(values.as_ptr(), original_data_ptr);
        }
        other => panic!("expected I64 buffer, got {other:?}"),
    }
}
