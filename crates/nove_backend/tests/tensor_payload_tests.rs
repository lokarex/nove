use nove_backend::{BackendError, DType, Shape, TensorBuffer, TensorPayload};

#[test]
fn payload_new_matching_length_succeeds() {
    let p =
        TensorPayload::new(TensorBuffer::F32(vec![1.0; 4]), Shape::from_dims(&[2, 2])).unwrap();
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
