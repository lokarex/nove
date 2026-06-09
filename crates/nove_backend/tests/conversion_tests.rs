use nove_backend::{DType, IntoTensorPayload};

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
    let result =
        vec![vec![vec![1u8; 2]; 2], vec![vec![5u8]]].into_tensor_payload();
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
