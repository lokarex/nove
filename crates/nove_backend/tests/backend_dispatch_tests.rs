use nove_backend::{backend::BackendStorage, device, BackendKind, DType, Device, Shape};

fn cpu_device() -> Device {
    Device::default()
}

#[test]
#[cfg(feature = "candle")]
fn candle_creates_zeros() {
    let s = BackendStorage::zeros(
        &Shape::from_dims(&[2, 3]),
        DType::F32,
        &device::candle::cpu().unwrap(),
    )
    .unwrap();
    assert_eq!(s.shape().unwrap().dims(), &[2, 3]);
}

#[test]
#[cfg(feature = "native")]
fn native_creates_zeros() {
    let s = BackendStorage::zeros(
        &Shape::from_dims(&[2, 3]),
        DType::F32,
        &device::native::cpu().unwrap(),
    )
    .unwrap();
    assert_eq!(s.shape().unwrap().dims(), &[2, 3]);
}

#[test]
#[cfg(feature = "candle")]
fn candle_ones_values() {
    let s = BackendStorage::ones(
        &Shape::from_dims(&[1, 4]),
        DType::F64,
        &device::candle::cpu().unwrap(),
    )
    .unwrap();
    let vals: Vec<f64> = s.to_vec().unwrap();
    assert_eq!(vals, vec![1.0; 4]);
}

#[test]
#[cfg(feature = "candle")]
fn relu_works() {
    let s =
        BackendStorage::from_slice(&[-1.0f32, 2.0], &Shape::from_dims(&[2]), &cpu_device()).unwrap();
    let vals: Vec<f32> = s.relu().unwrap().to_vec().unwrap();
    assert_eq!(vals, vec![0.0, 2.0]);
}

#[test]
#[cfg(feature = "candle")]
fn reshape_works() {
    let s = BackendStorage::from_slice(&[1.0f32; 4], &Shape::from_dims(&[2, 2]), &cpu_device())
        .unwrap();
    assert_eq!(
        s.reshape(&Shape::from_dims(&[1, 4]))
            .unwrap()
            .shape()
            .unwrap()
            .dims(),
        &[1, 4]
    );
}

#[test]
fn clone_is_independent() {
    let s =
        BackendStorage::from_slice(&[1.0f32, 2.0], &Shape::from_dims(&[2]), &cpu_device()).unwrap();
    let c = s.clone();
    assert_eq!(s.to_vec::<f32>().unwrap(), c.to_vec::<f32>().unwrap());
}

#[test]
fn detach_preserves_data() {
    let s =
        BackendStorage::from_slice(&[1.0f32, 2.0], &Shape::from_dims(&[2]), &cpu_device()).unwrap();
    assert_eq!(
        s.detach().unwrap().to_vec::<f32>().unwrap(),
        vec![1.0, 2.0]
    );
}

#[test]
fn assign_from_copies() {
    let mut t =
        BackendStorage::zeros(&Shape::from_dims(&[2]), DType::F32, &cpu_device()).unwrap();
    let src =
        BackendStorage::from_slice(&[3.0f32, 4.0], &Shape::from_dims(&[2]), &cpu_device()).unwrap();
    t.assign_from(&src).unwrap();
    assert_eq!(t.to_vec::<f32>().unwrap(), vec![3.0, 4.0]);
}

#[test]
fn to_scalar_single_element() {
    let s =
        BackendStorage::from_slice(&[42.0f64], &Shape::from(()), &cpu_device()).unwrap();
    let v: f64 = s.to_scalar().unwrap();
    assert!((v - 42.0).abs() < 1e-10);
}

#[test]
fn to_scalar_fails_multi_element() {
    let s =
        BackendStorage::from_slice(&[1.0f64, 2.0], &Shape::from_dims(&[2]), &cpu_device()).unwrap();
    assert!(s.to_scalar::<f64>().is_err());
}

#[test]
fn from_data_via_payload() {
    let s = BackendStorage::from_data(vec![1.0f32, 2.0, 3.0], &cpu_device()).unwrap();
    assert_eq!(s.shape().unwrap().dims(), &[3]);
}

#[test]
fn backend_kind_is_correct() {
    let k = BackendStorage::zeros(&Shape::from_dims(&[1]), DType::F32, &cpu_device())
        .unwrap()
        .backend_kind()
        .unwrap();
    assert!(matches!(k, BackendKind::Candle | BackendKind::Native));
}

#[test]
#[cfg(all(feature = "candle", feature = "native"))]
fn backend_mismatch_on_binary() {
    let d1 = device::candle::cpu().unwrap();
    let d2 = device::native::cpu().unwrap();
    let a = BackendStorage::ones(&Shape::from_dims(&[2]), DType::F32, &d1).unwrap();
    let b = BackendStorage::ones(&Shape::from_dims(&[2]), DType::F32, &d2).unwrap();
    assert!(matches!(
        a.broadcast_add(&b).unwrap_err(),
        BackendError::BackendMismatch { .. }
    ));
}
