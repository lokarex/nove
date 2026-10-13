use nove_backend::{BackendError, DType, Device, Shape, TensorBuffer, backend::BackendStorage};

const OPERATIONS: [&str; 24] = [
    "broadcast_add",
    "broadcast_mul",
    "broadcast_div",
    "broadcast_sub",
    "broadcast_eq",
    "broadcast_ne",
    "broadcast_gt",
    "broadcast_lt",
    "broadcast_ge",
    "broadcast_le",
    "broadcast_pow",
    "broadcast_matmul",
    "embedding",
    "conv1d",
    "conv2d",
    "conv_transpose1d",
    "conv_transpose2d",
    "gather",
    "index_select",
    "scatter_add",
    "index_add",
    "where_cond",
    "stack",
    "cat",
];

fn cpu_devices() -> Vec<Device> {
    vec![
        #[cfg(feature = "candle-cpu")]
        nove_backend::device::candle::cpu().unwrap(),
        #[cfg(feature = "native-cpu")]
        nove_backend::device::native::cpu().unwrap(),
    ]
}

struct Inputs {
    matrix: BackendStorage,
    indexes: BackendStorage,
    grid_indexes: BackendStorage,
    condition: BackendStorage,
    image1: BackendStorage,
    kernel1: BackendStorage,
    image2: BackendStorage,
    kernel2: BackendStorage,
}

impl Inputs {
    fn new(device: &Device) -> Self {
        Self {
            matrix: BackendStorage::from_vec(
                vec![1.0f32, 2.0, 3.0, 4.0],
                &Shape::from_dims(&[2, 2]),
                device,
            )
            .unwrap(),
            indexes: BackendStorage::from_data(vec![0i64, 1], device).unwrap(),
            grid_indexes: BackendStorage::from_vec(
                vec![0i64, 1, 0, 1],
                &Shape::from_dims(&[2, 2]),
                device,
            )
            .unwrap(),
            condition: BackendStorage::from_vec(
                vec![2u8, 0, 2, 0],
                &Shape::from_dims(&[2, 2]),
                device,
            )
            .unwrap(),
            image1: BackendStorage::ones(&Shape::from_dims(&[1, 1, 2]), DType::F32, device)
                .unwrap(),
            kernel1: BackendStorage::ones(&Shape::from_dims(&[1, 1, 1]), DType::F32, device)
                .unwrap(),
            image2: BackendStorage::ones(&Shape::from_dims(&[1, 1, 2, 2]), DType::F32, device)
                .unwrap(),
            kernel2: BackendStorage::ones(&Shape::from_dims(&[1, 1, 1, 1]), DType::F32, device)
                .unwrap(),
        }
    }

    fn snapshot(&self) -> Vec<TensorBuffer> {
        [
            &self.matrix,
            &self.indexes,
            &self.grid_indexes,
            &self.condition,
            &self.image1,
            &self.kernel1,
            &self.image2,
            &self.kernel2,
        ]
        .iter()
        .map(|storage| storage.to_tensor_buffer().unwrap())
        .collect()
    }
}

fn call(
    operation: &str,
    a: &Inputs,
    b: &Inputs,
    c: &Inputs,
) -> Result<BackendStorage, BackendError> {
    match operation {
        "broadcast_add" => a.matrix.broadcast_add(&b.matrix),
        "broadcast_mul" => a.matrix.broadcast_mul(&b.matrix),
        "broadcast_div" => a.matrix.broadcast_div(&b.matrix),
        "broadcast_sub" => a.matrix.broadcast_sub(&b.matrix),
        "broadcast_eq" => a.matrix.broadcast_eq(&b.matrix),
        "broadcast_ne" => a.matrix.broadcast_ne(&b.matrix),
        "broadcast_gt" => a.matrix.broadcast_gt(&b.matrix),
        "broadcast_lt" => a.matrix.broadcast_lt(&b.matrix),
        "broadcast_ge" => a.matrix.broadcast_ge(&b.matrix),
        "broadcast_le" => a.matrix.broadcast_le(&b.matrix),
        "broadcast_pow" => a.matrix.broadcast_pow(&b.matrix),
        "broadcast_matmul" => a.matrix.broadcast_matmul(&b.matrix),
        "embedding" => a.matrix.embedding(&b.indexes),
        "conv1d" => a.image1.conv1d(&b.kernel1, 0, 1, 1, 1),
        "conv2d" => a.image2.conv2d(&b.kernel2, 0, 1, 1, 1),
        "conv_transpose1d" => a.image1.conv_transpose1d(&b.kernel1, 0, 0, 1, 1, 1),
        "conv_transpose2d" => a.image2.conv_transpose2d(&b.kernel2, 0, (0, 0), 1, 1, 1),
        "gather" => a.matrix.gather(&b.grid_indexes, 1),
        "index_select" => a.matrix.index_select(&b.indexes, 1),
        "scatter_add" => a.matrix.scatter_add(&b.grid_indexes, &c.matrix, 1),
        "index_add" => a.matrix.index_add(&b.indexes, &c.matrix, 1),
        "where_cond" => BackendStorage::where_cond(&a.condition, &b.matrix, &c.matrix),
        "stack" => {
            BackendStorage::stack(&[a.matrix.clone(), b.matrix.clone(), c.matrix.clone()], 1)
        }
        "cat" => BackendStorage::cat(&[a.matrix.clone(), b.matrix.clone(), c.matrix.clone()], 1),
        _ => panic!("unknown operation {operation}"),
    }
}

#[test]
fn all_multi_input_operations_accept_same_cpu_device() {
    for device in cpu_devices() {
        let input = Inputs::new(&device);
        let snapshot = input.snapshot();
        for operation in OPERATIONS {
            let output = call(operation, &input, &input, &input)
                .unwrap_or_else(|error| panic!("{operation}: {error}"));
            assert_eq!(output.backend_kind().unwrap(), device.backend());
            assert!(output.same_device(&input.matrix));
            assert_eq!(input.snapshot(), snapshot, "{operation}");
        }
    }
}

#[cfg(all(feature = "candle-cpu", feature = "native-cpu"))]
#[test]
fn all_multi_input_operations_reject_mixed_backends() {
    let candle = Inputs::new(&nove_backend::device::candle::cpu().unwrap());
    let native = Inputs::new(&nove_backend::device::native::cpu().unwrap());
    for (a, b) in [(&candle, &native), (&native, &candle)] {
        let snapshot_a = a.snapshot();
        let snapshot_b = b.snapshot();
        for operation in OPERATIONS {
            let result = call(operation, a, b, a);
            match result {
                Err(BackendError::BackendMismatch { expected, found }) => {
                    assert_eq!(expected, a.matrix.backend_kind().unwrap(), "{operation}");
                    assert_eq!(found, b.matrix.backend_kind().unwrap(), "{operation}");
                }
                result => panic!("{operation}: {result:?}"),
            }
            if ["scatter_add", "index_add", "where_cond", "stack", "cat"].contains(&operation) {
                match call(operation, a, a, b) {
                    Err(BackendError::BackendMismatch { expected, found }) => {
                        assert_eq!(expected, a.matrix.backend_kind().unwrap(), "{operation}");
                        assert_eq!(found, b.matrix.backend_kind().unwrap(), "{operation}");
                    }
                    result => panic!("{operation} third input: {result:?}"),
                }
            }
            assert_eq!(a.snapshot(), snapshot_a, "{operation}");
            assert_eq!(b.snapshot(), snapshot_b, "{operation}");
        }
    }
}

#[cfg(all(feature = "candle-cpu", feature = "native-cpu"))]
#[test]
fn backend_mismatch_precedes_invalid_metadata() {
    let candle = Inputs::new(&nove_backend::device::candle::cpu().unwrap());
    let native = Inputs::new(&nove_backend::device::native::cpu().unwrap());
    for (a, b) in [(&candle, &native), (&native, &candle)] {
        for result in [
            a.matrix.gather(&b.matrix, usize::MAX),
            a.matrix.index_select(&b.matrix, usize::MAX),
            BackendStorage::where_cond(&a.indexes, &a.matrix, &b.matrix),
            BackendStorage::stack(&[a.indexes.clone(), b.matrix.clone()], usize::MAX),
        ] {
            assert!(
                matches!(result, Err(BackendError::BackendMismatch { .. })),
                "{result:?}"
            );
        }
    }
}

#[cfg(all(feature = "candle-cuda", feature = "candle-cpu"))]
#[test]
#[ignore = "requires CUDA hardware and toolkit; run explicitly with --ignored"]
fn cuda_dispatch_compares_actual_handles() {
    let device = nove_backend::device::candle::cuda(0).unwrap();
    let same_handle = Inputs::new(&device.clone());
    let a = Inputs::new(&device);
    let different_handle = Inputs::new(&nove_backend::device::candle::cuda(0).unwrap());
    let cpu = Inputs::new(&nove_backend::device::candle::cpu().unwrap());
    assert!(a.matrix.same_device(&same_handle.matrix));
    assert!(!a.matrix.same_device(&different_handle.matrix));
    assert!(a.matrix.broadcast_add(&same_handle.matrix).is_ok());
    assert!(BackendStorage::where_cond(&a.condition, &same_handle.matrix, &a.matrix).is_ok());
    assert!(BackendStorage::stack(&[a.matrix.clone(), same_handle.matrix.clone()], 0).is_ok());
    assert!(BackendStorage::cat(&[a.matrix.clone(), same_handle.matrix.clone()], 0).is_ok());
    for b in [&different_handle, &cpu] {
        for operation in OPERATIONS {
            assert!(
                matches!(
                    call(operation, &a, b, &a),
                    Err(BackendError::DeviceMismatch { operation: name, input_index: 1 }) if name == operation
                ),
                "{operation}"
            );
            if ["scatter_add", "index_add", "where_cond", "stack", "cat"].contains(&operation) {
                assert!(
                    matches!(
                        call(operation, &a, &a, b),
                        Err(BackendError::DeviceMismatch { operation: name, input_index: 2 }) if name == operation
                    ),
                    "{operation} third input"
                );
            }
        }
        #[cfg(feature = "native-cpu")]
        {
            let native = Inputs::new(&nove_backend::device::native::cpu().unwrap());
            for operation in ["scatter_add", "index_add", "where_cond", "stack", "cat"] {
                assert!(
                    matches!(
                        call(operation, &a, b, &native),
                        Err(BackendError::BackendMismatch { .. })
                    ),
                    "{operation}"
                );
            }
        }
    }
}
