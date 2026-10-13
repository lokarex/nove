use nove_backend::{BackendError, DType, Device, Shape, backend::BackendStorage};

fn cpu_devices() -> Vec<Device> {
    vec![
        #[cfg(feature = "candle-cpu")]
        nove_backend::device::candle::cpu().unwrap(),
        #[cfg(feature = "native-cpu")]
        nove_backend::device::native::cpu().unwrap(),
    ]
}

fn zeros(device: &Device, dims: &[usize]) -> BackendStorage {
    BackendStorage::zeros(&Shape::from_dims(dims), DType::F32, device).unwrap()
}

fn same_float(actual: f64, expected: f64) -> bool {
    actual == expected || (actual.is_nan() && expected.is_nan())
}

macro_rules! assert_error {
    ($result:expr, $pattern:pat $(if $guard:expr)?) => {{
        let error = $result.unwrap_err();
        assert!(matches!(&error, $pattern $(if $guard)?), "{error:?}");
    }};
}

#[test]
fn permutation_rejects_invalid_axes() {
    for device in cpu_devices() {
        let input = zeros(&device, &[2, 3]);
        for axis in [2, usize::MAX] {
            assert_error!(input.permute(&[axis, 0]), BackendError::AxisOutOfBounds {
                operation: "permute", axis: found, rank: 2
            } if *found == axis);
        }
        for dims in [&[0, 0][..], &[1]] {
            assert_error!(input.permute(dims), BackendError::InvalidPermutation {
                operation: "permute", permutation, rank: 2
            } if permutation.as_slice() == dims);
        }
    }
}

#[test]
fn transpose_rejects_invalid_axes() {
    for device in cpu_devices() {
        let input = zeros(&device, &[2, 3]);
        for (a, b, expected) in [(2, 0, 2), (0, 2, 2), (usize::MAX, 0, usize::MAX)] {
            assert_error!(input.transpose(a, b), BackendError::AxisOutOfBounds {
                operation: "transpose", axis, rank: 2
            } if *axis == expected);
        }
    }
}

#[test]
fn broadcast_rejects_rank_reduction() {
    for device in cpu_devices() {
        assert_error!(zeros(&device, &[2, 3]).broadcast_as(&Shape::from_dims(&[2])),
            BackendError::ShapeMismatch { operation: "broadcast_as", lhs, rhs, axis: None }
            if lhs.dims() == [2, 3] && rhs.dims() == [2]);
    }
}

#[test]
fn narrow_rejects_overflow() {
    for device in cpu_devices() {
        let input = zeros(&device, &[2, 3]);
        assert_error!(
            input.narrow(1, usize::MAX, 1),
            BackendError::SizeOverflow {
                operation: "narrow",
                parameter: "slice_end",
                lhs: usize::MAX,
                rhs: 1
            }
        );
        assert_error!(
            input.narrow(1, 2, 2),
            BackendError::SliceOutOfBounds {
                operation: "narrow",
                axis: 1,
                start: 2,
                length: 2,
                axis_size: 3
            }
        );
        assert_error!(
            input.narrow(2, 0, 1),
            BackendError::AxisOutOfBounds {
                operation: "narrow",
                axis: 2,
                rank: 2
            }
        );
    }
}

#[test]
fn reshape_rejects_invalid_count() {
    for device in cpu_devices() {
        assert_error!(zeros(&device, &[2, 3]).reshape(&Shape::from_dims(&[5])),
            BackendError::ShapeMismatch { operation: "reshape", lhs, rhs, axis: None }
            if lhs.dims() == [2, 3] && rhs.dims() == [5]);
    }
}

#[test]
fn shape_targets_reject_overflow() {
    for device in cpu_devices() {
        let scalar = zeros(&device, &[]);
        let target = Shape::from_dims(&[usize::MAX, 2]);
        assert_error!(scalar.reshape(&target), BackendError::ShapeOverflow {
            operation: "reshape", shape
        } if shape.dims() == [usize::MAX, 2]);
        assert_error!(scalar.broadcast_as(&target), BackendError::ShapeOverflow {
            operation: "broadcast_as", shape
        } if shape.dims() == [usize::MAX, 2]);
    }
}

#[test]
fn axis_reductions_reject_out_of_range_axis() {
    for device in cpu_devices() {
        let input = zeros(&device, &[2, 3]);
        for axis in [2, usize::MAX] {
            for (operation, result) in [
                ("sum", input.sum(axis)),
                ("sum_keepdim", input.sum_keepdim(axis)),
                ("max", input.max(axis)),
                ("max_keepdim", input.max_keepdim(axis)),
                ("min", input.min(axis)),
                ("min_keepdim", input.min_keepdim(axis)),
                ("mean", input.mean(axis)),
                ("mean_keepdim", input.mean_keepdim(axis)),
                ("var", input.var(axis)),
                ("var_keepdim", input.var_keepdim(axis)),
                ("argmax", input.argmax(axis)),
                ("argmax_keepdim", input.argmax_keepdim(axis)),
                ("argmin", input.argmin(axis)),
                ("argmin_keepdim", input.argmin_keepdim(axis)),
            ] {
                assert_error!(result, BackendError::AxisOutOfBounds {
                    operation: name, axis: found, rank: 2
                } if *name == operation && *found == axis);
            }
        }
    }
}

#[test]
fn layout_positions_reject_invalid_parameters() {
    for device in cpu_devices() {
        let input = zeros(&device, &[2, 3]);
        for (operation, result, start, end) in [
            ("flatten_from", input.flatten_from(2), 2, 1),
            ("flatten_to", input.flatten_to(2), 0, 2),
            ("flatten", input.flatten(1, 0), 1, 0),
        ] {
            assert_error!(result, BackendError::InvalidAxisRange {
                operation: name, start: a, end: b, rank: 2
            } if *name == operation && *a == start && *b == end);
        }
        assert_error!(
            input.squeeze(2),
            BackendError::AxisOutOfBounds {
                operation: "squeeze",
                axis: 2,
                rank: 2
            }
        );
        assert_error!(
            input.unsqueeze(3),
            BackendError::InsertionAxisOutOfBounds {
                operation: "unsqueeze",
                axis: 3,
                rank: 2
            }
        );
        assert_error!(
            zeros(&device, &[]).flatten_from(usize::MAX),
            BackendError::InvalidAxisRange {
                operation: "flatten_from",
                start: usize::MAX,
                end: 0,
                rank: 1
            }
        );
    }
}

#[test]
fn legal_shape_boundaries_remain_usable() {
    for device in cpu_devices() {
        let input = BackendStorage::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            &Shape::from_dims(&[2, 3]),
            &device,
        )
        .unwrap();
        assert_eq!(input.flatten(0, 1).unwrap().shape().unwrap().dims(), &[6]);
        assert_eq!(
            input.unsqueeze(2).unwrap().shape().unwrap().dims(),
            &[2, 3, 1]
        );
        assert_eq!(
            input.transpose(1, 1).unwrap().to_vec::<f32>().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
        assert_eq!(
            input.narrow(1, 3, 0).unwrap().shape().unwrap().dims(),
            &[2, 0]
        );
        let scalar = zeros(&device, &[]);
        assert_eq!(
            scalar
                .reshape(&Shape::from_dims(&[1]))
                .unwrap()
                .shape()
                .unwrap()
                .dims(),
            &[1]
        );
        assert_eq!(
            scalar.permute(&[]).unwrap().shape().unwrap().dims(),
            &[] as &[usize]
        );
        assert_eq!(scalar.unsqueeze(0).unwrap().shape().unwrap().dims(), &[1]);
        assert_eq!(
            zeros(&device, &[1, 3])
                .broadcast_as(&Shape::from_dims(&[2, 0, 3]))
                .unwrap()
                .shape()
                .unwrap()
                .dims(),
            &[2, 0, 3]
        );
    }
}

#[test]
fn empty_index_select_validates_metadata() {
    for device in cpu_devices() {
        let input = zeros(&device, &[2, 3]);
        for dtype in [DType::F32, DType::U32] {
            let indexes = BackendStorage::zeros(&Shape::from_dims(&[0]), dtype, &device).unwrap();
            assert_error!(input.index_select(&indexes, 1), BackendError::DTypeMismatch {
                operation: "index_select", input_index: 1, expected: DType::I64, found
            } if *found == dtype);
        }
        let rank2 = BackendStorage::zeros(&Shape::from_dims(&[0, 2]), DType::I64, &device).unwrap();
        assert_error!(
            input.index_select(&rank2, 1),
            BackendError::RankMismatch {
                operation: "index_select",
                input_index: 1,
                expected: 1,
                found: 2
            }
        );
        let indexes = BackendStorage::from_data(Vec::<i64>::new(), &device).unwrap();
        assert_error!(
            input.index_select(&indexes, 2),
            BackendError::AxisOutOfBounds {
                operation: "index_select",
                axis: 2,
                rank: 2
            }
        );
        let output = input.index_select(&indexes, 1).unwrap();
        assert_eq!(output.shape().unwrap().dims(), &[2, 0]);
        assert_eq!(output.dtype().unwrap(), DType::F32);
    }
}

#[test]
fn embedding_validates_documented_metadata() {
    for device in cpu_devices() {
        let table = zeros(&device, &[2, 3]);
        let indexes = BackendStorage::from_data(vec![1i64], &device).unwrap();
        assert_error!(
            zeros(&device, &[2, 2, 3]).embedding(&indexes),
            BackendError::RankMismatch {
                operation: "embedding",
                input_index: 0,
                expected: 2,
                found: 3
            }
        );
        let rank2 =
            BackendStorage::from_vec(vec![1i64], &Shape::from_dims(&[1, 1]), &device).unwrap();
        assert_error!(
            table.embedding(&rank2),
            BackendError::RankMismatch {
                operation: "embedding",
                input_index: 1,
                expected: 1,
                found: 2
            }
        );
        for dtype in [DType::U32, DType::F32] {
            for length in [0, 1] {
                let wrong =
                    BackendStorage::zeros(&Shape::from_dims(&[length]), dtype, &device).unwrap();
                assert_error!(table.embedding(&wrong), BackendError::DTypeMismatch {
                    operation: "embedding", input_index: 1, expected: DType::I64, found
                } if *found == dtype);
            }
        }
        assert_eq!(
            table.embedding(&indexes).unwrap().shape().unwrap().dims(),
            &[1, 3]
        );
        let empty = BackendStorage::from_data(Vec::<i64>::new(), &device).unwrap();
        assert_eq!(
            table.embedding(&empty).unwrap().shape().unwrap().dims(),
            &[0, 3]
        );
    }
}

#[test]
fn index_operations_validate_metadata() {
    for device in cpu_devices() {
        let input = zeros(&device, &[2, 3]);
        let indexes =
            BackendStorage::from_vec(vec![0i64; 4], &Shape::from_dims(&[2, 2]), &device).unwrap();
        let row = BackendStorage::from_data(vec![0i64, 1], &device).unwrap();
        let source = zeros(&device, &[2, 2]);
        for dtype in [DType::U32, DType::F32] {
            let wrong = BackendStorage::zeros(&Shape::from_dims(&[2, 2]), dtype, &device).unwrap();
            let vector = BackendStorage::zeros(&Shape::from_dims(&[2]), dtype, &device).unwrap();
            for (operation, result) in [
                ("gather", input.gather(&wrong, 1)),
                ("scatter_add", input.scatter_add(&wrong, &source, 1)),
                ("index_add", input.index_add(&vector, &source, 1)),
            ] {
                assert_error!(result, BackendError::DTypeMismatch {
                    operation: name, input_index: 1, expected: DType::I64, found
                } if *name == operation && *found == dtype);
            }
        }
        assert_error!(
            input.gather(&row, 1),
            BackendError::RankMismatch {
                operation: "gather",
                input_index: 1,
                expected: 2,
                found: 1
            }
        );
        assert_error!(
            input.index_add(&indexes, &source, 1),
            BackendError::RankMismatch {
                operation: "index_add",
                input_index: 1,
                expected: 1,
                found: 2
            }
        );
        for (operation, result) in [
            ("gather", input.gather(&indexes, 2)),
            ("scatter_add", input.scatter_add(&indexes, &source, 2)),
            ("index_add", input.index_add(&row, &source, 2)),
        ] {
            assert_error!(result, BackendError::AxisOutOfBounds {
                operation: name, axis: 2, rank: 2
            } if *name == operation);
        }
        let large =
            BackendStorage::from_vec(vec![0i64; 3], &Shape::from_dims(&[3, 1]), &device).unwrap();
        assert_error!(input.gather(&large, 1), BackendError::ShapeMismatch {
            operation: "gather", lhs, rhs, axis: Some(1)
        } if lhs.dims() == [2, 3] && rhs.dims() == [3, 1]);
        assert_error!(input.scatter_add(&indexes, &zeros(&device, &[2, 1]), 1),
            BackendError::ShapeMismatch { operation: "scatter_add", lhs, rhs, axis: Some(1) }
            if lhs.dims() == [2, 2] && rhs.dims() == [2, 1]);
        for (dims, expected) in [(&[2, 1][..], &[2][..]), (&[3, 2][..], &[2, 3][..])] {
            assert_error!(input.index_add(&row, &zeros(&device, dims), 1),
                BackendError::ShapeMismatch { operation: "index_add", lhs, rhs, axis: Some(1) }
                if rhs.dims() == dims && lhs.dims() == expected);
        }
        let mixed = BackendStorage::zeros(&Shape::from_dims(&[2, 2]), DType::F64, &device).unwrap();
        for (operation, result) in [
            ("scatter_add", input.scatter_add(&indexes, &mixed, 1)),
            ("index_add", input.index_add(&row, &mixed, 1)),
        ] {
            assert_error!(result, BackendError::DTypeMismatch {
                operation: name, input_index: 2, expected: DType::F32, found: DType::F64
            } if *name == operation);
        }
        let subset =
            BackendStorage::from_vec(vec![0i64, 1], &Shape::from_dims(&[1, 2]), &device).unwrap();
        assert_eq!(
            input.gather(&subset, 1).unwrap().shape().unwrap().dims(),
            &[1, 2]
        );
        let empty_wrong =
            BackendStorage::zeros(&Shape::from_dims(&[2, 0]), DType::F32, &device).unwrap();
        for (operation, result) in [
            ("gather", input.gather(&empty_wrong, 1)),
            (
                "scatter_add",
                input.scatter_add(&empty_wrong, &zeros(&device, &[2, 0]), 1),
            ),
        ] {
            assert_error!(result, BackendError::DTypeMismatch {
                operation: name, input_index: 1, expected: DType::I64, found: DType::F32
            } if *name == operation);
        }
        let empty = BackendStorage::from_data(Vec::<i64>::new(), &device).unwrap();
        assert_error!(input.index_add(&empty, &zeros(&device, &[2, 1]), 1),
            BackendError::ShapeMismatch { operation: "index_add", lhs, rhs, axis: Some(1) }
            if lhs.dims() == [0] && rhs.dims() == [2, 1]);
    }
}

#[test]
fn combination_rejects_invalid_metadata() {
    assert_error!(
        BackendStorage::stack(&[], 0),
        BackendError::EmptyInputs { operation: "stack" }
    );
    assert_error!(
        BackendStorage::cat(&[], 0),
        BackendError::EmptyInputs { operation: "cat" }
    );
    for device in cpu_devices() {
        let a = zeros(&device, &[2, 3]);
        let b = zeros(&device, &[2, 2]);
        assert_error!(BackendStorage::stack(&[a.clone(), b.clone()], 0),
            BackendError::ShapeMismatch { operation: "stack", lhs, rhs, axis: None }
            if lhs.dims() == [2, 3] && rhs.dims() == [2, 2]);
        assert_error!(BackendStorage::cat(&[a.clone(), b], 0),
            BackendError::ShapeMismatch { operation: "cat", lhs, rhs, axis: Some(0) }
            if lhs.dims() == [2, 3] && rhs.dims() == [2, 2]);
        assert_error!(
            BackendStorage::cat(&[a.clone(), zeros(&device, &[6])], 0),
            BackendError::RankMismatch {
                operation: "cat",
                input_index: 1,
                expected: 2,
                found: 1
            }
        );
        for (name, operation) in [
            (
                "stack",
                BackendStorage::stack
                    as fn(&[BackendStorage], usize) -> Result<BackendStorage, BackendError>,
            ),
            ("cat", BackendStorage::cat),
        ] {
            let mixed =
                BackendStorage::zeros(&Shape::from_dims(&[2, 3]), DType::F64, &device).unwrap();
            assert_error!(operation(&[a.clone(), mixed], 0), BackendError::DTypeMismatch {
                operation: found, input_index: 1, expected: DType::F32, found: DType::F64
            } if *found == name);
        }
        assert_error!(
            BackendStorage::stack(std::slice::from_ref(&a), 3),
            BackendError::InsertionAxisOutOfBounds {
                operation: "stack",
                axis: 3,
                rank: 2
            }
        );
        assert_error!(
            BackendStorage::cat(std::slice::from_ref(&a), 3),
            BackendError::AxisOutOfBounds {
                operation: "cat",
                axis: 3,
                rank: 2
            }
        );
    }
}

#[test]
fn legal_combinations_keep_shape_and_dtype() {
    for device in cpu_devices() {
        let scalar = BackendStorage::from_data(2.0f32, &device).unwrap();
        assert_eq!(
            BackendStorage::stack(&[scalar.clone(), scalar], 0)
                .unwrap()
                .to_vec::<f32>()
                .unwrap(),
            vec![2.0, 2.0]
        );
        let empty = zeros(&device, &[2, 0, 3]);
        assert_eq!(
            BackendStorage::stack(&[empty.clone(), empty.clone()], 1)
                .unwrap()
                .shape()
                .unwrap()
                .dims(),
            &[2, 2, 0, 3]
        );
        assert_eq!(
            BackendStorage::cat(&[empty.clone(), empty], 2)
                .unwrap()
                .shape()
                .unwrap()
                .dims(),
            &[2, 0, 6]
        );
    }
}

#[test]
fn binary_shapes_are_validated_before_dispatch() {
    for device in cpu_devices() {
        let a = zeros(&device, &[2, 3]);
        let b = zeros(&device, &[4, 3]);
        assert_error!(a.broadcast_add(&b), BackendError::ShapeMismatch {
            operation: "broadcast_add", lhs, rhs, axis: None
        } if lhs.dims() == [2, 3] && rhs.dims() == [4, 3]);
        assert_error!(a.broadcast_matmul(&a), BackendError::ShapeMismatch {
            operation: "broadcast_matmul", lhs, rhs, axis: None
        } if lhs.dims() == [2, 3] && rhs.dims() == [2, 3]);
        assert_error!(
            zeros(&device, &[3]).broadcast_matmul(&a),
            BackendError::RankTooSmall {
                operation: "broadcast_matmul",
                input_index: 0,
                minimum: 2,
                found: 1
            }
        );
        assert_error!(
            a.broadcast_matmul(&zeros(&device, &[3])),
            BackendError::RankTooSmall {
                operation: "broadcast_matmul",
                input_index: 1,
                minimum: 2,
                found: 1
            }
        );
        let empty = zeros(&device, &[2, 0, 3]);
        assert_eq!(
            empty
                .broadcast_add(&zeros(&device, &[1, 1, 3]))
                .unwrap()
                .shape()
                .unwrap()
                .dims(),
            &[2, 0, 3]
        );
        assert_eq!(
            a.broadcast_add(&zeros(&device, &[]))
                .unwrap()
                .shape()
                .unwrap()
                .dims(),
            &[2, 3]
        );
        assert_eq!(
            a.broadcast_matmul(&zeros(&device, &[3, 2]))
                .unwrap()
                .shape()
                .unwrap()
                .dims(),
            &[2, 2]
        );
        #[cfg(feature = "native-cpu")]
        if device.backend() == nove_backend::BackendKind::Native {
            assert!(matches!(
                zeros(&device, &[1, 2, 3]).broadcast_matmul(&zeros(&device, &[1, 3, 2])),
                Err(BackendError::UnsupportedOperation { .. })
            ));
        }
    }
}

#[test]
fn where_rejects_invalid_dtypes_before_dispatch() {
    for device in cpu_devices() {
        let values = zeros(&device, &[2]);
        for dtype in [DType::F32, DType::I64, DType::U32] {
            for length in [0, 2] {
                let condition =
                    BackendStorage::zeros(&Shape::from_dims(&[length]), dtype, &device).unwrap();
                assert_error!(BackendStorage::where_cond(&condition, &values, &values),
                    BackendError::DTypeMismatch { operation: "where_cond", input_index: 0,
                        expected: DType::U8, found } if *found == dtype);
            }
        }
        let condition = BackendStorage::from_data(vec![1u8, 0], &device).unwrap();
        let mixed = BackendStorage::zeros(&Shape::from_dims(&[2]), DType::F64, &device).unwrap();
        assert_error!(
            BackendStorage::where_cond(&condition, &values, &mixed),
            BackendError::DTypeMismatch {
                operation: "where_cond",
                input_index: 2,
                expected: DType::F32,
                found: DType::F64
            }
        );
        assert_error!(BackendStorage::where_cond(&condition, &values, &zeros(&device, &[3])),
            BackendError::ShapeMismatch { operation: "where_cond", lhs, rhs, axis: None }
            if lhs.dims() == [2] && rhs.dims() == [3]);
    }
}

#[test]
fn where_preserves_nonzero_condition_semantics() {
    for device in cpu_devices() {
        let condition = BackendStorage::from_data(vec![2u8, 0], &device).unwrap();
        let a = BackendStorage::from_data(vec![1.0f32, 2.0], &device).unwrap();
        let b = BackendStorage::from_data(vec![3.0f32, 4.0], &device).unwrap();
        assert_eq!(
            BackendStorage::where_cond(&condition, &a, &b)
                .unwrap()
                .to_vec::<f32>()
                .unwrap(),
            vec![1.0, 4.0]
        );
        #[cfg(feature = "native-cpu")]
        if device.backend() == nove_backend::BackendKind::Native {
            let scalar = BackendStorage::from_data(5.0f32, &device).unwrap();
            assert_eq!(
                BackendStorage::where_cond(&condition, &scalar, &b)
                    .unwrap()
                    .to_vec::<f32>()
                    .unwrap(),
                vec![5.0, 4.0]
            );
        }
    }
}

#[test]
fn clamp_rejects_invalid_bounds() {
    for device in cpu_devices() {
        for dims in [&[2][..], &[0]] {
            let input = zeros(&device, dims);
            for (min, max) in [(2.0, 1.0), (f64::NAN, 1.0), (0.0, f64::NAN)] {
                assert_error!(input.clamp(min, max), BackendError::InvalidBounds {
                    operation: "clamp", lower, upper
                } if same_float(*lower, min) && same_float(*upper, max));
            }
        }
        let input = BackendStorage::from_data(vec![-1.0f32, 3.0], &device).unwrap();
        assert_eq!(
            input.clamp(2.0, 2.0).unwrap().to_vec::<f32>().unwrap(),
            vec![2.0, 2.0]
        );
        assert_eq!(
            input
                .clamp(f64::NEG_INFINITY, f64::INFINITY)
                .unwrap()
                .to_vec::<f32>()
                .unwrap(),
            vec![-1.0, 3.0]
        );
    }
}

#[test]
fn random_parameters_are_checked_before_generation() {
    for device in cpu_devices() {
        for dims in [&[2][..], &[0]] {
            let shape = Shape::from_dims(dims);
            for (low, high) in [
                (2.0, 1.0),
                (1.0, 1.0),
                (f64::NAN, 1.0),
                (0.0, f64::INFINITY),
            ] {
                assert_error!(BackendStorage::rand(low, high, &shape, &device), BackendError::InvalidBounds {
                    operation: "rand", lower, upper
                } if same_float(*lower, low) && same_float(*upper, high));
                assert_error!(BackendStorage::rand(low as f32, high as f32, &shape, &device),
                    BackendError::InvalidBounds { operation: "rand", lower, upper }
                    if same_float(*lower, f64::from(low as f32)) && same_float(*upper, f64::from(high as f32)));
            }
            for (mean, std) in [
                (f64::NAN, 1.0),
                (f64::INFINITY, 1.0),
                (0.0, f64::INFINITY),
                (0.0, f64::NAN),
                (0.0, -1.0),
            ] {
                assert_error!(BackendStorage::randn(mean, std, &shape, &device), BackendError::InvalidNormalParameters {
                    operation: "randn", mean: found_mean, std: found_std
                } if same_float(*found_mean, mean) && same_float(*found_std, std));
                assert_error!(BackendStorage::randn(mean as f32, std as f32, &shape, &device),
                    BackendError::InvalidNormalParameters { operation: "randn", mean: found_mean, std: found_std }
                    if same_float(*found_mean, f64::from(mean as f32)) && same_float(*found_std, f64::from(std as f32)));
            }
            assert_eq!(
                BackendStorage::randn(2.0f32, 0.0, &shape, &device)
                    .unwrap()
                    .to_vec::<f32>()
                    .unwrap(),
                vec![2.0; dims[0]]
            );
            assert_eq!(
                BackendStorage::randn(2.0f64, 0.0, &shape, &device)
                    .unwrap()
                    .to_vec::<f64>()
                    .unwrap(),
                vec![2.0; dims[0]]
            );
            let values = BackendStorage::rand(-1.0f32, 1.0, &shape, &device)
                .unwrap()
                .to_vec::<f32>()
                .unwrap();
            assert!(values.iter().all(|value| (-1.0..1.0).contains(value)));
        }
    }
}

#[test]
fn convolution_rejects_invalid_parameters() {
    for device in cpu_devices() {
        let input1 = zeros(&device, &[1, 1, 2]);
        let kernel1 = zeros(&device, &[1, 1, 1]);
        let input2 = zeros(&device, &[1, 1, 2, 2]);
        let kernel2 = zeros(&device, &[1, 1, 1, 1]);
        for (stride, dilation, groups, parameter) in [
            (0, 1, 1, "stride"),
            (1, 0, 1, "dilation"),
            (1, 1, 0, "groups"),
        ] {
            for (operation, result) in [
                (
                    "conv1d",
                    input1.conv1d(&kernel1, 0, stride, dilation, groups),
                ),
                (
                    "conv_transpose1d",
                    input1.conv_transpose1d(&kernel1, 0, 0, stride, dilation, groups),
                ),
                (
                    "conv2d",
                    input2.conv2d(&kernel2, 0, stride, dilation, groups),
                ),
                (
                    "conv_transpose2d",
                    input2.conv_transpose2d(&kernel2, 0, (0, 0), stride, dilation, groups),
                ),
            ] {
                assert_error!(result, BackendError::ZeroParameter { operation: name, parameter: found }
                    if *name == operation && *found == parameter);
            }
        }
        for (input, kernel, input_index) in [(&input2, &kernel1, 0), (&input1, &kernel2, 1)] {
            for (operation, result) in [
                ("conv1d", input.conv1d(kernel, 0, 1, 1, 1)),
                (
                    "conv_transpose1d",
                    input.conv_transpose1d(kernel, 0, 0, 1, 1, 1),
                ),
            ] {
                assert_error!(result, BackendError::RankMismatch {
                    operation: name, input_index: found, expected: 3, found: 4
                } if *name == operation && *found == input_index);
            }
        }
        for (input, kernel, input_index) in [(&input1, &kernel2, 0), (&input2, &kernel1, 1)] {
            for (operation, result) in [
                ("conv2d", input.conv2d(kernel, 0, 1, 1, 1)),
                (
                    "conv_transpose2d",
                    input.conv_transpose2d(kernel, 0, (0, 0), 1, 1, 1),
                ),
            ] {
                assert_error!(result, BackendError::RankMismatch {
                    operation: name, input_index: found, expected: 4, found: 3
                } if *name == operation && *found == input_index);
            }
        }
        let empty_kernel1 = zeros(&device, &[1, 1, 0]);
        for (operation, result) in [
            ("conv1d", input1.conv1d(&empty_kernel1, 0, 1, 1, 1)),
            (
                "conv_transpose1d",
                input1.conv_transpose1d(&empty_kernel1, 0, 0, 1, 1, 1),
            ),
        ] {
            assert_error!(result, BackendError::EmptySpatialDimension {
                operation: name, input_index: 1, axis: 2
            } if *name == operation);
        }
        for (dims, expected_axis) in [([1, 1, 0, 1], 2), ([1, 1, 1, 0], 3)] {
            let empty = zeros(&device, &dims);
            for (operation, result) in [
                ("conv2d", input2.conv2d(&empty, 0, 1, 1, 1)),
                (
                    "conv_transpose2d",
                    input2.conv_transpose2d(&empty, 0, (0, 0), 1, 1, 1),
                ),
            ] {
                assert_error!(result, BackendError::EmptySpatialDimension {
                    operation: name, input_index: 1, axis
                } if *name == operation && *axis == expected_axis);
            }
        }
        assert_error!(
            zeros(&device, &[1, 1, 0]).conv_transpose1d(&kernel1, 0, 0, 1, 1, 1),
            BackendError::EmptySpatialDimension {
                operation: "conv_transpose1d",
                input_index: 0,
                axis: 2
            }
        );
        assert_error!(
            zeros(&device, &[1, 1, 0, 2]).conv_transpose2d(&kernel2, 0, (0, 0), 1, 1, 1),
            BackendError::EmptySpatialDimension {
                operation: "conv_transpose2d",
                input_index: 0,
                axis: 2
            }
        );
    }
}

#[test]
fn pooling_rejects_invalid_parameters() {
    for device in cpu_devices() {
        let input = zeros(&device, &[1, 1, 2, 2]);
        for (kernel, stride, parameter) in [
            ((1, 1), (0, 1), "stride.0"),
            ((1, 1), (1, 0), "stride.1"),
            ((0, 1), (1, 1), "kernel.0"),
            ((1, 0), (1, 1), "kernel.1"),
        ] {
            for (operation, result) in [
                (
                    "max_pool2d_with_stride",
                    input.max_pool2d_with_stride(kernel, stride),
                ),
                (
                    "avg_pool2d_with_stride",
                    input.avg_pool2d_with_stride(kernel, stride),
                ),
            ] {
                assert_error!(result, BackendError::ZeroParameter { operation: name, parameter: found }
                    if *name == operation && *found == parameter);
            }
        }
        for kernel in [(3, 1), (1, 3)] {
            for (operation, result) in [
                (
                    "max_pool2d_with_stride",
                    input.max_pool2d_with_stride(kernel, (1, 1)),
                ),
                (
                    "avg_pool2d_with_stride",
                    input.avg_pool2d_with_stride(kernel, (1, 1)),
                ),
            ] {
                assert_error!(result, BackendError::KernelTooLarge { operation: name, kernel: found, spatial: (2, 2) }
                    if *name == operation && *found == kernel);
            }
        }
        let rank3 = zeros(&device, &[1, 2, 2]);
        for (operation, result) in [
            (
                "max_pool2d_with_stride",
                rank3.max_pool2d_with_stride((1, 1), (1, 1)),
            ),
            (
                "avg_pool2d_with_stride",
                rank3.avg_pool2d_with_stride((1, 1), (1, 1)),
            ),
        ] {
            assert_error!(result, BackendError::RankMismatch {
                operation: name, input_index: 0, expected: 4, found: 3
            } if *name == operation);
        }
        assert_eq!(
            input
                .max_pool2d_with_stride((2, 2), (1, 1))
                .unwrap()
                .shape()
                .unwrap()
                .dims(),
            &[1, 1, 1, 1]
        );
        assert_eq!(
            input
                .avg_pool2d_with_stride((2, 2), (1, 1))
                .unwrap()
                .to_vec::<f32>()
                .unwrap(),
            vec![0.0]
        );
    }
}

#[test]
fn metadata_errors_preserve_check_order() {
    for device in cpu_devices() {
        let input = zeros(&device, &[2, 3]);
        let wrong_indexes = zeros(&device, &[1, 1]);
        assert_error!(
            input.index_select(&wrong_indexes, 2),
            BackendError::AxisOutOfBounds {
                operation: "index_select",
                axis: 2,
                rank: 2
            }
        );
        assert_error!(
            zeros(&device, &[2, 3, 1]).embedding(&wrong_indexes),
            BackendError::DTypeMismatch {
                operation: "embedding",
                input_index: 1,
                expected: DType::I64,
                found: DType::F32
            }
        );
        let wrong = BackendStorage::zeros(&Shape::from_dims(&[1]), DType::F64, &device).unwrap();
        assert_error!(
            BackendStorage::stack(&[input.clone(), wrong.clone()], 3),
            BackendError::InsertionAxisOutOfBounds {
                operation: "stack",
                axis: 3,
                rank: 2
            }
        );
        assert_error!(BackendStorage::stack(&[input.clone(), wrong.clone()], 0),
            BackendError::ShapeMismatch { operation: "stack", lhs, rhs, axis: None }
            if lhs.dims() == [2, 3] && rhs.dims() == [1]);
        assert_error!(
            BackendStorage::cat(&[input.clone(), wrong.clone()], 0),
            BackendError::RankMismatch {
                operation: "cat",
                input_index: 1,
                expected: 2,
                found: 1
            }
        );
        let indexes =
            BackendStorage::from_vec(vec![0i64; 4], &Shape::from_dims(&[2, 2]), &device).unwrap();
        assert_error!(
            input.scatter_add(&indexes, &wrong, 1),
            BackendError::DTypeMismatch {
                operation: "scatter_add",
                input_index: 2,
                expected: DType::F32,
                found: DType::F64
            }
        );
        assert_error!(
            BackendStorage::where_cond(&input, &input, &wrong),
            BackendError::DTypeMismatch {
                operation: "where_cond",
                input_index: 0,
                expected: DType::U8,
                found: DType::F32
            }
        );
    }
}

#[test]
fn structured_errors_remain_visible_through_public_paths() {
    for device in cpu_devices() {
        let error = zeros(&device, &[2, 3]).transpose(3, 0).unwrap_err();
        assert!(matches!(
            error,
            nove_backend::backend::BackendError::AxisOutOfBounds {
                operation: "transpose",
                axis: 3,
                rank: 2
            }
        ));
        let message = error.to_string();
        assert!(message.contains("transpose") && message.contains('3') && message.contains('2'));
    }
}

#[test]
fn random_shape_overflow_precedes_parameter_errors() {
    for device in cpu_devices() {
        let shape = Shape::from_dims(&[usize::MAX, 2]);
        assert_error!(
            BackendStorage::rand(1.0f32, 1.0, &shape, &device),
            BackendError::InvalidOperation(_)
        );
        assert_error!(
            BackendStorage::randn(f64::NAN, -1.0, &shape, &device),
            BackendError::InvalidOperation(_)
        );
    }
}

#[test]
fn parameter_errors_preserve_check_order() {
    for device in cpu_devices() {
        let input = zeros(&device, &[1, 1, 2]);
        let kernel = zeros(&device, &[1, 1, 0]);
        assert_error!(
            input.conv1d(&kernel, 0, 0, 0, 0),
            BackendError::EmptySpatialDimension {
                operation: "conv1d",
                input_index: 1,
                axis: 2
            }
        );
        let kernel = zeros(&device, &[1, 1, 1]);
        assert_error!(
            input.conv1d(&kernel, 0, 0, 0, 0),
            BackendError::ZeroParameter {
                operation: "conv1d",
                parameter: "stride"
            }
        );
        assert_error!(
            zeros(&device, &[1, 1, 0]).conv_transpose1d(&kernel, 0, 0, 1, 0, 0),
            BackendError::ZeroParameter {
                operation: "conv_transpose1d",
                parameter: "dilation"
            }
        );
        assert_error!(
            zeros(&device, &[1, 2, 2]).max_pool2d_with_stride((0, 0), (0, 0)),
            BackendError::RankMismatch {
                operation: "max_pool2d_with_stride",
                input_index: 0,
                expected: 4,
                found: 3
            }
        );
        assert_error!(
            zeros(&device, &[1, 1, 2, 2]).max_pool2d_with_stride((0, 0), (0, 0)),
            BackendError::ZeroParameter {
                operation: "max_pool2d_with_stride",
                parameter: "kernel.0"
            }
        );
    }
}
