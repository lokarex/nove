use nove::tensor::{DType, Device, Shape, Tensor};
use proptest::prelude::*;
use rand::{Rng, SeedableRng, rngs::StdRng};

/// Generate a random tensor and its corresponding data.
///
/// # Notes
/// * The dimension number of the tensor must be in the range [1, 4].
///
/// # Generic Type Parameters
/// * `S` - The element type of the vector. It supports `f32`, `f64`, `i64`, `u32`, `u8`.
///
/// # Arguments
/// * `shape` - The shape of the tensor.
/// * `dtype` - The data type of the tensor.
/// * `device` - The device of the tensor.
/// * `grad_enabled` - Whether the tensor requires gradient.
/// * `seed` - The seed for random number generator.
///
/// # Returns
/// * `(tensor, data)` - The tensor and its corresponding data.
pub fn generate_random_tensor_and_corresponding_data<S>(
    shape: &Shape,
    dtype: &DType,
    device: &Device,
    grad_enabled: bool,
    seed: u64,
) -> (Tensor, Vec<S>)
where
    S: candle_core::WithDType,
{
    // Check dimension number
    let num_dim = shape.rank();
    if num_dim > 4 {
        panic!(
            "Unsupported dimension number {} (only support [1, 4])",
            num_dim
        );
    }

    // Count total size
    let dims = shape.dims();
    let total_size = dims.iter().product::<usize>();

    // Generate random data
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data: Vec<f32> = Vec::with_capacity(total_size);
    for _ in 0..total_size {
        data.push(rng.random());
    }

    // Create tensor from data
    let mut tensor = match num_dim {
        1 => Tensor::from_data(&data[..dims[0]], device, grad_enabled).unwrap(),
        2 => {
            let data_2d: Vec<Vec<f32>> = data[..dims[0] * dims[1]]
                .chunks(dims[1])
                .map(|c| c.to_vec())
                .collect();
            Tensor::from_data(data_2d, device, grad_enabled).unwrap()
        }
        3 => {
            let data_3d: Vec<Vec<Vec<f32>>> = data[..dims[0] * dims[1] * dims[2]]
                .chunks(dims[1] * dims[2])
                .map(|chunk| chunk.chunks(dims[2]).map(|c| c.to_vec()).collect())
                .collect();
            Tensor::from_data(data_3d, device, grad_enabled).unwrap()
        }
        4 => {
            let data_4d: Vec<Vec<Vec<Vec<f32>>>> = data
                .chunks(dims[1] * dims[2] * dims[3])
                .map(|chunk| {
                    chunk
                        .chunks(dims[2] * dims[3])
                        .map(|c| c.chunks(dims[3]).map(|d| d.to_vec()).collect())
                        .collect()
                })
                .collect();
            Tensor::from_data(data_4d, device, grad_enabled).unwrap()
        }
        _ => unreachable!(),
    };

    if *dtype != tensor.dtype().unwrap() {
        tensor = tensor.to_dtype(dtype).unwrap();
    }

    // Convert data to the specified dtype
    let data = match dtype {
        DType::F32 => {
            assert_eq!(std::any::TypeId::of::<S>(), std::any::TypeId::of::<f32>());
            let concrete_data: Vec<f32> = data;
            unsafe { std::mem::transmute(concrete_data) }
        }
        DType::F64 => {
            assert_eq!(std::any::TypeId::of::<S>(), std::any::TypeId::of::<f64>());
            let concrete_data: Vec<f64> = data.into_iter().map(|x| x as f64).collect();
            unsafe { std::mem::transmute(concrete_data) }
        }
        DType::I64 => {
            assert_eq!(std::any::TypeId::of::<S>(), std::any::TypeId::of::<i64>());
            let concrete_data: Vec<i64> = data.into_iter().map(|x| x as i64).collect();
            unsafe { std::mem::transmute(concrete_data) }
        }
        DType::U32 => {
            assert_eq!(std::any::TypeId::of::<S>(), std::any::TypeId::of::<u32>());
            let concrete_data: Vec<u32> = data.into_iter().map(|x| x as u32).collect();
            unsafe { std::mem::transmute(concrete_data) }
        }
        DType::U8 => {
            assert_eq!(std::any::TypeId::of::<S>(), std::any::TypeId::of::<u8>());
            let concrete_data: Vec<u8> = data.into_iter().map(|x| x as u8).collect();
            unsafe { std::mem::transmute(concrete_data) }
        }
        _ => panic!("Unsupported dtype {:?}", dtype),
    };

    (tensor, data)
}

/// Generate a random Shape with an appropriate size.
///
/// # Note
/// * The maximum dimension number is 4.
/// * The maximum size of each dimension is 100.
/// * The product of all dimensions is less than 10000.
///
/// # Arguments
/// * `seed` - The seed for random number generator.
///
/// # Returns
/// * `shape` - The random shape.
pub fn generate_random_shape_with_appropriate_size(seed: u64) -> Shape {
    let mut rng = StdRng::seed_from_u64(seed);
    let num_dim = rng.random_range(1..=4);
    let mut dims = Vec::with_capacity(num_dim);
    let mut total_size = 1;
    for _ in 0..num_dim {
        let max_dim = (10000 / total_size).min(100);
        let dim = rng.random_range(1..=max_dim);
        dims.push(dim);
        total_size *= dim;
    }
    Shape::from_dims(dims.as_slice())
}

proptest! {
    #[test]
    fn test_generate_random_tensor_and_corresponding_data_with_random_shape_and_random_dtype(
        shape_seed in prop::num::u64::ANY,
        dtype in prop::sample::select(vec![DType::U8, DType::U32, DType::I64, DType::F32,DType::F64]),
        grad_enabled in prop::bool::ANY,
        data_seed in prop::num::u64::ANY,
    ) {
        let shape = generate_random_shape_with_appropriate_size(shape_seed);
        let dtype = DType::from(dtype);
        let device = Device::cpu();

        match dtype {
            DType::U8 => {
                let (tensor, data) = generate_random_tensor_and_corresponding_data::<u8>(&shape, &dtype, &device, grad_enabled, data_seed);
                assert_eq!(tensor.shape().unwrap(), shape);
                assert_eq!(tensor.dtype().unwrap(), dtype);
                assert_eq!(tensor.device().unwrap(), device);
                assert_eq!(tensor.grad_enabled().unwrap(), grad_enabled);
                assert_eq!(tensor.to_vec::<u8>().unwrap(), data);
            }
            DType::U32 => {
                let (tensor, data) = generate_random_tensor_and_corresponding_data::<u32>(&shape, &dtype, &device, grad_enabled, data_seed);
                assert_eq!(tensor.shape().unwrap(), shape);
                assert_eq!(tensor.dtype().unwrap(), dtype);
                assert_eq!(tensor.device().unwrap(), device);
                assert_eq!(tensor.grad_enabled().unwrap(), grad_enabled);
                assert_eq!(tensor.to_vec::<u32>().unwrap(), data);
            }
            DType::I64 => {
                let (tensor, data) = generate_random_tensor_and_corresponding_data::<i64>(&shape, &dtype, &device, grad_enabled, data_seed);
                assert_eq!(tensor.shape().unwrap(), shape);
                assert_eq!(tensor.dtype().unwrap(), dtype);
                assert_eq!(tensor.device().unwrap(), device);
                assert_eq!(tensor.grad_enabled().unwrap(), grad_enabled);
                assert_eq!(tensor.to_vec::<i64>().unwrap(), data);
            }
            DType::F32 => {
                let (tensor, data) = generate_random_tensor_and_corresponding_data::<f32>(&shape, &dtype, &device, grad_enabled, data_seed);
                assert_eq!(tensor.shape().unwrap(), shape);
                assert_eq!(tensor.dtype().unwrap(), dtype);
                assert_eq!(tensor.device().unwrap(), device);
                assert_eq!(tensor.grad_enabled().unwrap(), grad_enabled);
                assert_eq!(tensor.to_vec::<f32>().unwrap(), data);
            }
            DType::F64 => {
                let (tensor, data) = generate_random_tensor_and_corresponding_data::<f64>(&shape, &dtype, &device, grad_enabled, data_seed);
                assert_eq!(tensor.shape().unwrap(), shape);
                assert_eq!(tensor.dtype().unwrap(), dtype);
                assert_eq!(tensor.device().unwrap(), device);
                assert_eq!(tensor.grad_enabled().unwrap(), grad_enabled);
                assert_eq!(tensor.to_vec::<f64>().unwrap(), data);
            }
            _ => {
                panic!("Unsupported dtype");
            }
        };
    }

    #[test]
    fn test_generate_random_shape_with_appropriate_size(
        seed in prop::num::u64::ANY,
    ) {
        let shape = generate_random_shape_with_appropriate_size(seed);
        let dims = shape.dims();
        let num_dim = dims.len();
        assert!(num_dim <= 4);

        let mut total_size: usize = 1;
        for &dim in dims {
            assert!(dim <= 100);
            total_size = total_size.saturating_mul(dim);
        }
        assert!(total_size <= 10000);
    }
}
