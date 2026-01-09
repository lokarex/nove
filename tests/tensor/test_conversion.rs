use std::fmt::Debug;

use nove::tensor::{Device, Tensor};
use proptest::prelude::*;

// Test from_data and to_scalar with a scalar value.
fn test_from_data_and_to_scalar_with_scalar<T>(val: T) -> Result<(), TestCaseError>
where
    T: candle_core::NdArray + candle_core::WithDType + Debug,
{
    let device = Device::get_cuda_if_available(0);
    let tensor = Tensor::from_data(val, &device, false).unwrap();
    prop_assert_eq!(tensor.to_scalar::<T>().unwrap(), val);
    Ok(())
}

proptest! {
    #[test]
    // Test from_data and to_scalar with a scalar f32 value.
    fn test_from_data_and_to_scalar_with_scalar_f32(val in any::<f32>()) {
        test_from_data_and_to_scalar_with_scalar(val)?;
    }

    #[test]
    // Test from_data and to_scalar with a scalar f64 value.
    fn test_from_data_and_to_scalar_with_scalar_f64(val in any::<f64>()) {
        test_from_data_and_to_scalar_with_scalar(val)?;
    }

    // Test from_data and to_scalar with a scalar i64 value.
    #[test]
    fn test_from_data_and_to_scalar_with_scalar_i64(val in any::<i64>()) {
        test_from_data_and_to_scalar_with_scalar(val)?;
    }

    // Test from_data and to_scalar with a scalar u32 value.
    #[test]
    fn test_from_data_and_to_scalar_with_scalar_u32(val in any::<u32>()) {
        test_from_data_and_to_scalar_with_scalar(val)?;
    }

    // Test from_data and to_scalar with a scalar u8 value.
    #[test]
    fn test_from_data_and_to_scalar_with_scalar_u8(val in any::<u8>()) {
        test_from_data_and_to_scalar_with_scalar(val)?;
    }
}

// Test from_data and to_vec with a vector value.
fn test_from_data_and_to_vec_with_vector<T>(vec: Vec<T>) -> Result<(), TestCaseError>
where
    T: candle_core::NdArray + candle_core::WithDType + Debug + Copy,
{
    let device = Device::get_cuda_if_available(0);

    // Test from_data and to_vec with a 1D vector.
    let tensor = Tensor::from_data(vec.clone(), &device, false).unwrap();
    prop_assert_eq!(tensor.to_vec::<T>().unwrap(), vec.clone());

    // Test from_data and to_vec with a 2D vector.
    let vec_2d = vec![vec.clone(), vec.clone()];
    let tensor = Tensor::from_data(vec_2d.clone(), &device, false).unwrap();
    prop_assert_eq!(
        tensor.to_vec::<T>().unwrap(),
        vec.iter().chain(vec.iter()).copied().collect::<Vec<_>>()
    );

    // Test from_data and to_vec with a 3D vector.
    let vec_3d = vec![
        vec![vec.clone(), vec.clone()],
        vec![vec.clone(), vec.clone()],
    ];
    let tensor = Tensor::from_data(vec_3d, &device, false).unwrap();
    let expected: Vec<T> = vec.iter().cycle().take(vec.len() * 4).copied().collect();
    prop_assert_eq!(tensor.to_vec::<T>().unwrap(), expected);

    // Test from_data and to_vec with a 4D vector.
    let vec_4d = vec![
        vec![
            vec![vec.clone(), vec.clone()],
            vec![vec.clone(), vec.clone()],
        ],
        vec![
            vec![vec.clone(), vec.clone()],
            vec![vec.clone(), vec.clone()],
        ],
    ];
    let tensor = Tensor::from_data(vec_4d, &device, false).unwrap();
    let expected: Vec<T> = vec.iter().cycle().take(vec.len() * 8).copied().collect();
    prop_assert_eq!(tensor.to_vec::<T>().unwrap(), expected);

    Ok(())
}

proptest! {
    #[test]
    // Test from_data and to_vec with a vector f32 value.
    fn test_from_data_and_to_vec_with_vector_f32(vec in prop::collection::vec(any::<f32>(), 1..10)) {
        test_from_data_and_to_vec_with_vector(vec)?;
    }

    #[test]
    // Test from_data and to_vec with a vector f64 value.
    fn test_from_data_and_to_vec_with_vector_f64(vec in prop::collection::vec(any::<f64>(), 1..10)) {
        test_from_data_and_to_vec_with_vector(vec)?;
    }

    // Test from_data and to_vec with a vector i64 value.
    #[test]
    fn test_from_data_and_to_vec_with_vector_i64(vec in prop::collection::vec(any::<i64>(), 1..10)) {
        test_from_data_and_to_vec_with_vector(vec)?;
    }

    // Test from_data and to_vec with a vector u32 value.
    #[test]
    fn test_from_data_and_to_vec_with_vector_u32(vec in prop::collection::vec(any::<u32>(), 1..10)) {
        test_from_data_and_to_vec_with_vector(vec)?;
    }

    // Test from_data and to_vec with a vector u8 value.
    #[test]
    fn test_from_data_and_to_vec_with_vector_u8(vec in prop::collection::vec(any::<u8>(), 1..10)) {
        test_from_data_and_to_vec_with_vector(vec)?;
    }
}

// Test from_data and to_vec with a slice value.
fn test_from_data_and_to_vec_with_slice<T>(slice: &[T]) -> Result<(), TestCaseError>
where
    T: candle_core::NdArray + candle_core::WithDType + Debug + Copy,
{
    let device = Device::get_cuda_if_available(0);

    // Test from_data and to_vec with a 1D slice.
    let tensor = Tensor::from_data(slice, &device, false).unwrap();
    prop_assert_eq!(tensor.to_vec::<T>().unwrap(), slice.to_vec());

    // Test from_data and to_vec with a 2D slice (fixed size).
    if slice.len() >= 2 {
        let arr: [T; 2] = [slice[0], slice[1]];
        let arr_2d: [[T; 2]; 2] = [arr, arr];
        let tensor = Tensor::from_data(&arr_2d, &device, false).unwrap();
        let expected: Vec<T> = vec![slice[0], slice[1], slice[0], slice[1]];
        prop_assert_eq!(tensor.to_vec::<T>().unwrap(), expected);
    }

    // Test from_data and to_vec with a 3D slice (fixed size).
    if slice.len() >= 2 {
        let arr: [T; 2] = [slice[0], slice[1]];
        let arr_2d: [[T; 2]; 2] = [arr, arr];
        let arr_3d: [[[T; 2]; 2]; 2] = [arr_2d, arr_2d];
        let tensor = Tensor::from_data(&arr_3d, &device, false).unwrap();
        let expected: Vec<T> = vec![
            slice[0], slice[1], slice[0], slice[1], slice[0], slice[1], slice[0], slice[1],
        ];
        prop_assert_eq!(tensor.to_vec::<T>().unwrap(), expected);
    }

    // Test from_data and to_vec with a 4D slice (fixed size).
    if slice.len() >= 2 {
        let arr: [T; 2] = [slice[0], slice[1]];
        let arr_2d: [[T; 2]; 2] = [arr, arr];
        let arr_3d: [[[T; 2]; 2]; 2] = [arr_2d, arr_2d];
        let arr_4d: [[[[T; 2]; 2]; 2]; 2] = [arr_3d, arr_3d];
        let tensor = Tensor::from_data(&arr_4d, &device, false).unwrap();
        let expected: Vec<T> = vec![
            slice[0], slice[1], slice[0], slice[1], slice[0], slice[1], slice[0], slice[1],
            slice[0], slice[1], slice[0], slice[1], slice[0], slice[1], slice[0], slice[1],
        ];
        prop_assert_eq!(tensor.to_vec::<T>().unwrap(), expected);
    }

    Ok(())
}

proptest! {
    #[test]
    // Test from_data and to_vec with a slice f32 value.
    fn test_from_data_and_to_vec_with_slice_f32(slice in prop::collection::vec(any::<f32>(), 2..10)) {
        test_from_data_and_to_vec_with_slice(&slice)?;
    }

    #[test]
    // Test from_data and to_vec with a slice f64 value.
    fn test_from_data_and_to_vec_with_slice_f64(slice in prop::collection::vec(any::<f64>(), 2..10)) {
        test_from_data_and_to_vec_with_slice(&slice)?;
    }

    // Test from_data and to_vec with a slice i64 value.
    #[test]
    fn test_from_data_and_to_vec_with_slice_i64(slice in prop::collection::vec(any::<i64>(), 2..10)) {
        test_from_data_and_to_vec_with_slice(&slice)?;
    }

    // Test from_data and to_vec with a slice u32 value.
    #[test]
    fn test_from_data_and_to_vec_with_slice_u32(slice in prop::collection::vec(any::<u32>(), 2..10)) {
        test_from_data_and_to_vec_with_slice(&slice)?;
    }

    // Test from_data and to_vec with a slice u8 value.
    #[test]
    fn test_from_data_and_to_vec_with_slice_u8(slice in prop::collection::vec(any::<u8>(), 2..10)) {
        test_from_data_and_to_vec_with_slice(&slice)?;
    }
}
