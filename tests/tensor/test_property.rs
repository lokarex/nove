use nove::tensor::{DType, Device, Tensor};
use proptest::prelude::*;

#[test]
fn test_to_device_and_get_device_with_cpu() {
    let cpu = Device::get_cpu();
    let tensor = Tensor::from_data(&[1.0f32, 2.0f32], &cpu, false).unwrap();

    assert_eq!(tensor.get_device().unwrap(), cpu);
    assert_eq!(tensor.to_device(&cpu), Ok(()));
}

fn test_to_dtype_and_get_dtype(tensor: &Tensor, original_dtype: &DType, target_dtype: &DType) {
    // Check original dtype
    assert_eq!(tensor.get_dtype().unwrap(), *original_dtype);

    // Try to convert to target dtype
    assert_eq!(Ok(()), tensor.to_dtype(target_dtype));

    // Check the current dtype
    assert_eq!(tensor.get_dtype().unwrap(), *target_dtype);
}

proptest! {
    #[test]
    fn test_to_dtype_and_get_dtype_with_random_dtype(
        original_dtype in prop::sample::select(vec![DType::U8, DType::U32, DType::I64, DType::BF16, DType::F16, DType::F32,DType::F64]),
        target_dtype in prop::sample::select(vec![DType::U8, DType::U32, DType::I64, DType::BF16, DType::F16, DType::F32,DType::F64]),
    ) {
        let cpu = Device::get_cpu();
        let mut tensor = Tensor::from_data(&[1.0f32, 2.0f32], &cpu, false).unwrap();

        // If original_dtype is not F32, convert it to original_dtype
        if original_dtype != DType::F32 {
            tensor.to_dtype(&original_dtype).unwrap();
        }

        test_to_dtype_and_get_dtype(&mut tensor, &original_dtype, &target_dtype);
    }
}

fn test_get_shape_and_get_dim_num(tensor: &Tensor, expected_shape: &[usize]) {
    // Check shape
    let shape = tensor.get_shape().unwrap();
    assert_eq!(shape.dims(), expected_shape);

    // Check dim_num
    let dim_num = tensor.get_dim_num().unwrap();
    assert_eq!(dim_num, expected_shape.len());
}

proptest! {
    #[test]
    fn test_get_shape_and_get_dim_num_with_random_shapes(
        dim_num in 1usize..5,
        dim0 in 1usize..10,
        dim1 in 1usize..10,
        dim2 in 1usize..10,
        dim3 in 1usize..10,
    ) {
        let cpu = Device::get_cpu();
        let data: Vec<f32> = (0..dim0 * dim1 * dim2 * dim3).map(|i| i as f32).collect();

        let tensor = match dim_num {
            1 => Tensor::from_data(&data[..dim0], &cpu, false).unwrap(),
            2 => {
                let data_2d: Vec<Vec<f32>> = data[..dim0 * dim1]
                    .chunks(dim1)
                    .map(|c| c.to_vec())
                    .collect();
                Tensor::from_data(data_2d, &cpu, false).unwrap()
            }
            3 => {
                let data_3d: Vec<Vec<Vec<f32>>> = data[..dim0 * dim1 * dim2]
                    .chunks(dim1 * dim2)
                    .map(|chunk| {
                        chunk.chunks(dim2).map(|c| c.to_vec()).collect()
                    })
                    .collect();
                Tensor::from_data(data_3d, &cpu, false).unwrap()
            }
            4 => {
                let data_4d: Vec<Vec<Vec<Vec<f32>>>> = data
                    .chunks(dim1 * dim2 * dim3)
                    .map(|chunk| {
                        chunk.chunks(dim2 * dim3)
                            .map(|c| {
                                c.chunks(dim3).map(|d| d.to_vec()).collect()
                            })
                            .collect()
                    })
                    .collect();
                Tensor::from_data(data_4d, &cpu, false).unwrap()
            }
            _ => unreachable!(),
        };

        let expected_shape = match dim_num {
            1 => vec![dim0],
            2 => vec![dim0, dim1],
            3 => vec![dim0, dim1, dim2],
            4 => vec![dim0, dim1, dim2, dim3],
            _ => unreachable!(),
        };

        test_get_shape_and_get_dim_num(&tensor, &expected_shape);
    }
}
