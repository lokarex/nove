use nove::tensor::{DType, Device, Tensor};
use proptest::prelude::*;

use crate::tensor::util;

fn test_add_two_tensors<S>((tensor1, data1): (Tensor, Vec<S>), (tensor2, data2): (Tensor, Vec<S>))
where
    S: candle_core::WithDType + std::fmt::Debug,
{
    let tensor3 = tensor1.add(&tensor2);
    let data3 = tensor3.unwrap().to_vec::<S>().unwrap();
    assert_eq!(
        data3,
        data1
            .iter()
            .zip(data2.iter())
            .map(|(a, b)| *a + *b)
            .collect::<Vec<S>>()
    );
}

proptest! {
    #[test]
    fn test_add_two_tensors_with_same_shape(
        shape_seed in prop::num::u64::ANY,
        dtype in prop::sample::select(vec![DType::U8, DType::U32, DType::I64, DType::F32,DType::F64]),
        grad_enabled in prop::bool::ANY,
        data_seed in prop::num::u64::ANY,
    ) {
        let shape = util::generate_random_shape_with_appropriate_size(shape_seed);
        let device = Device::get_cpu();
        let mut data_seed = data_seed;

        match dtype {
            DType::F32 => {
                let (tensor1, data1) = util::generate_random_tensor_and_corresponding_data::<f32>(&shape, &dtype, &device, grad_enabled, data_seed);
                data_seed += 1;
                let (tensor2, data2) = util::generate_random_tensor_and_corresponding_data::<f32>(&shape, &dtype, &device, grad_enabled, data_seed);
                test_add_two_tensors((tensor1, data1), (tensor2, data2));
            }
            DType::F64 => {
                let (tensor1, data1) = util::generate_random_tensor_and_corresponding_data::<f64>(&shape, &dtype, &device, grad_enabled, data_seed);
                data_seed += 1;
                let (tensor2, data2) = util::generate_random_tensor_and_corresponding_data::<f64>(&shape, &dtype, &device, grad_enabled, data_seed);
                test_add_two_tensors((tensor1, data1), (tensor2, data2));
            }
            DType::I64 => {
                let (tensor1, data1) = util::generate_random_tensor_and_corresponding_data::<i64>(&shape, &dtype, &device, grad_enabled, data_seed);
                data_seed += 1;
                let (tensor2, data2) = util::generate_random_tensor_and_corresponding_data::<i64>(&shape, &dtype, &device, grad_enabled, data_seed);
                test_add_two_tensors((tensor1, data1), (tensor2, data2));
            }
            DType::U32 => {
                let (tensor1, data1) = util::generate_random_tensor_and_corresponding_data::<u32>(&shape, &dtype, &device, grad_enabled, data_seed);
                data_seed += 1;
                let (tensor2, data2) = util::generate_random_tensor_and_corresponding_data::<u32>(&shape, &dtype, &device, grad_enabled, data_seed);
                test_add_two_tensors((tensor1, data1), (tensor2, data2));
            }
            DType::U8 => {
                let (tensor1, data1) = util::generate_random_tensor_and_corresponding_data::<u8>(&shape, &dtype, &device, grad_enabled, data_seed);
                data_seed += 1;
                let (tensor2, data2) = util::generate_random_tensor_and_corresponding_data::<u8>(&shape, &dtype, &device, grad_enabled, data_seed);
                test_add_two_tensors((tensor1, data1), (tensor2, data2));
            }

            _ => panic!("unexpected dtype {:?}", dtype),
        }
    }
}

fn test_stack_tensors<S>(datas: Vec<(Tensor, Vec<S>)>)
where
    S: candle_core::WithDType + std::fmt::Debug,
{
    let mut stacked_data: Vec<S> = Vec::with_capacity(
        datas.len()
            * datas[0]
                .0
                .get_shape()
                .unwrap()
                .dims()
                .iter()
                .product::<usize>(),
    );
    let mut stacked_tensor = Vec::with_capacity(datas.len());

    datas.iter().for_each(|(tensor, data)| {
        stacked_tensor.push(tensor.clone());
        stacked_data.extend(data);
    });

    let stacked_tensor = Tensor::stack(&stacked_tensor, 0).unwrap();
    assert_eq!(stacked_tensor.to_vec::<S>().unwrap(), stacked_data);
}

proptest! {
    #[test]
    fn test_stack_tensors_on_axis_0_with_same_shape(
        num_tensors in 2..10,
        shape_seed in prop::num::u64::ANY,
        dtype in prop::sample::select(vec![DType::U8, DType::U32, DType::I64, DType::F32,DType::F64]),
        grad_enabled in prop::bool::ANY,
        data_seed in prop::num::u64::ANY,
    ) {
        let shape = util::generate_random_shape_with_appropriate_size(shape_seed);
        let device = Device::get_cpu();
        let mut data_seed= data_seed;

        match dtype {
            DType::F32 => {
                let datas = (0..num_tensors)
                    .map(|_| {
                        data_seed += 1;
                        util::generate_random_tensor_and_corresponding_data::<f32>(&shape, &dtype, &device, grad_enabled, data_seed)
                    })
                    .collect::<Vec<_>>();
                test_stack_tensors(datas);
            }
            DType::F64 => {
                let datas = (0..num_tensors)
                    .map(|_| {
                        data_seed += 1;
                        util::generate_random_tensor_and_corresponding_data::<f64>(&shape, &dtype, &device, grad_enabled, data_seed)
                    })
                    .collect::<Vec<_>>();
                test_stack_tensors(datas);
            }
            DType::I64 => {
                let datas = (0..num_tensors)
                    .map(|_| {
                        data_seed += 1;
                        util::generate_random_tensor_and_corresponding_data::<i64>(&shape, &dtype, &device, grad_enabled, data_seed)
                    })
                    .collect::<Vec<_>>();
                test_stack_tensors(datas);
            }
            DType::U32 => {
                let datas = (0..num_tensors)
                    .map(|_| {
                        data_seed += 1;
                        util::generate_random_tensor_and_corresponding_data::<u32>(&shape, &dtype, &device, grad_enabled, data_seed)
                    })
                    .collect::<Vec<_>>();
                test_stack_tensors(datas);
            }
            DType::U8 => {
                let datas = (0..num_tensors)
                    .map(|_| {
                        data_seed += 1;
                        util::generate_random_tensor_and_corresponding_data::<u8>(&shape, &dtype, &device, grad_enabled, data_seed)
                    })
                    .collect::<Vec<_>>();
                test_stack_tensors(datas);
            }
            _ => panic!("unexpected dtype {:?}", dtype),
        }
    }
}
