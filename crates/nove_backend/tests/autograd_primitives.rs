use nove_backend::{Device, backend::BackendStorage};

fn assert_index_accumulation(device: Device) {
    let zeros = BackendStorage::from_data(vec![vec![0.0f32; 3]; 2], &device).unwrap();
    let indexes = BackendStorage::from_data(vec![2i64, 0, 2], &device).unwrap();
    let source =
        BackendStorage::from_data(vec![vec![1.0f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device)
            .unwrap();

    let index_added = zeros.index_add(&indexes, &source, 1).unwrap();
    assert_eq!(
        index_added.to_vec::<f32>().unwrap(),
        vec![2.0, 0.0, 4.0, 5.0, 0.0, 10.0]
    );

    let scatter_indexes =
        BackendStorage::from_data(vec![vec![2i64, 0, 2], vec![1, 1, 0]], &device).unwrap();
    let scattered = zeros.scatter_add(&scatter_indexes, &source, 1).unwrap();
    assert_eq!(
        scattered.to_vec::<f32>().unwrap(),
        vec![2.0, 0.0, 4.0, 6.0, 9.0, 0.0]
    );
}

fn assert_transposed_convolution(device: Device) {
    let input_1d = BackendStorage::from_data(vec![vec![vec![1.0f32, 2.0]]], &device).unwrap();
    let kernel_1d = BackendStorage::from_data(vec![vec![vec![3.0f32, 4.0]]], &device).unwrap();
    let output_1d = input_1d
        .conv_transpose1d(&kernel_1d, 0, 0, 2, 1, 1)
        .unwrap();
    assert_eq!(output_1d.shape().unwrap().dims(), &[1, 1, 4]);
    assert_eq!(output_1d.to_vec::<f32>().unwrap(), vec![3.0, 4.0, 6.0, 8.0]);

    let input_2d =
        BackendStorage::from_data(vec![vec![vec![vec![1.0f32, 2.0], vec![3.0, 4.0]]]], &device)
            .unwrap();
    let kernel_2d =
        BackendStorage::from_data(vec![vec![vec![vec![1.0f32, 2.0], vec![3.0, 4.0]]]], &device)
            .unwrap();
    let output_2d = input_2d
        .conv_transpose2d(&kernel_2d, 0, (0, 1), 2, 1, 1)
        .unwrap();
    assert_eq!(output_2d.shape().unwrap().dims(), &[1, 1, 4, 5]);
    assert_eq!(
        output_2d.to_vec::<f32>().unwrap(),
        vec![
            1.0, 2.0, 2.0, 4.0, 0.0, 3.0, 4.0, 6.0, 8.0, 0.0, 3.0, 6.0, 4.0, 8.0, 0.0, 9.0, 12.0,
            12.0, 16.0, 0.0,
        ]
    );
}

#[cfg(feature = "candle-cpu")]
#[test]
fn candle_cpu_index_accumulation_primitives() {
    assert_index_accumulation(nove_backend::device::candle::cpu().unwrap());
}

#[cfg(feature = "candle-cpu")]
#[test]
fn candle_cpu_transposed_convolution_primitives() {
    assert_transposed_convolution(nove_backend::device::candle::cpu().unwrap());
}

#[cfg(feature = "native-cpu")]
#[test]
fn native_cpu_index_accumulation_primitives() {
    assert_index_accumulation(nove_backend::device::native::cpu().unwrap());
}

#[cfg(feature = "native-cpu")]
#[test]
fn native_cpu_transposed_convolution_primitives() {
    assert_transposed_convolution(nove_backend::device::native::cpu().unwrap());
}
