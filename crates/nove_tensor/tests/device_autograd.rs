use nove_tensor::{Device, Shape, Tensor};

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "value {index} differs: actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }
}

fn grouped_conv2d_loss(
    input_values: &[f32],
    kernel_values: &[f32],
    device: &Device,
) -> Result<f32, nove_tensor::TensorError> {
    let input = Tensor::from_slice(
        input_values,
        &Shape::from_dims(&[1, 2, 2, 3]),
        device,
        false,
    )?;
    let kernel = Tensor::from_slice(
        kernel_values,
        &Shape::from_dims(&[2, 1, 2, 2]),
        device,
        false,
    )?;
    input
        .conv2d(&kernel, 1, 2, 1, 2)?
        .sum(None)?
        .to_scalar::<f32>()
}

fn numerical_gradient(
    values: &[f32],
    epsilon: f32,
    mut loss: impl FnMut(&[f32]) -> Result<f32, nove_tensor::TensorError>,
) -> Result<Vec<f32>, nove_tensor::TensorError> {
    let mut gradient = Vec::with_capacity(values.len());
    for index in 0..values.len() {
        let mut upper = values.to_vec();
        upper[index] += epsilon;
        let mut lower = values.to_vec();
        lower[index] -= epsilon;
        gradient.push((loss(&upper)? - loss(&lower)?) / (2.0 * epsilon));
    }
    Ok(gradient)
}

fn grouped_conv1d_loss(
    input_values: &[f32],
    kernel_values: &[f32],
    device: &Device,
) -> Result<f32, nove_tensor::TensorError> {
    let input = Tensor::from_slice(input_values, &Shape::from_dims(&[1, 2, 6]), device, false)?;
    let kernel = Tensor::from_slice(kernel_values, &Shape::from_dims(&[2, 1, 2]), device, false)?;
    input
        .conv1d(&kernel, 1, 2, 2, 2)?
        .sum(None)?
        .to_scalar::<f32>()
}

fn assert_grouped_conv1d_gradient(device: Device) -> TestResult {
    let input_values = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5,
    ];
    let kernel_values = vec![1.0f32, -0.5, 0.75, 1.5];
    let input = Tensor::from_slice(&input_values, &Shape::from_dims(&[1, 2, 6]), &device, true)?;
    let kernel = Tensor::from_slice(&kernel_values, &Shape::from_dims(&[2, 1, 2]), &device, true)?;
    input.conv1d(&kernel, 1, 2, 2, 2)?.sum(None)?.backward()?;

    let epsilon = 1e-3f32;
    let numerical_input = numerical_gradient(&input_values, epsilon, |values| {
        grouped_conv1d_loss(values, &kernel_values, &device)
    })?;
    let numerical_kernel = numerical_gradient(&kernel_values, epsilon, |values| {
        grouped_conv1d_loss(&input_values, values, &device)
    })?;
    assert_close(
        &input.grad()?.unwrap().to_vec::<f32>()?,
        &numerical_input,
        2e-3,
    );
    assert_close(
        &kernel.grad()?.unwrap().to_vec::<f32>()?,
        &numerical_kernel,
        5e-3,
    );
    Ok(())
}

fn assert_grouped_conv2d_gradient(device: Device) -> TestResult {
    let input_values = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5,
    ];
    let kernel_values = vec![1.0f32, -0.5, 0.25, 2.0, -1.0, 0.75, 1.5, -0.25];
    let (analytic_input, analytic_kernel) =
        grouped_conv2d_analytic_gradients(&input_values, &kernel_values, &device)?;

    let epsilon = 1e-3f32;
    let numerical_input = numerical_gradient(&input_values, epsilon, |values| {
        grouped_conv2d_loss(values, &kernel_values, &device)
    })?;
    let numerical_kernel = numerical_gradient(&kernel_values, epsilon, |values| {
        grouped_conv2d_loss(&input_values, values, &device)
    })?;
    assert_close(&analytic_input, &numerical_input, 2e-3);
    assert_close(&analytic_kernel, &numerical_kernel, 5e-3);
    Ok(())
}

fn grouped_conv2d_analytic_gradients(
    input_values: &[f32],
    kernel_values: &[f32],
    device: &Device,
) -> Result<(Vec<f32>, Vec<f32>), nove_tensor::TensorError> {
    let input = Tensor::from_slice(input_values, &Shape::from_dims(&[1, 2, 2, 3]), device, true)?;
    let kernel = Tensor::from_slice(
        kernel_values,
        &Shape::from_dims(&[2, 1, 2, 2]),
        device,
        true,
    )?;
    input.conv2d(&kernel, 1, 2, 1, 2)?.sum(None)?.backward()?;
    Ok((
        input.grad()?.unwrap().to_vec::<f32>()?,
        kernel.grad()?.unwrap().to_vec::<f32>()?,
    ))
}

fn assert_pooling_overlap_and_ties(device: Device) -> TestResult {
    let values = vec![
        2.0f32, 2.0, 1.0, 0.0, 2.0, 2.0, 3.0, 3.0, 0.0, 1.0, 3.0, 3.0,
    ];
    let max_input = Tensor::from_slice(&values, &Shape::from_dims(&[1, 1, 3, 4]), &device, true)?;
    max_input
        .max_pool2d((2, 2), (1, 2))?
        .sum(None)?
        .backward()?;
    assert_close(
        &max_input.grad()?.unwrap().to_vec::<f32>()?,
        &[
            0.25, 0.25, 0.0, 0.0, 0.75, 0.75, 0.75, 0.75, 0.0, 0.0, 0.25, 0.25,
        ],
        1e-6,
    );

    let avg_input = Tensor::from_slice(&values, &Shape::from_dims(&[1, 1, 3, 4]), &device, true)?;
    avg_input
        .avg_pool2d((2, 2), (1, 2))?
        .sum(None)?
        .backward()?;
    assert_close(
        &avg_input.grad()?.unwrap().to_vec::<f32>()?,
        &[
            0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25,
        ],
        1e-6,
    );
    Ok(())
}

fn assert_indexed_gradient_accumulation(device: Device) -> TestResult {
    let selected = Tensor::from_data(
        vec![
            vec![1.0f32, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ],
        &device,
        true,
    )?;
    let row_indexes = Tensor::from_slice(&[2i64, 0, 2], &Shape::from_dims(&[3]), &device, false)?;
    selected
        .index_select(&row_indexes, 0)?
        .sum(None)?
        .backward()?;
    assert_eq!(
        selected.grad()?.unwrap().to_vec::<f32>()?,
        vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0]
    );

    let gathered = Tensor::from_data(
        vec![vec![1.0f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
        &device,
        true,
    )?;
    let gather_indexes = Tensor::from_data(vec![vec![2i64, 0], vec![1, 1]], &device, false)?;
    gathered.gather(&gather_indexes, 1)?.sum(None)?.backward()?;
    assert_eq!(
        gathered.grad()?.unwrap().to_vec::<f32>()?,
        vec![1.0, 0.0, 1.0, 0.0, 2.0, 0.0]
    );

    let table = Tensor::from_data(&[[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]], &device, true)?;
    let embedding_indexes =
        Tensor::from_slice(&[0i64, 2, 0], &Shape::from_dims(&[3]), &device, false)?;
    table.embedding(&embedding_indexes)?.sum(None)?.backward()?;
    assert_eq!(
        table.grad()?.unwrap().to_vec::<f32>()?,
        vec![2.0, 2.0, 0.0, 0.0, 1.0, 1.0]
    );
    Ok(())
}

#[cfg(feature = "candle-cpu")]
#[test]
fn candle_cpu_grouped_conv2d_matches_finite_difference() -> TestResult {
    assert_grouped_conv2d_gradient(nove_tensor::device::candle::cpu()?)
}

#[cfg(feature = "candle-cpu")]
#[test]
fn candle_cpu_grouped_conv1d_matches_finite_difference() -> TestResult {
    assert_grouped_conv1d_gradient(nove_tensor::device::candle::cpu()?)
}

#[cfg(feature = "candle-cpu")]
#[test]
fn candle_cpu_pooling_handles_overlap_asymmetric_stride_and_ties() -> TestResult {
    assert_pooling_overlap_and_ties(nove_tensor::device::candle::cpu()?)
}

#[cfg(feature = "candle-cpu")]
#[test]
fn candle_cpu_indexed_gradients_accumulate_repeated_indexes() -> TestResult {
    assert_indexed_gradient_accumulation(nove_tensor::device::candle::cpu()?)
}

#[cfg(feature = "native-cpu")]
#[test]
fn native_cpu_grouped_conv2d_matches_finite_difference() -> TestResult {
    assert_grouped_conv2d_gradient(nove_tensor::device::native::cpu()?)
}

#[cfg(feature = "native-cpu")]
#[test]
fn native_cpu_grouped_conv1d_matches_finite_difference() -> TestResult {
    assert_grouped_conv1d_gradient(nove_tensor::device::native::cpu()?)
}

#[cfg(feature = "native-cpu")]
#[test]
fn native_cpu_pooling_handles_overlap_asymmetric_stride_and_ties() -> TestResult {
    assert_pooling_overlap_and_ties(nove_tensor::device::native::cpu()?)
}

#[cfg(feature = "native-cpu")]
#[test]
fn native_cpu_indexed_gradients_accumulate_repeated_indexes() -> TestResult {
    assert_indexed_gradient_accumulation(nove_tensor::device::native::cpu()?)
}

#[cfg(all(feature = "candle-cpu", feature = "native-cpu"))]
#[test]
fn candle_and_native_grouped_conv2d_gradients_match() -> TestResult {
    let input_values = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5,
    ];
    let kernel_values = vec![1.0f32, -0.5, 0.25, 2.0, -1.0, 0.75, 1.5, -0.25];
    let candle = grouped_conv2d_analytic_gradients(
        &input_values,
        &kernel_values,
        &nove_tensor::device::candle::cpu()?,
    )?;
    let native = grouped_conv2d_analytic_gradients(
        &input_values,
        &kernel_values,
        &nove_tensor::device::native::cpu()?,
    )?;
    assert_close(&candle.0, &native.0, 1e-6);
    assert_close(&candle.1, &native.1, 1e-6);
    Ok(())
}
