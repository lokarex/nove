#[cfg(feature = "native-cpu")]
use nove_tensor::BackendKind;
use nove_tensor::{DType, Device, Shape, Tensor};

type TestResult = Result<(), Box<dyn std::error::Error>>;

#[test]
fn dtype_and_shape_are_nove_owned_public_types() {
    let dtype = "f32".parse::<DType>().unwrap();
    assert_eq!(dtype, DType::F32);
    assert_eq!(dtype.as_str(), "f32");
    assert_eq!(dtype.size_in_bytes(), 4);
    assert!(dtype.is_float());
    assert!(!dtype.is_int());

    let shape = Shape::from_dims(&[2, 3, 4]);
    assert_eq!(shape.dims(), &[2, 3, 4]);
    assert_eq!(shape.rank(), 3);
    assert_eq!(shape.elem_count(), 24);
    assert_eq!(format!("{shape:?}"), "[2, 3, 4]");
}

#[cfg(feature = "candle-cpu")]
#[test]
fn dtype_converts_to_and_from_the_candle_backend() {
    let candle_dtype: candle_core::DType = DType::F32.into();
    assert_eq!(candle_dtype, candle_core::DType::F32);

    let nove_dtype = DType::from(candle_core::DType::F64);
    assert_eq!(nove_dtype, DType::F64);
}

#[cfg(feature = "native-cpu")]
#[test]
fn explicit_nove_cpu_device_uses_native_cpu_backend() -> TestResult {
    let device = nove_tensor::device::native::cpu()?;
    assert_eq!(device.backend(), BackendKind::Native);
    assert!(device.is_cpu());

    let explicit = nove_tensor::device::native::cpu()?;
    assert_eq!(explicit.backend(), BackendKind::Native);

    Ok(())
}

#[test]
fn backward_tracks_a_representative_backend_graph() -> TestResult {
    let device = Device::default();
    let input = Tensor::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0],
        &Shape::from_dims(&[2, 2]),
        &device,
        true,
    )?;
    let weights = Tensor::from_slice(
        &[1.0f32, 0.0, 0.0, 1.0],
        &Shape::from_dims(&[2, 2]),
        &device,
        false,
    )?;
    let bias = Tensor::from_slice(&[1.0f32, -1.0], &Shape::from_dims(&[2]), &device, false)?;

    let loss = input
        .reshape(&Shape::from_dims(&[2, 2]))?
        .matmul(&weights)?
        .add(&bias)?
        .sum(None)?;

    assert!(loss.grad_enabled()?);
    loss.backward()?;

    let grad = input.grad()?.expect("input gradient should be populated");
    assert_eq!(grad.to_vec::<f32>()?, vec![1.0, 1.0, 1.0, 1.0]);

    Ok(())
}

#[test]
fn nove_cpu_runs_representative_forward_and_backward_chain() -> TestResult {
    let device = Device::default();
    let input = Tensor::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0],
        &Shape::from_dims(&[2, 2]),
        &device,
        true,
    )?;
    let weights = Tensor::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0],
        &Shape::from_dims(&[2, 2]),
        &device,
        false,
    )?;
    let bias = Tensor::from_slice(&[1.0f32, -1.0], &Shape::from_dims(&[2]), &device, false)?;

    let loss = input
        .reshape(&Shape::from_dims(&[2, 2]))?
        .matmul(&weights)?
        .add(&bias)?
        .sum(None)?;

    assert_eq!(loss.to_vec::<f32>()?, vec![54.0]);
    loss.backward()?;

    let grad = input.grad()?.expect("input gradient should be populated");
    assert_eq!(grad.to_vec::<f32>()?, vec![3.0, 7.0, 3.0, 7.0]);

    Ok(())
}

#[cfg(feature = "native-cpu")]
#[test]
fn nove_autograd_reduces_broadcast_gradients_on_cpu() -> TestResult {
    let device = nove_tensor::device::native::cpu()?;
    let lhs = Tensor::from_data(
        vec![vec![1.0f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
        &device,
        true,
    )?;
    let rhs = Tensor::from_data(vec![vec![2.0f32, 3.0, 4.0]], &device, true)?;

    let loss = lhs.mul(&rhs)?.sum(None)?;
    loss.backward()?;

    let lhs_grad = lhs.grad()?.expect("lhs gradient should be populated");
    assert_eq!(
        lhs_grad.to_vec::<f32>()?,
        vec![2.0, 3.0, 4.0, 2.0, 3.0, 4.0]
    );

    let rhs_grad = rhs.grad()?.expect("rhs gradient should be populated");
    assert_eq!(rhs_grad.to_vec::<f32>()?, vec![5.0, 7.0, 9.0]);
    assert_eq!(rhs_grad.shape()?, Shape::from_dims(&[1, 3]));

    Ok(())
}

#[cfg(feature = "native-cpu")]
#[test]
fn nove_cpu_autograd_handles_shape_clip_and_gelu_without_backend_fallback() -> TestResult {
    let device = nove_tensor::device::native::cpu()?;
    let input = Tensor::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        &Shape::from_dims(&[2, 3]),
        &device,
        true,
    )?;

    let loss = input
        .transpose(0, 1)?
        .permute(&[1, 0])?
        .flatten(None, None)?
        .reshape(&Shape::from_dims(&[2, 3]))?
        .clip(1.5, 5.5)?
        .gelu()?
        .sum(None)?;
    loss.backward()?;

    let grad = input.grad()?.expect("input gradient should be populated");
    let actual = grad.to_vec::<f32>()?;
    let expected = [
        0.0,
        gelu_tanh_derivative(2.0),
        gelu_tanh_derivative(3.0),
        gelu_tanh_derivative(4.0),
        gelu_tanh_derivative(5.0),
        0.0,
    ];
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert!((*actual as f64 - *expected).abs() < 1e-4);
    }

    Ok(())
}

#[cfg(feature = "native-cpu")]
#[test]
fn nove_cpu_autograd_handles_extrema_and_combination_without_backend_fallback() -> TestResult {
    let device = nove_tensor::device::native::cpu()?;
    let max_input = Tensor::from_data(
        vec![vec![1.0f32, 5.0, 3.0], vec![4.0, 2.0, 6.0]],
        &device,
        true,
    )?;
    max_input.max(Some((0, false)))?.sum(None)?.backward()?;
    let max_grad = max_input
        .grad()?
        .expect("max input gradient should be populated");
    assert_eq!(
        max_grad.to_vec::<f32>()?,
        vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    );

    let min_input = Tensor::from_data(
        vec![vec![1.0f32, 5.0, 3.0], vec![4.0, 2.0, 6.0]],
        &device,
        true,
    )?;
    min_input.min(Some((1, true)))?.sum(None)?.backward()?;
    let min_grad = min_input
        .grad()?
        .expect("min input gradient should be populated");
    assert_eq!(
        min_grad.to_vec::<f32>()?,
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    );

    let narrow_input = Tensor::from_data(
        vec![vec![1.0f32, 5.0, 3.0], vec![4.0, 2.0, 6.0]],
        &device,
        true,
    )?;
    narrow_input.narrow(1, 1, 2)?.sum(None)?.backward()?;
    let narrow_grad = narrow_input
        .grad()?
        .expect("narrow input gradient should be populated");
    assert_eq!(
        narrow_grad.to_vec::<f32>()?,
        vec![0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    );

    let lhs = Tensor::from_slice(&[1.0f32, 2.0], &Shape::from_dims(&[2]), &device, true)?;
    let rhs = Tensor::from_slice(&[3.0f32, 4.0], &Shape::from_dims(&[2]), &device, true)?;
    Tensor::stack(&[lhs.copy(), rhs.copy()], 0)?
        .sum(None)?
        .backward()?;
    assert_eq!(
        lhs.grad()?
            .expect("lhs gradient should be populated")
            .to_vec::<f32>()?,
        vec![1.0, 1.0]
    );
    assert_eq!(
        rhs.grad()?
            .expect("rhs gradient should be populated")
            .to_vec::<f32>()?,
        vec![1.0, 1.0]
    );

    let left = Tensor::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0],
        &Shape::from_dims(&[2, 2]),
        &device,
        true,
    )?;
    let right = Tensor::from_slice(
        &[5.0f32, 6.0, 7.0, 8.0],
        &Shape::from_dims(&[2, 2]),
        &device,
        true,
    )?;
    Tensor::cat(&[left.copy(), right.copy()], 1)?
        .sum(None)?
        .backward()?;
    assert_eq!(
        left.grad()?
            .expect("left gradient should be populated")
            .to_vec::<f32>()?,
        vec![1.0, 1.0, 1.0, 1.0]
    );
    assert_eq!(
        right
            .grad()?
            .expect("right gradient should be populated")
            .to_vec::<f32>()?,
        vec![1.0, 1.0, 1.0, 1.0]
    );

    let template = Tensor::from_slice(&[9.0f32, 8.0], &Shape::from_dims(&[2]), &device, true)?;
    template.zeros_like()?.sum(None)?.backward()?;
    assert_eq!(
        template
            .grad()?
            .expect("template gradient should be populated")
            .to_vec::<f32>()?,
        vec![0.0, 0.0]
    );

    let condition =
        Tensor::from_slice(&[1u8, 0, 1, 0], &Shape::from_dims(&[2, 2]), &device, false)?;
    let true_value = Tensor::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0],
        &Shape::from_dims(&[2, 2]),
        &device,
        true,
    )?;
    let false_value = Tensor::from_slice(
        &[5.0f32, 6.0, 7.0, 8.0],
        &Shape::from_dims(&[2, 2]),
        &device,
        true,
    )?;
    Tensor::where_cond(&condition, &true_value, &false_value)?
        .sum(None)?
        .backward()?;
    assert_eq!(
        true_value
            .grad()?
            .expect("true branch gradient should be populated")
            .to_vec::<f32>()?,
        vec![1.0, 0.0, 1.0, 0.0]
    );
    assert_eq!(
        false_value
            .grad()?
            .expect("false branch gradient should be populated")
            .to_vec::<f32>()?,
        vec![0.0, 1.0, 0.0, 1.0]
    );

    Ok(())
}

#[cfg(feature = "native-cpu")]
#[test]
fn nove_cpu_autograd_handles_var_and_nondifferentiable_zero_paths() -> TestResult {
    let device = nove_tensor::device::native::cpu()?;
    let input = Tensor::from_data(
        vec![vec![1.0f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
        &device,
        true,
    )?;
    input.var(1, true, true)?.sum(None)?.backward()?;
    let grad = input
        .grad()?
        .expect("variance gradient should be populated");
    assert_eq!(grad.to_vec::<f32>()?, vec![-1.0, 0.0, 1.0, -1.0, 0.0, 1.0]);

    let biased_input = Tensor::from_data(
        vec![vec![1.0f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
        &device,
        true,
    )?;
    biased_input.var(1, true, false)?.sum(None)?.backward()?;
    let biased_grad = biased_input
        .grad()?
        .expect("biased variance gradient should be populated");
    assert_eq!(
        biased_grad.to_vec::<f32>()?,
        vec![-0.5, 0.0, 0.5, -0.5, 0.0, 0.5]
    );

    let comparison_input =
        Tensor::from_slice(&[1.0f32, 2.0, 3.0], &Shape::from_dims(&[3]), &device, true)?;
    let threshold = Tensor::from_scalar(2.0f32, &device, false)?;
    comparison_input
        .gt(&threshold)?
        .to_dtype(&DType::F32)?
        .sum(None)?
        .backward()?;
    assert_eq!(
        comparison_input
            .grad()?
            .expect("comparison gradient should be populated")
            .to_vec::<f32>()?,
        vec![0.0, 0.0, 0.0]
    );

    let arg_input = Tensor::from_data(
        vec![vec![1.0f32, 5.0, 3.0], vec![4.0, 2.0, 6.0]],
        &device,
        true,
    )?;
    arg_input
        .argmax((1, false))?
        .to_dtype(&DType::F32)?
        .sum(None)?
        .backward()?;
    assert_eq!(
        arg_input
            .grad()?
            .expect("argmax gradient should be populated")
            .to_vec::<f32>()?,
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    );

    Ok(())
}

#[cfg(feature = "native-cpu")]
#[test]
fn nove_cpu_autograd_handles_index_scatter_add_paths() -> TestResult {
    let device = nove_tensor::device::native::cpu()?;

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
        selected
            .grad()?
            .expect("index_select gradient should be populated")
            .to_vec::<f32>()?,
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
        gathered
            .grad()?
            .expect("gather gradient should be populated")
            .to_vec::<f32>()?,
        vec![1.0, 0.0, 1.0, 0.0, 2.0, 0.0]
    );

    let table = Tensor::from_data(&[[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]], &device, true)?;
    let embedding_indexes =
        Tensor::from_slice(&[0i64, 2, 0], &Shape::from_dims(&[3]), &device, false)?;
    table.embedding(&embedding_indexes)?.sum(None)?.backward()?;
    assert_eq!(
        table
            .grad()?
            .expect("embedding gradient should be populated")
            .to_vec::<f32>()?,
        vec![2.0, 2.0, 0.0, 0.0, 1.0, 1.0]
    );

    Ok(())
}

#[cfg(feature = "native-cpu")]
#[test]
fn nove_cpu_autograd_handles_pooling_without_backend_fallback() -> TestResult {
    let device = nove_tensor::device::native::cpu()?;

    let max_1d = Tensor::from_data(vec![vec![vec![1.0f32, 4.0, 3.0, 2.0]]], &device, true)?;
    let max_1d_output = max_1d.max_pool1d(2, 2)?;
    assert_eq!(max_1d_output.to_vec::<f32>()?, vec![4.0, 3.0]);
    max_1d_output.sum(None)?.backward()?;
    assert_eq!(
        max_1d
            .grad()?
            .expect("max_pool1d gradient should be populated")
            .to_vec::<f32>()?,
        vec![0.0, 1.0, 1.0, 0.0]
    );

    let avg_1d = Tensor::from_data(vec![vec![vec![1.0f32, 2.0, 3.0, 4.0]]], &device, true)?;
    let avg_1d_output = avg_1d.avg_pool1d(2, 2)?;
    assert_eq!(avg_1d_output.to_vec::<f32>()?, vec![1.5, 3.5]);
    avg_1d_output.sum(None)?.backward()?;
    assert_eq!(
        avg_1d
            .grad()?
            .expect("avg_pool1d gradient should be populated")
            .to_vec::<f32>()?,
        vec![0.5, 0.5, 0.5, 0.5]
    );

    let max_2d = Tensor::from_data(
        vec![vec![vec![vec![1.0f32, 2.0], vec![3.0, 4.0]]]],
        &device,
        true,
    )?;
    let max_2d_output = max_2d.max_pool2d((2, 2), (2, 2))?;
    assert_eq!(max_2d_output.to_vec::<f32>()?, vec![4.0]);
    max_2d_output.backward()?;
    assert_eq!(
        max_2d
            .grad()?
            .expect("max_pool2d gradient should be populated")
            .to_vec::<f32>()?,
        vec![0.0, 0.0, 0.0, 1.0]
    );

    let avg_2d = Tensor::from_data(
        vec![vec![vec![vec![1.0f32, 2.0], vec![3.0, 4.0]]]],
        &device,
        true,
    )?;
    let avg_2d_output = avg_2d.avg_pool2d((2, 2), (2, 2))?;
    assert_eq!(avg_2d_output.to_vec::<f32>()?, vec![2.5]);
    avg_2d_output.backward()?;
    assert_eq!(
        avg_2d
            .grad()?
            .expect("avg_pool2d gradient should be populated")
            .to_vec::<f32>()?,
        vec![0.25, 0.25, 0.25, 0.25]
    );

    Ok(())
}

#[cfg(feature = "native-cpu")]
#[test]
fn nove_cpu_autograd_handles_convolution_without_backend_fallback() -> TestResult {
    let device = nove_tensor::device::native::cpu()?;

    let conv1d_input = Tensor::from_data(vec![vec![vec![1.0f32, 2.0, 3.0, 4.0]]], &device, true)?;
    let conv1d_kernel = Tensor::from_data(vec![vec![vec![1.0f32, 2.0]]], &device, true)?;
    let conv1d_output = conv1d_input.conv1d(&conv1d_kernel, 0, 1, 1, 1)?;
    assert_eq!(conv1d_output.to_vec::<f32>()?, vec![5.0, 8.0, 11.0]);
    conv1d_output.backward()?;
    assert_eq!(
        conv1d_input
            .grad()?
            .expect("conv1d input gradient should be populated")
            .to_vec::<f32>()?,
        vec![1.0, 3.0, 3.0, 2.0]
    );
    assert_eq!(
        conv1d_kernel
            .grad()?
            .expect("conv1d kernel gradient should be populated")
            .to_vec::<f32>()?,
        vec![6.0, 9.0]
    );

    let conv2d_input = Tensor::from_data(
        vec![vec![vec![vec![1.0f32, 2.0], vec![3.0, 4.0]]]],
        &device,
        true,
    )?;
    let conv2d_kernel = Tensor::from_data(
        vec![vec![vec![vec![1.0f32, 2.0], vec![3.0, 4.0]]]],
        &device,
        true,
    )?;
    let conv2d_output = conv2d_input.conv2d(&conv2d_kernel, 0, 1, 1, 1)?;
    assert_eq!(conv2d_output.to_vec::<f32>()?, vec![30.0]);
    conv2d_output.backward()?;
    assert_eq!(
        conv2d_input
            .grad()?
            .expect("conv2d input gradient should be populated")
            .to_vec::<f32>()?,
        vec![1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(
        conv2d_kernel
            .grad()?
            .expect("conv2d kernel gradient should be populated")
            .to_vec::<f32>()?,
        vec![1.0, 2.0, 3.0, 4.0]
    );

    Ok(())
}

#[cfg(feature = "native-cpu")]
#[test]
fn to_dtype_backward_casts_gradient_back_to_parent_dtype() -> TestResult {
    let device = nove_tensor::device::native::cpu()?;
    let input = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &Shape::from_dims(&[3]), &device, true)?;

    let loss = input.to_dtype(&DType::F64)?.sum(None)?;
    loss.backward()?;

    let grad = input.grad()?.expect("input gradient should be populated");
    assert_eq!(grad.dtype()?, DType::F32);
    assert_eq!(grad.to_vec::<f32>()?, vec![1.0, 1.0, 1.0]);

    Ok(())
}

#[cfg(feature = "native-cpu")]
fn gelu_tanh_derivative(value: f64) -> f64 {
    let coefficient = 0.797_884_560_802_865_4;
    let cubic_coefficient = 0.044_715;
    let inner = coefficient * (value + cubic_coefficient * value.powi(3));
    let tanh_inner = inner.tanh();
    0.5 * (1.0 + tanh_inner)
        + 0.5
            * value
            * (1.0 - tanh_inner.powi(2))
            * coefficient
            * (1.0 + 3.0 * cubic_coefficient * value.powi(2))
}

#[cfg(all(feature = "candle-cpu", feature = "native-cpu"))]
#[test]
fn to_device_backward_moves_gradient_back_to_parent_backend() -> TestResult {
    let candle_cpu = nove_tensor::device::candle::cpu()?;
    let nove_cpu = nove_tensor::device::native::cpu()?;
    let input = Tensor::from_slice(
        &[1.0f32, 2.0, 3.0],
        &Shape::from_dims(&[3]),
        &candle_cpu,
        true,
    )?;

    let loss = input.to_device(&nove_cpu)?.sum(None)?;
    loss.backward()?;

    let grad = input.grad()?.expect("input gradient should be populated");
    assert_eq!(grad.device()?, candle_cpu);
    assert_eq!(grad.to_vec::<f32>()?, vec![1.0, 1.0, 1.0]);

    Ok(())
}

#[cfg(feature = "candle-cpu")]
#[test]
fn gradient_controls_zero_clear_and_clear_graph() -> TestResult {
    let device = nove_tensor::device::candle::cpu()?;
    let mut input = Tensor::from_slice(&[2.0f32, 4.0], &Shape::from_dims(&[2]), &device, false)?
        .require_grad(true)?;
    let offset = Tensor::ones(&Shape::from_dims(&[2]), &DType::F32, &device, false)?;
    let loss = input.add(&offset)?.sum(None)?;

    loss.backward()?;
    let grad = input.grad()?.expect("input gradient should be populated");
    assert_eq!(grad.to_vec::<f32>()?, vec![1.0, 1.0]);

    input.zero_grad()?;
    let zero_grad = input.grad()?.expect("zeroed gradient should still exist");
    assert_eq!(zero_grad.to_vec::<f32>()?, vec![0.0, 0.0]);

    input.clear_grad()?;
    assert!(input.grad()?.is_none());

    loss.clear_graph()?;
    loss.clear_graph()?;

    Ok(())
}
