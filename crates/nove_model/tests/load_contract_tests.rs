#[cfg(feature = "candle-cpu")]
mod candle_tests {
    use nove_model::{
        Model, ModelError,
        nn::{Linear, LinearBuilder},
    };
    #[cfg(feature = "native-cpu")]
    use nove_tensor::BackendError;
    use nove_tensor::{DType, Device, Shape, Tensor, TensorError, safetensor};
    use std::collections::HashMap;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn linear(device: &Device) -> Result<Linear, ModelError> {
        LinearBuilder::new(2, 1)
            .bias_enabled(false)
            .device(device.clone())
            .dtype(DType::F32)
            .grad_enabled(true)
            .build()
    }

    fn checkpoint(
        model: &Linear,
        tensor: Tensor,
    ) -> Result<(tempfile::TempDir, String), Box<dyn std::error::Error>> {
        let root =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/task3-tests");
        std::fs::create_dir_all(&root)?;
        let dir = tempfile::tempdir_in(root)?;
        let path = dir
            .path()
            .join("parameters.safetensors")
            .to_str()
            .unwrap()
            .to_owned();
        let name = model.named_parameters()?.into_keys().next().unwrap();
        safetensor::save(&path, HashMap::from([(name, tensor)]))?;
        Ok((dir, path))
    }

    #[test]
    fn load_updates_matching_parameters_without_changing_identity() -> TestResult {
        let device = nove_tensor::device::candle::cpu()?;
        let mut model = linear(&device)?;
        let alias = model.weight();
        let id = alias.graph_id()?;
        let name = alias.name()?;
        let source =
            Tensor::from_slice(&[5.0f32, 6.0], &Shape::from_dims(&[2, 1]), &device, false)?;
        let (_dir, path) = checkpoint(&model, source)?;
        model.load(&path, &device)?;
        assert_eq!(alias.to_vec::<f32>()?, [5.0, 6.0]);
        assert_eq!(alias.graph_id()?, id);
        assert_eq!(alias.name()?, name);
        assert_eq!(alias.shape()?.dims(), &[2, 1]);
        assert_eq!(alias.dtype()?, DType::F32);
        assert_eq!(alias.device()?, device);
        assert!(alias.grad_enabled()?);
        assert_eq!(model.weight().graph_id()?, id);
        Ok(())
    }

    #[test]
    fn load_rejects_incompatible_parameter_metadata() -> TestResult {
        let device = nove_tensor::device::candle::cpu()?;
        let mut model = linear(&device)?;
        let alias = model.weight();
        let before = alias.to_vec::<f32>()?;
        let id = alias.graph_id()?;
        let shape = Shape::from_dims(&[2, 1]);
        let sources = [
            Tensor::from_slice(
                &[7.0f32, 8.0, 9.0],
                &Shape::from_dims(&[3, 1]),
                &device,
                false,
            )?,
            Tensor::from_slice(&[7.0f64, 8.0], &shape, &device, false)?,
        ];
        for (index, source) in sources.into_iter().enumerate() {
            let (_dir, path) = checkpoint(&model, source)?;
            match model.load(&path, &device).unwrap_err() {
                ModelError::TensorError(TensorError::ShapeMismatch(expected, found))
                    if index == 0 =>
                {
                    assert_eq!(expected, shape);
                    assert_eq!(found.dims(), &[3, 1]);
                }
                ModelError::TensorError(TensorError::DTypeMismatch(expected, found))
                    if index == 1 =>
                {
                    assert_eq!(expected, DType::F32);
                    assert_eq!(found, DType::F64);
                }
                error => panic!("unexpected loading error: {error:?}"),
            }
            assert_eq!(alias.to_vec::<f32>()?, before);
            assert_eq!(alias.graph_id()?, id);
            assert_eq!(alias.shape()?, shape);
            assert_eq!(alias.dtype()?, DType::F32);
            assert_eq!(alias.device()?, device);
            assert!(alias.grad_enabled()?);
        }
        Ok(())
    }

    #[cfg(feature = "native-cpu")]
    #[test]
    fn load_does_not_implicitly_move_native_parameters_to_candle() -> TestResult {
        let native = nove_tensor::device::native::cpu()?;
        let candle = nove_tensor::device::candle::cpu()?;
        let mut model = linear(&native)?;
        let alias = model.weight();
        let before = alias.to_vec::<f32>()?;
        let id = alias.graph_id()?;
        let source =
            Tensor::from_slice(&[5.0f32, 6.0], &Shape::from_dims(&[2, 1]), &candle, false)?;
        let (_dir, path) = checkpoint(&model, source)?;
        assert!(matches!(model.load(&path, &candle),
            Err(ModelError::TensorError(TensorError::BackendError(BackendError::BackendMismatch { expected, found })))
            if expected == native.backend() && found == candle.backend()));
        assert_eq!(alias.to_vec::<f32>()?, before);
        assert_eq!(alias.device()?, native);
        assert_eq!(alias.graph_id()?, id);
        Ok(())
    }
}

#[cfg(feature = "native-cpu")]
#[test]
fn native_safetensors_loading_remains_unsupported() -> Result<(), Box<dyn std::error::Error>> {
    use nove_model::{Model, ModelError, nn::LinearBuilder};
    use nove_tensor::{BackendError, TensorError};
    let device = nove_tensor::device::native::cpu()?;
    let mut model = LinearBuilder::new(2, 1)
        .bias_enabled(false)
        .device(device.clone())
        .build()?;
    assert!(matches!(model.load("unavailable.safetensors", &device),
        Err(ModelError::TensorError(TensorError::BackendError(BackendError::UnsupportedOperation { backend, .. })))
        if backend == device.backend()));
    Ok(())
}
