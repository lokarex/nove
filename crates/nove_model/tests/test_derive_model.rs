use nove_macro::Model;
use nove_model::{Model as ModelTrait, layer::Linear, layer::LinearBuilder};
use nove_tensor::{DType, Device, Shape, Tensor};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Model)]
#[model(input = Tensor, output = Tensor)]
struct TestModel {
    linear1: Linear,
    linear2: Linear,
    #[model(ignore)]
    timestamp: u64,
}

impl TestModel {
    fn forward(&mut self, input: Tensor) -> Result<Tensor, nove_model::ModelError> {
        let x = self.linear1.forward(input)?;
        let x = self.linear2.forward(x)?;
        Ok(x)
    }
}

#[test]
fn test_derive_model() {
    // Initialize the model
    let mut model = TestModel {
        linear1: LinearBuilder::default()
            .in_features(10)
            .out_features(20)
            .build()
            .unwrap(),
        linear2: LinearBuilder::default()
            .in_features(20)
            .out_features(100)
            .build()
            .unwrap(),
        timestamp: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    };
    println!("{}\n{}", model.timestamp, model);

    let device = Device::cpu();
    model.to_device(&device).unwrap();

    model.require_grad(false).unwrap();

    let params = model.parameters().unwrap();
    assert_eq!(params.len(), 4);

    let named_params = model.named_parameters().unwrap();
    assert_eq!(named_params.len(), 4);
    for (key, _) in &named_params {
        println!("Parameter name: {}", key);
    }
    assert!(named_params.contains_key("linear1.linear.0.weight"));
    assert!(named_params.contains_key("linear1.linear.0.bias"));
    assert!(named_params.contains_key("linear2.linear.1.weight"));
    assert!(named_params.contains_key("linear2.linear.1.bias"));

    assert_eq!(model.linear1.weight().dtype().unwrap(), DType::F32);
    assert_eq!(model.linear1.bias().unwrap().dtype().unwrap(), DType::F32);
    assert_eq!(model.linear2.weight().dtype().unwrap(), DType::F32);
    assert_eq!(model.linear2.bias().unwrap().dtype().unwrap(), DType::F32);
    model.to_dtype(&DType::F64).unwrap();
    assert_eq!(model.linear1.bias().unwrap().dtype().unwrap(), DType::F64);
    assert_eq!(model.linear1.weight().dtype().unwrap(), DType::F64);
    assert_eq!(model.linear2.weight().dtype().unwrap(), DType::F64);
    assert_eq!(model.linear2.bias().unwrap().dtype().unwrap(), DType::F64);

    let input = Tensor::randn(0.0, 1.0, &Shape::from(&[1, 10]), &device, false).unwrap();
    let output = model.forward(input.clone()).unwrap();
    assert_eq!(output.shape().unwrap(), Shape::from(&[1, 100]));

    let temp_dir = tempfile::tempdir().unwrap();
    let file_path = temp_dir.path().join("model.safetensors");
    model.save(file_path.to_str().unwrap()).unwrap();

    let mut loaded_model = TestModel {
        linear1: LinearBuilder::default()
            .in_features(10)
            .out_features(20)
            .build()
            .unwrap(),
        linear2: LinearBuilder::default()
            .in_features(20)
            .out_features(100)
            .build()
            .unwrap(),
        timestamp: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    };
    loaded_model.to_dtype(&DType::F64).unwrap();
    loaded_model
        .load(file_path.to_str().unwrap(), &device)
        .unwrap();
    println!("{}\n{}", loaded_model.timestamp, loaded_model);

    // The output of the loaded model should be the same as the original model
    let loaded_output = loaded_model.forward(input.clone()).unwrap();
    assert_eq!(
        loaded_output.to_vec::<f64>().unwrap(),
        output.to_vec::<f64>().unwrap()
    );
}
