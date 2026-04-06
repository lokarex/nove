use nove::model::Model;
use nove::model::nn::EmbeddingBuilder;
use nove::tensor::{DType, Device, Shape, Tensor};

#[test]
fn test_embedding_builder_creation() {
    let embedding = EmbeddingBuilder::new(100, 50).build().unwrap();

    assert_eq!(embedding.num_embeddings(), 100);
    assert_eq!(embedding.embedding_dim(), 50);
    assert_eq!(embedding.padding_idx(), None);

    let weight_shape = embedding.weight().shape().unwrap();
    assert_eq!(weight_shape, Shape::from_dims(&[100, 50]));
}

#[test]
fn test_embedding_builder_with_padding_idx() {
    let embedding = EmbeddingBuilder::new(100, 50)
        .padding_idx(Some(0))
        .build()
        .unwrap();

    assert_eq!(embedding.num_embeddings(), 100);
    assert_eq!(embedding.embedding_dim(), 50);
    assert_eq!(embedding.padding_idx(), Some(0));

    let weight_shape = embedding.weight().shape().unwrap();
    assert_eq!(weight_shape, Shape::from_dims(&[100, 50]));
}

#[test]
fn test_embedding_builder_configuration() {
    let embedding = EmbeddingBuilder::new(100, 50)
        .padding_idx(Some(0))
        .device(Device::cpu())
        .dtype(DType::F32)
        .grad_enabled(true)
        .build()
        .unwrap();

    assert_eq!(embedding.num_embeddings(), 100);
    assert_eq!(embedding.embedding_dim(), 50);
    assert_eq!(embedding.padding_idx(), Some(0));

    let weight = embedding.weight();
    assert_eq!(weight.dtype().unwrap(), DType::F32);
    assert!(weight.grad_enabled().unwrap());
}

#[test]
fn test_embedding_builder_method_chaining() {
    let mut builder = EmbeddingBuilder::new(100, 50);
    builder
        .num_embeddings(200)
        .embedding_dim(100)
        .padding_idx(Some(0))
        .device(Device::cpu())
        .dtype(DType::F32)
        .grad_enabled(true);

    let embedding = builder.build().unwrap();

    assert_eq!(embedding.num_embeddings(), 200);
    assert_eq!(embedding.embedding_dim(), 100);
    assert_eq!(embedding.padding_idx(), Some(0));
}

#[test]
fn test_embedding_forward_1d() {
    let mut embedding = EmbeddingBuilder::new(100, 50).build().unwrap();

    let input = Tensor::from_data(vec![1i64, 2, 3, 4], &Device::cpu(), false).unwrap();
    let output = embedding.forward(input).unwrap();

    assert_eq!(output.shape().unwrap(), Shape::from_dims(&[4, 50]));
}

#[test]
fn test_embedding_forward_2d() {
    let mut embedding = EmbeddingBuilder::new(100, 50).build().unwrap();

    let input = Tensor::from_data(vec![1i64, 2, 3, 4], &Device::cpu(), false)
        .unwrap()
        .reshape(&Shape::from_dims(&[2, 2]))
        .unwrap();
    let output = embedding.forward(input).unwrap();

    assert_eq!(output.shape().unwrap(), Shape::from_dims(&[2, 2, 50]));
}

#[test]
fn test_embedding_forward_3d() {
    let mut embedding = EmbeddingBuilder::new(100, 50).build().unwrap();

    let input = Tensor::from_data(vec![1i64, 2, 3, 4, 5, 6, 7, 8], &Device::cpu(), false)
        .unwrap()
        .reshape(&Shape::from_dims(&[2, 2, 2]))
        .unwrap();
    let output = embedding.forward(input).unwrap();

    assert_eq!(output.shape().unwrap(), Shape::from_dims(&[2, 2, 2, 50]));
}

#[test]
fn test_embedding_forward_with_padding_idx() {
    let mut embedding = EmbeddingBuilder::new(100, 50)
        .padding_idx(Some(0))
        .build()
        .unwrap();

    let input = Tensor::from_data(vec![0i64, 1, 2, 0], &Device::cpu(), false).unwrap();
    let output = embedding.forward(input).unwrap();

    assert_eq!(output.shape().unwrap(), Shape::from_dims(&[4, 50]));

    let output_vec = output.to_vec::<f32>().unwrap();
    for i in 0..4 {
        let start_idx = i * 50;
        let end_idx = start_idx + 50;
        let slice = &output_vec[start_idx..end_idx];
        if i == 0 || i == 3 {
            assert!(
                slice.iter().all(|&x| x == 0.0),
                "Padding indices should produce zero vectors"
            );
        } else {
            assert!(
                slice.iter().any(|&x| x != 0.0),
                "Non-padding indices should produce non-zero vectors"
            );
        }
    }
}

#[test]
fn test_embedding_parameters() {
    let embedding = EmbeddingBuilder::new(100, 50).build().unwrap();

    let params = embedding.parameters().unwrap();
    assert_eq!(params.len(), 1);
    assert_eq!(params[0].shape().unwrap(), Shape::from_dims(&[100, 50]));
}

#[test]
fn test_embedding_named_parameters() {
    let embedding = EmbeddingBuilder::new(100, 50).build().unwrap();

    let named_params = embedding.named_parameters().unwrap();
    assert_eq!(named_params.len(), 1);

    let param_names: Vec<String> = named_params.keys().cloned().collect();
    assert!(param_names[0].contains("weight"));
}

#[test]
fn test_embedding_require_grad() {
    let mut embedding = EmbeddingBuilder::new(100, 50)
        .grad_enabled(false)
        .build()
        .unwrap();

    let weight = embedding.weight();
    assert!(!weight.grad_enabled().unwrap());

    embedding.require_grad(true).unwrap();

    let weight = embedding.weight();
    assert!(weight.grad_enabled().unwrap());
}

#[test]
fn test_embedding_to_dtype() {
    let mut embedding = EmbeddingBuilder::new(100, 50).build().unwrap();

    let weight_dtype = embedding.weight().dtype().unwrap();
    assert_eq!(weight_dtype, DType::F32);

    embedding.to_dtype(&DType::F64).unwrap();

    let weight_dtype = embedding.weight().dtype().unwrap();
    assert_eq!(weight_dtype, DType::F64);
}

#[test]
fn test_embedding_to_device() {
    let mut embedding = EmbeddingBuilder::new(100, 50).build().unwrap();

    let device = embedding.weight().device().unwrap();
    assert_eq!(device, Device::cpu());

    let target_device = Device::cpu();
    embedding.to_device(&target_device).unwrap();

    let new_device = embedding.weight().device().unwrap();
    assert_eq!(new_device, target_device);
}

#[test]
fn test_embedding_error_wrong_dtype() {
    let mut embedding = EmbeddingBuilder::new(100, 50).build().unwrap();

    let input = Tensor::from_data(vec![1.0f32, 2.0, 3.0], &Device::cpu(), false).unwrap();
    let result = embedding.forward(input);

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("dtype I64"));
}

#[test]
fn test_embedding_error_out_of_range_indices() {
    let mut embedding = EmbeddingBuilder::new(10, 50).build().unwrap();

    let input = Tensor::from_data(vec![5i64, 10, 15], &Device::cpu(), false).unwrap();
    let result = embedding.forward(input);

    assert!(result.is_err());
}

#[test]
fn test_embedding_error_invalid_padding_idx() {
    let result = EmbeddingBuilder::new(10, 50).padding_idx(Some(10)).build();

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("padding_idx"));
}

#[test]
fn test_embedding_display() {
    let embedding = EmbeddingBuilder::new(100, 50)
        .padding_idx(Some(0))
        .build()
        .unwrap();

    let display_str = format!("{}", embedding);
    assert!(display_str.contains("embedding"));
    assert!(display_str.contains("num_embeddings=100"));
    assert!(display_str.contains("embedding_dim=50"));
    assert!(display_str.contains("padding_idx=Some(0)"));
}

#[test]
fn test_embedding_edge_case_single_index() {
    let mut embedding = EmbeddingBuilder::new(100, 50).build().unwrap();

    let input = Tensor::from_scalar(5i64, &Device::cpu(), false).unwrap();
    let output = embedding.forward(input).unwrap();

    assert_eq!(output.shape().unwrap(), Shape::from_dims(&[50]));
}

#[test]
fn test_embedding_edge_case_empty_batch() {
    let mut embedding = EmbeddingBuilder::new(100, 50).build().unwrap();

    let input = Tensor::from_data(Vec::<i64>::new(), &Device::cpu(), false)
        .unwrap()
        .reshape(&Shape::from_dims(&[0]))
        .unwrap();
    let output = embedding.forward(input).unwrap();

    assert_eq!(output.shape().unwrap(), Shape::from_dims(&[0, 50]));
}

#[test]
fn test_embedding_gradient_flow_without_padding() {
    let mut embedding = EmbeddingBuilder::new(10, 5)
        .grad_enabled(true)
        .build()
        .unwrap();

    let input = Tensor::from_data(vec![1i64, 2, 3], &Device::cpu(), false).unwrap();
    let output = embedding.forward(input).unwrap();
    eprintln!(
        "weight grad before backward: {:?}",
        embedding.weight().grad()
    );

    let loss = output.sum(None).unwrap();
    eprintln!("loss grad before backward: {:?}", loss.grad());
    loss.backward().unwrap();

    let weight_grad = embedding.weight().grad().unwrap();
    assert!(weight_grad.is_some());
}

#[test]
fn test_embedding_gradient_flow_with_padding() {
    let mut embedding = EmbeddingBuilder::new(10, 5)
        .padding_idx(Some(0))
        .grad_enabled(true)
        .build()
        .unwrap();

    let input = Tensor::from_data(vec![0i64, 1, 2, 0], &Device::cpu(), false).unwrap();
    let output = embedding.forward(input).unwrap();
    eprintln!(
        "weight grad before backward: {:?}",
        embedding.weight().grad()
    );

    let loss = output.sum(None).unwrap();
    eprintln!("loss grad before backward: {:?}", loss.grad());
    loss.backward().unwrap();

    let weight_grad = embedding.weight().grad().unwrap();
    assert!(weight_grad.is_some());
}
