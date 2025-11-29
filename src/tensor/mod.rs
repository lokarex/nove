pub mod kind;
pub type Tensor<const D: usize, K = burn::tensor::Float> =
    burn::tensor::Tensor<burn::backend::Autodiff<burn::backend::Wgpu>, D, K>;
pub type Shape = burn::tensor::Shape;
