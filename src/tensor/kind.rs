/// The 32-bit floating point number type for tensor.
/// Note: The corresponding type of tensor element in rust is `f32`.
pub type F32 = burn::tensor::Float;
/// The 32-bit integer number type for tensor.
/// Note: The corresponding type of tensor element in rust is `i32`.
pub type I32 = burn::tensor::Int;
/// The boolean type for tensor.
/// Warning: The corresponding type of tensor element in rust is `u32`, not `bool`.
pub type Bool = burn::tensor::Bool;
