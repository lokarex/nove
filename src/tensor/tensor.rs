use crate::tensor::{Device, Shape};

/// Fix the backend of BurnTensor to Wgpu.
type BurnTensor<const D: usize, K = burn::tensor::Float> =
    burn::tensor::Tensor<burn::backend::Autodiff<burn::backend::Wgpu>, D, K>;

/// The `Tensor` struct represents a tensor with a specified dimension and numeric type.
///
/// # Generic Type Parameters
/// * `D` - The dimension of the tensor.
/// * `K` - The numeric type of the tensor(F32, I32 or Bool).
///
/// # Fields
/// * `inner` - The inner tensor(`BurnTensor<D, K>`) of the `Tensor` struct.
#[derive(Debug)]
pub struct Tensor<
    const D: usize,
    K: burn::tensor::Numeric<burn::backend::Autodiff<burn::backend::Wgpu>>,
> {
    inner: BurnTensor<D, K>,
}

impl<const D: usize, K: burn::tensor::Numeric<burn::backend::Autodiff<burn::backend::Wgpu>>>
    Tensor<D, K>
{
    /// Create a new tensor filled with zeros.
    ///
    /// # Arguments
    /// * `shape` - The shape of the tensor.
    /// * `device` - The device to create the tensor on.
    ///
    /// # Returns
    /// A new `Tensor` instance filled with zeros.
    pub fn zeros(shape: Shape, device: &Device) -> Self {
        let tensor = BurnTensor::<D, K>::zeros(shape, device);
        Self { inner: tensor }
    }

    /// Add two tensors element-wise.
    ///
    /// # Arguments
    /// * `other` - The tensor to add.
    ///
    /// # Returns
    /// A new `Tensor` instance with the element-wise sum of the two tensors.
    pub fn add(&self, other: &Self) -> Self {
        let tensor = self.inner.clone().add(other.inner.clone());
        Self { inner: tensor }
    }

    /// Create a new tensor from data.
    ///
    /// # Arguments
    /// * `data` - The data to create the tensor from.
    /// * `device` - The device to create the tensor on.
    ///
    /// # Returns
    /// A new `Tensor` instance with the data.
    pub fn from_data<T>(data: T, device: &Device) -> Self
    where
        T: Into<burn::tensor::TensorData>,
    {
        let tensor = BurnTensor::<D, K>::from_data(data, device);
        Self { inner: tensor }
    }

    /// Get the shape of the tensor.
    ///
    /// # Returns
    /// The shape of the tensor.
    pub fn shape(&self) -> Shape {
        self.inner.shape().clone()
    }

    /// Get the bytes of the tensor.
    ///
    /// # Returns
    /// The bytes of the tensor.
    pub fn get_bytes(&self) -> Vec<u8> {
        self.inner.to_data().bytes.to_vec()
    }
}

impl<const D: usize, K: burn::tensor::Numeric<burn::backend::Autodiff<burn::backend::Wgpu>>>
    Tensor<D, K>
{
    /// Stack tensors along a new dimension.
    ///
    /// # Arguments
    /// * `tensors` - The tensors to stack.
    /// * `dim` - The dimension to stack along.
    ///
    /// # Returns
    /// A new `Tensor` instance with the stacked tensors.
    pub fn stack<const D2: usize>(tensors: Vec<Tensor<D2, K>>, dim: usize) -> Tensor<D, K> {
        let tensors = tensors
            .into_iter()
            .map(|t| t.inner.clone())
            .collect::<Vec<_>>();
        let tensor = BurnTensor::stack::<D>(tensors, dim);
        Self { inner: tensor }
    }
}
