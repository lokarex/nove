use crate::{
    Device, Shape, Tensor, TensorError,
    backend::{BackendStorage, IntoTensorPayload, TensorElement},
    backpropagation::graph::OpKind,
};

impl Tensor {
    /// Create a new tensor from the given data.
    ///
    /// # Notes
    /// * The type of data supported by this function includes:
    ///   - Scalar: `S` (single value)
    ///   - Vec types: `Vec<S>`, `Vec<Vec<S>>`, `Vec<Vec<Vec<S>>>`, `Vec<Vec<Vec<Vec<S>>>>>`, `Vec<&[S]>`
    ///   - Slice/Array types: `&[S]`, `&[S; N]`, `&[[S; N]; M]`, `&[[[S; N3]; N2]; N1]`, `&[[[[S; N4]; N3]; N2]; N1]`
    /// * The element type `S` supported by this function includes `f32`, `f64`, `i64`, `u32`, `u8`.
    ///
    /// # Arguments
    /// * `data` - The data to create the tensor from.
    /// * `device` - The device to place the tensor on.
    /// * `requires_grad` - Whether to enable gradient tracking for the tensor.
    ///
    /// # Returns
    /// * `Ok(tensor)` - The created tensor if successful.
    /// * `Err(TensorError)` - The error when creating the tensor.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::tensor::Device;
    /// use nove::tensor::Tensor;
    /// let device = Device::default();
    ///
    /// // Vec<&[S]> - Vec of 1D slices
    /// let data: Vec<&[f32]> = vec![&[1.0f32, 2.0f32, 3.0f32], &[4.0f32, 5.0f32, 6.0f32]];
    /// let tensor = Tensor::from_data(data, &device, false).unwrap();
    /// println!("{:?}", tensor);
    ///
    /// // Vec<S> - Vec of scalars (flattened)
    /// let data: Vec<f32> = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let tensor = Tensor::from_data(data, &device, false).unwrap();
    /// println!("{:?}", tensor);
    ///
    /// // Vec<Vec<S>> - 2D Vec
    /// let data: Vec<Vec<f32>> = vec![vec![1.0f32, 2.0f32, 3.0f32], vec![4.0f32, 5.0f32, 6.0f32]];
    /// let tensor = Tensor::from_data(data, &device, false).unwrap();
    /// println!("{:?}", tensor);
    ///
    /// // Vec<Vec<Vec<S>>> - 3D Vec
    /// let data: Vec<Vec<Vec<f32>>> = vec![
    ///     vec![vec![1.0f32, 2.0f32], vec![3.0f32, 4.0f32]],
    ///     vec![vec![5.0f32, 6.0f32], vec![7.0f32, 8.0f32]],
    /// ];
    /// let tensor = Tensor::from_data(data, &device, false).unwrap();
    /// println!("{:?}", tensor);
    ///
    /// // Vec<Vec<Vec<Vec<S>>>> - 4D Vec
    /// let data: Vec<Vec<Vec<Vec<f32>>>> = vec![
    ///     vec![vec![vec![1.0f32, 2.0f32], vec![3.0f32, 4.0f32]]],
    ///     vec![vec![vec![5.0f32, 6.0f32], vec![7.0f32, 8.0f32]]],
    /// ];
    /// let tensor = Tensor::from_data(data, &device, false).unwrap();
    /// println!("{:?}", tensor);
    ///
    /// // &[S; N] - 1D array reference
    /// let data: &[f32; 6] = &[1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32];
    /// let tensor = Tensor::from_data(data, &device, false).unwrap();
    /// println!("{:?}", tensor);
    ///
    /// // &[S] - 1D slice reference
    /// let data: &[f32] = &[1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32];
    /// let tensor = Tensor::from_data(data, &device, false).unwrap();
    /// println!("{:?}", tensor);
    ///
    /// // &[[S; N]; M] - 2D array reference
    /// let data: &[[f32; 3]; 2] = &[[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]];
    /// let tensor = Tensor::from_data(data, &device, false).unwrap();
    /// println!("{:?}", tensor);
    ///
    /// // &[[[S; N3]; N2]; N1] - 3D array reference
    /// let data: &[[[f32; 2]; 2]; 2] = &[
    ///     [[1.0f32, 2.0f32], [3.0f32, 4.0f32]],
    ///     [[5.0f32, 6.0f32], [7.0f32, 8.0f32]],
    /// ];
    /// let tensor = Tensor::from_data(data, &device, false).unwrap();
    /// println!("{:?}", tensor);
    ///
    /// // &[[[[S; N4]; N3]; N2]; N1] - 4D array reference
    /// let data: &[[[[f32; 2]; 2]; 2]; 2] = &[
    ///     [[[1.0f32, 2.0f32], [3.0f32, 4.0f32]], [[5.0f32, 6.0f32], [7.0f32, 8.0f32]]],
    ///     [[[9.0f32, 10.0f32], [11.0f32, 12.0f32]], [[13.0f32, 14.0f32], [15.0f32, 16.0f32]]],
    /// ];
    /// let tensor = Tensor::from_data(data, &device, false).unwrap();
    /// println!("{:?}", tensor);
    /// ```
    pub fn from_data<A>(data: A, device: &Device, requires_grad: bool) -> Result<Self, TensorError>
    where
        A: IntoTensorPayload,
    {
        let storage = BackendStorage::from_data(data, device, requires_grad)?;
        Ok(Self::from_backend_storage(
            storage,
            device.clone(),
            requires_grad,
            vec![],
            OpKind::Leaf,
        ))
    }

    /// Create a new tensor from a vector with the specified shape.
    ///
    /// # Notes
    /// * The element type of the data supported by this function includes `f32`, `f64`, `i64`, `u32`, `u8`.
    /// * The total number of elements in the vector must match the product of the shape dimensions.
    ///
    /// # Arguments
    /// * `data` - The vector of data to create the tensor from.
    /// * `shape` - The shape of the tensor.
    /// * `device` - The device to place the tensor on.
    /// * `requires_grad` - Whether to enable gradient tracking for the tensor.
    ///
    /// # Returns
    /// * `Ok(tensor)` - The created tensor if successful.
    /// * `Err(TensorError)` - The error when creating the tensor.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::default();
    ///
    /// let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let shape = Shape::from(&[2, 3]);
    /// let tensor = Tensor::from_vec(data, &shape, &device, false).unwrap();
    /// println!("{}", tensor);
    /// ```
    pub fn from_vec<D>(
        data: Vec<D>,
        shape: &Shape,
        device: &Device,
        requires_grad: bool,
    ) -> Result<Self, TensorError>
    where
        D: TensorElement,
    {
        let storage = BackendStorage::from_vec(data, shape, device, requires_grad)?;
        Ok(Self::from_backend_storage(
            storage,
            device.clone(),
            requires_grad,
            vec![],
            OpKind::Leaf,
        ))
    }

    /// Create a new tensor from a slice with the specified shape.
    ///
    /// # Notes
    /// * The element type of the data supported by this function includes `f32`, `f64`, `i64`, `u32`, `u8`.
    /// * The total number of elements in the slice must match the product of the shape dimensions.
    ///
    /// # Arguments
    /// * `data` - The slice of data to create the tensor from.
    /// * `shape` - The shape of the tensor.
    /// * `device` - The device to place the tensor on.
    /// * `requires_grad` - Whether to enable gradient tracking for the tensor.
    ///
    /// # Returns
    /// * `Ok(tensor)` - The created tensor if successful.
    /// * `Err(TensorError)` - The error when creating the tensor.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::default();
    ///
    /// let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let shape = Shape::from(&[2, 3]);
    /// let tensor = Tensor::from_slice(&data, &shape, &device, false).unwrap();
    /// println!("{:?}", tensor);
    /// ```
    pub fn from_slice<D>(
        data: &[D],
        shape: &Shape,
        device: &Device,
        requires_grad: bool,
    ) -> Result<Self, TensorError>
    where
        D: TensorElement,
    {
        let storage = BackendStorage::from_slice(data, shape, device, requires_grad)?;
        Ok(Self::from_backend_storage(
            storage,
            device.clone(),
            requires_grad,
            vec![],
            OpKind::Leaf,
        ))
    }

    /// Create a new tensor from a scalar.
    ///
    /// # Notes
    /// * This function is an alias of `from_data` but accepts a scalar value with more explicit type.
    /// * The type of the scalar supported by this function includes `f32`, `f64`, `i64`, `u32`, `u8`.
    ///
    /// # Arguments
    /// * `scalar` - The scalar to create the tensor from. It should be a single value not a vector or array.
    /// * `device` - The device to place the tensor on.
    /// * `requires_grad` - Whether to enable gradient tracking for the tensor.
    ///
    /// # Returns
    /// * `Ok(tensor)` - The created tensor if successful.
    /// * `Err(TensorError)` - The error when creating the tensor.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::tensor::Device;
    /// use nove::tensor::Tensor;
    /// let device = Device::default();
    ///
    /// let tensor = Tensor::from_scalar(1.0f32, &device, false).unwrap();
    /// println!("{:?}", tensor);
    /// ```
    pub fn from_scalar<S>(
        scalar: S,
        device: &Device,
        requires_grad: bool,
    ) -> Result<Self, TensorError>
    where
        S: IntoTensorPayload + TensorElement,
    {
        Self::from_data(scalar, device, requires_grad)
    }

    /// Convert the tensor to a scalar.
    ///
    /// # Generic Type Parameters
    /// * `S` - The element type of the scalar. It supports `f32`, `f64`, `i64`, `u32`, `u8`.
    ///
    /// # Notes
    /// * The tensor must only have one element.
    ///
    /// # Returns
    /// * `Ok(scalar)` - The scalar value if the tensor is a scalar.
    /// * `Err(TensorError)` - The error when converting the tensor to a scalar.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::tensor::Device;
    /// use nove::tensor::Tensor;
    /// let device = Device::default();
    ///
    /// let tensor = Tensor::from_scalar(1.0f32, &device, false).unwrap();
    /// println!("{}", tensor.to_scalar::<f32>().unwrap());
    /// ```
    pub fn to_scalar<S>(&self) -> Result<S, TensorError>
    where
        S: TensorElement,
    {
        Ok(self.backend_storage()?.to_scalar::<S>()?)
    }

    /// Convert the tensor to a one-dimensional vector.
    ///
    /// # Generic Type Parameters
    /// * `S` - The element type of the vector. It supports `f32`, `f64`, `i64`, `u32`, `u8`.
    ///
    /// # Notes
    /// * The tensor could be any shape, and it will be flattened to a one-dimensional vector.
    ///
    /// # Returns
    /// * `Ok(vec)` - The vector value if the tensor can be converted to a vector.
    /// * `Err(TensorError)` - The error when converting the tensor to a vector.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::tensor::Device;
    /// use nove::tensor::Tensor;
    /// let device = Device::default();
    ///
    /// let tensor = Tensor::from_data(&[[1.0f64, 2.0f64, 3.0f64], [4.0f64, 5.0f64, 6.0f64]], &device, false).unwrap();
    /// println!("{:?}", tensor.to_vec::<f64>().unwrap());
    /// ```
    ///
    pub fn to_vec<S>(&self) -> Result<Vec<S>, TensorError>
    where
        S: TensorElement,
    {
        Ok(self.backend_storage()?.to_vec::<S>()?)
    }
}
