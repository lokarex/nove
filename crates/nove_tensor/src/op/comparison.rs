use crate::{Tensor, TensorError, backpropagation::graph::OpKind};

impl Tensor {
    /// Element-wise equal (==) comparison with broadcasting, returning a boolean tensor.
    ///
    /// # Notes
    /// * The [`crate::DType`] of the result tensor will be [`crate::DType::U8`] representing boolean values.
    ///
    /// # Arguments
    /// * `rhs` - The tensor to compare with.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor with [`crate::DType::U8`] DType.
    /// * `Err(TensorError)` - The error when comparing the tensors.
    ///
    /// # Examples
    /// * Compare two 2x3 matrices without broadcasting
    /// ```
    /// use nove::tensor::{Device, Tensor, Shape};
    ///
    /// let device = Device::default();
    ///
    /// let t1 = Tensor::from_data(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device, false).unwrap();
    /// let t2 = Tensor::from_data(vec![vec![1.0, 2.0, 0.0], vec![4.0, 5.0, 0.0]], &device, false).unwrap();
    /// let result = t1.eq(&t2).unwrap();
    /// let expected_values = vec![1, 1, 0, 1, 1, 0];
    /// let expected_shape: Shape = (&[2, 3]).into();
    /// assert_eq!(result.to_vec::<u8>().unwrap(), expected_values);
    /// assert_eq!(result.shape().unwrap(), expected_shape);
    /// ```
    /// * Compare a 1x3 vector with a 2x3 matrix with broadcasting
    /// ```
    /// use nove::tensor::{Device, Tensor, Shape};
    ///
    /// let device = Device::default();
    ///
    /// let t1 = Tensor::from_data(vec![1.0, 2.0, 3.0], &device, false).unwrap();
    /// let t2 = Tensor::from_data(vec![vec![1.0, 2.0, 0.0], vec![0.0, 0.0, 0.0]], &device, false).unwrap();
    /// let result = t1.eq(&t2).unwrap();
    /// let expected_values = vec![1, 1, 0, 0, 0, 0];
    /// let expected_shape: Shape = (&[2, 3]).into();
    /// assert_eq!(result.to_vec::<u8>().unwrap(), expected_values);
    /// assert_eq!(result.shape().unwrap(), expected_shape);
    /// ```
    pub fn eq(&self, rhs: &Self) -> Result<Self, TensorError> {
        comparison_op(self, rhs, "eq", |lhs, rhs| lhs.broadcast_eq(rhs))
    }

    /// Element-wise not-equal (!=) comparison with broadcasting, returning a boolean tensor.
    ///
    /// # Notes
    /// * The [`crate::DType`] of the result tensor will be [`crate::DType::U8`] representing boolean values.
    ///
    /// # Arguments
    /// * `rhs` - The tensor to compare with.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor with [`crate::DType::U8`] DType.
    /// * `Err(TensorError)` - The error when comparing the tensors.
    ///
    /// # Examples
    /// * Compare two 2x3 matrices without broadcasting
    /// ```
    /// use nove::tensor::{Device, Tensor, Shape};
    ///
    /// let device = Device::default();
    ///
    /// let t1 = Tensor::from_data(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device, false).unwrap();
    /// let t2 = Tensor::from_data(vec![vec![1.0, 2.0, 0.0], vec![4.0, 5.0, 0.0]], &device, false).unwrap();
    /// let result = t1.ne(&t2).unwrap();
    /// let expected_values = vec![0, 0, 1, 0, 0, 1];
    /// let expected_shape: Shape = (&[2, 3]).into();
    /// assert_eq!(result.to_vec::<u8>().unwrap(), expected_values);
    /// assert_eq!(result.shape().unwrap(), expected_shape);
    /// ```
    /// * Compare a 1x3 vector with a 2x3 matrix with broadcasting
    /// ```
    /// use nove::tensor::{Device, Tensor, Shape};
    ///
    /// let device = Device::default();
    ///
    /// let t1 = Tensor::from_data(vec![1.0, 2.0, 3.0], &device, false).unwrap();
    /// let t2 = Tensor::from_data(vec![vec![1.0, 2.0, 0.0], vec![0.0, 0.0, 0.0]], &device, false).unwrap();
    /// let result = t1.ne(&t2).unwrap();
    /// let expected_values = vec![0, 0, 1, 1, 1, 1];
    /// let expected_shape: Shape = (&[2, 3]).into();
    /// assert_eq!(result.to_vec::<u8>().unwrap(), expected_values);
    /// assert_eq!(result.shape().unwrap(), expected_shape);
    /// ```
    pub fn ne(&self, rhs: &Self) -> Result<Self, TensorError> {
        comparison_op(self, rhs, "ne", |lhs, rhs| lhs.broadcast_ne(rhs))
    }

    /// Element-wise greater-than (>) comparison with broadcasting, returning a boolean tensor.
    ///
    /// # Notes
    /// * The [`crate::DType`] of the result tensor will be [`crate::DType::U8`] representing boolean values.
    ///
    /// # Arguments
    /// * `rhs` - The tensor to compare with.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor with [`crate::DType::U8`] DType.
    /// * `Err(TensorError)` - The error when comparing the tensors.
    ///
    /// # Examples
    /// * Compare two 2x3 matrices without broadcasting
    /// ```
    /// use nove::tensor::{Device, Tensor, Shape};
    ///
    /// let device = Device::default();
    ///
    /// let t1 = Tensor::from_data(vec![vec![1.5, -2.0, 3.0], vec![0.0, 5.0, -1.0]], &device, false).unwrap();
    /// let t2 = Tensor::from_data(vec![vec![1.0, -1.0, 2.5], vec![-0.5, 4.0, -2.0]], &device, false).unwrap();
    /// let result = t1.gt(&t2).unwrap();
    /// let expected_values = vec![1, 0, 1, 1, 1, 1];
    /// let expected_shape: Shape = (&[2, 3]).into();
    /// assert_eq!(result.to_vec::<u8>().unwrap(), expected_values);
    /// assert_eq!(result.shape().unwrap(), expected_shape);
    /// ```
    /// * Compare a 2x1 column vector with a 2x3 matrix with broadcasting
    /// ```
    /// use nove::tensor::{Device, Tensor, Shape};
    ///
    /// let device = Device::default();
    ///
    /// let t1 = Tensor::from_data(vec![vec![1.0], vec![2.0]], &device, false).unwrap();
    /// let t2 = Tensor::from_data(vec![vec![10.0, 20.0, 30.0], vec![40.0, 50.0, 60.0]], &device, false).unwrap();
    /// let result = t1.gt(&t2).unwrap();
    /// let expected_values = vec![0, 0, 0, 0, 0, 0];
    /// let expected_shape: Shape = (&[2, 3]).into();
    /// assert_eq!(result.to_vec::<u8>().unwrap(), expected_values);
    /// assert_eq!(result.shape().unwrap(), expected_shape);
    /// ```
    pub fn gt(&self, rhs: &Self) -> Result<Self, TensorError> {
        comparison_op(self, rhs, "gt", |lhs, rhs| lhs.broadcast_gt(rhs))
    }

    /// Element-wise less-than (<) comparison with broadcasting, returning a boolean tensor.
    ///
    /// # Notes
    /// * The [`crate::DType`] of the result tensor will be [`crate::DType::U8`] representing boolean values.
    ///
    /// # Arguments
    /// * `rhs` - The tensor to compare with.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor with [`crate::DType::U8`] DType.
    /// * `Err(TensorError)` - The error when comparing the tensors.
    ///
    /// # Examples
    /// * Compare two 2x3 matrices without broadcasting
    /// ```
    /// use nove::tensor::{Device, Tensor, Shape};
    ///
    /// let device = Device::default();
    ///
    /// let t1 = Tensor::from_data(vec![vec![1.5, -2.0, 3.0], vec![0.0, 5.0, -1.0]], &device, false).unwrap();
    /// let t2 = Tensor::from_data(vec![vec![1.0, -1.0, 2.5], vec![-0.5, 4.0, -2.0]], &device, false).unwrap();
    /// let result = t1.lt(&t2).unwrap();
    /// let expected_values = vec![0, 1, 0, 0, 0, 0];
    /// let expected_shape: Shape = (&[2, 3]).into();
    /// assert_eq!(result.to_vec::<u8>().unwrap(), expected_values);
    /// assert_eq!(result.shape().unwrap(), expected_shape);
    /// ```
    /// * Compare a 1x3 row vector with a 2x3 matrix with broadcasting
    /// ```
    /// use nove::tensor::{Device, Tensor, Shape};
    ///
    /// let device = Device::default();
    ///
    /// let t1 = Tensor::from_data(vec![1.0, 5.0, 100.0], &device, false).unwrap();
    /// let t2 = Tensor::from_data(vec![vec![10.0, 20.0, 30.0], vec![40.0, 50.0, 60.0]], &device, false).unwrap();
    /// let result = t1.lt(&t2).unwrap();
    /// let expected_values = vec![1, 1, 0, 1, 1, 0];
    /// let expected_shape: Shape = (&[2, 3]).into();
    /// assert_eq!(result.to_vec::<u8>().unwrap(), expected_values);
    /// assert_eq!(result.shape().unwrap(), expected_shape);
    /// ```
    pub fn lt(&self, rhs: &Self) -> Result<Self, TensorError> {
        comparison_op(self, rhs, "lt", |lhs, rhs| lhs.broadcast_lt(rhs))
    }

    /// Element-wise greater-than-or-equal (>=) comparison with broadcasting, returning a boolean tensor.
    ///
    /// # Notes
    /// * The [`crate::DType`] of the result tensor will be [`crate::DType::U8`] representing boolean values.
    ///
    /// # Arguments
    /// * `rhs` - The tensor to compare with.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor with [`crate::DType::U8`] DType.
    /// * `Err(TensorError)` - The error when comparing the tensors.
    ///
    /// # Examples
    /// * Compare two 2x3 matrices without broadcasting
    /// ```
    /// use nove::tensor::{Device, Tensor, Shape};
    ///
    /// let device = Device::default();
    ///
    /// let t1 = Tensor::from_data(vec![vec![1.0, -1.0, 3.0], vec![0.0, 5.0, -2.0]], &device, false).unwrap();
    /// let t2 = Tensor::from_data(vec![vec![1.0, -2.0, 2.5], vec![0.0, 4.0, -1.0]], &device, false).unwrap();
    /// let result = t1.ge(&t2).unwrap();
    /// let expected_values = vec![1, 1, 1, 1, 1, 0];
    /// let expected_shape: Shape = (&[2, 3]).into();
    /// assert_eq!(result.to_vec::<u8>().unwrap(), expected_values);
    /// assert_eq!(result.shape().unwrap(), expected_shape);
    /// ```
    /// * Compare a 2x1 column vector with a 2x3 matrix with broadcasting
    /// ```
    /// use nove::tensor::{Device, Tensor, Shape};
    ///
    /// let device = Device::default();
    ///
    /// let t1 = Tensor::from_data(vec![vec![10.0], vec![2.0]], &device, false).unwrap();
    /// let t2 = Tensor::from_data(vec![vec![10.0, 20.0, 30.0], vec![40.0, 50.0, 60.0]], &device, false).unwrap();
    /// let result = t1.ge(&t2).unwrap();
    /// let expected_values = vec![1, 0, 0, 0, 0, 0];
    /// let expected_shape: Shape = (&[2, 3]).into();
    /// assert_eq!(result.to_vec::<u8>().unwrap(), expected_values);
    /// assert_eq!(result.shape().unwrap(), expected_shape);
    /// ```
    pub fn ge(&self, rhs: &Self) -> Result<Self, TensorError> {
        comparison_op(self, rhs, "ge", |lhs, rhs| lhs.broadcast_ge(rhs))
    }

    /// Element-wise less-than-or-equal (<=) comparison with broadcasting, returning a boolean tensor.
    ///
    /// # Notes
    /// * The [`crate::DType`] of the result tensor will be [`crate::DType::U8`] representing boolean values.
    ///
    /// # Arguments
    /// * `rhs` - The tensor to compare with.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor with [`crate::DType::U8`] DType.
    /// * `Err(TensorError)` - The error when comparing the tensors.
    ///
    /// # Examples
    /// * Compare two 2x3 matrices without broadcasting
    /// ```
    /// use nove::tensor::{Device, Tensor, Shape};
    ///
    /// let device = Device::default();
    ///
    /// let t1 = Tensor::from_data(vec![vec![1.0, -1.0, 3.0], vec![0.0, 5.0, -2.0]], &device, false).unwrap();
    /// let t2 = Tensor::from_data(vec![vec![1.0, -2.0, 2.5], vec![0.0, 4.0, -1.0]], &device, false).unwrap();
    /// let result = t1.le(&t2).unwrap();
    /// let expected_values = vec![1, 0, 0, 1, 0, 1];
    /// let expected_shape: Shape = (&[2, 3]).into();
    /// assert_eq!(result.to_vec::<u8>().unwrap(), expected_values);
    /// assert_eq!(result.shape().unwrap(), expected_shape);
    /// ```
    /// * Compare a 1x3 row vector with a 2x3 matrix with broadcasting
    /// ```
    /// use nove::tensor::{Device, Tensor, Shape};
    ///
    /// let device = Device::default();
    ///
    /// let t1 = Tensor::from_data(vec![10.0, 5.0, 100.0], &device, false).unwrap();
    /// let t2 = Tensor::from_data(vec![vec![10.0, 20.0, 30.0], vec![40.0, 50.0, 60.0]], &device, false).unwrap();
    /// let result = t1.le(&t2).unwrap();
    /// let expected_values = vec![1, 1, 0, 1, 1, 0];
    /// let expected_shape: Shape = (&[2, 3]).into();
    /// assert_eq!(result.to_vec::<u8>().unwrap(), expected_values);
    /// assert_eq!(result.shape().unwrap(), expected_shape);
    /// ```
    pub fn le(&self, rhs: &Self) -> Result<Self, TensorError> {
        comparison_op(self, rhs, "le", |lhs, rhs| lhs.broadcast_le(rhs))
    }
}

fn comparison_op<F>(
    lhs: &Tensor,
    rhs: &Tensor,
    name: &'static str,
    f: F,
) -> Result<Tensor, TensorError>
where
    F: FnOnce(
        &crate::backend::BackendStorage,
        &crate::backend::BackendStorage,
    ) -> Result<crate::backend::BackendStorage, crate::backend::BackendError>,
{
    let lhs_storage = lhs.backend_storage()?;
    let rhs_storage = rhs.backend_storage()?;
    let storage = f(&lhs_storage, &rhs_storage)?;
    Ok(Tensor::op_result_with_kind(
        storage,
        lhs.device()?,
        vec![lhs.copy(), rhs.copy()],
        OpKind::Comparison(name),
    ))
}
