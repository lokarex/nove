use std::sync::{Arc, RwLock};

use crate::{
    Tensor, TensorError,
    tensor::{TensorData, TensorInner},
};

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
    /// ```
    /// use nove::tensor::{Device, Tensor, Shape};
    /// let device = Device::cpu();
    ///
    /// let shape = Shape::from(&[2, 3]);
    /// // Create 2x3 matrices with typical data distribution
    /// let lhs = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &shape, &device, false).unwrap();
    /// let rhs = Tensor::from_vec(vec![1.0, 2.0, 0.0, 4.0, 5.0, 0.0], &shape, &device, false).unwrap();
    /// // Compare element-wise equality without broadcasting
    /// let result = lhs.eq(&rhs).unwrap();
    /// // Result will be [[true, true, false], [true, true, false]]
    /// let expected = vec![1, 1, 0, 1, 1, 0];
    /// assert_eq!(result.to_vec::<u8>().unwrap(), expected);
    ///
    /// // Create a 1x3 vector and a 2x3 matrix
    /// let lhs = Tensor::from_data(vec![1.0, 2.0, 3.0], &device, false).unwrap();
    /// let rhs = Tensor::from_data(vec![vec![1.0, 2.0, 0.0], vec![0.0, 0.0, 0.0]], &device, false).unwrap();
    /// // Compare element-wise equality with broadcasting
    /// let result = lhs.eq(&rhs).unwrap();
    /// // Result will be [[true, true, false], [false, false, false]]
    /// let expected = vec![1, 1, 0, 0, 0, 0];
    /// assert_eq!(result.to_vec::<u8>().unwrap(), expected);
    /// assert_eq!(result.shape().unwrap(), (&[2, 3]).into());
    /// ```
    pub fn eq(&self, rhs: &Self) -> Result<Self, TensorError> {
        let lhs_inner = self.data.read()?;
        let lhs_inner_tensor = match &lhs_inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let rhs_inner = rhs.data.read()?;
        let rhs_inner_tensor = match &rhs_inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = TensorInner::Tensor(lhs_inner_tensor.broadcast_eq(rhs_inner_tensor)?);

        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                device: self.data.read()?.device.clone(),
                parents: vec![self.copy(), rhs.copy()],
                grad: None,
                name: None,
            })),
        })
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
    /// ```
    /// use nove::tensor::{Device, Tensor, Shape};
    /// let device = Device::cpu();
    /// let shape = Shape::from(&[2, 3]);
    /// // Create 2x3 matrices with typical data distribution
    /// let lhs = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &shape, &device, false).unwrap();
    /// let rhs = Tensor::from_vec(vec![1.0, 2.0, 0.0, 4.0, 5.0, 0.0], &shape, &device, false).unwrap();
    /// // Compare element-wise inequality
    /// let result = lhs.ne(&rhs).unwrap();
    /// // Result will be [[false, false, true], [false, false, true]]
    /// let expected = vec![0, 0, 1, 0, 0, 1];
    /// assert_eq!(result.to_vec::<u8>().unwrap(), expected);
    ///
    /// // Create a 1x3 vector and a 2x3 matrix
    /// let lhs = Tensor::from_data(vec![1.0, 2.0, 3.0], &device, false).unwrap();
    /// let rhs = Tensor::from_data(vec![vec![1.0, 2.0, 0.0], vec![0.0, 0.0, 0.0]], &device, false).unwrap();
    /// // Compare element-wise inequality with broadcasting
    /// let result = lhs.ne(&rhs).unwrap();
    /// // Result will be [[false, false, true], [true, true, true]]
    /// let expected = vec![0, 0, 1, 1, 1, 1];
    /// assert_eq!(result.to_vec::<u8>().unwrap(), expected);
    /// assert_eq!(result.shape().unwrap(), (&[2, 3]).into());
    /// ```
    pub fn ne(&self, rhs: &Self) -> Result<Self, TensorError> {
        let lhs_inner = self.data.read()?;
        let lhs_inner_tensor = match &lhs_inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let rhs_inner = rhs.data.read()?;
        let rhs_inner_tensor = match &rhs_inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = TensorInner::Tensor(lhs_inner_tensor.broadcast_ne(rhs_inner_tensor)?);

        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                device: self.data.read()?.device.clone(),
                parents: vec![self.copy(), rhs.copy()],
                grad: None,
                name: None,
            })),
        })
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
    /// ```
    /// use nove::tensor::{Device, Tensor, Shape};
    /// let device = Device::cpu();
    /// let shape = Shape::from(&[2, 3]);
    /// // Create 2x3 matrices with mixed positive and negative values
    /// let lhs = Tensor::from_vec(vec![1.5, -2.0, 3.0, 0.0, 5.0, -1.0], &shape, &device, false).unwrap();
    /// let rhs = Tensor::from_vec(vec![1.0, -1.0, 2.5, -0.5, 4.0, -2.0], &shape, &device, false).unwrap();
    /// // Compare element-wise greater-than
    /// let result = lhs.gt(&rhs).unwrap();
    /// // Result will be [[true, false, true], [true, true, true]]
    /// let expected = vec![1, 0, 1, 1, 1, 1];
    /// assert_eq!(result.to_vec::<u8>().unwrap(), expected);
    ///
    /// // Create a 2x1 column vector and a 2x3 matrix
    /// let lhs = Tensor::from_data(vec![vec![1.0], vec![2.0]], &device, false).unwrap();
    /// let rhs = Tensor::from_data(vec![vec![10.0, 20.0, 30.0], vec![40.0, 50.0, 60.0]], &device, false).unwrap();
    /// // Compare element-wise greater-than with broadcasting
    /// let result = lhs.gt(&rhs).unwrap();
    /// // Result will be [[false, false, false], [false, false, false]]
    /// let expected = vec![0, 0, 0, 0, 0, 0];
    /// assert_eq!(result.to_vec::<u8>().unwrap(), expected);
    /// assert_eq!(result.shape().unwrap(), (&[2, 3]).into());
    /// ```
    pub fn gt(&self, rhs: &Self) -> Result<Self, TensorError> {
        let lhs_inner = self.data.read()?;
        let lhs_inner_tensor = match &lhs_inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let rhs_inner = rhs.data.read()?;
        let rhs_inner_tensor = match &rhs_inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = TensorInner::Tensor(lhs_inner_tensor.broadcast_gt(rhs_inner_tensor)?);

        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                device: self.data.read()?.device.clone(),
                parents: vec![self.copy(), rhs.copy()],
                grad: None,
                name: None,
            })),
        })
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
    /// ```
    /// use nove::tensor::{Device, Tensor, Shape};
    /// let device = Device::cpu();
    /// let shape = Shape::from(&[2, 3]);
    /// // Create 2x3 matrices with values that have clear ordering relationships
    /// let lhs = Tensor::from_vec(vec![1.5, -2.0, 3.0, 0.0, 5.0, -1.0], &shape, &device, false).unwrap();
    /// let rhs = Tensor::from_vec(vec![1.0, -1.0, 2.5, -0.5, 4.0, -2.0], &shape, &device, false).unwrap();
    /// // Compare element-wise less-than
    /// let result = lhs.lt(&rhs).unwrap();
    /// // Result will be [[false, true, false], [false, false, false]]
    /// let expected = vec![0, 1, 0, 0, 0, 0];
    /// assert_eq!(result.to_vec::<u8>().unwrap(), expected);
    ///
    /// // Create a 1x2 row vector and a 2x3 matrix
    /// let lhs = Tensor::from_data(vec![1.0, 5.0, 100.0], &device, false).unwrap();
    /// let rhs = Tensor::from_data(vec![vec![10.0, 20.0, 30.0], vec![40.0, 50.0, 60.0]], &device, false).unwrap();
    /// // Compare element-wise less-than with broadcasting
    /// let result = lhs.lt(&rhs).unwrap();
    /// // Result will be [[true, true, false], [true, true, false]]
    /// let expected = vec![1, 1, 0, 1, 1, 0];
    /// assert_eq!(result.to_vec::<u8>().unwrap(), expected);
    /// assert_eq!(result.shape().unwrap(), (&[2, 3]).into());
    /// ```
    pub fn lt(&self, rhs: &Self) -> Result<Self, TensorError> {
        let lhs_inner = self.data.read()?;
        let lhs_inner_tensor = match &lhs_inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let rhs_inner = rhs.data.read()?;
        let rhs_inner_tensor = match &rhs_inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = TensorInner::Tensor(lhs_inner_tensor.broadcast_lt(rhs_inner_tensor)?);

        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                device: self.data.read()?.device.clone(),
                parents: vec![self.copy(), rhs.copy()],
                grad: None,
                name: None,
            })),
        })
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
    /// ```
    /// use nove::tensor::{Device, Tensor, Shape};
    /// let device = Device::cpu();
    /// let shape = Shape::from(&[2, 3]);
    /// // Create 2x3 matrices with mixed values including equalities
    /// let lhs = Tensor::from_vec(vec![1.0, -1.0, 3.0, 0.0, 5.0, -2.0], &shape, &device, false).unwrap();
    /// let rhs = Tensor::from_vec(vec![1.0, -2.0, 2.5, 0.0, 4.0, -1.0], &shape, &device, false).unwrap();
    /// // Compare element-wise greater-than-or-equal
    /// let result = lhs.ge(&rhs).unwrap();
    /// // Result will be [[true, true, true], [true, true, false]]
    /// let expected = vec![1, 1, 1, 1, 1, 0];
    /// assert_eq!(result.to_vec::<u8>().unwrap(), expected);
    ///
    /// // Create a 2x1 column vector and a 2x3 matrix
    /// let lhs = Tensor::from_data(vec![vec![10.0], vec![2.0]], &device, false).unwrap();
    /// let rhs = Tensor::from_data(vec![vec![10.0, 20.0, 30.0], vec![40.0, 50.0, 60.0]], &device, false).unwrap();
    /// // Compare element-wise greater-than-or-equal with broadcasting
    /// let result = lhs.ge(&rhs).unwrap();
    /// // Result will be [[true, false, false], [false, false, false]]
    /// let expected = vec![1, 0, 0, 0, 0, 0];
    /// assert_eq!(result.to_vec::<u8>().unwrap(), expected);
    /// assert_eq!(result.shape().unwrap(), (&[2, 3]).into());
    /// ```
    pub fn ge(&self, rhs: &Self) -> Result<Self, TensorError> {
        let lhs_inner = self.data.read()?;
        let lhs_inner_tensor = match &lhs_inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let rhs_inner = rhs.data.read()?;
        let rhs_inner_tensor = match &rhs_inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = TensorInner::Tensor(lhs_inner_tensor.broadcast_ge(rhs_inner_tensor)?);

        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                device: self.data.read()?.device.clone(),
                parents: vec![self.copy(), rhs.copy()],
                grad: None,
                name: None,
            })),
        })
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
    /// ```
    /// use nove::tensor::{Device, Tensor, Shape};
    /// let device = Device::cpu();
    /// let shape = Shape::from(&[2, 3]);
    /// // Create 2x3 matrices with mixed values including equalities
    /// let lhs = Tensor::from_vec(vec![1.0, -1.0, 3.0, 0.0, 5.0, -2.0], &shape, &device, false).unwrap();
    /// let rhs = Tensor::from_vec(vec![1.0, -2.0, 2.5, 0.0, 4.0, -1.0], &shape, &device, false).unwrap();
    /// // Compare element-wise less-than-or-equal
    /// let result = lhs.le(&rhs).unwrap();
    /// // Result will be [[true, false, false], [true, false, true]]
    /// let expected = vec![1, 0, 0, 1, 0, 1];
    /// assert_eq!(result.to_vec::<u8>().unwrap(), expected);
    ///
    /// // Create a 1x2 row vector and a 2x3 matrix
    /// let lhs = Tensor::from_data(vec![10.0, 5.0, 100.0], &device, false).unwrap();
    /// let rhs = Tensor::from_data(vec![vec![10.0, 20.0, 30.0], vec![40.0, 50.0, 60.0]], &device, false).unwrap();
    /// // Compare element-wise less-than-or-equal with broadcasting
    /// let result = lhs.le(&rhs).unwrap();
    /// // Result will be [[true, true, false], [true, true, false]]
    /// let expected = vec![1, 1, 0, 1, 1, 0];
    /// assert_eq!(result.to_vec::<u8>().unwrap(), expected);
    /// assert_eq!(result.shape().unwrap(), (&[2, 3]).into());
    /// ```
    pub fn le(&self, rhs: &Self) -> Result<Self, TensorError> {
        let lhs_inner = self.data.read()?;
        let lhs_inner_tensor = match &lhs_inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let rhs_inner = rhs.data.read()?;
        let rhs_inner_tensor = match &rhs_inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = TensorInner::Tensor(lhs_inner_tensor.broadcast_le(rhs_inner_tensor)?);

        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                device: self.data.read()?.device.clone(),
                parents: vec![self.copy(), rhs.copy()],
                grad: None,
                name: None,
            })),
        })
    }
}
