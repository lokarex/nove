use std::sync::{Arc, RwLock};

use crate::{
    Tensor, TensorError,
    tensor::{TensorData, TensorInner},
};

impl Tensor {
    /// Element-wise equal (==) comparison with broadcasting, returning a boolean tensor.
    ///
    /// # Arguments
    /// * `rhs` - The tensor to compare with.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor with boolean values(dtype: U8).
    /// * `Err(TensorError)` - The error when comparing the tensors.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::cpu();
    /// let lhs = Tensor::from_data(vec![1.0, 2.0, 0.0, 0.0], &device, false).unwrap();
    /// let rhs = Tensor::from_data(vec![0.0, 0.0, 3.0, 4.0], &device, false).unwrap();
    ///
    /// let result = lhs.eq(&rhs).unwrap();
    /// println!("{:?}", result);
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
    /// # Arguments
    /// * `rhs` - The tensor to compare with.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor with boolean values(dtype: U8).
    /// * `Err(TensorError)` - The error when comparing the tensors.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::cpu();
    /// let lhs = Tensor::from_data(vec![1.0, 2.0, 0.0, 0.0], &device, false).unwrap();
    /// let rhs = Tensor::from_data(vec![0.0, 0.0, 3.0, 4.0], &device, false).unwrap();
    ///
    /// let result = lhs.ne(&rhs).unwrap();
    /// println!("{:?}", result);
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
    /// # Arguments
    /// * `rhs` - The tensor to compare with.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor with boolean values(dtype: U8).
    /// * `Err(TensorError)` - The error when comparing the tensors.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::cpu();
    /// let lhs = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &device, false).unwrap();
    /// let rhs = Tensor::from_data(vec![2.0, 2.0, 2.0, 2.0], &device, false).unwrap();
    ///
    /// let result = lhs.gt(&rhs).unwrap();
    /// println!("{:?}", result);
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
    /// # Arguments
    /// * `rhs` - The tensor to compare with.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor with boolean values(dtype: U8).
    /// * `Err(TensorError)` - The error when comparing the tensors.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::cpu();
    /// let lhs = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &device, false).unwrap();
    /// let rhs = Tensor::from_data(vec![2.0, 2.0, 2.0, 2.0], &device, false).unwrap();
    ///
    /// let result = lhs.lt(&rhs).unwrap();
    /// println!("{:?}", result);
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
    /// # Arguments
    /// * `rhs` - The tensor to compare with.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor with boolean values(dtype: U8).
    /// * `Err(TensorError)` - The error when comparing the tensors.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::cpu();
    /// let lhs = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &device, false).unwrap();
    /// let rhs = Tensor::from_data(vec![2.0, 2.0, 2.0, 2.0], &device, false).unwrap();
    ///
    /// let result = lhs.ge(&rhs).unwrap();
    /// println!("{:?}", result);
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
    /// # Arguments
    /// * `rhs` - The tensor to compare with.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor with boolean values(dtype: U8).
    /// * `Err(TensorError)` - The error when comparing the tensors.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::cpu();
    /// let lhs = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &device, false).unwrap();
    /// let rhs = Tensor::from_data(vec![2.0, 2.0, 2.0, 2.0], &device, false).unwrap();
    ///
    /// let result = lhs.le(&rhs).unwrap();
    /// println!("{:?}", result);
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