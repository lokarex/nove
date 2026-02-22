use std::sync::{Arc, RwLock};

use crate::{
    Shape, Tensor, TensorError,
    tensor::{TensorData, TensorInner},
};

impl Tensor {
    /// Add two tensors with broadcasting.
    ///
    /// # Arguments
    /// * `rhs` - The tensor to add.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor after addition.
    /// * `Err(TensorError)` - The error when adding the tensors.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::cpu();
    /// let t1 = Tensor::from_data(vec![1.0, 2.0], &device, false).unwrap();
    /// let t2 = Tensor::from_data(vec![3.0, 4.0], &device, false).unwrap();
    ///
    /// let t3 = t1.add(&t2).unwrap();
    /// println!("{:?}", t3);
    /// ```
    pub fn add(&self, rhs: &Self) -> Result<Self, TensorError> {
        let inner1 = self.data.read()?;
        let inner1_tensor = match &inner1.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };
        let inner2 = rhs.data.read()?;
        let inner2_tensor = match &inner2.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        // Get the device from the first tensor
        let device = self.data.read()?.device.clone();

        let new_inner = TensorInner::Tensor(inner1_tensor.broadcast_add(inner2_tensor)?);

        let parents = vec![self.clone(), rhs.clone()];

        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                device,
                parents,
                grad: None,
                name: None,
            })),
        })
    }

    /// Multiply two tensors with broadcasting.
    ///
    /// # Arguments
    /// * `rhs` - The tensor to multiply.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor after multiplication.
    /// * `Err(TensorError)` - The error when multiplying the tensors.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::cpu();
    /// let t1 = Tensor::from_data(vec![1.0, 2.0], &device, false).unwrap();
    /// let t2 = Tensor::from_data(vec![3.0, 4.0], &device, false).unwrap();
    ///
    /// let t3 = t1.mul(&t2).unwrap();
    /// println!("{:?}", t3);
    /// ```
    pub fn mul(&self, rhs: &Self) -> Result<Self, TensorError> {
        let inner1 = self.data.read()?;
        let inner1_tensor = match &inner1.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };
        let inner2 = rhs.data.read()?;
        let inner2_tensor = match &inner2.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = TensorInner::Tensor(inner1_tensor.broadcast_mul(inner2_tensor)?);

        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                device: self.data.read()?.device.clone(),
                parents: vec![self.clone(), rhs.clone()],
                grad: None,
                name: None,
            })),
        })
    }

    /// Stack a list of tensors along a new dimension.
    ///
    /// # Arguments
    /// * `tensors` - The list of tensors to stack.
    /// * `dim` - The dimension along which to stack the tensors.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor after stacking.
    /// * `Err(TensorError)` - The error when stacking the tensors.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::cpu();
    /// let t1 = Tensor::from_data(vec![1.0, 2.0], &device, false).unwrap();
    /// let t2 = Tensor::from_data(vec![3.0, 4.0], &device, false).unwrap();
    /// let t3 = Tensor::from_data(vec![5.0, 6.0], &device, false).unwrap();
    ///
    /// let t4 = Tensor::stack(&[t1, t2, t3], 0).unwrap();
    /// println!("{:?}", t4);
    /// ```
    pub fn stack<A, D>(tensors: &[A], dim: D) -> Result<Self, TensorError>
    where
        A: AsRef<Tensor> + std::clone::Clone,
        D: candle_core::shape::Dim,
    {
        let inner_tensors = tensors
            .iter()
            .map(|tensor| {
                let data = tensor.as_ref().data.read()?;
                match &data.inner {
                    TensorInner::Tensor(tensor) => Ok(tensor.clone()),
                    TensorInner::Var(var) => Ok(var.as_tensor().clone()),
                }
            })
            .collect::<Result<Vec<_>, TensorError>>()?;
        // Stack the tensors
        let new_inner_tensor = candle_core::Tensor::stack(&inner_tensors, dim)?;

        // Get the device from the first tensor
        let device = tensors
            .first()
            .map(|t| t.as_ref().data.read().unwrap().device.clone())
            .unwrap();

        let new_inner = TensorInner::Tensor(new_inner_tensor);

        //  Set the parents
        let parents = tensors
            .iter()
            .map(|tensor| tensor.as_ref().clone())
            .collect::<Vec<_>>();

        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                device,
                parents,
                grad: None,
                name: None,
            })),
        })
    }

    /// Matrix multiplication between two tensors with broadcasting.
    ///
    /// # Arguments
    /// * `rhs` - The tensor to multiply.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor after matrix multiplication.
    /// * `Err(TensorError)` - The error when multiplying the tensors.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::cpu();
    /// let t1 = Tensor::from_data(&[[1.0, 2.0]], &device, false).unwrap();
    /// let t2 = Tensor::from_data(&[[5.0, 6.0], [7.0, 8.0]], &device, false).unwrap();
    ///
    /// let t3 = t1.matmul(&t2).unwrap();
    /// println!("{:?}", t3);
    /// ```
    pub fn matmul(&self, rhs: &Self) -> Result<Self, TensorError> {
        let inner1 = self.data.read()?;
        let inner1_tensor = match &inner1.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };
        let inner2 = rhs.data.read()?;
        let inner2_tensor = match &inner2.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = TensorInner::Tensor(inner1_tensor.broadcast_matmul(inner2_tensor)?);

        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                device: self.data.read()?.device.clone(),
                parents: vec![self.clone(), rhs.clone()],
                grad: None,
                name: None,
            })),
        })
    }

    /// Compute the maximum value along a specified dimension or across all elements.
    ///
    /// # Arguments
    ///
    /// * `axis` - Optional `(dim, keep_dim)` tuple.
    ///   - `Some((dim, keep_dim))`: compute along `dim`
    ///     - `keep_dim = true`: keep dimension (size becomes 1)
    ///     - `keep_dim = false`: remove dimension
    ///   - `None`: compute across all elements
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor containing the maximum values.
    /// * `Err(TensorError)` - The error when computing the maximum.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::cpu();
    /// let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &device, false).unwrap();
    ///
    /// let max_all = t.max(None).unwrap();
    /// println!("{:?}", max_all);
    /// ```
    pub fn max(&self, axis: Option<(usize, bool)>) -> Result<Self, TensorError> {
        let inner = self.data.read()?;
        let inner_tensor = match &inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = match axis {
            Some((dim, keep_dim)) => match keep_dim {
                true => TensorInner::Tensor(inner_tensor.max_keepdim(dim)?),
                false => TensorInner::Tensor(inner_tensor.max(dim)?),
            },
            None => TensorInner::Tensor(inner_tensor.max_all()?),
        };

        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                device: self.data.read()?.device.clone(),
                parents: vec![self.clone()],
                grad: None,
                name: None,
            })),
        })
    }

    /// Compute the indices of the maximum values along a specified dimension.
    ///
    /// # Arguments
    /// * `axis` - `(dim, keep_dim)` tuple.
    ///   - `dim`: dimension to compute argmax
    ///   - `keep_dim`:
    ///     - `true`: keep dimension (size becomes 1)
    ///     - `false`: remove dimension
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor with dtype `u64` containing the indices of maximum values.
    /// * `Err(TensorError)` - The error when computing the argmax.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::cpu();
    /// let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &device, false).unwrap();
    ///
    /// let argmax = t.argmax((0, false)).unwrap();
    /// println!("{:?}", argmax);
    /// ```
    pub fn argmax(&self, axis: (usize, bool)) -> Result<Self, TensorError> {
        let inner = self.data.read()?;
        let inner_tensor = match &inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let (dim, keep_dim) = axis;
        let new_inner = match keep_dim {
            true => TensorInner::Tensor(inner_tensor.argmax_keepdim(dim)?),
            false => TensorInner::Tensor(inner_tensor.argmax(dim)?),
        };

        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                device: self.data.read()?.device.clone(),
                parents: vec![self.clone()],
                grad: None,
                name: None,
            })),
        })
    }

    /// Compute the exponential (e^x) of each element in the tensor.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor with exponential values.
    /// * `Err(TensorError)` - The error when computing the exponential.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::cpu();
    /// let t = Tensor::from_data(vec![0.0, 1.0, 2.0], &device, false).unwrap();
    ///
    /// let exp = t.exp().unwrap();
    /// println!("{:?}", exp);
    /// ```
    pub fn exp(&self) -> Result<Self, TensorError> {
        let inner = self.data.read()?;
        let inner_tensor = match &inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = TensorInner::Tensor(inner_tensor.exp()?);

        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                device: self.data.read()?.device.clone(),
                parents: vec![self.clone()],
                grad: None,
                name: None,
            })),
        })
    }

    /// Compute the sum of elements along a specified dimension or across all elements.
    ///
    /// # Arguments
    /// * `axis` - Optional `(dim, keep_dim)` tuple.
    ///   - `Some((dim, keep_dim))`: compute along `dim`
    ///     - `keep_dim = true`: keep dimension (size becomes 1)
    ///     - `keep_dim = false`: remove dimension
    ///   - `None`: compute across all elements
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor containing the sum values.
    /// * `Err(TensorError)` - The error when computing the sum.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::cpu();
    /// let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &device, false).unwrap();
    ///
    /// let sum_all = t.sum(None).unwrap();
    /// println!("{:?}", sum_all);
    /// ```
    pub fn sum(&self, axis: Option<(usize, bool)>) -> Result<Self, TensorError> {
        let inner = self.data.read()?;
        let inner_tensor = match &inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = match axis {
            Some((dim, keep_dim)) => match keep_dim {
                true => TensorInner::Tensor(inner_tensor.sum_keepdim(dim)?),
                false => TensorInner::Tensor(inner_tensor.sum(dim)?),
            },
            None => TensorInner::Tensor(inner_tensor.sum_all()?),
        };

        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                device: self.data.read()?.device.clone(),
                parents: vec![self.clone()],
                grad: None,
                name: None,
            })),
        })
    }

    /// Compute the natural logarithm (ln) of each element in the tensor.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor with logarithm values.
    /// * `Err(TensorError)` - The error when computing the logarithm.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::cpu();
    /// let t = Tensor::from_data(vec![1.0, 2.0, 3.0], &device, false).unwrap();
    ///
    /// let log = t.log().unwrap();
    /// println!("{:?}", log);
    /// ```
    pub fn log(&self) -> Result<Self, TensorError> {
        let inner = self.data.read()?;
        let inner_tensor = match &inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = TensorInner::Tensor(inner_tensor.log()?);

        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                device: self.data.read()?.device.clone(),
                parents: vec![self.clone()],
                grad: None,
                name: None,
            })),
        })
    }

    /// Subtract two tensors with broadcasting.
    ///
    /// # Arguments
    /// * `rhs` - The tensor to subtract.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor after subtraction.
    /// * `Err(TensorError)` - The error when subtracting the tensors.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::cpu();
    /// let t1 = Tensor::from_data(vec![5.0, 6.0], &device, false).unwrap();
    /// let t2 = Tensor::from_data(vec![1.0, 2.0], &device, false).unwrap();
    ///
    /// let t3 = t1.sub(&t2).unwrap();
    /// println!("{:?}", t3);
    /// ```
    pub fn sub(&self, rhs: &Self) -> Result<Self, TensorError> {
        let inner = self.data.read()?;
        let inner_tensor = match &inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let rhs_inner = rhs.data.read()?;
        let rhs_inner_tensor = match &rhs_inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = TensorInner::Tensor(inner_tensor.broadcast_sub(rhs_inner_tensor)?);

        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                device: self.data.read()?.device.clone(),
                parents: vec![self.clone(), rhs.clone()],
                grad: None,
                name: None,
            })),
        })
    }

    /// Insert a dimension of size 1 at the specified position.
    ///
    /// # Arguments
    /// * `dim` - The dimension at which to insert the new dimension.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor with the new dimension added.
    /// * `Err(TensorError)` - The error when adding the dimension.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::cpu();
    /// let t = Tensor::from_data(vec![1.0, 2.0, 3.0], &device, false).unwrap();
    ///
    /// let t2 = t.unsqueeze(0).unwrap();
    /// println!("{:?}", t2);
    /// ```
    pub fn unsqueeze(&self, dim: usize) -> Result<Self, TensorError> {
        let inner = self.data.read()?;
        let inner_tensor = match &inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = TensorInner::Tensor(inner_tensor.unsqueeze(dim)?);

        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                device: self.data.read()?.device.clone(),
                parents: vec![self.clone()],
                grad: None,
                name: None,
            })),
        })
    }

    /// Gather values from the tensor along the specified dimension using the provided indices.
    ///
    /// # Notes
    /// * The data type(`DType`) of the indices tensor must be i64(`DType::I64`).
    ///
    /// # Arguments
    /// * `indices` - The tensor containing the indices to gather.
    /// * `dim` - The dimension along which to gather values.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor with gathered values.
    /// * `Err(TensorError)` - The error when gathering values.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::cpu();
    /// let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &device, false).unwrap();
    /// let indices = Tensor::from_data(vec![0i64, 2i64], &device, false).unwrap();
    ///
    /// let result = t.gather(&indices, 0).unwrap();
    /// println!("{:?}", result);
    /// ```
    pub fn gather(&self, indices: &Self, dim: usize) -> Result<Self, TensorError> {
        let inner = self.data.read()?;
        let inner_tensor = match &inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let indices_inner = indices.data.read()?;
        let indices_inner_tensor = match &indices_inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = TensorInner::Tensor(inner_tensor.gather(indices_inner_tensor, dim)?);

        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                device: self.data.read()?.device.clone(),
                parents: vec![self.clone(), indices.clone()],
                grad: None,
                name: None,
            })),
        })
    }

    /// Apply an affine transformation to each element in the tensor: `output = weight * input + bias`.
    ///
    /// # Arguments
    /// * `weight` - The multiplicative weight coefficient.
    /// * `bias` - The additive bias coefficient.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor after applying the affine transformation.
    /// * `Err(TensorError)` - The error when applying the affine transformation.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::cpu();
    /// let t = Tensor::from_data(vec![1.0, 2.0, 3.0], &device, false).unwrap();
    ///
    /// let result = t.affine(2.0, 1.0).unwrap();
    /// println!("{:?}", result);
    /// ```
    pub fn affine(&self, weight: f64, bias: f64) -> Result<Self, TensorError> {
        let inner = self.data.read()?;
        let inner_tensor = match &inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = TensorInner::Tensor(inner_tensor.affine(weight, bias)?);

        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                device: self.data.read()?.device.clone(),
                parents: vec![self.clone()],
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
    /// * `Ok(Tensor)` - The result tensor with boolean values.
    /// * `Err(TensorError)` - The error when comparing the tensors.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::cpu();
    /// let lhs = Tensor::from_data(vec![1.0, 2.0, 0.0, 0.0], &device, false).unwrap();
    /// let rhs = Tensor::from_data(vec![0.0, 0.0, 3.0, 4.0], &device, false).unwrap();
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
                parents: vec![self.clone(), rhs.clone()],
                grad: None,
                name: None,
            })),
        })
    }

    /// Element-wise equal (==) comparison with broadcasting, returning a boolean tensor.
    ///
    /// # Arguments
    /// * `rhs` - The tensor to compare with.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor with boolean values.
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
                parents: vec![self.clone(), rhs.clone()],
                grad: None,
                name: None,
            })),
        })
    }

    /// Broadcast the tensor to the specified shape.
    ///
    /// # Parameters
    /// * `shape` - The shape to broadcast the tensor to.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The broadcasted tensor if successful.
    /// * `Err(TensorError)` - The error when broadcasting the tensor.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    /// let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &device, false).unwrap();
    /// let shape = Shape::from_dims(&[2, 4]);
    ///
    /// let result = t.broadcast(&shape).unwrap();
    /// println!("{:?}", result);
    /// ```
    pub fn broadcast(&self, shape: &Shape) -> Result<Self, TensorError> {
        let inner = match &self.data.read()?.inner {
            TensorInner::Var(var) => {
                TensorInner::Var(candle_core::Var::from_tensor(&var.broadcast_as(shape)?)?)
            }
            TensorInner::Tensor(tensor) => TensorInner::Tensor(tensor.broadcast_as(shape)?),
        };
        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner,
                device: self.data.read()?.device.clone(),
                parents: vec![],
                grad: None,
                name: None,
            })),
        })
    }

    /// Apply the Rectified Linear Unit (ReLU) activation function element-wise.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The tensor after applying the ReLU activation function.
    /// * `Err(TensorError)` - The error when applying the ReLU activation function.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::cpu();
    /// let t = Tensor::from_data(vec![-1.0, 2.0, -3.0, 4.0], &device, false).unwrap();
    ///
    /// let result = t.relu().unwrap();
    /// println!("{:?}", result);
    /// ```
    pub fn relu(&self) -> Result<Self, TensorError> {
        let inner = self.data.read()?;
        let inner_tensor = match &inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = TensorInner::Tensor(inner_tensor.relu()?);

        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                device: self.data.read()?.device.clone(),
                parents: vec![self.clone()],
                grad: None,
                name: None,
            })),
        })
    }

    /// Apply the 2D convolutional operation.
    ///
    /// # Parameters
    /// * `kernel` - The kernel tensor.
    /// * `padding` - The padding size.
    /// * `stride` - The stride size.
    /// * `dilation` - The dilation size.
    /// * `groups` - The number of groups.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The tensor after applying the convolutional operation.
    /// * `Err(TensorError)` - The error when applying the convolutional operation.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    /// let t = Tensor::rand(0.0f32, 1.0f32, &Shape::from_dims(&[1, 3, 5, 5]), &device, false).unwrap();
    /// let kernel = Tensor::rand(0.0f32, 1.0f32, &Shape::from_dims(&[7, 3, 3, 3]), &device, false).unwrap();
    /// let result = t.conv2d(&kernel, 1, 1, 1, 1).unwrap();
    /// println!("{:?}", result);
    /// ```
    pub fn conv2d(
        &self,
        kernel: &Self,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self, TensorError> {
        let inner = self.data.read()?;
        let inner_tensor = match &inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let kernel_inner = kernel.data.read()?;
        let kernel_inner_tensor = match &kernel_inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = TensorInner::Tensor(inner_tensor.conv2d(
            kernel_inner_tensor,
            padding,
            stride,
            dilation,
            groups,
        )?);

        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                device: self.data.read()?.device.clone(),
                parents: vec![self.clone(), kernel.clone()],
                grad: None,
                name: None,
            })),
        })
    }

    /// Apply the mean operation along the specified axis.
    ///
    /// # Parameters
    /// * `axis` - Optional `(dim, keep_dim)` tuple.
    ///   - `Some((dim, keep_dim))`: compute along `dim`
    ///     - `keep_dim = true`: keep dimension (size becomes 1)
    ///     - `keep_dim = false`: remove dimension
    ///   - `None`: compute across all elements
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The tensor after applying the mean operation.
    /// * `Err(TensorError)` - The error when applying the mean operation.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    /// let t = Tensor::rand(0.0f32, 1.0f32, &Shape::from_dims(&[1, 3, 5, 5]), &device, false).unwrap();
    /// let result = t.mean(Some((1, false))).unwrap();
    /// println!("{:?}", result);
    /// ```
    pub fn mean(&self, axis: Option<(usize, bool)>) -> Result<Self, TensorError> {
        let inner = self.data.read()?;
        let inner_tensor = match &inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = match axis {
            Some((axis, false)) => TensorInner::Tensor(inner_tensor.mean(axis)?),
            Some((axis, true)) => TensorInner::Tensor(inner_tensor.mean_keepdim(axis)?),
            None => TensorInner::Tensor(inner_tensor.mean_all()?),
        };

        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                device: self.data.read()?.device.clone(),
                parents: vec![self.clone()],
                grad: None,
                name: None,
            })),
        })
    }

    /// Apply the 2D max pooling operation.
    ///
    /// # Parameters
    /// * `kernel_size` - The kernel size.
    /// * `stride` - The stride size.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The tensor after applying the max pooling operation.
    /// * `Err(TensorError)` - The error when applying the max pooling operation.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    /// let t = Tensor::rand(0.0f32, 1.0f32, &Shape::from_dims(&[1, 3, 5, 5]), &device, false).unwrap();
    /// let result = t.max_pool2d((2, 2), (2, 2)).unwrap();
    /// println!("{:?}", result);
    /// ```
    pub fn max_pool2d(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self, TensorError> {
        let inner = self.data.read()?;
        let inner_tensor = match &inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner =
            TensorInner::Tensor(inner_tensor.max_pool2d_with_stride(kernel_size, stride)?);

        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                device: self.data.read()?.device.clone(),
                parents: vec![self.clone()],
                grad: None,
                name: None,
            })),
        })
    }
}
