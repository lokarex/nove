use std::sync::{Arc, RwLock};

use crate::{
    Tensor, TensorError,
    tensor::{TensorData, TensorInner},
};

impl Tensor {
    /// Compute the sum of tensor elements along a specified dimension or across all elements.
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
                parents: vec![self.copy()],
                grad: None,
                name: None,
            })),
        })
    }

    /// Compute the maximum value along a specified dimension or across all elements.
    ///
    /// # Arguments
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
                parents: vec![self.copy()],
                grad: None,
                name: None,
            })),
        })
    }

    /// Compute the minimum value along a specified dimension or across all elements.
    ///
    /// # Arguments
    /// * `axis` - Optional `(dim, keep_dim)` tuple.
    ///   - `Some((dim, keep_dim))`: compute along `dim`
    ///     - `keep_dim = true`: keep dimension (size becomes 1)
    ///     - `keep_dim = false`: remove dimension
    ///   - `None`: compute across all elements
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor containing the minimum values.
    /// * `Err(TensorError)` - The error when computing the minimum.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::cpu();
    /// let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &device, false).unwrap();
    ///
    /// let min_all = t.min(None).unwrap();
    /// println!("{:?}", min_all);
    /// ```
    pub fn min(&self, axis: Option<(usize, bool)>) -> Result<Self, TensorError> {
        let inner = self.data.read()?;
        let inner_tensor = match &inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = match axis {
            Some((dim, keep_dim)) => match keep_dim {
                true => TensorInner::Tensor(inner_tensor.min_keepdim(dim)?),
                false => TensorInner::Tensor(inner_tensor.min(dim)?),
            },
            None => TensorInner::Tensor(inner_tensor.min_all()?),
        };

        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                device: self.data.read()?.device.clone(),
                parents: vec![self.copy()],
                grad: None,
                name: None,
            })),
        })
    }

    /// Compute the mean along the specified axis.
    ///
    /// # Arguments
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
                parents: vec![self.copy()],
                grad: None,
                name: None,
            })),
        })
    }

    /// Compute the variance along the specified axis.
    ///
    /// # Arguments
    /// * `dim` - The dimension to compute variance.
    /// * `keep_dim` - Whether to keep the dimension (size becomes 1).
    /// * `unbiased` - Whether to use unbiased estimation.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The tensor after applying the variance operation.
    /// * `Err(TensorError)` - The error when applying the variance operation.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    /// let t = Tensor::rand(0.0f32, 1.0f32, &Shape::from_dims(&[1, 3, 5, 5]), &device, false).unwrap();
    /// let result = t.var(1, false, true).unwrap();
    /// println!("{:?}", result);
    /// ```
    pub fn var(&self, dim: usize, keep_dim: bool, unbiased: bool) -> Result<Self, TensorError> {
        let inner = self.data.read()?;
        let inner_tensor = match &inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner_tensor = match (dim, keep_dim) {
            (dim, false) => inner_tensor.var(dim)?,
            (dim, true) => inner_tensor.var_keepdim(dim)?,
        };

        let new_inner = match unbiased {
            false => {
                let total_size = inner_tensor.shape().dims().iter().product::<usize>();
                let num_features = inner_tensor.shape().dims()[dim];
                let n = total_size / num_features;
                TensorInner::Tensor(new_inner_tensor.affine((n as f64 - 1f64) / n as f64, 0f64)?)
            }
            true => TensorInner::Tensor(new_inner_tensor),
        };

        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                device: self.data.read()?.device.clone(),
                parents: vec![self.copy()],
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
                parents: vec![self.copy()],
                grad: None,
                name: None,
            })),
        })
    }

    /// Compute the indices of the minimum values along a specified dimension.
    ///
    /// # Arguments
    /// * `axis` - `(dim, keep_dim)` tuple.
    ///   - `dim`: dimension to compute argmin
    ///   - `keep_dim`:
    ///     - `true`: keep dimension (size becomes 1)
    ///     - `false`: remove dimension
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor with dtype `u64` containing the indices of minimum values.
    /// * `Err(TensorError)` - The error when computing the argmin.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::cpu();
    /// let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &device, false).unwrap();
    ///
    /// let argmin = t.argmin((0, false)).unwrap();
    /// println!("{:?}", argmin);
    /// ```
    pub fn argmin(&self, axis: (usize, bool)) -> Result<Self, TensorError> {
        let inner = self.data.read()?;
        let inner_tensor = match &inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let (dim, keep_dim) = axis;
        let new_inner = match keep_dim {
            true => TensorInner::Tensor(inner_tensor.argmin_keepdim(dim)?),
            false => TensorInner::Tensor(inner_tensor.argmin(dim)?),
        };

        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                device: self.data.read()?.device.clone(),
                parents: vec![self.copy()],
                grad: None,
                name: None,
            })),
        })
    }
}
