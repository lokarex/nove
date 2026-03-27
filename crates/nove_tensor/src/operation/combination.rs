use std::sync::{Arc, RwLock};

use crate::{
    Tensor, TensorError,
    tensor::{TensorData, TensorInner},
};

impl Tensor {
    /// Stack a list of tensors along a new dimension.
    ///
    /// # Arguments
    /// * `tensors` - The list of tensors to stack.
    /// * `dim` - The dimension along which to stack the tensors. It must be greater than or equal to `-1`.
    /// When `dim` is equal to `-1`, the tensors are stacked along the last dimension.
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
    /// // Stack tensors along the first dimension
    /// let t4 = Tensor::stack(&[t1.copy(), t2.copy(), t3.copy()], 0).unwrap();
    /// // Result should be: [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    /// let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// assert_eq!(t4.to_vec::<f64>().unwrap(), expected);
    /// assert_eq!(t4.shape().unwrap(), (&[3, 2]).into());
    ///
    /// // Stack tensors along the last dimension
    /// let t5 = Tensor::stack(&[t1, t2, t3], -1).unwrap();
    /// // Result should be: [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]
    /// let expected = vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0];
    /// assert_eq!(t5.to_vec::<f64>().unwrap(), expected);
    /// assert_eq!(t5.shape().unwrap(), (&[2, 3]).into());
    /// ```
    pub fn stack<A>(tensors: &[A], dim: isize) -> Result<Self, TensorError>
    where
        A: AsRef<Tensor> + std::clone::Clone,
    {
        let dim: usize = match dim {
            d if d >= 0 => d as usize,
            -1 => tensors[0].as_ref().shape()?.dims().len() as usize,
            _ => return Err(TensorError::InvalidDimension(dim)),
        };

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
            .ok_or(TensorError::CandleError(candle_core::Error::Msg(
                "empty tensor slice".to_string(),
            )))?
            .as_ref()
            .data
            .read()?
            .device
            .clone();

        let new_inner = TensorInner::Tensor(new_inner_tensor);

        //  Set the parents
        let parents = tensors
            .iter()
            .map(|tensor| tensor.as_ref().copy())
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

    /// Concatenate a sequence of tensors along the specified dimension.
    ///
    /// # Arguments
    /// * `tensors` - The tensors to concatenate.
    /// * `dim` - The dimension along which to concatenate. It must be greater than or equal to `-1`.
    /// When `dim` is equal to `-1`, the tensors are concatenated along the last dimension.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The concatenated tensor.
    /// * `Err(TensorError)` - The error when concatenating the tensors.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::cpu();
    /// let t1 = Tensor::from_data(&[[1.0, 2.0], [5.0, 6.0]], &device, false).unwrap();
    /// let t2 = Tensor::from_data(&[[3.0, 4.0], [7.0, 8.0]], &device, false).unwrap();
    ///
    /// // Concatenate tensors along the first dimension
    /// let concatenated = Tensor::concat(&[t1.copy(), t2.copy()], 0).unwrap();
    /// // Result should be: [[1.0, 2.0], [5.0, 6.0], [3.0, 4.0], [7.0, 8.0]]
    /// let expected = vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0];
    /// assert_eq!(concatenated.to_vec::<f64>().unwrap(), expected);
    /// assert_eq!(concatenated.shape().unwrap(), (&[4, 2]).into());
    ///
    /// // Concatenate tensors along the last dimension
    /// let concatenated = Tensor::concat(&[t1, t2], -1).unwrap();
    /// // Result should be: [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    /// let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    /// assert_eq!(concatenated.to_vec::<f64>().unwrap(), expected);
    /// assert_eq!(concatenated.shape().unwrap(), (&[2, 4]).into());
    /// ```
    pub fn concat<A>(tensors: &[A], dim: isize) -> Result<Self, TensorError>
    where
        A: AsRef<Tensor> + std::clone::Clone,
    {
        let dim: usize = match dim {
            d if d >= 0 => d as usize,
            -1 => tensors[0].as_ref().shape()?.dims().len() as usize - 1,
            _ => return Err(TensorError::InvalidDimension(dim)),
        };

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

        let new_inner_tensor = candle_core::Tensor::cat(&inner_tensors, dim)?;

        let device = tensors
            .first()
            .ok_or(TensorError::CandleError(candle_core::Error::Msg(
                "empty tensor slice".to_string(),
            )))?
            .as_ref()
            .data
            .read()?
            .device
            .clone();

        let new_inner = TensorInner::Tensor(new_inner_tensor);

        let parents = tensors
            .iter()
            .map(|tensor| tensor.as_ref().copy())
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

    /// Concatenate a sequence of tensors along the specified dimension.
    ///
    /// # Arguments
    /// * `tensors` - The tensors to concatenate.
    /// * `dim` - The dimension along which to concatenate. It must be greater than or equal to `-1`.
    /// When `dim` is equal to `-1`, the tensors are concatenated along the last dimension.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The concatenated tensor.
    /// * `Err(TensorError)` - The error when concatenating the tensors.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::cpu();
    /// let t1 = Tensor::from_data(&[[1.0, 2.0], [5.0, 6.0]], &device, false).unwrap();
    /// let t2 = Tensor::from_data(&[[3.0, 4.0], [7.0, 8.0]], &device, false).unwrap();
    ///
    /// // Concatenate tensors along the first dimension
    /// let concatenated = Tensor::cat(&[t1.copy(), t2.copy()], 0).unwrap();
    /// // Result should be: [[1.0, 2.0], [5.0, 6.0], [3.0, 4.0], [7.0, 8.0]]
    /// let expected = vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0];
    /// assert_eq!(concatenated.to_vec::<f64>().unwrap(), expected);
    /// assert_eq!(concatenated.shape().unwrap(), (&[4, 2]).into());
    ///
    /// // Concatenate tensors along the last dimension
    /// let concatenated = Tensor::concat(&[t1, t2], -1).unwrap();
    /// // Result should be: [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    /// let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    /// assert_eq!(concatenated.to_vec::<f64>().unwrap(), expected);
    /// assert_eq!(concatenated.shape().unwrap(), (&[2, 4]).into());
    /// ```
    pub fn cat<A>(tensors: &[A], dim: isize) -> Result<Self, TensorError>
    where
        A: AsRef<Tensor> + std::clone::Clone,
    {
        Self::concat(tensors, dim)
    }

    /// Gather values from the tensor along the specified dimension using the provided indices.
    ///
    /// # Notes
    /// * The data type([`crate::DType`]) of the indices tensor must be i64([`crate::DType::I64`]).
    ///
    /// # Arguments
    /// * `indices` - The tensor with i64 data type([`crate::DType::I64`]) containing the indices to gather.
    /// * `dim` - The dimension along which to gather values. It must be greater than or equal to `-1`.
    /// When `dim` is equal to `-1`, the tensors are gathered along the last dimension.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor with gathered values.
    /// * `Err(TensorError)` - The error when gathering values.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::cpu();
    ///
    /// // Create a 1D tensor and an indices tensor
    /// let t1 = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &device, false).unwrap();
    /// let indices = Tensor::from_data(vec![0i64, 2i64], &device, false).unwrap();
    /// // Gather values along the first dimension
    /// let result = t1.gather(&indices, 0).unwrap();
    /// // Result should be: [1.0, 3.0]
    /// let expected = vec![1.0, 3.0];
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected);
    /// assert_eq!(result.shape().unwrap(), (&[2]).into());
    ///
    /// // Create a 2D tensor and an indices tensor
    /// // For 2D tensor [3, 2] with gather on last dim (dim=1):
    /// // - Input tensor shape: [3, 2]
    /// // - Indices shape must be [3, k] where k is the number of indices per row
    /// let t2d = Tensor::from_data(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]], &device, false).unwrap();
    /// let indices_2d = Tensor::from_data(vec![vec![0i64], vec![1i64], vec![0i64]], &device, false).unwrap();
    /// // Gather values along the last dimension (dim=1)
    /// let result_2d = t2d.gather(&indices_2d, -1).unwrap();
    /// // Result should be: [[1.0], [4.0], [5.0]]
    /// let expected_2d = vec![1.0, 4.0, 5.0];
    /// assert_eq!(result_2d.to_vec::<f64>().unwrap(), expected_2d);
    /// assert_eq!(result_2d.shape().unwrap(), (&[3, 1]).into());
    /// ```
    pub fn gather(&self, indices: &Self, dim: isize) -> Result<Self, TensorError> {
        let dim: usize = match dim {
            d if d >= 0 => d as usize,
            -1 => self.as_ref().shape()?.dims().len() as usize - 1,
            _ => return Err(TensorError::InvalidDimension(dim)),
        };

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
                parents: vec![self.copy()],
                grad: None,
                name: None,
            })),
        })
    }

    pub fn index_select(&self, indices: &Self, dim: isize) -> Result<Self, TensorError> {
        let dim: usize = match dim {
            d if d >= 0 => d as usize,
            -1 => self.as_ref().shape()?.dims().len() as usize - 1,
            _ => return Err(TensorError::InvalidDimension(dim)),
        };

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

        let new_inner = TensorInner::Tensor(inner_tensor.index_select(indices_inner_tensor, dim)?);

        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                device: inner.device.clone(),
                parents: vec![self.copy()],
                grad: None,
                name: None,
            })),
        })
    }

    pub fn embedding(&self, indices: &Self) -> Result<Self, TensorError> {
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

        let new_inner = TensorInner::Tensor(inner_tensor.embedding(indices_inner_tensor)?);

        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                device: inner.device.clone(),
                parents: vec![self.copy()],
                grad: None,
                name: None,
            })),
        })
    }
}
