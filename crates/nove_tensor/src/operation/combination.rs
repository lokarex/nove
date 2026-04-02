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
    ///   When `dim` is equal to `-1`, the tensors are stacked along the last dimension.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor after stacking.
    /// * `Err(TensorError)` - The error when stacking the tensors.
    ///
    /// # Examples
    /// * Stack 1D tensors along the first dimension
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    /// let t1 = Tensor::from_data(vec![1.0, 2.0], &device, false).unwrap();
    /// let t2 = Tensor::from_data(vec![3.0, 4.0], &device, false).unwrap();
    /// let t3 = Tensor::from_data(vec![5.0, 6.0], &device, false).unwrap();
    ///
    /// let result = Tensor::stack(&[t1.copy(), t2.copy(), t3.copy()], 0).unwrap();
    /// // Result should be: [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    /// let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected);
    /// assert_eq!(result.shape().unwrap(), (&[3, 2]).into());
    /// ```
    ///
    /// * Stack 1D tensors along the last dimension
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    /// let t1 = Tensor::from_data(vec![1.0, 2.0], &device, false).unwrap();
    /// let t2 = Tensor::from_data(vec![3.0, 4.0], &device, false).unwrap();
    /// let t3 = Tensor::from_data(vec![5.0, 6.0], &device, false).unwrap();
    ///
    /// let result = Tensor::stack(&[t1, t2, t3], -1).unwrap();
    /// // Result should be: [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]
    /// let expected = vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0];
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected);
    /// assert_eq!(result.shape().unwrap(), (&[2, 3]).into());
    /// ```
    ///
    /// * Backpropagate for stack operation
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    /// let t1 = Tensor::from_data(vec![1.0, 2.0], &device, true).unwrap();
    /// let t2 = Tensor::from_data(vec![3.0, 4.0], &device, true).unwrap();
    /// let t3 = Tensor::from_data(vec![5.0, 6.0], &device, true).unwrap();
    ///
    /// let result = Tensor::stack(&[t1.copy(), t2.copy(), t3.copy()], 0).unwrap();
    /// result.backward().unwrap();
    ///
    /// let t1_grad = t1.grad().unwrap().unwrap();
    /// assert_eq!(t1_grad.to_vec::<f64>().unwrap(), vec![1.0, 1.0]);
    /// assert_eq!(t1_grad.shape().unwrap(), (&[2]).into());
    ///
    /// let t2_grad = t2.grad().unwrap().unwrap();
    /// assert_eq!(t2_grad.to_vec::<f64>().unwrap(), vec![1.0, 1.0]);
    /// assert_eq!(t2_grad.shape().unwrap(), (&[2]).into());
    ///
    /// let t3_grad = t3.grad().unwrap().unwrap();
    /// assert_eq!(t3_grad.to_vec::<f64>().unwrap(), vec![1.0, 1.0]);
    /// assert_eq!(t3_grad.shape().unwrap(), (&[2]).into());
    /// ```
    pub fn stack<A>(tensors: &[A], dim: isize) -> Result<Self, TensorError>
    where
        A: AsRef<Tensor> + std::clone::Clone,
    {
        let dim: usize = match dim {
            d if d >= 0 => d as usize,
            -1 => tensors[0].as_ref().shape()?.dims().len(),
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
    ///   When `dim` is equal to `-1`, the tensors are concatenated along the last dimension.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The concatenated tensor.
    /// * `Err(TensorError)` - The error when concatenating the tensors.
    ///
    /// # Examples
    /// * Concatenate 2D tensors along the first dimension
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    /// let t1 = Tensor::from_data(&[[1.0, 2.0], [5.0, 6.0]], &device, false).unwrap();
    /// let t2 = Tensor::from_data(&[[3.0, 4.0], [7.0, 8.0]], &device, false).unwrap();
    ///
    /// let result = Tensor::concat(&[t1.copy(), t2.copy()], 0).unwrap();
    /// // Result should be: [[1.0, 2.0], [5.0, 6.0], [3.0, 4.0], [7.0, 8.0]]
    /// let expected = vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0];
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected);
    /// assert_eq!(result.shape().unwrap(), (&[4, 2]).into());
    /// ```
    ///
    /// * Concatenate 2D tensors along the last dimension
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    /// let t1 = Tensor::from_data(&[[1.0, 2.0], [5.0, 6.0]], &device, false).unwrap();
    /// let t2 = Tensor::from_data(&[[3.0, 4.0], [7.0, 8.0]], &device, false).unwrap();
    ///
    /// let result = Tensor::concat(&[t1, t2], -1).unwrap();
    /// // Result should be: [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    /// let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected);
    /// assert_eq!(result.shape().unwrap(), (&[2, 4]).into());
    /// ```
    ///
    /// * Backpropagate for concat operation
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    /// let t1 = Tensor::from_data(&[[1.0, 2.0], [5.0, 6.0]], &device, true).unwrap();
    /// let t2 = Tensor::from_data(&[[3.0, 4.0], [7.0, 8.0]], &device, true).unwrap();
    ///
    /// let result = Tensor::concat(&[t1.copy(), t2.copy()], 0).unwrap();
    /// result.backward().unwrap();
    ///
    /// let t1_grad = t1.grad().unwrap().unwrap();
    /// assert_eq!(t1_grad.to_vec::<f64>().unwrap(), vec![1.0, 1.0, 1.0, 1.0]);
    /// assert_eq!(t1_grad.shape().unwrap(), (&[2, 2]).into());
    ///
    /// let t2_grad = t2.grad().unwrap().unwrap();
    /// assert_eq!(t2_grad.to_vec::<f64>().unwrap(), vec![1.0, 1.0, 1.0, 1.0]);
    /// assert_eq!(t2_grad.shape().unwrap(), (&[2, 2]).into());
    /// ```
    pub fn concat<A>(tensors: &[A], dim: isize) -> Result<Self, TensorError>
    where
        A: AsRef<Tensor> + std::clone::Clone,
    {
        let dim: usize = match dim {
            d if d >= 0 => d as usize,
            -1 => tensors[0].as_ref().shape()?.dims().len() - 1,
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
    ///   When `dim` is equal to `-1`, the tensors are concatenated along the last dimension.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The concatenated tensor.
    /// * `Err(TensorError)` - The error when concatenating the tensors.
    ///
    /// # Examples
    /// * Concatenate 2D tensors along the first dimension
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    /// let t1 = Tensor::from_data(&[[1.0, 2.0], [5.0, 6.0]], &device, false).unwrap();
    /// let t2 = Tensor::from_data(&[[3.0, 4.0], [7.0, 8.0]], &device, false).unwrap();
    ///
    /// let result = Tensor::cat(&[t1.copy(), t2.copy()], 0).unwrap();
    /// // Result should be: [[1.0, 2.0], [5.0, 6.0], [3.0, 4.0], [7.0, 8.0]]
    /// let expected = vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0];
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected);
    /// assert_eq!(result.shape().unwrap(), (&[4, 2]).into());
    /// ```
    ///
    /// * Concatenate 2D tensors along the last dimension
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    /// let t1 = Tensor::from_data(&[[1.0, 2.0], [5.0, 6.0]], &device, false).unwrap();
    /// let t2 = Tensor::from_data(&[[3.0, 4.0], [7.0, 8.0]], &device, false).unwrap();
    ///
    /// let result = Tensor::cat(&[t1, t2], -1).unwrap();
    /// // Result should be: [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    /// let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected);
    /// assert_eq!(result.shape().unwrap(), (&[2, 4]).into());
    /// ```
    ///
    /// * Backpropagate for cat operation
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    /// let t1 = Tensor::from_data(&[[1.0, 2.0], [5.0, 6.0]], &device, true).unwrap();
    /// let t2 = Tensor::from_data(&[[3.0, 4.0], [7.0, 8.0]], &device, true).unwrap();
    ///
    /// let result = Tensor::cat(&[t1.copy(), t2.copy()], 0).unwrap();
    /// result.backward().unwrap();
    ///
    /// let t1_grad = t1.grad().unwrap().unwrap();
    /// assert_eq!(t1_grad.to_vec::<f64>().unwrap(), vec![1.0, 1.0, 1.0, 1.0]);
    /// assert_eq!(t1_grad.shape().unwrap(), (&[2, 2]).into());
    ///
    /// let t2_grad = t2.grad().unwrap().unwrap();
    /// assert_eq!(t2_grad.to_vec::<f64>().unwrap(), vec![1.0, 1.0, 1.0, 1.0]);
    /// assert_eq!(t2_grad.shape().unwrap(), (&[2, 2]).into());
    /// ```
    pub fn cat<A>(tensors: &[A], dim: isize) -> Result<Self, TensorError>
    where
        A: AsRef<Tensor> + std::clone::Clone,
    {
        Self::concat(tensors, dim)
    }

    /// Gather values from the tensor along the specified dimension using the provided indexes.
    ///
    /// # Notes
    /// * The data type([`crate::DType`]) of the indexes tensor must be i64([`crate::DType::I64`]).
    ///
    /// # Arguments
    /// * `indexes` - The tensor with i64 data type([`crate::DType::I64`]) containing the indexes to gather.
    /// * `dim` - The dimension along which to gather values. It must be greater than or equal to `-1`.
    ///   When `dim` is equal to `-1`, the tensors are gathered along the last dimension.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor with gathered values.
    /// * `Err(TensorError)` - The error when gathering values.
    ///
    /// # Examples
    /// * Gather values from 1D tensor along the first dimension
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    /// let t1 = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &device, false).unwrap();
    /// let indexes = Tensor::from_data(vec![0i64, 2i64], &device, false).unwrap();
    ///
    /// let result = t1.gather(&indexes, 0).unwrap();
    /// // Result should be: [1.0, 3.0]
    /// let expected = vec![1.0, 3.0];
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected);
    /// assert_eq!(result.shape().unwrap(), (&[2]).into());
    /// ```
    ///
    /// * Gather values from 2D tensor along the last dimension
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    /// let t2d = Tensor::from_data(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]], &device, false).unwrap();
    /// let indexes_2d = Tensor::from_data(vec![vec![0i64], vec![1i64], vec![0i64]], &device, false).unwrap();
    ///
    /// let result = t2d.gather(&indexes_2d, -1).unwrap();
    /// // Result should be: [[1.0], [4.0], [5.0]]
    /// let expected = vec![1.0, 4.0, 5.0];
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected);
    /// assert_eq!(result.shape().unwrap(), (&[3, 1]).into());
    /// ```
    ///
    /// * Backpropagate for gather operation
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    /// let t1 = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &device, true).unwrap();
    /// let indexes = Tensor::from_data(vec![0i64, 2i64], &device, false).unwrap();
    ///
    /// let result = t1.gather(&indexes, 0).unwrap();
    /// result.backward().unwrap();
    ///
    /// let grad = t1.grad().unwrap().unwrap();
    /// assert_eq!(grad.to_vec::<f64>().unwrap(), vec![1.0, 0.0, 1.0, 0.0]);
    /// assert_eq!(grad.shape().unwrap(), (&[4]).into());
    /// ```
    pub fn gather(&self, indexes: &Self, dim: isize) -> Result<Self, TensorError> {
        let dim: usize = match dim {
            d if d >= 0 => d as usize,
            -1 => self.as_ref().shape()?.dims().len() - 1,
            _ => return Err(TensorError::InvalidDimension(dim)),
        };

        let inner = self.data.read()?;
        let inner_tensor = match &inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let indexes_inner = indexes.data.read()?;
        let indexes_inner_tensor = match &indexes_inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = TensorInner::Tensor(inner_tensor.gather(indexes_inner_tensor, dim)?);

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

    /// Select values from the tensor along the specified dimension using the provided indexes.
    ///
    /// # Notes
    /// * The data type([`crate::DType`]) of the indexes tensor must be i64([`crate::DType::I64`]).
    ///
    /// # Arguments
    /// * `indexes` - The tensor with i64 data type([`crate::DType::I64`]) containing the indexes to select.
    /// * `dim` - The dimension along which to select values. It must be greater than or equal to `-1`.
    ///   When `dim` is equal to `-1`, the values are selected along the last dimension.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor with selected values.
    /// * `Err(TensorError)` - The error when selecting values.
    ///
    /// # Examples
    /// * Select values from 2D tensor along the first dimension
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    /// let t = Tensor::from_data(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], &device, false).unwrap();
    /// let indexes = Tensor::from_data(vec![0i64, 2i64], &device, false).unwrap();
    ///
    /// let result = t.index_select(&indexes, 0).unwrap();
    /// // Result should be: [[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]]
    /// let expected = vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0];
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected);
    /// assert_eq!(result.shape().unwrap(), (&[2, 3]).into());
    /// ```
    ///
    /// * Select values from 2D tensor along the last dimension
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    /// let t = Tensor::from_data(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device, false).unwrap();
    /// let indexes = Tensor::from_data(vec![0i64, 2i64], &device, false).unwrap();
    ///
    /// let result = t.index_select(&indexes, -1).unwrap();
    /// // Result should be: [[1.0, 3.0], [4.0, 6.0]]
    /// let expected = vec![1.0, 3.0, 4.0, 6.0];
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected);
    /// assert_eq!(result.shape().unwrap(), (&[2, 2]).into());
    /// ```
    ///
    /// * Backpropagate for index_select operation
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    /// let t = Tensor::from_data(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], &device, true).unwrap();
    /// let indexes = Tensor::from_data(vec![0i64, 2i64], &device, false).unwrap();
    ///
    /// let result = t.index_select(&indexes, 0).unwrap();
    /// result.backward().unwrap();
    ///
    /// let grad = t.grad().unwrap().unwrap();
    /// assert_eq!(grad.to_vec::<f64>().unwrap(), vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    /// assert_eq!(grad.shape().unwrap(), (&[3, 3]).into());
    /// ```
    pub fn index_select(&self, indexes: &Self, dim: isize) -> Result<Self, TensorError> {
        let dim: usize = match dim {
            d if d >= 0 => d as usize,
            -1 => self.as_ref().shape()?.dims().len() - 1,
            _ => return Err(TensorError::InvalidDimension(dim)),
        };

        let inner = self.data.read()?;
        let inner_tensor = match &inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let indexes_inner = indexes.data.read()?;
        let indexes_inner_tensor = match &indexes_inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = TensorInner::Tensor(inner_tensor.index_select(indexes_inner_tensor, dim)?);

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

    /// Look up embeddings from a fixed dictionary and size.
    ///
    /// # Notes
    /// * The data type([`crate::DType`]) of the indexes tensor must be i64([`crate::DType::I64`]).
    /// * The embedding table must be a 2D tensor with shape [num_embeddings, embedding_dim].
    /// * The indexes must be a 1D tensor with shape [batch_size].
    ///
    /// # Arguments
    /// * `indexes` - The tensor with i64 data type([`crate::DType::I64`]) containing the indexes to look up.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor with the embeddings.
    /// * `Err(TensorError)` - The error when looking up embeddings.
    ///
    /// # Examples
    /// * Look up embeddings from a dictionary
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    /// let embedding_table = Tensor::from_data(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], &device, false).unwrap();
    /// let indexes = Tensor::from_data(vec![0i64, 2i64, 1i64], &device, false).unwrap();
    ///
    /// let result = embedding_table.embedding(&indexes).unwrap();
    /// // Result should be: [[1.0, 2.0], [5.0, 6.0], [3.0, 4.0]]
    /// let expected = vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0];
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected);
    /// assert_eq!(result.shape().unwrap(), (&[3, 2]).into());
    /// ```
    ///
    /// * Look up a single embedding
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    /// let embedding_table = Tensor::from_data(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], &device, false).unwrap();
    /// let indexes = Tensor::from_data(vec![1i64], &device, false).unwrap();
    ///
    /// let result = embedding_table.embedding(&indexes).unwrap();
    /// // Result should be: [[3.0, 4.0]]
    /// let expected = vec![3.0, 4.0];
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected);
    /// assert_eq!(result.shape().unwrap(), (&[1, 2]).into());
    /// ```
    ///
    /// * Backpropagate for embedding operation
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    /// let embedding_table = Tensor::from_data(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], &device, true).unwrap();
    /// let indexes = Tensor::from_data(vec![0i64, 2i64], &device, false).unwrap();
    ///
    /// let result = embedding_table.embedding(&indexes).unwrap();
    /// result.backward().unwrap();
    ///
    /// let grad = embedding_table.grad().unwrap().unwrap();
    /// assert_eq!(grad.to_vec::<f64>().unwrap(), vec![1.0, 1.0, 0.0, 0.0, 1.0, 1.0]);
    /// assert_eq!(grad.shape().unwrap(), (&[3, 2]).into());
    /// ```
    pub fn embedding(&self, indexes: &Self) -> Result<Self, TensorError> {
        let inner = self.data.read()?;
        let inner_tensor = match &inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let indexes_inner = indexes.data.read()?;
        let indexes_inner_tensor = match &indexes_inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = TensorInner::Tensor(inner_tensor.embedding(indexes_inner_tensor)?);

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
