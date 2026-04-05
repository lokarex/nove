use std::sync::{Arc, RwLock};

use crate::{
    Tensor, TensorError,
    tensor::{TensorData, TensorInner},
};

impl Tensor {
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

    /// Apply the where operation: `condition ? true_value : false_value` element-wise.
    ///
    /// # Arguments
    /// * `condition` - The boolean tensor (dtype U8) where non-zero values indicate true.
    /// * `true_value` - The tensor to select where condition is true.
    /// * `false_value` - The tensor to select where condition is false.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor with selected values.
    /// * `Err(TensorError)` - The error when applying the where operation.
    ///
    /// # Examples
    /// * Apply where operation on 2x3 matrices
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    /// let shape = Shape::from(&[2, 3]);
    ///
    /// let condition = Tensor::from_data(vec![vec![1u8, 0u8, 1u8], vec![0u8, 1u8, 0u8]], &device, false).unwrap();
    /// let true_val = Tensor::from_data(vec![vec![10.0, 20.0, 30.0], vec![40.0, 50.0, 60.0]], &device, false).unwrap();
    /// let false_val = Tensor::from_data(vec![vec![100.0, 200.0, 300.0], vec![400.0, 500.0, 600.0]], &device, false).unwrap();
    ///
    /// let result = Tensor::where_cond(&condition, &true_val, &false_val).unwrap();
    /// // Condition: [[true, false, true], [false, true, false]]
    /// // Result should be: [[10.0, 200.0, 30.0], [400.0, 50.0, 600.0]]
    /// let expected = vec![10.0, 200.0, 30.0, 400.0, 50.0, 600.0];
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected);
    /// assert_eq!(result.shape().unwrap(), (&[2, 3]).into());
    /// ```
    ///
    /// * Backpropagate for where operation
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    /// let shape = Shape::from(&[2, 3]);
    ///
    /// let condition = Tensor::from_data(vec![vec![1u8, 0u8, 1u8], vec![0u8, 1u8, 0u8]], &device, false).unwrap();
    /// let true_val = Tensor::from_data(vec![vec![10.0, 20.0, 30.0], vec![40.0, 50.0, 60.0]], &device, true).unwrap();
    /// let false_val = Tensor::from_data(vec![vec![100.0, 200.0, 300.0], vec![400.0, 500.0, 600.0]], &device, true).unwrap();
    ///
    /// let result = Tensor::where_cond(&condition, &true_val, &false_val).unwrap();
    /// result.backward().unwrap();
    ///
    /// // The gradient of true_val should be 1.0 where condition is true, 0.0 otherwise
    /// let true_grad = true_val.grad().unwrap().unwrap();
    /// let expected_true_grad = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    /// assert_eq!(true_grad.to_vec::<f64>().unwrap(), expected_true_grad);
    /// assert_eq!(true_grad.shape().unwrap(), (&[2, 3]).into());
    ///
    /// // The gradient of false_val should be 1.0 where condition is false, 0.0 otherwise
    /// let false_grad = false_val.grad().unwrap().unwrap();
    /// let expected_false_grad = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    /// assert_eq!(false_grad.to_vec::<f64>().unwrap(), expected_false_grad);
    /// assert_eq!(false_grad.shape().unwrap(), (&[2, 3]).into());
    /// ```
    pub fn where_cond(
        condition: &Self,
        true_value: &Self,
        false_value: &Self,
    ) -> Result<Self, TensorError> {
        let cond_inner = condition.data.read()?;
        let cond_tensor = match &cond_inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };
        let true_inner = true_value.data.read()?;
        let true_tensor = match &true_inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };
        let false_inner = false_value.data.read()?;
        let false_tensor = match &false_inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner_tensor = cond_tensor.where_cond(true_tensor, false_tensor)?;
        let new_inner = TensorInner::Tensor(new_inner_tensor);

        let device = condition.data.read()?.device.clone();

        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                device,
                parents: vec![true_value.copy(), false_value.copy()],
                grad: None,
                name: None,
            })),
        })
    }
}
