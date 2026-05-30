use crate::{Tensor, TensorError, backend::BackendStorage, backpropagation::graph::OpKind};

impl Tensor {
    /// Stack a list of tensors along a new dimension.
    ///
    /// # Arguments
    /// * `tensors` - The list of tensors to stack.
    /// * `dim` - The dimension along which to stack the tensors (0-based).
    ///   Supports negative indexing: -1 means the last dimension, -2 means second last, etc.
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
    /// let device = Device::default();
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
    /// let device = Device::default();
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
    /// let device = Device::default();
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
        A: AsRef<Tensor> + Clone,
    {
        let first = tensors
            .first()
            .ok_or_else(|| TensorError::InvalidOperation("empty tensor slice".to_string()))?
            .as_ref();
        let num_dims = first.shape()?.rank();
        let dim = if dim < 0 {
            (num_dims as isize + dim + 1) as usize
        } else {
            dim as usize
        };

        let storages = tensors
            .iter()
            .map(|tensor| tensor.as_ref().backend_storage())
            .collect::<Result<Vec<_>, TensorError>>()?;
        let storage = BackendStorage::stack(&storages, dim)?;
        let input_shapes = tensors
            .iter()
            .map(|tensor| tensor.as_ref().shape())
            .collect::<Result<Vec<_>, TensorError>>()?;
        let parents = tensors
            .iter()
            .map(|tensor| tensor.as_ref().copy())
            .collect::<Vec<_>>();

        Ok(Self::op_result_with_kind(
            storage,
            first.device()?,
            parents,
            OpKind::Stack { dim, input_shapes },
        ))
    }

    /// Concatenate a sequence of tensors along the specified dimension.
    ///
    /// # Arguments
    /// * `tensors` - The tensors to concatenate.
    /// * `dim` - The dimension along which to concatenate (0-based).
    ///   Supports negative indexing: -1 means the last dimension, -2 means second last, etc.
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
    /// let device = Device::default();
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
    /// let device = Device::default();
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
    /// let device = Device::default();
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
        A: AsRef<Tensor> + Clone,
    {
        let first = tensors
            .first()
            .ok_or_else(|| TensorError::InvalidOperation("empty tensor slice".to_string()))?
            .as_ref();
        let num_dims = first.shape()?.rank();
        let dim = if dim < 0 {
            (num_dims as isize + dim) as usize
        } else {
            dim as usize
        };

        let storages = tensors
            .iter()
            .map(|tensor| tensor.as_ref().backend_storage())
            .collect::<Result<Vec<_>, TensorError>>()?;
        let storage = BackendStorage::cat(&storages, dim)?;
        let input_shapes = tensors
            .iter()
            .map(|tensor| tensor.as_ref().shape())
            .collect::<Result<Vec<_>, TensorError>>()?;
        let parents = tensors
            .iter()
            .map(|tensor| tensor.as_ref().copy())
            .collect::<Vec<_>>();

        Ok(Self::op_result_with_kind(
            storage,
            first.device()?,
            parents,
            OpKind::Cat { dim, input_shapes },
        ))
    }

    /// Concatenate a sequence of tensors along the specified dimension.
    ///
    /// This is an alias for [`Tensor::concat`].
    ///
    /// # Arguments
    /// * `tensors` - The tensors to concatenate.
    /// * `dim` - The dimension along which to concatenate (0-based).
    ///   Supports negative indexing: -1 means the last dimension, -2 means second last, etc.
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
    /// let device = Device::default();
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
    /// let device = Device::default();
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
    /// let device = Device::default();
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
        A: AsRef<Tensor> + Clone,
    {
        Self::concat(tensors, dim)
    }
}
