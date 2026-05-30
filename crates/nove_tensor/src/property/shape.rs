use crate::{Shape, Tensor, TensorError, backpropagation::graph::OpKind};

impl Tensor {
    /// Create a new tensor like the current tensor with the specified shape.
    ///
    /// # Notes
    /// * In the nove framework, gradients of intermediate Tensors are not stored.
    ///   When `reshape` is called on a Tensor with `requires_grad=true`, it returns an intermediate Tensor whose `grad` is always `None`.
    ///
    /// # Arguments
    /// * `shape` - The shape to reshape the tensor to.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The new tensor with the specified shape.
    /// * `Err(TensorError)` - The error when reshaping the tensor.
    ///
    /// # Examples
    /// * Reshape 1D tensor to 2D
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::default();
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32, 3.0f32, 4.0f32], &device, false).unwrap();
    ///
    /// let result = tensor.reshape(&Shape::from(&[2, 2])).unwrap();
    /// assert_eq!(result.shape().unwrap(), (&[2, 2]).into());
    /// assert_eq!(result.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    /// ```
    ///
    /// * Reshape 1D tensor to column vector
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::default();
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32, 3.0f32], &device, false).unwrap();
    ///
    /// let result = tensor.reshape(&Shape::from(&[3, 1])).unwrap();
    /// assert_eq!(result.shape().unwrap(), (&[3, 1]).into());
    /// assert_eq!(result.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
    /// ```
    ///
    /// * Backpropagate through reshape with gradient
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::default();
    ///
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32, 3.0f32, 4.0f32], &device, true).unwrap();
    /// let result = tensor.reshape(&Shape::from(&[2, 2])).unwrap();
    /// result.backward().unwrap();
    /// let grad = tensor.grad().unwrap().unwrap();
    /// assert_eq!(grad.shape().unwrap(), (&[4]).into());
    /// assert_eq!(grad.to_vec::<f32>().unwrap(), vec![1.0, 1.0, 1.0, 1.0]);
    ///
    /// // The following Tensor is actually an intermediate Tensor, its grad is always None
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32, 3.0f32, 4.0f32], &device, true).unwrap().reshape(&Shape::from(&[2, 2])).unwrap();
    /// tensor.backward().unwrap();
    /// assert!(tensor.grad().unwrap().is_none());
    ///
    /// // Keep a handle to the source tensor when chaining operations that should receive gradients.
    /// let source = Tensor::from_data(&[1.0f32, 2.0f32, 3.0f32, 4.0f32], &device, true).unwrap();
    /// let tensor = source.reshape(&Shape::from(&[2, 2])).unwrap();
    /// tensor.backward().unwrap();
    /// let grad = source.grad().unwrap().unwrap();
    /// assert_eq!(grad.shape().unwrap(), (&[4]).into());
    /// assert_eq!(grad.to_vec::<f32>().unwrap(), vec![1.0, 1.0, 1.0, 1.0]);
    /// ```
    pub fn reshape(&self, shape: &Shape) -> Result<Tensor, TensorError> {
        let data = self.data.read()?;
        let storage = data.storage.reshape(shape)?;
        let grad = match &data.grad {
            Some(grad) => Some(grad.reshape(shape)?),
            None => None,
        };
        let device = data.device.clone();
        let name = data.name.clone();
        let requires_grad = data.requires_grad;
        drop(data);

        Ok(Tensor::from_backend_parts(
            storage,
            device,
            requires_grad,
            vec![self.copy()],
            OpKind::Reshape {
                from: self.shape()?,
                to: shape.clone(),
            },
            grad,
            name,
        ))
    }

    /// Get the shape of the tensor.
    ///
    /// # Returns
    /// * `Ok(shape)` - The shape of the tensor.
    /// * `Err(TensorError)` - The error when getting the shape of the tensor.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::default();
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32, 3.0f32], &device, false).unwrap();
    ///
    /// let shape = tensor.shape().unwrap();
    /// assert_eq!(shape, (&[3]).into());
    /// ```
    pub fn shape(&self) -> Result<Shape, TensorError> {
        Ok(self.data.read()?.storage.shape()?)
    }

    /// Get the number of dimensions of the tensor.
    ///
    /// # Returns
    /// * `Ok(num_dim)` - The number of dimensions of the tensor.
    /// * `Err(TensorError)` - The error when getting the number of dimensions of the tensor.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::default();
    ///
    /// // 1-dimensional tensor (vector)
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32, 3.0f32], &device, false).unwrap();
    /// let num_dim = tensor.num_dim().unwrap();
    /// assert_eq!(num_dim, 1);
    ///
    /// // 2-dimensional tensor (matrix)
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32, 3.0f32, 4.0f32], &device, false).unwrap();
    /// let result = tensor.reshape(&(&[2, 2]).into()).unwrap();
    /// let num_dim = result.num_dim().unwrap();
    /// assert_eq!(num_dim, 2);
    ///
    /// // 3-dimensional tensor
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32], &device, false).unwrap();
    /// let result = tensor.reshape(&(&[2, 2, 2]).into()).unwrap();
    /// let num_dim = result.num_dim().unwrap();
    /// assert_eq!(num_dim, 3);
    ///
    /// // 4-dimensional tensor
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32], &device, false).unwrap();
    /// let result = tensor.reshape(&(&[2, 2, 1, 2]).into()).unwrap();
    /// let num_dim = result.num_dim().unwrap();
    /// assert_eq!(num_dim, 4);
    /// ```
    pub fn num_dim(&self) -> Result<usize, TensorError> {
        Ok(self.shape()?.rank())
    }

    /// Broadcast the tensor to the specified shape.
    ///
    /// # Arguments
    /// * `shape` - The shape to broadcast the tensor to.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The broadcasted tensor if successful.
    /// * `Err(TensorError)` - The error when broadcasting the tensor.
    ///
    /// # Examples
    /// * Forward pass with shape and value verification
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::default();
    ///
    /// let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &device, false).unwrap();
    /// let shape = Shape::from_dims(&[2, 4]);
    ///
    /// let result = t.broadcast(&shape).unwrap();
    /// assert_eq!(result.shape().unwrap(), Shape::from_dims(&[2, 4]));
    /// assert_eq!(result.to_vec::<f64>().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]);
    /// ```
    ///
    /// * Backward pass with gradient verification
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::default();
    ///
    /// let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &device, true).unwrap();
    ///
    /// let result = t.broadcast(&Shape::from_dims(&[2, 4])).unwrap();
    /// result.backward().unwrap();
    ///
    /// let grad = t.grad().unwrap().unwrap();
    /// assert_eq!(grad.shape().unwrap(), Shape::from_dims(&[4]));
    /// assert_eq!(grad.to_vec::<f64>().unwrap(), vec![2.0, 2.0, 2.0, 2.0]);
    /// ```
    pub fn broadcast(&self, shape: &Shape) -> Result<Self, TensorError> {
        let data = self.data.read()?;
        let storage = data.storage.broadcast_as(shape)?;
        let device = data.device.clone();
        drop(data);
        Ok(Self::op_result_with_kind(
            storage,
            device,
            vec![self.copy()],
            OpKind::BroadcastAs {
                from: self.shape()?,
                to: shape.clone(),
            },
        ))
    }

    /// Flatten the tensor by merging multiple dimensions into one.
    ///
    /// # Arguments
    /// * `start_dim` - Optional starting dimension index to begin flattening (0-based).
    ///   If `None`, flattening starts from the first dimension (0).
    ///   Supports negative indexing: -1 means the last dimension, -2 means second last, etc.
    /// * `end_dim` - Optional ending dimension index to stop flattening (0-based, inclusive).
    ///   If `None`, flattening continues to the last dimension.
    ///   Supports negative indexing: -1 means the last dimension, -2 means second last, etc.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The flattened tensor.
    /// * `Err(TensorError)` - The error when flattening the tensor.
    ///
    /// # Examples
    /// * Forward pass with shape and value verification
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::default();
    ///
    /// let t = Tensor::from_data(vec![vec![1.0, 2.0], vec![3.0, 4.0]], &device, false).unwrap();
    ///
    /// let result = t.flatten(None, None).unwrap();
    /// assert_eq!(result.shape().unwrap(), Shape::from_dims(&[4]));
    /// assert_eq!(result.to_vec::<f64>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    /// ```
    ///
    /// * Forward pass with start_dim
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::default();
    ///
    /// let t = Tensor::from_data(vec![vec![1.0, 2.0], vec![3.0, 4.0]], &device, false).unwrap();
    ///
    /// let result = t.flatten(Some(1), None).unwrap();
    /// assert_eq!(result.shape().unwrap(), Shape::from_dims(&[2, 2]));
    /// assert_eq!(result.to_vec::<f64>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    /// ```
    ///
    /// * Forward pass with negative indices
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::default();
    ///
    /// let t = Tensor::from_data(vec![vec![1.0, 2.0], vec![3.0, 4.0]], &device, false).unwrap();
    ///
    /// let result = t.flatten(Some(0), Some(-1)).unwrap();
    /// assert_eq!(result.shape().unwrap(), Shape::from_dims(&[4]));
    /// assert_eq!(result.to_vec::<f64>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    /// ```
    ///
    /// * Backward pass with gradient verification
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::default();
    ///
    /// let t = Tensor::from_data(vec![vec![1.0, 2.0], vec![3.0, 4.0]], &device, true).unwrap();
    ///
    /// let result = t.flatten(None, None).unwrap();
    /// result.backward().unwrap();
    ///
    /// let grad = t.grad().unwrap().unwrap();
    /// assert_eq!(grad.shape().unwrap(), Shape::from_dims(&[2, 2]));
    /// assert_eq!(grad.to_vec::<f64>().unwrap(), vec![1.0, 1.0, 1.0, 1.0]);
    /// ```
    pub fn flatten(
        &self,
        start_dim: Option<isize>,
        end_dim: Option<isize>,
    ) -> Result<Self, TensorError> {
        let num_dims = self.num_dim()?;
        let input_shape = self.shape()?;
        let storage = self.backend_storage()?;

        let start = start_dim.map(|dim| normalize_dim(dim, num_dims));
        let end = end_dim.map(|dim| normalize_dim(dim, num_dims));

        let storage = match (start, end) {
            (None, None) => storage.flatten_all()?,
            (Some(start_dim), None) => storage.flatten_from(start_dim)?,
            (None, Some(end_dim)) => storage.flatten_to(end_dim)?,
            (Some(start_dim), Some(end_dim)) => storage.flatten(start_dim, end_dim)?,
        };

        let output_shape = storage.shape()?;
        Ok(Self::op_result_with_kind(
            storage,
            self.device()?,
            vec![self.copy()],
            OpKind::Reshape {
                from: input_shape,
                to: output_shape,
            },
        ))
    }

    /// Remove dimensions of size 1 from the tensor.
    ///
    /// # Arguments
    /// * `dim` - Optional dimension to squeeze. If `None`, all dimensions of size 1 are removed.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The squeezed tensor.
    /// * `Err(TensorError)` - The error when squeezing the tensor.
    ///
    /// # Examples
    /// * Forward pass with shape and value verification
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::default();
    ///
    /// let t = Tensor::from_data(vec![vec![1.0], vec![2.0]], &device, false).unwrap();
    ///
    /// let result = t.squeeze(None).unwrap();
    /// assert_eq!(result.shape().unwrap(), Shape::from_dims(&[2]));
    /// assert_eq!(result.to_vec::<f64>().unwrap(), vec![1.0, 2.0]);
    /// ```
    ///
    /// * Forward pass with specific dimension
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::default();
    ///
    /// let t = Tensor::from_data(vec![vec![1.0, 2.0]], &device, false).unwrap();
    ///
    /// let result = t.squeeze(Some(0)).unwrap();
    /// assert_eq!(result.shape().unwrap(), Shape::from_dims(&[2]));
    /// assert_eq!(result.to_vec::<f64>().unwrap(), vec![1.0, 2.0]);
    /// ```
    ///
    /// * Backward pass with gradient verification
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::default();
    ///
    /// let t = Tensor::from_data(vec![vec![1.0], vec![2.0]], &device, true).unwrap();
    ///
    /// let result = t.squeeze(None).unwrap();
    /// result.backward().unwrap();
    ///
    /// let grad = t.grad().unwrap().unwrap();
    /// assert_eq!(grad.shape().unwrap(), Shape::from_dims(&[2, 1]));
    /// assert_eq!(grad.to_vec::<f64>().unwrap(), vec![1.0, 1.0]);
    /// ```
    pub fn squeeze(&self, dim: Option<usize>) -> Result<Self, TensorError> {
        let input_shape = self.shape()?;
        let storage = self.backend_storage()?;
        let storage = match dim {
            Some(dim) => storage.squeeze(dim)?,
            None => {
                let mut result = storage;
                let dims = result.shape()?.dims().to_vec();
                for (index, dim_size) in dims.iter().enumerate().rev() {
                    if *dim_size == 1 {
                        result = result.squeeze(index)?;
                    }
                }
                result
            }
        };

        let output_shape = storage.shape()?;
        Ok(Self::op_result_with_kind(
            storage,
            self.device()?,
            vec![self.copy()],
            OpKind::Reshape {
                from: input_shape,
                to: output_shape,
            },
        ))
    }

    /// Add a dimension of size 1 at the specified position.
    ///
    /// # Arguments
    /// * `dim` - The dimension index to insert.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The tensor with added dimension.
    /// * `Err(TensorError)` - The error when unsqueezing the tensor.
    ///
    /// # Examples
    /// * Forward pass with shape and value verification
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::default();
    ///
    /// let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &device, false).unwrap();
    ///
    /// let result = t.unsqueeze(0).unwrap();
    /// assert_eq!(result.shape().unwrap(), Shape::from_dims(&[1, 4]));
    /// assert_eq!(result.to_vec::<f64>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    /// ```
    ///
    /// * Forward pass at different dimension
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::default();
    ///
    /// let t = Tensor::from_data(vec![1.0, 2.0], &device, false).unwrap();
    ///
    /// let result = t.unsqueeze(1).unwrap();
    /// assert_eq!(result.shape().unwrap(), Shape::from_dims(&[2, 1]));
    /// assert_eq!(result.to_vec::<f64>().unwrap(), vec![1.0, 2.0]);
    /// ```
    ///
    /// * Backward pass with gradient verification
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::default();
    ///
    /// let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &device, true).unwrap();
    ///
    /// let result = t.unsqueeze(0).unwrap();
    /// result.backward().unwrap();
    ///
    /// let grad = t.grad().unwrap().unwrap();
    /// assert_eq!(grad.shape().unwrap(), Shape::from_dims(&[4]));
    /// assert_eq!(grad.to_vec::<f64>().unwrap(), vec![1.0, 1.0, 1.0, 1.0]);
    /// ```
    pub fn unsqueeze(&self, dim: usize) -> Result<Self, TensorError> {
        let input_shape = self.shape()?;
        let storage = self.backend_storage()?.unsqueeze(dim)?;
        let output_shape = storage.shape()?;
        Ok(Self::op_result_with_kind(
            storage,
            self.device()?,
            vec![self.copy()],
            OpKind::Reshape {
                from: input_shape,
                to: output_shape,
            },
        ))
    }

    /// Transpose the tensor by swapping the two specified dimensions.
    ///
    /// # Arguments
    /// * `dim0` - The first dimension to swap (0-based). Supports negative indexing: -1 means the last dimension, -2 means second last, etc.
    /// * `dim1` - The second dimension to swap (0-based). Supports negative indexing: -1 means the last dimension, -2 means second last, etc.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The transposed tensor.
    /// * `Err(TensorError)` - The error when transposing the tensor.
    ///
    /// # Examples
    /// * Forward pass with shape and value verification
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::default();
    ///
    /// let t = Tensor::from_data(vec![vec![1.0, 2.0], vec![3.0, 4.0]], &device, false).unwrap();
    ///
    /// let result = t.transpose(0, 1).unwrap();
    /// assert_eq!(result.shape().unwrap(), Shape::from_dims(&[2, 2]));
    /// assert_eq!(result.to_vec::<f64>().unwrap(), vec![1.0, 3.0, 2.0, 4.0]);
    /// ```
    ///
    /// * Forward pass with negative indices
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::default();
    ///
    /// let t = Tensor::from_data(vec![vec![1.0, 2.0], vec![3.0, 4.0]], &device, false).unwrap();
    ///
    /// let result = t.transpose(-2, -1).unwrap();
    /// assert_eq!(result.shape().unwrap(), Shape::from_dims(&[2, 2]));
    /// assert_eq!(result.to_vec::<f64>().unwrap(), vec![1.0, 3.0, 2.0, 4.0]);
    /// ```
    ///
    /// * Backward pass with gradient verification
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::default();
    ///
    /// let t = Tensor::from_data(vec![vec![1.0, 2.0], vec![3.0, 4.0]], &device, true).unwrap();
    ///
    /// let result = t.transpose(0, 1).unwrap();
    /// result.backward().unwrap();
    ///
    /// let grad = t.grad().unwrap().unwrap();
    /// assert_eq!(grad.shape().unwrap(), Shape::from_dims(&[2, 2]));
    /// assert_eq!(grad.to_vec::<f64>().unwrap(), vec![1.0, 1.0, 1.0, 1.0]);
    /// ```
    pub fn transpose(&self, dim0: isize, dim1: isize) -> Result<Self, TensorError> {
        let num_dims = self.num_dim()?;
        let dim0 = normalize_dim(dim0, num_dims);
        let dim1 = normalize_dim(dim1, num_dims);
        let storage = self.backend_storage()?.transpose(dim0, dim1)?;
        Ok(Self::op_result_with_kind(
            storage,
            self.device()?,
            vec![self.copy()],
            OpKind::Transpose { dim0, dim1 },
        ))
    }

    /// Returns a tensor with the same data but in a contiguous memory layout.
    ///
    /// This is necessary for operations like `matmul` on CUDA backends which
    /// require contiguous inputs. Transpose and other view operations can
    /// produce non-contiguous tensors.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - A contiguous copy of the tensor, or the original if already contiguous.
    /// * `Err(TensorError)` - The backend failed to make the tensor contiguous.
    ///
    /// # Examples
    /// * Forward pass with shape and value verification
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::default();
    ///
    /// // transpose produces a non-contiguous view
    /// let t = Tensor::from_data(vec![vec![1.0, 2.0], vec![3.0, 4.0]], &device, false).unwrap();
    /// let transposed = t.transpose(0, 1).unwrap();
    /// let result = transposed.contiguous().unwrap();
    /// assert_eq!(result.shape().unwrap(), Shape::from_dims(&[2, 2]));
    /// assert_eq!(result.to_vec::<f64>().unwrap(), vec![1.0, 3.0, 2.0, 4.0]);
    /// ```
    ///
    /// * Backward pass with gradient verification
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::default();
    ///
    /// let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &device, true).unwrap();
    ///
    /// let result = t.contiguous().unwrap();
    /// result.backward().unwrap();
    ///
    /// let grad = t.grad().unwrap().unwrap();
    /// assert_eq!(grad.shape().unwrap(), Shape::from_dims(&[4]));
    /// assert_eq!(grad.to_vec::<f64>().unwrap(), vec![1.0, 1.0, 1.0, 1.0]);
    /// ```
    pub fn contiguous(&self) -> Result<Self, TensorError> {
        let storage = self.backend_storage()?.contiguous()?;
        Ok(Self::op_result_with_kind(
            storage,
            self.device()?,
            vec![self.copy()],
            OpKind::Contiguous,
        ))
    }

    /// Permute the dimensions of the tensor according to the given order.
    ///
    /// # Arguments
    /// * `dims` - The new order of dimensions (0-based). Supports negative indexing: -1 means the last dimension, -2 means second last, etc.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The permuted tensor.
    /// * `Err(TensorError)` - The error when permuting the tensor.
    ///
    /// # Examples
    /// * Forward pass with shape and value verification
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::default();
    ///
    /// let t = Tensor::from_data(vec![vec![vec![1.0, 2.0]], vec![vec![3.0, 4.0]]], &device, false).unwrap();
    ///
    /// let result = t.permute(&[1, 0, 2]).unwrap();
    /// assert_eq!(result.shape().unwrap(), Shape::from_dims(&[1, 2, 2]));
    /// assert_eq!(result.to_vec::<f64>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    /// ```
    ///
    /// * Forward pass with negative indices
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::default();
    ///
    /// // Shape [2, 1, 2] permuted with [-1, 0, 1] (same as [2, 0, 1])
    /// let t = Tensor::from_data(vec![vec![vec![1.0, 2.0]], vec![vec![3.0, 4.0]]], &device, false).unwrap();
    ///
    /// let result = t.permute(&[-1, 0, 1]).unwrap();
    /// assert_eq!(result.shape().unwrap(), Shape::from_dims(&[2, 2, 1]));
    /// assert_eq!(result.to_vec::<f64>().unwrap(), vec![1.0, 3.0, 2.0, 4.0]);
    /// ```
    ///
    /// * Backward pass with gradient verification
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::default();
    ///
    /// let t = Tensor::from_data(vec![vec![vec![1.0, 2.0]], vec![vec![3.0, 4.0]]], &device, true).unwrap();
    ///
    /// let result = t.permute(&[1, 0, 2]).unwrap();
    /// result.backward().unwrap();
    ///
    /// let grad = t.grad().unwrap().unwrap();
    /// assert_eq!(grad.shape().unwrap(), Shape::from_dims(&[2, 1, 2]));
    /// assert_eq!(grad.to_vec::<f64>().unwrap(), vec![1.0, 1.0, 1.0, 1.0]);
    /// ```
    pub fn permute(&self, dims: &[isize]) -> Result<Self, TensorError> {
        let num_dims = self.num_dim()?;
        let dims = dims
            .iter()
            .map(|dim| normalize_dim(*dim, num_dims))
            .collect::<Vec<_>>();
        let mut inverse_dims = vec![0; dims.len()];
        for (new_axis, old_axis) in dims.iter().copied().enumerate() {
            inverse_dims[old_axis] = new_axis;
        }
        let storage = self.backend_storage()?.permute(&dims)?;
        Ok(Self::op_result_with_kind(
            storage,
            self.device()?,
            vec![self.copy()],
            OpKind::Permute { dims, inverse_dims },
        ))
    }

    /// Narrow the tensor along a dimension by selecting a range of indices.
    ///
    /// # Arguments
    /// * `dim` - The dimension to narrow along (0-based). Supports negative indexing: -1 means the last dimension, -2 means second last, etc.
    /// * `start` - The starting index (inclusive) for the narrowing operation. Supports negative indexing: -1 means the last dimension, -2 means second last, etc.
    /// * `length` - The number of elements to include in the narrowed dimension.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The narrowed tensor.
    /// * `Err(TensorError)` - The error when narrowing the tensor.
    ///
    /// # Examples
    /// * Forward pass with shape and value verification
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::default();
    ///
    /// let t = Tensor::from_data(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device, false).unwrap();
    ///
    /// let result = t.narrow(1, 1, 2).unwrap();
    /// assert_eq!(result.shape().unwrap(), Shape::from_dims(&[2, 2]));
    /// assert_eq!(result.to_vec::<f64>().unwrap(), vec![2.0, 3.0, 5.0, 6.0]);
    /// ```
    ///
    /// * Forward pass with negative indices
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::default();
    ///
    /// let t = Tensor::from_data(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device, false).unwrap();
    ///
    /// let result = t.narrow(-1, -2, 2).unwrap();
    /// assert_eq!(result.shape().unwrap(), Shape::from_dims(&[2, 2]));
    /// assert_eq!(result.to_vec::<f64>().unwrap(), vec![2.0, 3.0, 5.0, 6.0]);
    /// ```
    ///
    /// * Forward pass along first dimension
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::default();
    ///
    /// let t = Tensor::from_data(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device, false).unwrap();
    ///
    /// let result = t.narrow(0, 0, 1).unwrap();
    /// assert_eq!(result.shape().unwrap(), Shape::from_dims(&[1, 3]));
    /// assert_eq!(result.to_vec::<f64>().unwrap(), vec![1.0, 2.0, 3.0]);
    /// ```
    ///
    /// * Backward pass with gradient verification
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::default();
    ///
    /// let t = Tensor::from_data(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device, true).unwrap();
    ///
    /// let result = t.narrow(1, 1, 2).unwrap();
    /// result.backward().unwrap();
    ///
    /// let grad = t.grad().unwrap().unwrap();
    /// assert_eq!(grad.shape().unwrap(), Shape::from_dims(&[2, 3]));
    /// assert_eq!(grad.to_vec::<f64>().unwrap(), vec![0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
    /// ```
    pub fn narrow(&self, dim: isize, start: isize, length: usize) -> Result<Self, TensorError> {
        let shape = self.shape()?;
        let dims = shape.dims();
        let dim = normalize_dim(dim, dims.len());
        let dim_size = dims[dim] as isize;
        let start = if start < 0 {
            (dim_size + start) as usize
        } else {
            start as usize
        };

        let storage = self.backend_storage()?.narrow(dim, start, length)?;
        Ok(Self::op_result_with_kind(
            storage,
            self.device()?,
            vec![self.copy()],
            OpKind::Narrow {
                input_shape: shape,
                dim,
                start,
                length,
            },
        ))
    }
}

fn normalize_dim(dim: isize, rank: usize) -> usize {
    if dim < 0 {
        (rank as isize + dim) as usize
    } else {
        dim as usize
    }
}
