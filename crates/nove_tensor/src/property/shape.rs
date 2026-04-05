use std::sync::{Arc, RwLock};

use crate::{
    Shape, Tensor, TensorError,
    tensor::{TensorData, TensorInner},
};

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
    /// let cpu = Device::cpu();
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32, 3.0f32, 4.0f32], &cpu, false).unwrap();
    ///
    /// let result = tensor.reshape(&Shape::from(&[2, 2])).unwrap();
    /// assert_eq!(result.shape().unwrap(), (&[2, 2]).into());
    /// assert_eq!(result.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    /// ```
    ///
    /// * Reshape 1D tensor to column vector
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let cpu = Device::cpu();
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32, 3.0f32], &cpu, false).unwrap();
    ///
    /// let result = tensor.reshape(&Shape::from(&[3, 1])).unwrap();
    /// assert_eq!(result.shape().unwrap(), (&[3, 1]).into());
    /// assert_eq!(result.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
    /// ```
    ///
    /// * Backpropagate through reshape with gradient
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let cpu = Device::cpu();
    ///
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32, 3.0f32, 4.0f32], &cpu, true).unwrap();
    /// let result = tensor.reshape(&Shape::from(&[2, 2])).unwrap();
    /// result.backward().unwrap();
    /// let grad = tensor.grad().unwrap().unwrap();
    /// assert_eq!(grad.shape().unwrap(), (&[4]).into());
    /// assert_eq!(grad.to_vec::<f32>().unwrap(), vec![1.0, 1.0, 1.0, 1.0]);
    ///
    /// // The following Tensor is actually an intermediate Tensor, its grad is always None
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32, 3.0f32, 4.0f32], &cpu, true).unwrap().reshape(&Shape::from(&[2, 2])).unwrap();
    /// tensor.backward().unwrap();
    /// assert!(tensor.grad().unwrap().is_none());
    ///
    /// // If you need to reshape the tensor immediately using chained calls while preserving gradients, you can do so as follows
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32, 3.0f32, 4.0f32], &cpu, true).unwrap().reshape(&Shape::from(&[2, 2])).unwrap().require_grad(true).unwrap();
    /// tensor.backward().unwrap();
    /// let grad = tensor.grad().unwrap().unwrap();
    /// assert_eq!(grad.shape().unwrap(), (&[2, 2]).into());
    /// assert_eq!(grad.to_vec::<f32>().unwrap(), vec![1.0, 1.0, 1.0, 1.0]);
    /// ```
    pub fn reshape(&self, shape: &Shape) -> Result<Tensor, TensorError> {
        let new_inner = match &self.data.read()?.inner {
            TensorInner::Tensor(tensor) => TensorInner::Tensor(tensor.reshape(shape)?),
            TensorInner::Var(var) => TensorInner::Tensor(var.reshape(shape)?),
        };

        let new_grad = match &self.data.read()?.grad {
            Some(grad) => Some(grad.reshape(shape)?),
            None => None,
        };

        Ok(Tensor {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                grad: new_grad,
                device: self.data.read()?.device.clone(),
                parents: vec![self.copy()],
                name: self.data.read()?.name.clone(),
            })),
        })
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
    /// let cpu = Device::cpu();
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32, 3.0f32], &cpu, false).unwrap();
    ///
    /// let shape = tensor.shape().unwrap();
    /// assert_eq!(shape, (&[3]).into());
    /// ```
    pub fn shape(&self) -> Result<Shape, TensorError> {
        let data = self.data.read()?;
        let shape = match &data.inner {
            TensorInner::Tensor(tensor) => tensor.shape(),
            TensorInner::Var(var) => var.shape(),
        };
        Ok(Shape::from(shape))
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
    /// let cpu = Device::cpu();
    ///
    /// // 1-dimensional tensor (vector)
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32, 3.0f32], &cpu, false).unwrap();
    /// let num_dim = tensor.num_dim().unwrap();
    /// assert_eq!(num_dim, 1);
    ///
    /// // 2-dimensional tensor (matrix)
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32, 3.0f32, 4.0f32], &cpu, false).unwrap();
    /// let result = tensor.reshape(&(&[2, 2]).into()).unwrap();
    /// let num_dim = result.num_dim().unwrap();
    /// assert_eq!(num_dim, 2);
    ///
    /// // 3-dimensional tensor
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32], &cpu, false).unwrap();
    /// let result = tensor.reshape(&(&[2, 2, 2]).into()).unwrap();
    /// let num_dim = result.num_dim().unwrap();
    /// assert_eq!(num_dim, 3);
    ///
    /// // 4-dimensional tensor
    /// let tensor = Tensor::from_data(&[1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32], &cpu, false).unwrap();
    /// let result = tensor.reshape(&(&[2, 2, 1, 2]).into()).unwrap();
    /// let num_dim = result.num_dim().unwrap();
    /// assert_eq!(num_dim, 4);
    /// ```
    pub fn num_dim(&self) -> Result<usize, TensorError> {
        let shape = self.shape()?;
        Ok(shape.rank())
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
                parents: vec![self.copy()],
                grad: None,
                name: None,
            })),
        })
    }

    /// Flatten the tensor by merging multiple dimensions into one.
    ///
    /// # Arguments
    /// * `start_dim` - Optional starting dimension index (0-based) to begin flattening.
    ///   If `None`, flattening starts from the first dimension.
    /// * `end_dim` - Optional ending dimension index (0-based) to stop flattening.
    ///   If `None`, flattening continues to the last dimension.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The flattened tensor.
    /// * `Err(TensorError)` - The error when flattening the tensor.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::cpu();
    /// let t = Tensor::from_data(&[[1.0, 2.0], [3.0, 4.0]], &device, false).unwrap();
    ///
    /// // Flatten entire tensor
    /// let flattened = t.flatten(None, None).unwrap();
    /// println!("{:?}", flattened);
    ///
    /// // Flatten from dimension 1 to the end
    /// let flattened = t.flatten(Some(1), None).unwrap();
    /// println!("{:?}", flattened);
    ///
    /// // Flatten from start to dimension 0
    /// let flattened = t.flatten(None, Some(0)).unwrap();
    /// println!("{:?}", flattened);
    ///
    /// // Flatten dimensions 1 to 2
    /// let t3d = Tensor::from_data(&[[[1.0, 2.0]], [[3.0, 4.0]]], &device, false).unwrap();
    /// let flattened = t3d.flatten(Some(1), Some(2)).unwrap();
    /// println!("{:?}", flattened);
    /// ```
    pub fn flatten(
        &self,
        start_dim: Option<usize>,
        end_dim: Option<usize>,
    ) -> Result<Self, TensorError> {
        let inner = self.data.read()?;
        let inner_tensor = match &inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = match (start_dim, end_dim) {
            (None, None) => TensorInner::Tensor(inner_tensor.flatten_all()?),
            (Some(start_dim), None) => TensorInner::Tensor(inner_tensor.flatten_from(start_dim)?),
            (None, Some(end_dim)) => TensorInner::Tensor(inner_tensor.flatten_to(end_dim)?),
            (Some(start_dim), Some(end_dim)) => {
                TensorInner::Tensor(inner_tensor.flatten(start_dim, end_dim)?)
            }
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
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::cpu();
    /// let t = Tensor::from_data(&[[[1.0]], [[2.0]]], &device, false).unwrap();
    ///
    /// let squeezed = t.squeeze(None).unwrap();
    /// println!("{:?}", squeezed);
    /// ```
    pub fn squeeze(&self, dim: Option<usize>) -> Result<Self, TensorError> {
        let inner = self.data.read()?;
        let inner_tensor = match &inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = match dim {
            Some(dim) => TensorInner::Tensor(inner_tensor.squeeze(dim)?),
            None => {
                // Squeeze all dimensions of size 1
                let shape = inner_tensor.shape();
                let dims = shape.dims();
                // Iterate dimensions in reverse order to avoid index shifting
                let mut result = inner_tensor.clone();
                for (i, &dim_size) in dims.iter().enumerate().rev() {
                    if dim_size == 1 {
                        result = result.squeeze(i)?;
                    }
                }
                TensorInner::Tensor(result)
            }
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
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::cpu();
    /// let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &device, false).unwrap();
    ///
    /// let unsqueezed = t.unsqueeze(0).unwrap();
    /// println!("{:?}", unsqueezed);
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
                parents: vec![self.copy()],
                grad: None,
                name: None,
            })),
        })
    }

    /// Transpose the tensor by swapping the two specified dimensions.
    ///
    /// # Arguments
    /// * `dim0` - The first dimension to swap.
    /// * `dim1` - The second dimension to swap.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The transposed tensor.
    /// * `Err(TensorError)` - The error when transposing the tensor.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::cpu();
    /// let t = Tensor::from_data(&[[1.0, 2.0], [3.0, 4.0]], &device, false).unwrap();
    ///
    /// let transposed = t.transpose(0, 1).unwrap();
    /// println!("{:?}", transposed);
    /// ```
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Self, TensorError> {
        let inner = self.data.read()?;
        let inner_tensor = match &inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = TensorInner::Tensor(inner_tensor.transpose(dim0, dim1)?);

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

    /// Permute the dimensions of the tensor according to the given order.
    ///
    /// # Arguments
    /// * `dims` - The new order of dimensions.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The permuted tensor.
    /// * `Err(TensorError)` - The error when permuting the tensor.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::cpu();
    /// let t = Tensor::from_data(&[[[1.0, 2.0]], [[3.0, 4.0]]], &device, false).unwrap();
    ///
    /// let permuted = t.permute(&[2, 0, 1]).unwrap();
    /// println!("{:?}", permuted);
    /// ```
    pub fn permute(&self, dims: &[usize]) -> Result<Self, TensorError> {
        let inner = self.data.read()?;
        let inner_tensor = match &inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = TensorInner::Tensor(inner_tensor.permute(dims)?);

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

    /// Narrow the tensor along a dimension by selecting a range of indices.
    ///
    /// # Arguments
    /// * `dim` - The dimension to narrow along.
    /// * `start` - The starting index (inclusive) for the narrowing operation.
    /// * `length` - The number of elements to include in the narrowed dimension.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The narrowed tensor.
    /// * `Err(TensorError)` - The error when narrowing the tensor.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::cpu();
    /// let t = Tensor::from_data(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device, false).unwrap();
    ///
    /// // Narrow along dimension 1 (columns), starting at index 1, length 2
    /// let narrowed = t.narrow(1, 1, 2).unwrap();
    /// println!("{:?}", narrowed);
    /// ```
    pub fn narrow(&self, dim: usize, start: usize, length: usize) -> Result<Self, TensorError> {
        let inner = self.data.read()?;
        let inner_tensor = match &inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = TensorInner::Tensor(inner_tensor.narrow(dim, start, length)?);

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
