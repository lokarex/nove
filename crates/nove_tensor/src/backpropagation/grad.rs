use crate::{
    Tensor, TensorError,
    tensor::{TensorData, TensorInner},
};
use crossbeam::queue::SegQueue;
use dashmap::DashMap;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::sync::{Arc, RwLock};

impl Tensor {
    /// Get the gradient enabled status of the tensor.
    ///
    /// # Returns
    /// * `Ok(true)` - The tensor has gradient enabled.
    /// * `Ok(false)` - The tensor has gradient disabled.
    /// * `Err(TensorError)` - The error when getting the tensor's gradient enabled status.
    pub fn grad_enabled(&self) -> Result<bool, TensorError> {
        let data = self.data.read()?;
        Ok(match &data.inner {
            TensorInner::Var(_) => true,
            TensorInner::Tensor(_) => false,
        })
    }

    /// Create a new tensor likes the current tensor, but with the desired gradient status.
    ///
    /// # Notes
    /// * If the tensor already has the desired gradient status, the method will return the current tensor.
    /// * Switching the gradient enabled status will disconnect the tensor from the computational graph.
    ///
    /// # Arguments
    /// * `grad_enabled` - The desired gradient enabled status.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - A new tensor with the desired gradient status.
    /// * `Err(TensorError)` - The error when setting the tensor's gradient enabled status.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    ///
    /// // Create a tensor with gradient tracking disabled
    /// let data = [1.0f32, 2.0, 3.0, 4.0];
    /// let mut tensor = Tensor::from_slice(&data, &Shape::from(&[2, 2]), &device, false).unwrap();
    ///
    /// // Initially, gradient tracking is disabled
    /// assert_eq!(tensor.grad_enabled().unwrap(), false);
    ///
    /// // Enable gradient tracking
    /// let mut tensor_with_grad = tensor.require_grad(true).unwrap();
    /// assert_eq!(tensor_with_grad.grad_enabled().unwrap(), true);
    ///
    /// // Disable gradient tracking
    /// let mut tensor_without_grad = tensor_with_grad.require_grad(false).unwrap();
    /// assert_eq!(tensor_without_grad.grad_enabled().unwrap(), false);
    ///
    /// // If the tensor already has the desired gradient status, it returns a copy
    /// let another_copy = tensor_without_grad.require_grad(false).unwrap();
    /// assert_eq!(another_copy.grad_enabled().unwrap(), false);
    ///
    /// // Switching gradient status disconnects the tensor from computational graph
    /// // This means any operations on the original tensor won't affect the new one
    /// ```
    pub fn require_grad(&mut self, grad_enabled: bool) -> Result<Tensor, TensorError> {
        // Check if the gradient status is already as required
        if self.grad_enabled()? == grad_enabled {
            return Ok(self.copy());
        }

        let inner_tensor = {
            let data = self.data.read()?;
            match &data.inner {
                TensorInner::Tensor(tensor) => tensor.clone(),
                TensorInner::Var(var) => var.as_tensor().clone(),
            }
        }
        .copy()?
        .detach();

        let new_inner = match grad_enabled {
            true => {
                // Convert to Var type
                let var = candle_core::Var::from_tensor(&inner_tensor)?;
                TensorInner::Var(var)
            }

            false => {
                // Convert to Tensor type
                TensorInner::Tensor(inner_tensor)
            }
        };

        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                device: self.device()?.clone(),
                parents: vec![],
                grad: None,
                name: self.data.read()?.name.clone(),
            })),
        })
    }

    /// Get the gradient tensor of the tensor.
    ///
    /// # Returns
    /// * `Ok(Some(Tensor))` - The tensor's gradient tensor.
    /// * `Ok(None)` - The tensor has no gradient tensor.
    /// * `Err(TensorError)` - The error when getting the tensor's gradient tensor.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    ///
    /// // Create a tensor with gradient tracking enabled
    /// let data = [1.0f32, 2.0, 3.0, 4.0];
    /// let tensor = Tensor::from_slice(&data, &Shape::from(&[2, 2]), &device, true).unwrap();
    ///
    /// // Initially, no gradient exists
    /// let grad_before = tensor.grad().unwrap();
    /// assert!(grad_before.is_none());
    ///
    /// // Perform computation to create gradient
    /// let scalar = Tensor::from_scalar(2.0f32, &device, true).unwrap();
    /// let result = tensor.mul(&scalar).unwrap();
    /// result.backward().unwrap();
    ///
    /// // After backward pass, gradient should exist
    /// let grad_after = tensor.grad().unwrap();
    /// assert!(grad_after.is_some());
    ///
    /// // Gradient should have the same shape as the original tensor
    /// if let Some(ref grad) = grad_after {
    ///     assert_eq!(grad.shape().unwrap(), Shape::from(&[2, 2]));
    /// }
    ///
    /// // Gradient can be cleared and will return None again
    /// let mut mutable_tensor = tensor.copy();
    /// mutable_tensor.clear_grad().unwrap();
    /// let grad_cleared = mutable_tensor.grad().unwrap();
    /// assert!(grad_cleared.is_none());
    /// ```
    pub fn grad(&self) -> Result<Option<Tensor>, TensorError> {
        let data = self.data.read()?;
        Ok(data.grad.as_ref().map(|grad| grad.copy()))
    }

    /// Zero the gradient tensor of the tensor.
    ///
    /// # Returns
    /// * `Ok(())` - The tensor's gradient tensor is successfully set to zero.
    /// * `Err(TensorError)` - The error when setting the tensor's gradient tensor to zero.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    ///
    /// // Create a tensor with gradient tracking enabled
    /// let data = [1.0f32, 2.0, 3.0, 4.0];
    /// let mut tensor = Tensor::from_slice(&data, &Shape::from(&[2, 2]), &device, true).unwrap();
    ///
    /// // Perform computation to create gradient
    /// let scalar = Tensor::from_scalar(2.0f32, &device, true).unwrap();
    /// let result = tensor.mul(&scalar).unwrap();
    /// result.backward().unwrap();
    ///
    /// // Verify gradient exists and is non-zero
    /// let grad_before = tensor.grad().unwrap().unwrap();
    ///
    /// // Get the gradient tensor and check its shape and values
    /// assert_eq!(grad_before.shape().unwrap(), Shape::from(&[2, 2]));
    /// assert_eq!(grad_before.to_vec::<f32>().unwrap(), vec![2.0, 2.0, 2.0, 2.0]);
    ///
    /// // Zero the gradient
    /// tensor.zero_grad().unwrap();
    ///
    /// // Gradient still exists but should be all zeros
    /// let grad_after = tensor.grad().unwrap().unwrap();
    /// assert_eq!(grad_after.shape().unwrap(), Shape::from(&[2, 2]));
    /// assert_eq!(grad_after.to_vec::<f32>().unwrap(), vec![0.0, 0.0, 0.0, 0.0]);
    /// ```
    ///
    /// # See Also
    /// * [`Tensor::clear_grad`] - Clears the gradient tensor of the tensor.
    pub fn zero_grad(&mut self) -> Result<(), TensorError> {
        let mut data = self.data.write()?;
        if let Some(grad) = &mut data.grad {
            match &mut grad.data.write()?.inner {
                TensorInner::Var(var) => var.zero_set()?,
                TensorInner::Tensor(tensor) => tensor.zero_set()?,
            }
        }
        Ok(())
    }

    /// Clear the gradient tensor of the tensor.
    ///
    /// # Returns
    /// * `Ok(())` - The tensor's gradient tensor is successfully cleared.
    /// * `Err(TensorError)` - The error when clearing the tensor's gradient tensor.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    ///
    /// // Create a tensor with gradient tracking enabled
    /// let data = [1.0f32, 2.0, 3.0, 4.0];
    /// let mut tensor = Tensor::from_slice(&data, &Shape::from(&[2, 2]), &device, true).unwrap();
    ///
    /// // Create a simple computational graph and compute gradients
    /// let other = Tensor::from_scalar(2.0f32, &device, true).unwrap();
    /// let result = tensor.mul(&other).unwrap();
    /// result.backward().unwrap();
    ///
    /// // Verify gradient exists and is non-zero
    /// let grad_before = tensor.grad().unwrap().unwrap();
    ///
    /// // Get the gradient tensor and check its shape and values
    /// assert_eq!(grad_before.shape().unwrap(), Shape::from(&[2, 2]));
    /// assert_eq!(grad_before.to_vec::<f32>().unwrap(), vec![2.0, 2.0, 2.0, 2.0]);
    ///
    /// // Clear the gradient
    /// tensor.clear_grad().unwrap();
    ///
    /// // Verify gradient is cleared
    /// let grad_after = tensor.grad().unwrap();
    /// assert!(grad_after.is_none());
    /// ```
    pub fn clear_grad(&mut self) -> Result<(), TensorError> {
        let mut data = self.data.write()?;
        data.grad = None;
        Ok(())
    }

    /// Backpropagate the gradient of the tensor to its parent tensors.
    ///
    /// # Returns
    /// * `Ok(())` - The tensor's gradient is successfully backpropagated.
    /// * `Err(TensorError)` - The error when backpropagating the tensor's gradient.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    ///
    /// // Create two tensors with gradient tracking enabled
    /// let x_data = [1.0f32, 2.0, 3.0, 4.0];
    /// let x = Tensor::from_slice(&x_data, &Shape::from(&[2, 2]), &device, true).unwrap();
    ///
    /// let y_data = [0.5f32, 1.5, 2.5, 3.5];
    /// let y = Tensor::from_slice(&y_data, &Shape::from(&[2, 2]), &device, true).unwrap();
    ///
    /// // Perform operations to create a computational graph
    /// let z = x.add(&y).unwrap();
    /// // Create a scalar tensor for multiplication
    /// let scalar = Tensor::from_scalar(2.0f32, &device, true).unwrap();
    /// let w = z.mul(&scalar).unwrap();
    ///
    /// // Compute gradient by calling backward on the final tensor
    /// w.backward().unwrap();
    ///
    /// // Check that gradients are computed for input tensors
    /// let x_grad = x.grad().unwrap();
    /// assert!(x_grad.is_some());
    ///
    /// let y_grad = y.grad().unwrap();
    /// assert!(y_grad.is_some());
    /// ```
    pub fn backward(&self) -> Result<(), TensorError> {
        // Get grad_store
        let grad_store = {
            let data = self.data.read()?;
            match &data.inner {
                TensorInner::Var(var) => {
                    // Clone the var to release the read lock before calling backward
                    let var_clone = var.clone();
                    drop(data);
                    var_clone.backward()?
                }
                TensorInner::Tensor(tensor) => {
                    // Clone the tensor to release the read lock before calling backward
                    let tensor_clone = tensor.clone();
                    drop(data);
                    tensor_clone.backward()?
                }
            }
        };

        println!(
            "The length of grad_store is {}",
            grad_store.get_ids().count()
        );

        // The parent tensors(including indirect parents) that have gradient enabled
        let mut parents = Vec::new();
        // The queue for pending tensors
        let queue = SegQueue::new();
        // The visited tensors
        let visited = DashMap::new();

        // Add the self tensor to the queue and mark it as visited
        queue.push(self.copy());
        visited.insert(Arc::as_ptr(&self.data) as usize, true);
        if self.grad_enabled().unwrap_or(false) {
            parents.push(self.copy());
        }

        while let Some(current) = queue.pop() {
            // Get the parent tensors of the current tensor
            let current_parents = {
                let data = current.data.read()?;
                data.parents
                    .iter()
                    .map(|parent| parent.copy())
                    .collect::<Vec<_>>()
            };

            // Filter out the gradient enabled parent tensors
            let grad_enabled_parents = current_parents
                .par_iter()
                .filter_map(|parent| {
                    let parent_id = Arc::as_ptr(&parent.data) as usize;
                    match visited.insert(parent_id, true).is_some() {
                        true => None,
                        false => {
                            // Add the parent tensor to the queue for further processing
                            queue.push(parent.copy());

                            match parent.grad_enabled().unwrap_or(false) {
                                true => Some(parent.copy()),
                                false => None,
                            }
                        }
                    }
                })
                .collect::<Vec<_>>();

            // Add the parent tensors that have gradient enabled to the list of parents
            parents.extend(grad_enabled_parents);
        }

        // for parent in &parents {
        //     println!("{}:{}", parent.name().unwrap().unwrap(), parent);
        // }

        // Update the gradient of each parent tensor
        parents
            .par_iter()
            .map(|parent| {
                let mut parent_write = parent.data.write()?;
                let inner_tensor = match &parent_write.inner {
                    TensorInner::Tensor(tensor) => tensor,
                    TensorInner::Var(var) => var.as_tensor(),
                };

                // Get the new gradient of the parent tensor from the grad_store
                let new_grad = grad_store
                    .get(inner_tensor)
                    .ok_or(TensorError::NoTensorGradient)?;

                println!("new_grad: {:?}", new_grad);

                match &parent_write.grad {
                    // If the parent tensor already has a gradient, add the new gradient to it
                    Some(parent_grad) => {
                        let mut parent_grad_write = parent_grad.data.write()?;
                        let parent_grad_inner_tensor = match &parent_grad_write.inner {
                            TensorInner::Tensor(tensor) => tensor,
                            TensorInner::Var(var) => var.as_tensor(),
                        };
                        parent_grad_write.inner =
                            TensorInner::Tensor(new_grad.add(parent_grad_inner_tensor)?.detach());
                    }
                    // If the parent tensor does not have a gradient, set it to the new gradient
                    None => {
                        parent_write.grad = Some(Tensor::from_candle_tensor(
                            new_grad.clone(),
                            &parent_write.device,
                            false,
                        )?);
                    }
                };

                Ok(())
            })
            .collect::<Result<(), TensorError>>()?;

        Ok(())
    }
}
