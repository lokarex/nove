use crate::{DType, Device, Shape, Tensor, TensorError, backpropagation::graph::OpKind};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

impl Tensor {
    /// Get the gradient tracking status of the tensor.
    ///
    /// # Returns
    /// * `Ok(true)` - The tensor has gradient enabled.
    /// * `Ok(false)` - The tensor has gradient disabled.
    /// * `Err(TensorError)` - The error when getting the tensor's gradient tracking status.
    pub fn grad_enabled(&self) -> Result<bool, TensorError> {
        Ok(self.data.read()?.requires_grad)
    }

    /// Create a new tensor likes the current tensor, but with the desired gradient status.
    ///
    /// # Notes
    /// * If the tensor already has the desired gradient status, the method will return the current tensor.
    /// * Switching the gradient tracking status will disconnect the tensor from the computational graph.
    ///
    /// # Arguments
    /// * `grad_enabled` - The desired gradient tracking status.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - A new tensor with the desired gradient status.
    /// * `Err(TensorError)` - The error when setting the tensor's gradient tracking status.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::default();
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
    pub fn require_grad(&mut self, requires_grad: bool) -> Result<Tensor, TensorError> {
        if self.grad_enabled()? == requires_grad {
            return Ok(self.copy());
        }

        let data = self.data.read()?;
        let storage = data.storage.with_requires_grad(requires_grad)?;
        let device = data.device.clone();
        let name = data.name.clone();
        drop(data);

        Ok(Self::from_backend_parts(
            storage,
            device,
            requires_grad,
            vec![],
            OpKind::RequireGrad,
            None,
            name,
        ))
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
    /// let device = Device::default();
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
        Ok(self.data.read()?.grad.as_ref().map(|grad| grad.copy()))
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
    /// let device = Device::default();
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
            grad.data.write()?.storage.zero_set()?;
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
    /// let device = Device::default();
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
        self.data.write()?.grad = None;
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
    /// let device = Device::default();
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
        self.backward_nove()
    }

    fn backward_nove(&self) -> Result<(), TensorError> {
        ensure_float_gradient_dtype(self.dtype()?)?;

        let topology = collect_topology(self)?;
        let mut gradients = HashMap::new();
        gradients.insert(tensor_key(self), ones_like_gradient(self)?);

        for current in topology.iter().rev() {
            let Some(grad_output) = gradients.get(&tensor_key(current)).cloned() else {
                continue;
            };
            let (op, parents) = {
                let data = current.data.read()?;
                (
                    data.graph.op.clone(),
                    data.graph
                        .parents
                        .iter()
                        .map(|parent| parent.copy())
                        .collect::<Vec<_>>(),
                )
            };

            for (parent, parent_grad) in current.local_backward(op, &parents, &grad_output)? {
                let parent_key = tensor_key(&parent);
                let parent_grad = parent_grad.detach()?;
                match gradients.remove(&parent_key) {
                    Some(existing) => {
                        let accumulated = existing.add(&parent_grad)?.detach()?;
                        gradients.insert(parent_key, accumulated);
                    }
                    None => {
                        gradients.insert(parent_key, parent_grad);
                    }
                }
            }
        }

        for tensor in topology {
            let key = tensor_key(&tensor);
            let Some(grad) = gradients.remove(&key) else {
                continue;
            };
            let should_store = {
                let data = tensor.data.read()?;
                data.requires_grad && data.graph.parents.is_empty()
            };
            if should_store {
                tensor.accumulate_stored_grad(grad.detach()?)?;
            }
        }

        Ok(())
    }

    fn accumulate_stored_grad(&self, grad: Tensor) -> Result<(), TensorError> {
        let mut data = self.data.write()?;
        match &data.grad {
            Some(existing) => {
                let mut existing = existing.data.write()?;
                existing.storage = existing
                    .storage
                    .broadcast_add(&grad.backend_storage()?)?
                    .detach()?;
            }
            None => {
                data.grad = Some(Tensor::from_backend_storage(
                    grad.backend_storage()?.detach()?,
                    data.device.clone(),
                    false,
                    vec![],
                    OpKind::Gradient,
                ));
            }
        }
        Ok(())
    }

    fn local_backward(
        &self,
        op: OpKind,
        parents: &[Tensor],
        grad_output: &Tensor,
    ) -> Result<Vec<(Tensor, Tensor)>, TensorError> {
        match op {
            OpKind::Leaf
            | OpKind::Clone
            | OpKind::Detach
            | OpKind::RequireGrad
            | OpKind::Gradient => Ok(vec![]),
            OpKind::ZerosLike | OpKind::OnesLike => {
                unary_parent_grad(parents, grad_output, |input, _| {
                    Tensor::zeros(&input.shape()?, &input.dtype()?, &input.device()?, false)
                })
            }
            OpKind::Comparison(_) | OpKind::ArgMax | OpKind::ArgMin => zero_parent_grads(parents),
            OpKind::Add => binary_parent_grads(parents, grad_output, |_, _, grad| {
                Ok((grad.copy(), grad.copy()))
            }),
            OpKind::Sub => binary_parent_grads(parents, grad_output, |_, _, grad| {
                Ok((grad.copy(), grad.neg()?))
            }),
            OpKind::Mul => binary_parent_grads(parents, grad_output, |lhs, rhs, grad| {
                Ok((grad.mul(rhs)?, grad.mul(lhs)?))
            }),
            OpKind::Div => binary_parent_grads(parents, grad_output, |lhs, rhs, grad| {
                let lhs_grad = grad.div(rhs)?;
                let rhs_grad = grad.mul(lhs)?.div(&rhs.mul(rhs)?)?.neg()?;
                Ok((lhs_grad, rhs_grad))
            }),
            OpKind::Matmul => binary_parent_grads(parents, grad_output, |lhs, rhs, grad| {
                let lhs_grad = grad.matmul(&rhs.transpose(-1, -2)?)?;
                let rhs_grad = lhs.transpose(-1, -2)?.matmul(grad)?;
                Ok((lhs_grad, rhs_grad))
            }),
            OpKind::Pow => binary_parent_grads(parents, grad_output, |base, exponent, grad| {
                let base_grad = grad
                    .mul(exponent)?
                    .mul(&base.pow(&exponent.affine(1.0, -1.0)?)?)?;
                let exponent_grad = grad.mul(&base.pow(exponent)?)?.mul(&base.log()?)?;
                Ok((base_grad, exponent_grad))
            }),
            OpKind::Powf(exponent) => unary_parent_grad(parents, grad_output, |input, grad| {
                if exponent == 0.0 {
                    return Ok(grad.affine(0.0, 0.0)?);
                }
                grad.mul(&input.powf(exponent - 1.0)?.affine(exponent, 0.0)?)
            }),
            OpKind::Neg => unary_parent_grad(parents, grad_output, |_, grad| grad.neg()),
            OpKind::Sqrt => {
                let output = self.detach()?;
                unary_parent_grad(parents, grad_output, |_, grad| {
                    grad.div(&output.affine(2.0, 0.0)?)
                })
            }
            OpKind::Recip => unary_parent_grad(parents, grad_output, |input, grad| {
                grad.mul(&input.mul(input)?.recip()?.neg()?)
            }),
            OpKind::Abs => unary_parent_grad(parents, grad_output, |input, grad| {
                let zero = scalar_like(0.0, input.dtype()?, &input.device()?)?;
                let positive = input.gt(&zero)?.to_dtype(&input.dtype()?)?;
                let negative = input.lt(&zero)?.to_dtype(&input.dtype()?)?;
                grad.mul(&positive.sub(&negative)?)
            }),
            OpKind::Exp => {
                let output = self.detach()?;
                unary_parent_grad(parents, grad_output, |_, grad| grad.mul(&output))
            }
            OpKind::Log => unary_parent_grad(parents, grad_output, |input, grad| grad.div(input)),
            OpKind::Tanh => {
                let output = self.detach()?;
                unary_parent_grad(parents, grad_output, |_, grad| {
                    grad.mul(&output.powf(2.0)?.affine(-1.0, 1.0)?)
                })
            }
            OpKind::Relu => unary_parent_grad(parents, grad_output, |input, grad| {
                let zero = scalar_like(0.0, input.dtype()?, &input.device()?)?;
                let mask = input.gt(&zero)?.to_dtype(&input.dtype()?)?;
                grad.mul(&mask)
            }),
            OpKind::Silu => unary_parent_grad(parents, grad_output, |input, grad| {
                let sigmoid = input.neg()?.exp()?.affine(1.0, 1.0)?.recip()?;
                let one_minus_sigmoid = sigmoid.affine(-1.0, 1.0)?;
                let derivative = sigmoid.mul(&input.mul(&one_minus_sigmoid)?.affine(1.0, 1.0)?)?;
                grad.mul(&derivative)
            }),
            OpKind::Gelu => unary_parent_grad(parents, grad_output, |input, grad| {
                grad.mul(&gelu_tanh_derivative(input)?)
            }),
            OpKind::Clip { min, max } => unary_parent_grad(parents, grad_output, |input, grad| {
                let min = scalar_like(min, input.dtype()?, &input.device()?)?;
                let max = scalar_like(max, input.dtype()?, &input.device()?)?;
                let lower = input.ge(&min)?.to_dtype(&input.dtype()?)?;
                let upper = input.le(&max)?.to_dtype(&input.dtype()?)?;
                grad.mul(&lower.mul(&upper)?)
            }),
            OpKind::Affine { weight, .. } => {
                unary_parent_grad(parents, grad_output, |_, grad| grad.affine(weight, 0.0))
            }
            OpKind::Reshape { from, .. } => {
                unary_parent_grad(parents, grad_output, |_, grad| grad.reshape(&from))
            }
            OpKind::BroadcastAs { from, .. } => {
                unary_parent_grad(parents, grad_output, |_, grad| reduce_to_shape(grad, &from))
            }
            OpKind::Transpose { dim0, dim1 } => {
                unary_parent_grad(parents, grad_output, |_, grad| {
                    grad.transpose(dim0 as isize, dim1 as isize)
                })
            }
            OpKind::Permute { inverse_dims, .. } => {
                let inverse_dims = inverse_dims
                    .into_iter()
                    .map(|dim| dim as isize)
                    .collect::<Vec<_>>();
                unary_parent_grad(parents, grad_output, |_, grad| grad.permute(&inverse_dims))
            }
            OpKind::Contiguous => {
                unary_parent_grad(parents, grad_output, |_, grad| Ok(grad.copy()))
            }
            OpKind::Narrow {
                input_shape,
                dim,
                start,
                length,
            } => unary_parent_grad(parents, grad_output, |_, grad| {
                narrow_backward_grad(grad, &input_shape, dim, start, length)
            }),
            OpKind::Gather { dim, input_shape } => {
                indexed_parent_grad(parents, grad_output, |indexes, grad| {
                    gather_backward_grad(grad, indexes, &input_shape, dim)
                })
            }
            OpKind::IndexSelect { dim, input_shape } => {
                indexed_parent_grad(parents, grad_output, |indexes, grad| {
                    index_select_backward_grad(grad, indexes, &input_shape, dim)
                })
            }
            OpKind::Embedding { table_shape } => {
                indexed_parent_grad(parents, grad_output, |indexes, grad| {
                    embedding_backward_grad(grad, indexes, &table_shape)
                })
            }
            OpKind::Stack { dim, input_shapes } => {
                stack_parent_grads(parents, grad_output, dim, &input_shapes)
            }
            OpKind::Cat { dim, input_shapes } => {
                cat_parent_grads(parents, grad_output, dim, &input_shapes)
            }
            OpKind::WhereCond => where_cond_parent_grads(parents, grad_output),
            OpKind::Sum { axis, input_shape } => {
                unary_parent_grad(parents, grad_output, |_, grad| {
                    expand_reduction_grad(grad, axis, &input_shape)
                })
            }
            OpKind::Max { axis, input_shape } => extrema_parent_grad(
                self,
                parents,
                grad_output,
                axis,
                &input_shape,
                ExtremaKind::Max,
            ),
            OpKind::Min { axis, input_shape } => extrema_parent_grad(
                self,
                parents,
                grad_output,
                axis,
                &input_shape,
                ExtremaKind::Min,
            ),
            OpKind::Conv1d {
                padding,
                stride,
                dilation,
                groups,
                input_shape,
                kernel_shape,
            } => conv1d_parent_grads(
                parents,
                grad_output,
                padding,
                stride,
                dilation,
                groups,
                &input_shape,
                &kernel_shape,
            ),
            OpKind::Conv2d {
                padding,
                stride,
                dilation,
                groups,
                input_shape,
                kernel_shape,
            } => conv2d_parent_grads(
                parents,
                grad_output,
                padding,
                stride,
                dilation,
                groups,
                &input_shape,
                &kernel_shape,
            ),
            OpKind::MaxPool1d {
                kernel_size,
                stride,
                input_shape,
            } => unary_parent_grad(parents, grad_output, |input, grad| {
                max_pool1d_backward_grad(input, grad, &input_shape, kernel_size, stride)
            }),
            OpKind::AvgPool1d {
                kernel_size,
                stride,
                input_shape,
            } => unary_parent_grad(parents, grad_output, |_, grad| {
                avg_pool1d_backward_grad(grad, &input_shape, kernel_size, stride)
            }),
            OpKind::MaxPool2d {
                kernel_size,
                stride,
                input_shape,
            } => unary_parent_grad(parents, grad_output, |input, grad| {
                max_pool2d_backward_grad(input, grad, &input_shape, kernel_size, stride)
            }),
            OpKind::AvgPool2d {
                kernel_size,
                stride,
                input_shape,
            } => unary_parent_grad(parents, grad_output, |_, grad| {
                avg_pool2d_backward_grad(grad, &input_shape, kernel_size, stride)
            }),
            OpKind::Mean { axis, input_shape } => {
                unary_parent_grad(parents, grad_output, |_, grad| {
                    let divisor = reduction_factor(axis, &input_shape) as f64;
                    expand_reduction_grad(grad, axis, &input_shape)?.affine(1.0 / divisor, 0.0)
                })
            }
            OpKind::Var {
                dim,
                keep_dim,
                unbiased,
                input_shape,
            } => unary_parent_grad(parents, grad_output, |input, grad| {
                var_parent_grad(input, grad, dim, keep_dim, unbiased, &input_shape)
            }),
            OpKind::ToDevice { from, .. } => {
                unary_parent_grad(parents, grad_output, |_, grad| grad.to_device(&from))
            }
            OpKind::ToDType { from, .. } => {
                unary_parent_grad(parents, grad_output, |input, grad| {
                    if from.is_float() {
                        grad.to_dtype(&from)
                    } else {
                        Tensor::zeros(&input.shape()?, &input.dtype()?, &input.device()?, false)
                    }
                })
            }
        }
    }
}

fn collect_topology(root: &Tensor) -> Result<Vec<Tensor>, TensorError> {
    fn visit(
        tensor: &Tensor,
        visited: &mut HashSet<usize>,
        order: &mut Vec<Tensor>,
    ) -> Result<(), TensorError> {
        let key = tensor_key(tensor);
        if !visited.insert(key) {
            return Ok(());
        }

        let parents = tensor
            .data
            .read()?
            .graph
            .parents
            .iter()
            .map(|parent| parent.copy())
            .collect::<Vec<_>>();
        for parent in parents {
            visit(&parent, visited, order)?;
        }
        order.push(tensor.copy());
        Ok(())
    }

    let mut visited = HashSet::new();
    let mut order = Vec::new();
    visit(root, &mut visited, &mut order)?;
    Ok(order)
}

fn tensor_key(tensor: &Tensor) -> usize {
    Arc::as_ptr(&tensor.data) as usize
}

fn ones_like_gradient(tensor: &Tensor) -> Result<Tensor, TensorError> {
    Tensor::ones(&tensor.shape()?, &tensor.dtype()?, &tensor.device()?, false)
}

fn ensure_float_gradient_dtype(dtype: DType) -> Result<(), TensorError> {
    if dtype.is_float() {
        Ok(())
    } else {
        Err(TensorError::InvalidOperation(format!(
            "gradients are only supported for floating point tensors, got {dtype}"
        )))
    }
}

fn unary_parent_grad<F>(
    parents: &[Tensor],
    grad_output: &Tensor,
    f: F,
) -> Result<Vec<(Tensor, Tensor)>, TensorError>
where
    F: FnOnce(&Tensor, &Tensor) -> Result<Tensor, TensorError>,
{
    if parents.len() != 1 {
        return Err(TensorError::InvalidOperation(format!(
            "unary autograd operation expected 1 parent, got {}",
            parents.len()
        )));
    }
    let parent = parents[0].copy();
    let parent_value = parent.detach()?;
    let parent_grad = f(&parent_value, grad_output)?;
    Ok(vec![(
        parent.copy(),
        reduce_to_shape(&parent_grad, &parent.shape()?)?,
    )])
}

fn binary_parent_grads<F>(
    parents: &[Tensor],
    grad_output: &Tensor,
    f: F,
) -> Result<Vec<(Tensor, Tensor)>, TensorError>
where
    F: FnOnce(&Tensor, &Tensor, &Tensor) -> Result<(Tensor, Tensor), TensorError>,
{
    if parents.len() != 2 {
        return Err(TensorError::InvalidOperation(format!(
            "binary autograd operation expected 2 parents, got {}",
            parents.len()
        )));
    }

    let lhs = parents[0].copy();
    let rhs = parents[1].copy();
    let lhs_value = lhs.detach()?;
    let rhs_value = rhs.detach()?;
    let (lhs_grad, rhs_grad) = f(&lhs_value, &rhs_value, grad_output)?;
    Ok(vec![
        (lhs.copy(), reduce_to_shape(&lhs_grad, &lhs.shape()?)?),
        (rhs.copy(), reduce_to_shape(&rhs_grad, &rhs.shape()?)?),
    ])
}

fn indexed_parent_grad<F>(
    parents: &[Tensor],
    grad_output: &Tensor,
    f: F,
) -> Result<Vec<(Tensor, Tensor)>, TensorError>
where
    F: FnOnce(&Tensor, &Tensor) -> Result<Tensor, TensorError>,
{
    if parents.len() != 2 {
        return Err(TensorError::InvalidOperation(format!(
            "indexed autograd operation expected 2 parents, got {}",
            parents.len()
        )));
    }

    let values = parents[0].copy();
    let indexes = parents[1].detach()?;
    let value_grad = f(&indexes, grad_output)?;
    Ok(vec![(
        values.copy(),
        reduce_to_shape(&value_grad, &values.shape()?)?,
    )])
}

fn zero_parent_grads(parents: &[Tensor]) -> Result<Vec<(Tensor, Tensor)>, TensorError> {
    parents
        .iter()
        .map(|parent| {
            Ok((
                parent.copy(),
                Tensor::zeros(&parent.shape()?, &parent.dtype()?, &parent.device()?, false)?,
            ))
        })
        .collect()
}

fn stack_parent_grads(
    parents: &[Tensor],
    grad_output: &Tensor,
    dim: usize,
    input_shapes: &[Shape],
) -> Result<Vec<(Tensor, Tensor)>, TensorError> {
    if parents.len() != input_shapes.len() {
        return Err(TensorError::InvalidOperation(format!(
            "stack autograd operation expected {} parents, got {}",
            input_shapes.len(),
            parents.len()
        )));
    }

    parents
        .iter()
        .enumerate()
        .map(|(index, parent)| {
            let grad = grad_output
                .narrow(dim as isize, index as isize, 1)?
                .squeeze(Some(dim))?;
            Ok((
                parent.copy(),
                reduce_to_shape(&grad, input_shapes.get(index).expect("shape checked above"))?,
            ))
        })
        .collect()
}

fn cat_parent_grads(
    parents: &[Tensor],
    grad_output: &Tensor,
    dim: usize,
    input_shapes: &[Shape],
) -> Result<Vec<(Tensor, Tensor)>, TensorError> {
    if parents.len() != input_shapes.len() {
        return Err(TensorError::InvalidOperation(format!(
            "cat autograd operation expected {} parents, got {}",
            input_shapes.len(),
            parents.len()
        )));
    }

    let mut start = 0;
    let mut grads = Vec::with_capacity(parents.len());
    for (parent, shape) in parents.iter().zip(input_shapes.iter()) {
        let Some(length) = shape.dims().get(dim).copied() else {
            return Err(TensorError::InvalidOperation(format!(
                "cat autograd dimension {dim} is invalid for shape {shape:?}"
            )));
        };
        let grad = grad_output.narrow(dim as isize, start as isize, length)?;
        grads.push((parent.copy(), reduce_to_shape(&grad, shape)?));
        start += length;
    }

    Ok(grads)
}

fn where_cond_parent_grads(
    parents: &[Tensor],
    grad_output: &Tensor,
) -> Result<Vec<(Tensor, Tensor)>, TensorError> {
    if parents.len() != 3 {
        return Err(TensorError::InvalidOperation(format!(
            "where_cond autograd operation expected 3 parents, got {}",
            parents.len()
        )));
    }

    let condition = parents[0].detach()?;
    let true_value = parents[1].copy();
    let false_value = parents[2].copy();
    let true_mask = condition.to_dtype(&true_value.dtype()?)?;
    let false_mask = true_mask.affine(-1.0, 1.0)?;
    let true_grad = grad_output.mul(&true_mask)?;
    let false_grad = grad_output.mul(&false_mask)?;

    Ok(vec![
        (
            true_value.copy(),
            reduce_to_shape(&true_grad, &true_value.shape()?)?,
        ),
        (
            false_value.copy(),
            reduce_to_shape(&false_grad, &false_value.shape()?)?,
        ),
    ])
}

fn narrow_backward_grad(
    grad: &Tensor,
    input_shape: &Shape,
    dim: usize,
    start: usize,
    length: usize,
) -> Result<Tensor, TensorError> {
    let source_shape = grad.shape()?;
    if dim >= input_shape.rank() {
        return Err(TensorError::InvalidOperation(format!(
            "narrow autograd dimension {dim} is invalid for shape {input_shape:?}"
        )));
    }
    if source_shape.rank() != input_shape.rank() || source_shape.dims()[dim] != length {
        return Err(TensorError::InvalidOperation(format!(
            "narrow autograd gradient shape {source_shape:?} is incompatible with input shape {input_shape:?}"
        )));
    }

    match grad.dtype()? {
        DType::F32 => {
            let values = scatter_narrow_values(
                &grad.to_vec::<f32>()?,
                source_shape.dims(),
                input_shape.dims(),
                dim,
                start,
            );
            Tensor::from_slice(&values, input_shape, &grad.device()?, false)
        }
        DType::F64 => {
            let values = scatter_narrow_values(
                &grad.to_vec::<f64>()?,
                source_shape.dims(),
                input_shape.dims(),
                dim,
                start,
            );
            Tensor::from_slice(&values, input_shape, &grad.device()?, false)
        }
        dtype => Err(TensorError::InvalidOperation(format!(
            "narrow autograd supports floating gradients only, got {dtype}"
        ))),
    }
}

fn gather_backward_grad(
    grad: &Tensor,
    indexes: &Tensor,
    input_shape: &Shape,
    dim: usize,
) -> Result<Tensor, TensorError> {
    let index_shape = indexes.shape()?;
    let indexes = if indexes.dtype()? != DType::I64 {
        indexes.to_dtype(&DType::I64)?
    } else {
        indexes.copy()
    };
    let index_values = indexes.to_vec::<i64>()?;
    let input_strides = contiguous_strides(input_shape.dims());
    match grad.dtype()? {
        DType::F32 => {
            let values = scatter_gather_values(
                &grad.to_vec::<f32>()?,
                &index_values,
                index_shape.dims(),
                input_shape.dims(),
                dim,
                &input_strides,
            )?;
            Tensor::from_slice(&values, input_shape, &grad.device()?, false)
        }
        DType::F64 => {
            let values = scatter_gather_values(
                &grad.to_vec::<f64>()?,
                &index_values,
                index_shape.dims(),
                input_shape.dims(),
                dim,
                &input_strides,
            )?;
            Tensor::from_slice(&values, input_shape, &grad.device()?, false)
        }
        dtype => Tensor::zeros(input_shape, &dtype, &grad.device()?, false),
    }
}

fn index_select_backward_grad(
    grad: &Tensor,
    indexes: &Tensor,
    input_shape: &Shape,
    dim: usize,
) -> Result<Tensor, TensorError> {
    let indexes = if indexes.dtype()? != DType::I64 {
        indexes.to_dtype(&DType::I64)?
    } else {
        indexes.copy()
    };
    let index_values = indexes.to_vec::<i64>()?;
    let grad_shape = grad.shape()?;
    let input_strides = contiguous_strides(input_shape.dims());
    match grad.dtype()? {
        DType::F32 => {
            let values = scatter_index_select_values(
                &grad.to_vec::<f32>()?,
                &index_values,
                grad_shape.dims(),
                input_shape.dims(),
                dim,
                &input_strides,
            )?;
            Tensor::from_slice(&values, input_shape, &grad.device()?, false)
        }
        DType::F64 => {
            let values = scatter_index_select_values(
                &grad.to_vec::<f64>()?,
                &index_values,
                grad_shape.dims(),
                input_shape.dims(),
                dim,
                &input_strides,
            )?;
            Tensor::from_slice(&values, input_shape, &grad.device()?, false)
        }
        dtype => Tensor::zeros(input_shape, &dtype, &grad.device()?, false),
    }
}

fn embedding_backward_grad(
    grad: &Tensor,
    indexes: &Tensor,
    table_shape: &Shape,
) -> Result<Tensor, TensorError> {
    let indexes = if indexes.dtype()? != DType::I64 {
        indexes.to_dtype(&DType::I64)?
    } else {
        indexes.copy()
    };
    let index_values = indexes.to_vec::<i64>()?;
    let row_size = table_shape.dims()[1..].iter().product::<usize>();
    match grad.dtype()? {
        DType::F32 => {
            let values = scatter_embedding_values(
                &grad.to_vec::<f32>()?,
                &index_values,
                table_shape.dims(),
                row_size,
            )?;
            Tensor::from_slice(&values, table_shape, &grad.device()?, false)
        }
        DType::F64 => {
            let values = scatter_embedding_values(
                &grad.to_vec::<f64>()?,
                &index_values,
                table_shape.dims(),
                row_size,
            )?;
            Tensor::from_slice(&values, table_shape, &grad.device()?, false)
        }
        dtype => Tensor::zeros(table_shape, &dtype, &grad.device()?, false),
    }
}

fn scatter_narrow_values<T: Copy + Default>(
    values: &[T],
    source_shape: &[usize],
    target_shape: &[usize],
    dim: usize,
    start: usize,
) -> Vec<T> {
    let source_strides = contiguous_strides(source_shape);
    let target_strides = contiguous_strides(target_shape);
    let mut result = vec![T::default(); target_shape.iter().product()];

    for (source_index, value) in values.iter().copied().enumerate() {
        let mut remaining = source_index;
        let mut target_index = 0;
        for axis in 0..source_shape.len() {
            let stride = source_strides[axis];
            let coord = if stride == 0 { 0 } else { remaining / stride };
            if stride != 0 {
                remaining %= stride;
            }
            let target_coord = if axis == dim { coord + start } else { coord };
            target_index += target_coord * target_strides[axis];
        }
        result[target_index] = value;
    }

    result
}

fn scatter_gather_values<T>(
    values: &[T],
    indexes: &[i64],
    index_shape: &[usize],
    target_shape: &[usize],
    dim: usize,
    target_strides: &[usize],
) -> Result<Vec<T>, TensorError>
where
    T: Copy + Default + std::ops::AddAssign,
{
    if values.len() != indexes.len() {
        return Err(TensorError::InvalidOperation(format!(
            "gather autograd expected {} gradient values, got {}",
            indexes.len(),
            values.len()
        )));
    }

    let mut result = vec![T::default(); target_shape.iter().product()];
    for (output_index, value) in values.iter().copied().enumerate() {
        let mut coords = unravel_index(output_index, index_shape);
        let selected = checked_tensor_index(indexes[output_index], target_shape[dim], "gather")?;
        coords[dim] = selected;
        let target_index = ravel_index_with_strides(&coords, target_strides);
        result[target_index] += value;
    }
    Ok(result)
}

fn scatter_index_select_values<T>(
    values: &[T],
    indexes: &[i64],
    source_shape: &[usize],
    target_shape: &[usize],
    dim: usize,
    target_strides: &[usize],
) -> Result<Vec<T>, TensorError>
where
    T: Copy + Default + std::ops::AddAssign,
{
    let mut result = vec![T::default(); target_shape.iter().product()];
    for (source_index, value) in values.iter().copied().enumerate() {
        let mut coords = unravel_index(source_index, source_shape);
        let selected =
            checked_tensor_index(indexes[coords[dim]], target_shape[dim], "index_select")?;
        coords[dim] = selected;
        let target_index = ravel_index_with_strides(&coords, target_strides);
        result[target_index] += value;
    }
    Ok(result)
}

fn scatter_embedding_values<T>(
    values: &[T],
    indexes: &[i64],
    table_shape: &[usize],
    row_size: usize,
) -> Result<Vec<T>, TensorError>
where
    T: Copy + Default + std::ops::AddAssign,
{
    let mut result = vec![T::default(); table_shape.iter().product()];
    if values.len() != indexes.len() * row_size {
        return Err(TensorError::InvalidOperation(format!(
            "embedding autograd expected {} gradient values, got {}",
            indexes.len() * row_size,
            values.len()
        )));
    }

    for (position, row_index) in indexes.iter().copied().enumerate() {
        let row = checked_tensor_index(row_index, table_shape[0], "embedding")?;
        let source_start = position * row_size;
        let target_start = row * row_size;
        for offset in 0..row_size {
            result[target_start + offset] += values[source_start + offset];
        }
    }
    Ok(result)
}

fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut strides = vec![1; shape.len()];
    for index in (1..shape.len()).rev() {
        strides[index - 1] = strides[index] * shape[index];
    }
    strides
}

fn unravel_index(mut index: usize, shape: &[usize]) -> Vec<usize> {
    let mut coords = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        coords[axis] = index % shape[axis];
        index /= shape[axis];
    }
    coords
}

fn ravel_index_with_strides(coords: &[usize], strides: &[usize]) -> usize {
    coords
        .iter()
        .zip(strides.iter())
        .map(|(coord, stride)| coord * stride)
        .sum()
}

fn checked_tensor_index(
    index: i64,
    upper_bound: usize,
    operation: &str,
) -> Result<usize, TensorError> {
    if index < 0 || index as usize >= upper_bound {
        return Err(TensorError::InvalidOperation(format!(
            "{operation} index {index} is out of bounds for dimension size {upper_bound}"
        )));
    }
    Ok(index as usize)
}

fn conv1d_parent_grads(
    parents: &[Tensor],
    grad_output: &Tensor,
    padding: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
    input_shape: &Shape,
    kernel_shape: &Shape,
) -> Result<Vec<(Tensor, Tensor)>, TensorError> {
    if parents.len() != 2 {
        return Err(TensorError::InvalidOperation(format!(
            "conv1d autograd operation expected 2 parents, got {}",
            parents.len()
        )));
    }

    let input_parent = parents[0].copy();
    let kernel_parent = parents[1].copy();
    let input = input_parent.detach()?;
    let kernel = kernel_parent.detach()?;
    let (input_grad, kernel_grad) = conv1d_backward_grads(
        &input,
        &kernel,
        grad_output,
        padding,
        stride,
        dilation,
        groups,
        input_shape,
        kernel_shape,
    )?;

    Ok(vec![
        (
            input_parent.copy(),
            reduce_to_shape(&input_grad, &input_parent.shape()?)?,
        ),
        (
            kernel_parent.copy(),
            reduce_to_shape(&kernel_grad, &kernel_parent.shape()?)?,
        ),
    ])
}

fn conv2d_parent_grads(
    parents: &[Tensor],
    grad_output: &Tensor,
    padding: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
    input_shape: &Shape,
    kernel_shape: &Shape,
) -> Result<Vec<(Tensor, Tensor)>, TensorError> {
    if parents.len() != 2 {
        return Err(TensorError::InvalidOperation(format!(
            "conv2d autograd operation expected 2 parents, got {}",
            parents.len()
        )));
    }

    let input_parent = parents[0].copy();
    let kernel_parent = parents[1].copy();
    let input = input_parent.detach()?;
    let kernel = kernel_parent.detach()?;
    let (input_grad, kernel_grad) = conv2d_backward_grads(
        &input,
        &kernel,
        grad_output,
        padding,
        stride,
        dilation,
        groups,
        input_shape,
        kernel_shape,
    )?;

    Ok(vec![
        (
            input_parent.copy(),
            reduce_to_shape(&input_grad, &input_parent.shape()?)?,
        ),
        (
            kernel_parent.copy(),
            reduce_to_shape(&kernel_grad, &kernel_parent.shape()?)?,
        ),
    ])
}

fn conv1d_backward_grads(
    input: &Tensor,
    kernel: &Tensor,
    grad: &Tensor,
    padding: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
    input_shape: &Shape,
    kernel_shape: &Shape,
) -> Result<(Tensor, Tensor), TensorError> {
    ensure_conv_rank(input_shape, 3, "conv1d")?;
    ensure_conv_rank(kernel_shape, 3, "conv1d")?;
    if stride == 0 || dilation == 0 || groups == 0 {
        return Err(TensorError::InvalidOperation(
            "conv1d autograd stride, dilation, and groups must be non-zero".to_string(),
        ));
    }

    let input_dims = input_shape.dims();
    let kernel_dims = kernel_shape.dims();
    let [batch, input_channels, input_length] = [input_dims[0], input_dims[1], input_dims[2]];
    let [output_channels, kernel_channels, kernel_size] =
        [kernel_dims[0], kernel_dims[1], kernel_dims[2]];
    if input_channels % groups != 0 || output_channels % groups != 0 {
        return Err(TensorError::InvalidOperation(format!(
            "conv1d autograd channels must be divisible by groups={groups}, got input_channels={input_channels}, output_channels={output_channels}"
        )));
    }
    let input_channels_per_group = input_channels / groups;
    let output_channels_per_group = output_channels / groups;
    if kernel_channels != input_channels_per_group {
        return Err(TensorError::InvalidOperation(format!(
            "conv1d autograd expected kernel channels {input_channels_per_group}, got {kernel_channels}"
        )));
    }
    let output_length = conv_output_size(
        input_length,
        kernel_size,
        padding,
        stride,
        dilation,
        "conv1d",
    )?;
    let grad_shape = grad.shape()?;
    let expected_grad_shape = Shape::from_dims(&[batch, output_channels, output_length]);
    if grad_shape != expected_grad_shape {
        return Err(TensorError::InvalidOperation(format!(
            "conv1d autograd expected gradient shape {expected_grad_shape:?}, got {grad_shape:?}"
        )));
    }

    let input_values = tensor_float_values(input)?;
    let kernel_values = tensor_float_values(kernel)?;
    let grad_values = tensor_float_values(grad)?;
    let input_strides = contiguous_strides(input_dims);
    let kernel_strides = contiguous_strides(kernel_dims);
    let mut input_grad = vec![0.0; input_shape.elem_count()];
    let mut kernel_grad = vec![0.0; kernel_shape.elem_count()];

    for n in 0..batch {
        for oc in 0..output_channels {
            let group = oc / output_channels_per_group;
            let input_channel_start = group * input_channels_per_group;
            for out_pos in 0..output_length {
                let grad_index = (n * output_channels + oc) * output_length + out_pos;
                let grad_value = grad_values[grad_index];
                for ic_local in 0..kernel_channels {
                    let ic = input_channel_start + ic_local;
                    for k in 0..kernel_size {
                        let padded_pos = out_pos * stride + k * dilation;
                        if padded_pos < padding {
                            continue;
                        }
                        let input_pos = padded_pos - padding;
                        if input_pos >= input_length {
                            continue;
                        }
                        let input_index = n * input_strides[0] + ic * input_strides[1] + input_pos;
                        let kernel_index =
                            oc * kernel_strides[0] + ic_local * kernel_strides[1] + k;
                        input_grad[input_index] += grad_value * kernel_values[kernel_index];
                        kernel_grad[kernel_index] += grad_value * input_values[input_index];
                    }
                }
            }
        }
    }

    Ok((
        tensor_from_float_values(input_grad, input_shape, input)?,
        tensor_from_float_values(kernel_grad, kernel_shape, kernel)?,
    ))
}

fn conv2d_backward_grads(
    input: &Tensor,
    kernel: &Tensor,
    grad: &Tensor,
    padding: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
    input_shape: &Shape,
    kernel_shape: &Shape,
) -> Result<(Tensor, Tensor), TensorError> {
    ensure_conv_rank(input_shape, 4, "conv2d")?;
    ensure_conv_rank(kernel_shape, 4, "conv2d")?;
    if stride == 0 || dilation == 0 || groups == 0 {
        return Err(TensorError::InvalidOperation(
            "conv2d autograd stride, dilation, and groups must be non-zero".to_string(),
        ));
    }

    let input_dims = input_shape.dims();
    let kernel_dims = kernel_shape.dims();
    let [batch, input_channels, input_h, input_w] =
        [input_dims[0], input_dims[1], input_dims[2], input_dims[3]];
    let [output_channels, kernel_channels, kernel_h, kernel_w] = [
        kernel_dims[0],
        kernel_dims[1],
        kernel_dims[2],
        kernel_dims[3],
    ];
    if input_channels % groups != 0 || output_channels % groups != 0 {
        return Err(TensorError::InvalidOperation(format!(
            "conv2d autograd channels must be divisible by groups={groups}, got input_channels={input_channels}, output_channels={output_channels}"
        )));
    }
    let input_channels_per_group = input_channels / groups;
    let output_channels_per_group = output_channels / groups;
    if kernel_channels != input_channels_per_group {
        return Err(TensorError::InvalidOperation(format!(
            "conv2d autograd expected kernel channels {input_channels_per_group}, got {kernel_channels}"
        )));
    }
    let output_h = conv_output_size(input_h, kernel_h, padding, stride, dilation, "conv2d")?;
    let output_w = conv_output_size(input_w, kernel_w, padding, stride, dilation, "conv2d")?;
    let grad_shape = grad.shape()?;
    let expected_grad_shape = Shape::from_dims(&[batch, output_channels, output_h, output_w]);
    if grad_shape != expected_grad_shape {
        return Err(TensorError::InvalidOperation(format!(
            "conv2d autograd expected gradient shape {expected_grad_shape:?}, got {grad_shape:?}"
        )));
    }

    let input_values = tensor_float_values(input)?;
    let kernel_values = tensor_float_values(kernel)?;
    let grad_values = tensor_float_values(grad)?;
    let input_strides = contiguous_strides(input_dims);
    let kernel_strides = contiguous_strides(kernel_dims);
    let mut input_grad = vec![0.0; input_shape.elem_count()];
    let mut kernel_grad = vec![0.0; kernel_shape.elem_count()];

    for n in 0..batch {
        for oc in 0..output_channels {
            let group = oc / output_channels_per_group;
            let input_channel_start = group * input_channels_per_group;
            for oh in 0..output_h {
                for ow in 0..output_w {
                    let grad_index = ((n * output_channels + oc) * output_h + oh) * output_w + ow;
                    let grad_value = grad_values[grad_index];
                    for ic_local in 0..kernel_channels {
                        let ic = input_channel_start + ic_local;
                        for kh in 0..kernel_h {
                            let padded_h = oh * stride + kh * dilation;
                            if padded_h < padding {
                                continue;
                            }
                            let ih = padded_h - padding;
                            if ih >= input_h {
                                continue;
                            }
                            for kw in 0..kernel_w {
                                let padded_w = ow * stride + kw * dilation;
                                if padded_w < padding {
                                    continue;
                                }
                                let iw = padded_w - padding;
                                if iw >= input_w {
                                    continue;
                                }
                                let input_index = n * input_strides[0]
                                    + ic * input_strides[1]
                                    + ih * input_strides[2]
                                    + iw;
                                let kernel_index = oc * kernel_strides[0]
                                    + ic_local * kernel_strides[1]
                                    + kh * kernel_strides[2]
                                    + kw;
                                input_grad[input_index] += grad_value * kernel_values[kernel_index];
                                kernel_grad[kernel_index] += grad_value * input_values[input_index];
                            }
                        }
                    }
                }
            }
        }
    }

    Ok((
        tensor_from_float_values(input_grad, input_shape, input)?,
        tensor_from_float_values(kernel_grad, kernel_shape, kernel)?,
    ))
}

fn avg_pool1d_backward_grad(
    grad: &Tensor,
    input_shape: &Shape,
    kernel_size: usize,
    stride: usize,
) -> Result<Tensor, TensorError> {
    ensure_pool_rank(input_shape, 3, "avg_pool1d")?;
    let dims = input_shape.dims();
    let [batch, channels, length] = [dims[0], dims[1], dims[2]];
    let output_length = (length - kernel_size) / stride + 1;
    let grad_values = tensor_float_values(grad)?;
    let mut result = vec![0.0; input_shape.elem_count()];
    let input_strides = contiguous_strides(dims);
    let share = 1.0 / kernel_size as f64;

    for n in 0..batch {
        for c in 0..channels {
            for out_pos in 0..output_length {
                let grad_index = (n * channels + c) * output_length + out_pos;
                let value = grad_values[grad_index] * share;
                let start = out_pos * stride;
                for offset in 0..kernel_size {
                    let input_index =
                        n * input_strides[0] + c * input_strides[1] + (start + offset);
                    result[input_index] += value;
                }
            }
        }
    }

    tensor_from_float_values(result, input_shape, grad)
}

fn max_pool1d_backward_grad(
    input: &Tensor,
    grad: &Tensor,
    input_shape: &Shape,
    kernel_size: usize,
    stride: usize,
) -> Result<Tensor, TensorError> {
    ensure_pool_rank(input_shape, 3, "max_pool1d")?;
    let dims = input_shape.dims();
    let [batch, channels, length] = [dims[0], dims[1], dims[2]];
    let output_length = (length - kernel_size) / stride + 1;
    let input_values = tensor_float_values(input)?;
    let grad_values = tensor_float_values(grad)?;
    let input_strides = contiguous_strides(dims);
    let mut result = vec![0.0; input_shape.elem_count()];

    for n in 0..batch {
        for c in 0..channels {
            for out_pos in 0..output_length {
                let start = out_pos * stride;
                let mut max_value = f64::NEG_INFINITY;
                for offset in 0..kernel_size {
                    let input_index =
                        n * input_strides[0] + c * input_strides[1] + (start + offset);
                    max_value = max_value.max(input_values[input_index]);
                }

                let mut ties = 0;
                for offset in 0..kernel_size {
                    let input_index =
                        n * input_strides[0] + c * input_strides[1] + (start + offset);
                    if input_values[input_index] == max_value {
                        ties += 1;
                    }
                }

                let grad_index = (n * channels + c) * output_length + out_pos;
                let value = grad_values[grad_index] / ties as f64;
                for offset in 0..kernel_size {
                    let input_index =
                        n * input_strides[0] + c * input_strides[1] + (start + offset);
                    if input_values[input_index] == max_value {
                        result[input_index] += value;
                    }
                }
            }
        }
    }

    tensor_from_float_values(result, input_shape, grad)
}

fn avg_pool2d_backward_grad(
    grad: &Tensor,
    input_shape: &Shape,
    kernel_size: (usize, usize),
    stride: (usize, usize),
) -> Result<Tensor, TensorError> {
    ensure_pool_rank(input_shape, 4, "avg_pool2d")?;
    let dims = input_shape.dims();
    let [batch, channels, height, width] = [dims[0], dims[1], dims[2], dims[3]];
    let (kernel_h, kernel_w) = kernel_size;
    let (stride_h, stride_w) = stride;
    let output_h = (height - kernel_h) / stride_h + 1;
    let output_w = (width - kernel_w) / stride_w + 1;
    let grad_values = tensor_float_values(grad)?;
    let input_strides = contiguous_strides(dims);
    let mut result = vec![0.0; input_shape.elem_count()];
    let share = 1.0 / (kernel_h * kernel_w) as f64;

    for n in 0..batch {
        for c in 0..channels {
            for oh in 0..output_h {
                for ow in 0..output_w {
                    let grad_index = ((n * channels + c) * output_h + oh) * output_w + ow;
                    let value = grad_values[grad_index] * share;
                    let h_start = oh * stride_h;
                    let w_start = ow * stride_w;
                    for kh in 0..kernel_h {
                        for kw in 0..kernel_w {
                            let input_index = n * input_strides[0]
                                + c * input_strides[1]
                                + (h_start + kh) * input_strides[2]
                                + (w_start + kw) * input_strides[3];
                            result[input_index] += value;
                        }
                    }
                }
            }
        }
    }

    tensor_from_float_values(result, input_shape, grad)
}

fn max_pool2d_backward_grad(
    input: &Tensor,
    grad: &Tensor,
    input_shape: &Shape,
    kernel_size: (usize, usize),
    stride: (usize, usize),
) -> Result<Tensor, TensorError> {
    ensure_pool_rank(input_shape, 4, "max_pool2d")?;
    let dims = input_shape.dims();
    let [batch, channels, height, width] = [dims[0], dims[1], dims[2], dims[3]];
    let (kernel_h, kernel_w) = kernel_size;
    let (stride_h, stride_w) = stride;
    let output_h = (height - kernel_h) / stride_h + 1;
    let output_w = (width - kernel_w) / stride_w + 1;
    let input_values = tensor_float_values(input)?;
    let grad_values = tensor_float_values(grad)?;
    let input_strides = contiguous_strides(dims);
    let mut result = vec![0.0; input_shape.elem_count()];

    for n in 0..batch {
        for c in 0..channels {
            for oh in 0..output_h {
                for ow in 0..output_w {
                    let h_start = oh * stride_h;
                    let w_start = ow * stride_w;
                    let mut max_value = f64::NEG_INFINITY;
                    for kh in 0..kernel_h {
                        for kw in 0..kernel_w {
                            let input_index = n * input_strides[0]
                                + c * input_strides[1]
                                + (h_start + kh) * input_strides[2]
                                + (w_start + kw) * input_strides[3];
                            max_value = max_value.max(input_values[input_index]);
                        }
                    }

                    let mut ties = 0;
                    for kh in 0..kernel_h {
                        for kw in 0..kernel_w {
                            let input_index = n * input_strides[0]
                                + c * input_strides[1]
                                + (h_start + kh) * input_strides[2]
                                + (w_start + kw) * input_strides[3];
                            if input_values[input_index] == max_value {
                                ties += 1;
                            }
                        }
                    }

                    let grad_index = ((n * channels + c) * output_h + oh) * output_w + ow;
                    let value = grad_values[grad_index] / ties as f64;
                    for kh in 0..kernel_h {
                        for kw in 0..kernel_w {
                            let input_index = n * input_strides[0]
                                + c * input_strides[1]
                                + (h_start + kh) * input_strides[2]
                                + (w_start + kw) * input_strides[3];
                            if input_values[input_index] == max_value {
                                result[input_index] += value;
                            }
                        }
                    }
                }
            }
        }
    }

    tensor_from_float_values(result, input_shape, grad)
}

fn ensure_pool_rank(shape: &Shape, rank: usize, operation: &str) -> Result<(), TensorError> {
    if shape.rank() != rank {
        return Err(TensorError::InvalidOperation(format!(
            "{operation} autograd expects rank-{rank} input, got {shape:?}"
        )));
    }
    Ok(())
}

fn ensure_conv_rank(shape: &Shape, rank: usize, operation: &str) -> Result<(), TensorError> {
    if shape.rank() != rank {
        return Err(TensorError::InvalidOperation(format!(
            "{operation} autograd expects rank-{rank} tensor, got {shape:?}"
        )));
    }
    Ok(())
}

fn conv_output_size(
    input_size: usize,
    kernel_size: usize,
    padding: usize,
    stride: usize,
    dilation: usize,
    operation: &str,
) -> Result<usize, TensorError> {
    if kernel_size == 0 {
        return Err(TensorError::InvalidOperation(format!(
            "{operation} autograd kernel dimensions must be non-zero"
        )));
    }
    let effective_kernel = dilation
        .checked_mul(kernel_size - 1)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| {
            TensorError::InvalidOperation(format!(
                "{operation} autograd effective kernel size overflowed"
            ))
        })?;
    let padded_input = padding
        .checked_mul(2)
        .and_then(|value| input_size.checked_add(value))
        .ok_or_else(|| {
            TensorError::InvalidOperation(format!(
                "{operation} autograd padded input size overflowed"
            ))
        })?;
    if padded_input < effective_kernel {
        return Err(TensorError::InvalidOperation(format!(
            "{operation} autograd effective kernel size {effective_kernel} is larger than padded input size {padded_input}"
        )));
    }
    Ok((padded_input - effective_kernel) / stride + 1)
}

fn tensor_float_values(tensor: &Tensor) -> Result<Vec<f64>, TensorError> {
    match tensor.dtype()? {
        DType::F32 => Ok(tensor
            .to_vec::<f32>()?
            .into_iter()
            .map(|value| value as f64)
            .collect()),
        DType::F64 => tensor.to_vec::<f64>(),
        dtype => Err(TensorError::InvalidOperation(format!(
            "native autograd supports floating tensors only, got {dtype}"
        ))),
    }
}

fn tensor_from_float_values(
    values: Vec<f64>,
    shape: &Shape,
    reference: &Tensor,
) -> Result<Tensor, TensorError> {
    match reference.dtype()? {
        DType::F32 => {
            let values = values
                .into_iter()
                .map(|value| value as f32)
                .collect::<Vec<_>>();
            Tensor::from_slice(&values, shape, &reference.device()?, false)
        }
        DType::F64 => Tensor::from_slice(&values, shape, &reference.device()?, false),
        dtype => Err(TensorError::InvalidOperation(format!(
            "native autograd supports floating tensors only, got {dtype}"
        ))),
    }
}

fn var_parent_grad(
    input: &Tensor,
    grad_output: &Tensor,
    dim: usize,
    keep_dim: bool,
    unbiased: bool,
    input_shape: &Shape,
) -> Result<Tensor, TensorError> {
    let reduced = input_shape.dims()[dim];
    if reduced <= 1 {
        return Err(TensorError::InvalidOperation(
            "variance autograd requires a reduced dimension larger than 1".to_string(),
        ));
    }

    let mean = input.mean(Some((dim, true)))?;
    let centered = input.sub(&mean)?;
    let grad = expand_reduction_grad(grad_output, Some((dim, keep_dim)), input_shape)?;
    let mut scale = 2.0 / (reduced - 1) as f64;

    if !unbiased {
        let total_size = input_shape.elem_count();
        let num_features = input_shape.dims()[dim];
        let n = total_size / num_features;
        scale *= (n as f64 - 1.0) / n as f64;
    }

    grad.mul(&centered)?.affine(scale, 0.0)
}

#[derive(Debug, Clone, Copy)]
enum ExtremaKind {
    Max,
    Min,
}

fn extrema_parent_grad(
    output: &Tensor,
    parents: &[Tensor],
    grad_output: &Tensor,
    axis: Option<(usize, bool)>,
    input_shape: &Shape,
    kind: ExtremaKind,
) -> Result<Vec<(Tensor, Tensor)>, TensorError> {
    unary_parent_grad(parents, grad_output, |input, grad| {
        let output = expand_reduction_grad(&output.detach()?, axis, input_shape)?;
        let grad = expand_reduction_grad(grad, axis, input_shape)?;
        let mask = match kind {
            ExtremaKind::Max => input.eq(&output)?,
            ExtremaKind::Min => input.eq(&output)?,
        }
        .to_dtype(&input.dtype()?)?;
        let count = expand_reduction_grad(&mask.sum(axis)?, axis, input_shape)?;
        grad.mul(&mask)?.div(&count)
    })
}

fn gelu_tanh_derivative(input: &Tensor) -> Result<Tensor, TensorError> {
    let coefficient = 0.797_884_560_802_865_4;
    let cubic_coefficient = 0.044_715;
    let inner = input
        .add(&input.powf(3.0)?.affine(cubic_coefficient, 0.0)?)?
        .affine(coefficient, 0.0)?;
    let tanh_inner = inner.tanh()?;
    let left = tanh_inner.affine(0.5, 0.5)?;
    let right = input
        .mul(&tanh_inner.powf(2.0)?.affine(-1.0, 1.0)?)?
        .mul(&input.powf(2.0)?.affine(3.0 * cubic_coefficient, 1.0)?)?
        .affine(0.5 * coefficient, 0.0)?;
    left.add(&right)
}

fn reduce_to_shape(grad: &Tensor, target: &Shape) -> Result<Tensor, TensorError> {
    let mut result = grad.copy();

    while result.shape()?.rank() > target.rank() {
        result = result.sum(Some((0, false)))?;
    }

    for dim in 0..target.rank() {
        let current_shape = result.shape()?;
        let current_dim = current_shape.dims()[dim];
        let target_dim = target.dims()[dim];
        if target_dim == 1 && current_dim != 1 {
            result = result.sum(Some((dim, true)))?;
        }
    }

    if result.shape()? != *target {
        result = result.reshape(target)?;
    }

    Ok(result.detach()?)
}

fn expand_reduction_grad(
    grad: &Tensor,
    axis: Option<(usize, bool)>,
    input_shape: &Shape,
) -> Result<Tensor, TensorError> {
    let mut result = grad.copy();
    if let Some((dim, false)) = axis {
        let mut dims = input_shape.dims().to_vec();
        dims[dim] = 1;
        result = result.reshape(&Shape::from_dims(&dims))?;
    }
    if result.shape()? != *input_shape {
        result = result.broadcast(input_shape)?;
    }
    Ok(result.detach()?)
}

fn reduction_factor(axis: Option<(usize, bool)>, input_shape: &Shape) -> usize {
    match axis {
        Some((dim, _)) => input_shape.dims()[dim],
        None => input_shape.elem_count(),
    }
}

fn scalar_like(value: f64, dtype: DType, device: &Device) -> Result<Tensor, TensorError> {
    match dtype {
        DType::F32 => Tensor::from_scalar(value as f32, device, false),
        DType::F64 => Tensor::from_scalar(value, device, false),
        dtype => Err(TensorError::InvalidOperation(format!(
            "cannot create floating gradient scalar for dtype {dtype}"
        ))),
    }
}
