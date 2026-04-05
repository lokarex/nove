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
    /// * Forward pass with value verification
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    ///
    /// // Create a 2x3 tensor for testing
    /// let tensor = Tensor::from_data(vec![vec![1.0f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device, false).unwrap();
    ///
    /// // Case 1: Sum all elements (global sum)
    /// let result_all = tensor.sum(None).unwrap();
    /// assert_eq!(result_all.shape().unwrap(), Shape::from_dims(&[]));
    /// assert_eq!(result_all.to_vec::<f32>().unwrap(), vec![21.0]);
    ///
    /// // Case 2: Sum along dimension 0 (remove dimension)
    /// let result_dim0 = tensor.sum(Some((0, false))).unwrap();
    /// assert_eq!(result_dim0.shape().unwrap(), Shape::from_dims(&[3]));
    /// assert_eq!(result_dim0.to_vec::<f32>().unwrap(), vec![5.0, 7.0, 9.0]);
    ///
    /// // Case 3: Sum along dimension 1 (keep dimension)
    /// let result_dim1 = tensor.sum(Some((1, true))).unwrap();
    /// assert_eq!(result_dim1.shape().unwrap(), Shape::from_dims(&[2, 1]));
    /// assert_eq!(result_dim1.to_vec::<f32>().unwrap(), vec![6.0, 15.0]);
    /// ```
    ///
    /// * Backward pass with gradient verification
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    ///
    /// // Create a 2x3 tensor with requires_grad=true for gradient test
    /// let mut tensor = Tensor::from_data(vec![vec![1.0f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device, true).unwrap();
    ///
    /// // Test gradient for global sum
    /// let result_all = tensor.sum(None).unwrap();
    /// result_all.backward().unwrap();
    /// let grad_all = tensor.grad().unwrap().unwrap();
    /// assert_eq!(grad_all.shape().unwrap(), Shape::from_dims(&[2, 3]));
    /// assert_eq!(grad_all.to_vec::<f32>().unwrap(), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    ///
    /// // Clear gradient for next test
    /// tensor.zero_grad();
    ///
    /// // Test gradient for sum along dimension 0
    /// let result_dim0 = tensor.sum(Some((0, false))).unwrap();
    /// result_dim0.backward().unwrap();
    /// let grad_dim0 = tensor.grad().unwrap().unwrap();
    /// assert_eq!(grad_dim0.shape().unwrap(), Shape::from_dims(&[2, 3]));
    /// assert_eq!(grad_dim0.to_vec::<f32>().unwrap(), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    ///
    /// // Clear gradient for next test
    /// tensor.zero_grad();
    ///
    /// // Test gradient for sum along dimension 1 (keep dimension)
    /// let result_dim1 = tensor.sum(Some((1, true))).unwrap();
    /// result_dim1.backward().unwrap();
    /// let grad_dim1 = tensor.grad().unwrap().unwrap();
    /// assert_eq!(grad_dim1.shape().unwrap(), Shape::from_dims(&[2, 3]));
    /// assert_eq!(grad_dim1.to_vec::<f32>().unwrap(), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
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
    /// * Forward pass with value verification
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    ///
    /// // Create a 2x3 tensor for testing
    /// let tensor = Tensor::from_data(vec![vec![1.0f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device, false).unwrap();
    ///
    /// // Case 1: Max of all elements (global maximum)
    /// let result_all = tensor.max(None).unwrap();
    /// assert_eq!(result_all.shape().unwrap(), Shape::from_dims(&[]));
    /// assert_eq!(result_all.to_vec::<f32>().unwrap(), vec![6.0]);
    ///
    /// // Case 2: Max along dimension 0 (remove dimension)
    /// let result_dim0 = tensor.max(Some((0, false))).unwrap();
    /// assert_eq!(result_dim0.shape().unwrap(), Shape::from_dims(&[3]));
    /// assert_eq!(result_dim0.to_vec::<f32>().unwrap(), vec![4.0, 5.0, 6.0]);
    ///
    /// // Case 3: Max along dimension 1 (keep dimension)
    /// let result_dim1 = tensor.max(Some((1, true))).unwrap();
    /// assert_eq!(result_dim1.shape().unwrap(), Shape::from_dims(&[2, 1]));
    /// assert_eq!(result_dim1.to_vec::<f32>().unwrap(), vec![3.0, 6.0]);
    /// ```
    ///
    /// * Backward pass with gradient verification
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    ///
    /// // Create a 2x3 tensor with requires_grad=true for gradient test
    /// let mut tensor = Tensor::from_data(vec![vec![1.0f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device, true).unwrap();
    ///
    /// // Test gradient for global max
    /// let result_all = tensor.max(None).unwrap();
    /// result_all.backward().unwrap();
    /// let grad_all = tensor.grad().unwrap().unwrap();
    /// // Gradient should be 1 at the maximum element (6.0), 0 elsewhere
    /// assert_eq!(grad_all.shape().unwrap(), Shape::from_dims(&[2, 3]));
    /// assert_eq!(grad_all.to_vec::<f32>().unwrap(), vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
    ///
    /// // Clear gradient for next test
    /// tensor.zero_grad();
    ///
    /// // Test gradient for max along dimension 0
    /// let result_dim0 = tensor.max(Some((0, false))).unwrap();
    /// result_dim0.backward().unwrap();
    /// let grad_dim0 = tensor.grad().unwrap().unwrap();
    /// // Gradient should be 1 at maximum elements in each column: (row1,col0)=1, (row1,col1)=1, (row1,col2)=1
    /// assert_eq!(grad_dim0.shape().unwrap(), Shape::from_dims(&[2, 3]));
    /// assert_eq!(grad_dim0.to_vec::<f32>().unwrap(), vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    ///
    /// // Clear gradient for next test
    /// tensor.zero_grad();
    ///
    /// // Test gradient for max along dimension 1 (keep dimension)
    /// let result_dim1 = tensor.max(Some((1, true))).unwrap();
    /// result_dim1.backward().unwrap();
    /// let grad_dim1 = tensor.grad().unwrap().unwrap();
    /// // Gradient should be 1 at maximum elements in each row: [0,0,1], [0,0,1]
    /// assert_eq!(grad_dim1.shape().unwrap(), Shape::from_dims(&[2, 3]));
    /// assert_eq!(grad_dim1.to_vec::<f32>().unwrap(), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
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
    /// * Forward pass with value verification
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    ///
    /// // Create a 2x3 tensor for testing
    /// let tensor = Tensor::from_data(vec![vec![1.0f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device, false).unwrap();
    ///
    /// // Case 1: Min of all elements (global minimum)
    /// let result_all = tensor.min(None).unwrap();
    /// assert_eq!(result_all.shape().unwrap(), Shape::from_dims(&[]));
    /// assert_eq!(result_all.to_vec::<f32>().unwrap(), vec![1.0]);
    ///
    /// // Case 2: Min along dimension 0 (remove dimension)
    /// let result_dim0 = tensor.min(Some((0, false))).unwrap();
    /// assert_eq!(result_dim0.shape().unwrap(), Shape::from_dims(&[3]));
    /// assert_eq!(result_dim0.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
    ///
    /// // Case 3: Min along dimension 1 (keep dimension)
    /// let result_dim1 = tensor.min(Some((1, true))).unwrap();
    /// assert_eq!(result_dim1.shape().unwrap(), Shape::from_dims(&[2, 1]));
    /// assert_eq!(result_dim1.to_vec::<f32>().unwrap(), vec![1.0, 4.0]);
    /// ```
    ///
    /// * Backward pass with gradient verification
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    ///
    /// // Create a 2x3 tensor with requires_grad=true for gradient test
    /// let mut tensor = Tensor::from_data(vec![vec![1.0f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device, true).unwrap();
    ///
    /// // Test gradient for global min
    /// let result_all = tensor.min(None).unwrap();
    /// result_all.backward().unwrap();
    /// let grad_all = tensor.grad().unwrap().unwrap();
    /// // Gradient should be 1 at the minimum element (1.0), 0 elsewhere
    /// assert_eq!(grad_all.shape().unwrap(), Shape::from_dims(&[2, 3]));
    /// assert_eq!(grad_all.to_vec::<f32>().unwrap(), vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    ///
    /// // Clear gradient for next test
    /// tensor.zero_grad();
    ///
    /// // Test gradient for min along dimension 0
    /// let result_dim0 = tensor.min(Some((0, false))).unwrap();
    /// result_dim0.backward().unwrap();
    /// let grad_dim0 = tensor.grad().unwrap().unwrap();
    /// // Gradient should be 1 at minimum elements in each column: (row0,col0)=1, (row0,col1)=1, (row0,col2)=1
    /// assert_eq!(grad_dim0.shape().unwrap(), Shape::from_dims(&[2, 3]));
    /// assert_eq!(grad_dim0.to_vec::<f32>().unwrap(), vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
    ///
    /// // Clear gradient for next test
    /// tensor.zero_grad();
    ///
    /// // Test gradient for min along dimension 1 (keep dimension)
    /// let result_dim1 = tensor.min(Some((1, true))).unwrap();
    /// result_dim1.backward().unwrap();
    /// let grad_dim1 = tensor.grad().unwrap().unwrap();
    /// // Gradient should be 1 at minimum elements in each row: (row0,col0)=1, (row1,col0)=1
    /// assert_eq!(grad_dim1.shape().unwrap(), Shape::from_dims(&[2, 3]));
    /// assert_eq!(grad_dim1.to_vec::<f32>().unwrap(), vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
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
    /// * Forward pass with value verification
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    ///
    /// // Create a 2x3 tensor for testing
    /// let tensor = Tensor::from_data(vec![vec![1.0f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device, false).unwrap();
    ///
    /// // Case 1: Mean of all elements (global mean)
    /// let result_all = tensor.mean(None).unwrap();
    /// assert_eq!(result_all.shape().unwrap(), Shape::from_dims(&[]));
    /// assert!((result_all.to_vec::<f32>().unwrap()[0] - 3.5).abs() < 1e-6);
    ///
    /// // Case 2: Mean along dimension 0 (remove dimension)
    /// let result_dim0 = tensor.mean(Some((0, false))).unwrap();
    /// assert_eq!(result_dim0.shape().unwrap(), Shape::from_dims(&[3]));
    /// let expected_dim0 = vec![2.5, 3.5, 4.5];
    /// let actual_dim0 = result_dim0.to_vec::<f32>().unwrap();
    /// for i in 0..3 {
    ///     assert!((actual_dim0[i] - expected_dim0[i]).abs() < 1e-6, "Mismatch at index {}: {} != {}", i, actual_dim0[i], expected_dim0[i]);
    /// }
    ///
    /// // Case 3: Mean along dimension 1 (keep dimension)
    /// let result_dim1 = tensor.mean(Some((1, true))).unwrap();
    /// assert_eq!(result_dim1.shape().unwrap(), Shape::from_dims(&[2, 1]));
    /// let expected_dim1 = vec![2.0, 5.0];
    /// let actual_dim1 = result_dim1.to_vec::<f32>().unwrap();
    /// for i in 0..2 {
    ///     assert!((actual_dim1[i] - expected_dim1[i]).abs() < 1e-6, "Mismatch at index {}: {} != {}", i, actual_dim1[i], expected_dim1[i]);
    /// }
    /// ```
    ///
    /// * Backward pass with gradient verification
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    ///
    /// // Create a 2x3 tensor with requires_grad=true for gradient test
    /// let mut tensor = Tensor::from_data(vec![vec![1.0f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device, true).unwrap();
    ///
    /// // Test gradient for global mean
    /// let result_all = tensor.mean(None).unwrap();
    /// result_all.backward().unwrap();
    /// let grad_all = tensor.grad().unwrap().unwrap();
    /// // Gradient for mean is 1/n for each element, where n = total elements = 6
    /// assert_eq!(grad_all.shape().unwrap(), Shape::from_dims(&[2, 3]));
    /// let expected_grad_all = vec![1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0];
    /// let actual_grad_all = grad_all.to_vec::<f32>().unwrap();
    /// for i in 0..6 {
    ///     assert!((actual_grad_all[i] - expected_grad_all[i]).abs() < 1e-6, "Mismatch at index {}: {} != {}", i, actual_grad_all[i], expected_grad_all[i]);
    /// }
    ///
    /// // Clear gradient for next test
    /// tensor.zero_grad();
    ///
    /// // Test gradient for mean along dimension 0
    /// let result_dim0 = tensor.mean(Some((0, false))).unwrap();
    /// result_dim0.backward().unwrap();
    /// let grad_dim0 = tensor.grad().unwrap().unwrap();
    /// // Gradient for mean along dim 0 is 1/n for each element in that column, where n = rows = 2
    /// assert_eq!(grad_dim0.shape().unwrap(), Shape::from_dims(&[2, 3]));
    /// let expected_grad_dim0 = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
    /// let actual_grad_dim0 = grad_dim0.to_vec::<f32>().unwrap();
    /// for i in 0..6 {
    ///     assert!((actual_grad_dim0[i] - expected_grad_dim0[i]).abs() < 1e-6, "Mismatch at index {}: {} != {}", i, actual_grad_dim0[i], expected_grad_dim0[i]);
    /// }
    ///
    /// // Clear gradient for next test
    /// tensor.zero_grad();
    ///
    /// // Test gradient for mean along dimension 1 (keep dimension)
    /// let result_dim1 = tensor.mean(Some((1, true))).unwrap();
    /// result_dim1.backward().unwrap();
    /// let grad_dim1 = tensor.grad().unwrap().unwrap();
    /// // Gradient for mean along dim 1 is 1/n for each element in that row, where n = columns = 3
    /// assert_eq!(grad_dim1.shape().unwrap(), Shape::from_dims(&[2, 3]));
    /// let expected_grad_dim1 = vec![1.0/3.0, 1.0/3.0, 1.0/3.0, 1.0/3.0, 1.0/3.0, 1.0/3.0];
    /// let actual_grad_dim1 = grad_dim1.to_vec::<f32>().unwrap();
    /// for i in 0..6 {
    ///     assert!((actual_grad_dim1[i] - expected_grad_dim1[i]).abs() < 1e-6, "Mismatch at index {}: {} != {}", i, actual_grad_dim1[i], expected_grad_dim1[i]);
    /// }
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
    /// * Forward pass with value verification
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    ///
    /// // Create a 2x3 tensor for testing
    /// let tensor = Tensor::from_data(vec![vec![1.0f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device, false).unwrap();
    ///
    /// // Case 1: Variance along dimension 0, remove dimension, unbiased=true
    /// let result_dim0_unbiased = tensor.var(0, false, true).unwrap();
    /// assert_eq!(result_dim0_unbiased.shape().unwrap(), Shape::from_dims(&[3]));
    /// // Unbiased variance for each column: [4.5, 4.5, 4.5]
    /// let expected_dim0_unbiased = vec![4.5, 4.5, 4.5];
    /// let actual_dim0_unbiased = result_dim0_unbiased.to_vec::<f32>().unwrap();
    /// for i in 0..3 {
    ///     assert!((actual_dim0_unbiased[i] - expected_dim0_unbiased[i]).abs() < 1e-6, "Mismatch at index {}: {} != {}", i, actual_dim0_unbiased[i], expected_dim0_unbiased[i]);
    /// }
    ///
    /// // Case 2: Variance along dimension 0, remove dimension, unbiased=false
    /// let result_dim0_biased = tensor.var(0, false, false).unwrap();
    /// assert_eq!(result_dim0_biased.shape().unwrap(), Shape::from_dims(&[3]));
    /// // Biased variance for each column: squared sum * (n-1)/n where n=3, squared sum=4.5, result=4.5*(2/3)=3.0
    /// let expected_dim0_biased = vec![3.0, 3.0, 3.0];
    /// let actual_dim0_biased = result_dim0_biased.to_vec::<f32>().unwrap();
    /// for i in 0..3 {
    ///     assert!((actual_dim0_biased[i] - expected_dim0_biased[i]).abs() < 1e-6, "Mismatch at index {}: {} != {}", i, actual_dim0_biased[i], expected_dim0_biased[i]);
    /// }
    ///
    /// // Case 3: Variance along dimension 1, keep dimension, unbiased=false
    /// let result_dim1_biased = tensor.var(1, true, false).unwrap();
    /// assert_eq!(result_dim1_biased.shape().unwrap(), Shape::from_dims(&[2, 1]));
    /// // Biased variance for each row: squared sum/(n-1) * (n'-1)/n' where n=3, n'=2
    /// let expected_dim1_biased = vec![0.5, 0.5];
    /// let actual_dim1_biased = result_dim1_biased.to_vec::<f32>().unwrap();
    /// for i in 0..2 {
    ///     assert!((actual_dim1_biased[i] - expected_dim1_biased[i]).abs() < 1e-6, "Mismatch at index {}: {} != {}", i, actual_dim1_biased[i], expected_dim1_biased[i]);
    /// }
    ///
    /// // Case 4: Variance along dimension 1, keep dimension, unbiased=true
    /// let result_dim1_unbiased = tensor.var(1, true, true).unwrap();
    /// assert_eq!(result_dim1_unbiased.shape().unwrap(), Shape::from_dims(&[2, 1]));
    /// // Unbiased variance for each row: [1.0, 1.0]
    /// let expected_dim1_unbiased = vec![1.0, 1.0];
    /// let actual_dim1_unbiased = result_dim1_unbiased.to_vec::<f32>().unwrap();
    /// for i in 0..2 {
    ///     assert!((actual_dim1_unbiased[i] - expected_dim1_unbiased[i]).abs() < 1e-6, "Mismatch at index {}: {} != {}", i, actual_dim1_unbiased[i], expected_dim1_unbiased[i]);
    /// }
    /// ```
    ///
    /// * Backward pass with gradient verification
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    ///
    /// // Create a 2x3 tensor with requires_grad=true for gradient test
    /// let mut tensor = Tensor::from_data(vec![vec![1.0f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device, true).unwrap();
    ///
    /// // Test gradient for variance along dimension 0 (unbiased)
    /// let result_dim0 = tensor.var(0, false, true).unwrap();
    /// result_dim0.backward().unwrap();
    /// let grad_dim0 = tensor.grad().unwrap().unwrap();
    /// assert_eq!(grad_dim0.shape().unwrap(), Shape::from_dims(&[2, 3]));
    ///
    /// // Clear gradient for next test
    /// tensor.zero_grad();
    ///
    /// // Test gradient for variance along dimension 1 (biased)
    /// let result_dim1 = tensor.var(1, true, false).unwrap();
    /// result_dim1.backward().unwrap();
    /// let grad_dim1 = tensor.grad().unwrap().unwrap();
    /// assert_eq!(grad_dim1.shape().unwrap(), Shape::from_dims(&[2, 3]));
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
    /// * Forward pass with value verification
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    ///
    /// // Create a 2x3 tensor for testing
    /// let tensor = Tensor::from_data(vec![vec![1.0f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device, false).unwrap();
    ///
    /// // Case 1: Argmax along dimension 0 (remove dimension)
    /// let result_dim0 = tensor.argmax((0, false)).unwrap();
    /// assert_eq!(result_dim0.shape().unwrap(), Shape::from_dims(&[3]));
    /// // Maximum indices for each column: column0 max=4 at row1 (index 1), column1 max=5 at row1 (index 1), column2 max=6 at row1 (index 1)
    /// assert_eq!(result_dim0.to_vec::<u32>().unwrap(), vec![1, 1, 1]);
    ///
    /// // Case 2: Argmax along dimension 1 (keep dimension)
    /// let result_dim1 = tensor.argmax((1, true)).unwrap();
    /// assert_eq!(result_dim1.shape().unwrap(), Shape::from_dims(&[2, 1]));
    /// // Maximum indices for each row: row0 max=3 at column2 (index 2), row1 max=6 at column2 (index 2)
    /// assert_eq!(result_dim1.to_vec::<u32>().unwrap(), vec![2, 2]);
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
    /// * Forward pass with value verification
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    ///
    /// // Create a 2x3 tensor for testing
    /// let tensor = Tensor::from_data(vec![vec![1.0f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device, false).unwrap();
    ///
    /// // Case 1: Argmin along dimension 0 (remove dimension)
    /// let result_dim0 = tensor.argmin((0, false)).unwrap();
    /// assert_eq!(result_dim0.shape().unwrap(), Shape::from_dims(&[3]));
    /// // Minimum indices for each column: column0 min=1 at row0 (index 0), column1 min=2 at row0 (index 0), column2 min=3 at row0 (index 0)
    /// assert_eq!(result_dim0.to_vec::<u32>().unwrap(), vec![0, 0, 0]);
    ///
    /// // Case 2: Argmin along dimension 1 (keep dimension)
    /// let result_dim1 = tensor.argmin((1, true)).unwrap();
    /// assert_eq!(result_dim1.shape().unwrap(), Shape::from_dims(&[2, 1]));
    /// // Minimum indices for each row: row0 min=1 at column0 (index 0), row1 min=4 at column0 (index 0)
    /// assert_eq!(result_dim1.to_vec::<u32>().unwrap(), vec![0, 0]);
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
