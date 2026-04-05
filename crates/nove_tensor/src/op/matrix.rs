use std::sync::{Arc, RwLock};

use crate::{
    Tensor, TensorError,
    tensor::{TensorData, TensorInner},
};

impl Tensor {
    /// Matrix multiplication between two tensors with broadcasting.
    ///
    /// # Arguments
    /// * `rhs` - The tensor to multiply.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor after matrix multiplication.
    /// * `Err(TensorError)` - The error when multiplying the tensors.
    ///
    /// # Examples
    /// * Matrix multiplication of 2x3 and 3x4 matrices
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    ///
    /// // Directly create 2x3 matrix without reshape
    /// let t1 = Tensor::from_data(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device, false).unwrap();
    /// // Directly create 3x4 matrix without reshape  
    /// let t2 = Tensor::from_data(vec![vec![7.0, 8.0, 9.0, 10.0], vec![11.0, 12.0, 13.0, 14.0], vec![15.0, 16.0, 17.0, 18.0]], &device, false).unwrap();
    ///
    /// let result = t1.matmul(&t2).unwrap();
    /// assert_eq!(result.shape().unwrap(), Shape::from(&[2, 4]));
    /// // Expected: [[74, 80, 86, 92], [173, 188, 203, 218]]
    /// let expected = vec![74.0, 80.0, 86.0, 92.0, 173.0, 188.0, 203.0, 218.0];
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected);
    /// ```
    ///
    /// * Backpropagation for matrix multiplication
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    ///
    /// // Create tensors with gradient tracking enabled
    /// let t1 = Tensor::from_data(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device, true).unwrap();
    /// let t2 = Tensor::from_data(vec![vec![7.0, 8.0, 9.0, 10.0], vec![11.0, 12.0, 13.0, 14.0], vec![15.0, 16.0, 17.0, 18.0]], &device, true).unwrap();
    ///
    /// let result = t1.matmul(&t2).unwrap();
    /// result.backward().unwrap();
    ///
    /// // Verify gradient shapes match input shapes
    /// let t1_grad = t1.grad().unwrap().unwrap();
    /// assert_eq!(t1_grad.shape().unwrap(), Shape::from(&[2, 3]));
    ///
    /// let t2_grad = t2.grad().unwrap().unwrap();
    /// assert_eq!(t2_grad.shape().unwrap(), Shape::from(&[3, 4]));
    ///
    /// // Verify gradient values (calculated manually)
    /// // For matrix multiplication C = A @ B with dC/dC = 1 (all ones gradient from output),
    /// // dL/dA = dL/dC @ B^T, dL/dB = A^T @ dL/dC
    /// // With dL/dC being all ones (2x4), we get:
    /// // dL/dA = ones(2x4) @ B^T(4x3) = [sum of each column of B] repeated for each row
    /// // B columns sums: [7+11+15=33, 8+12+16=36, 9+13+17=39, 10+14+18=42]
    /// // dL/dA row 0: [33+36+39+42 = 150] repeated 3 times? Actually matrix multiplication:
    /// // Let's compute properly: dL/dA_ij = Σ_k dL/dC_ik * B_jk^T = Σ_k 1 * B_jk
    /// // Since dL/dC is all ones, dL/dA_ij = Σ_k B_jk (sum over columns of B for each j)
    /// // For j=0: Σ_k B_0k = 7+8+9+10 = 34
    /// // j=1: Σ_k B_1k = 11+12+13+14 = 50
    /// // j=2: Σ_k B_2k = 15+16+17+18 = 66
    /// // So dL/dA = [[34, 50, 66], [34, 50, 66]] (2x3)
    /// let expected_t1_grad = vec![34.0, 50.0, 66.0, 34.0, 50.0, 66.0];
    /// assert_eq!(t1_grad.to_vec::<f64>().unwrap(), expected_t1_grad);
    ///
    /// // dL/dB = A^T @ dL/dC
    /// // A^T is 3x2, dL/dC is 2x4 all ones, so dL/dB_ij = Σ_k A^T_ik * dL/dC_kj = Σ_k A_ki * 1
    /// // For i=0: Σ_k A_k0 = 1+4 = 5
    /// // i=1: Σ_k A_k1 = 2+5 = 7  
    /// // i=2: Σ_k A_k2 = 3+6 = 9
    /// // So each row of dL/dB is [5, 7, 9] repeated 4 times? Actually matrix multiplication:
    /// // dL/dB_ij = Σ_k A_ki * dL/dC_kj = Σ_k A_ki * 1 = Σ_k A_ki (sum over rows of A for each column i)
    /// // So dL/dB has shape 3x4, each row i is constant with value Σ_k A_ki
    /// // Row 0: 5, 5, 5, 5
    /// // Row 1: 7, 7, 7, 7
    /// // Row 2: 9, 9, 9, 9
    /// let expected_t2_grad = vec![5.0, 5.0, 5.0, 5.0, 7.0, 7.0, 7.0, 7.0, 9.0, 9.0, 9.0, 9.0];
    /// assert_eq!(t2_grad.to_vec::<f64>().unwrap(), expected_t2_grad);
    /// ```
    pub fn matmul(&self, rhs: &Self) -> Result<Self, TensorError> {
        let inner1 = self.data.read()?;
        let inner1_tensor = match &inner1.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };
        let inner2 = rhs.data.read()?;
        let inner2_tensor = match &inner2.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = TensorInner::Tensor(inner1_tensor.broadcast_matmul(inner2_tensor)?);

        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                device: self.data.read()?.device.clone(),
                parents: vec![self.copy(), rhs.copy()],
                grad: None,
                name: None,
            })),
        })
    }

    /// Apply batch normalization to the tensor.
    ///
    /// # Arguments
    /// * `mean` - The mean tensor.
    /// * `var` - The variance tensor.
    /// * `epsilon` - A small value added to variance for numerical stability.
    /// * `gamma` - The scale tensor for scaling.
    /// * `beta` - The shift tensor for shifting.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor after batch normalization.
    /// * `Err(TensorError)` - The error when applying batch normalization.
    ///
    /// # Examples
    /// * Batch normalization with 4D input tensor
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    ///
    /// // Create 4D input tensor with shape [2, 2, 2, 2] directly without reshape
    /// // Using nested Vec to represent 4D structure: [batch, channels, height, width]
    /// let x = Tensor::from_data(vec![
    ///     // Batch 0
    ///     vec![
    ///         // Channel 0
    ///         vec![vec![1.0, 2.0], vec![3.0, 4.0]],
    ///         // Channel 1
    ///         vec![vec![5.0, 6.0], vec![7.0, 8.0]],
    ///     ],
    ///     // Batch 1
    ///     vec![
    ///         // Channel 0
    ///         vec![vec![9.0, 10.0], vec![11.0, 12.0]],
    ///         // Channel 1
    ///         vec![vec![13.0, 14.0], vec![15.0, 16.0]],
    ///     ],
    /// ], &device, false).unwrap();
    ///
    /// // Create mean tensor with shape [1, 2, 1, 1] for broadcasting
    /// let mean = Tensor::from_data(vec![vec![vec![vec![0.5]], vec![vec![1.5]]]], &device, false).unwrap();
    ///
    /// // Create variance tensor with shape [1, 2, 1, 1]
    /// let var = Tensor::from_data(vec![vec![vec![vec![1.0]], vec![vec![1.0]]]], &device, false).unwrap();
    ///
    /// // Create gamma tensor with shape [1, 2, 1, 1]
    /// let gamma = Tensor::from_data(vec![vec![vec![vec![1.0]], vec![vec![1.0]]]], &device, false).unwrap();
    ///
    /// // Create beta tensor with shape [1, 2, 1, 1]
    /// let beta = Tensor::from_data(vec![vec![vec![vec![0.0]], vec![vec![0.0]]]], &device, false).unwrap();
    ///
    /// let result = x.batch_norm(&mean, &var, 1e-5, &gamma, &beta).unwrap();
    /// assert_eq!(result.shape().unwrap(), Shape::from(&[2, 2, 2, 2]));
    ///
    /// // Compute expected values: (x - mean) / sqrt(var + epsilon)
    /// // Since gamma=1, beta=0, var=1, epsilon=1e-5, result ≈ x - mean
    /// let expected = vec![
    ///     1.0 - 0.5, 2.0 - 0.5, 3.0 - 0.5, 4.0 - 0.5,  // batch 0, channel 0
    ///     5.0 - 1.5, 6.0 - 1.5, 7.0 - 1.5, 8.0 - 1.5,  // batch 0, channel 1
    ///     9.0 - 0.5, 10.0 - 0.5, 11.0 - 0.5, 12.0 - 0.5, // batch 1, channel 0
    ///     13.0 - 1.5, 14.0 - 1.5, 15.0 - 1.5, 16.0 - 1.5, // batch 1, channel 1
    /// ];
    /// // Adjust for epsilon: divide by sqrt(1 + 1e-5)
    /// let epsilon = 1e-5;
    /// let scale = (1.0f64 + epsilon).sqrt();
    /// let expected_scaled: Vec<f64> = expected.iter().map(|&v| v / scale).collect();
    ///
    /// let result_vec = result.to_vec::<f64>().unwrap();
    /// for i in 0..result_vec.len() {
    ///     assert!((result_vec[i] - expected_scaled[i]).abs() < 1e-10, "Mismatch at index {}: result={}, expected={}", i, result_vec[i], expected_scaled[i]);
    /// }
    /// ```
    ///
    /// * Backpropagation for batch normalization
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    ///
    /// // Create tensors with gradient tracking enabled
    /// let x = Tensor::from_data(vec![
    ///     vec![
    ///         vec![vec![1.0, 2.0], vec![3.0, 4.0]],
    ///         vec![vec![5.0, 6.0], vec![7.0, 8.0]],
    ///     ],
    ///     vec![
    ///         vec![vec![9.0, 10.0], vec![11.0, 12.0]],
    ///         vec![vec![13.0, 14.0], vec![15.0, 16.0]],
    ///     ],
    /// ], &device, true).unwrap();
    ///
    /// let mean = Tensor::from_data(vec![vec![vec![vec![0.5]], vec![vec![1.5]]]], &device, true).unwrap();
    /// let var = Tensor::from_data(vec![vec![vec![vec![1.0]], vec![vec![1.0]]]], &device, true).unwrap();
    /// let gamma = Tensor::from_data(vec![vec![vec![vec![1.0]], vec![vec![1.0]]]], &device, true).unwrap();
    /// let beta = Tensor::from_data(vec![vec![vec![vec![0.0]], vec![vec![0.0]]]], &device, true).unwrap();
    ///
    /// let result = x.batch_norm(&mean, &var, 1e-5, &gamma, &beta).unwrap();
    /// result.backward().unwrap();
    ///
    /// // Verify gradient shapes match input shapes
    /// let x_grad = x.grad().unwrap().unwrap();
    /// assert_eq!(x_grad.shape().unwrap(), Shape::from(&[2, 2, 2, 2]));
    ///
    /// let mean_grad = mean.grad().unwrap().unwrap();
    /// assert_eq!(mean_grad.shape().unwrap(), Shape::from(&[1, 2, 1, 1]));
    ///
    /// let var_grad = var.grad().unwrap().unwrap();
    /// assert_eq!(var_grad.shape().unwrap(), Shape::from(&[1, 2, 1, 1]));
    ///
    /// let gamma_grad = gamma.grad().unwrap().unwrap();
    /// assert_eq!(gamma_grad.shape().unwrap(), Shape::from(&[1, 2, 1, 1]));
    ///
    /// let beta_grad = beta.grad().unwrap().unwrap();
    /// assert_eq!(beta_grad.shape().unwrap(), Shape::from(&[1, 2, 1, 1]));
    /// ```
    pub fn batch_norm(
        &self,
        mean: &Self,
        var: &Self,
        epsilon: f64,
        gamma: &Self,
        beta: &Self,
    ) -> Result<Self, TensorError> {
        let normalized = Tensor::div(
            &self.sub(mean)?,
            &Tensor::sqrt(&var.affine(1f64, epsilon)?)?,
        )?;

        let affined = normalized.mul(gamma)?.add(beta)?;

        Ok(affined)
    }
}
