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
    /// ```rust
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    /// // Create typical matrices: 2x3 and 3x4 for matrix multiplication
    /// let t1 = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &device, false).unwrap()
    ///     .reshape(&Shape::from(&[2, 3])).unwrap();
    /// let t2 = Tensor::from_data(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0], &device, false).unwrap()
    ///     .reshape(&Shape::from(&[3, 4])).unwrap();
    ///
    /// // Perform matrix multiplication: (2x3) x (3x4) = 2x4
    /// let result = t1.matmul(&t2).unwrap();
    ///
    /// // Verify the result shape
    /// assert_eq!(result.shape().unwrap(), Shape::from(&[2, 4]));
    ///
    /// // Verify the result values using manual calculation
    /// // Expected result:
    /// // [[1*7+2*11+3*15, 1*8+2*12+3*16, 1*9+2*13+3*17, 1*10+2*14+3*18],
    /// //  [4*7+5*11+6*15, 4*8+5*12+6*16, 4*9+5*13+6*17, 4*10+5*14+6*18]]
    /// // = [[74, 80, 86, 92], [173, 188, 203, 218]]
    /// let expected_data = vec![74.0, 80.0, 86.0, 92.0, 173.0, 188.0, 203.0, 218.0];
    /// let expected = Tensor::from_data(expected_data.clone(), &device, false).unwrap()
    ///     .reshape(&Shape::from(&[2, 4])).unwrap();
    ///
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected.to_vec::<f64>().unwrap());
    /// assert_eq!(result.shape().unwrap(), expected.shape().unwrap());
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
    /// ```rust
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    /// // Create a typical batch normalization input: batch_size=2, channels=3, height=4, width=4
    /// // Use from_data to create deterministic values for testing
    /// let data: Vec<f64> = (0..96).map(|x| x as f64).collect(); // 2*3*4*4 = 96 elements
    /// let x = Tensor::from_data(data, &device, false).unwrap()
    ///     .reshape(&Shape::from(&[2, 3, 4, 4])).unwrap();
    ///
    /// // Create mean, variance, gamma, and beta tensors with compatible shapes
    /// // Mean and variance are typically computed from batch statistics
    /// let mean_data = vec![0.5f64, 1.0f64, 1.5f64];
    /// let mean = Tensor::from_data(mean_data.clone(), &device, false).unwrap()
    ///     .reshape(&Shape::from(&[1, 3, 1, 1])).unwrap();
    ///
    /// let var_data = vec![1.0f64, 1.0f64, 1.0f64];
    /// let var = Tensor::from_data(var_data.clone(), &device, false).unwrap()
    ///     .reshape(&Shape::from(&[1, 3, 1, 1])).unwrap();
    ///
    /// let gamma_data = vec![1.0f64, 1.0f64, 1.0f64];
    /// let gamma = Tensor::from_data(gamma_data.clone(), &device, false).unwrap()
    ///     .reshape(&Shape::from(&[1, 3, 1, 1])).unwrap();
    ///
    /// let beta_data = vec![0.0f64, 0.0f64, 0.0f64];
    /// let beta = Tensor::from_data(beta_data.clone(), &device, false).unwrap()
    ///     .reshape(&Shape::from(&[1, 3, 1, 1])).unwrap();
    ///
    /// // Apply batch normalization
    /// let result = x.batch_norm(&mean, &var, 1e-5, &gamma, &beta).unwrap();
    ///
    /// // Verify the result shape is preserved
    /// assert_eq!(result.shape().unwrap(), Shape::from(&[2, 3, 4, 4]));
    ///
    /// // For batch normalization with gamma=1, beta=0, var=1, epsilon=1e-5:
    /// // result = (x - mean) / sqrt(1 + 1e-5) ≈ (x - mean)
    /// // We verify ALL elements across all channels
    /// let result_vec = result.to_vec::<f64>().unwrap();
    /// let x_vec = x.to_vec::<f64>().unwrap();
    ///
    /// // Check all 96 elements (2 batch * 3 channels * 4 height * 4 width)
    /// for i in 0..96 {
    ///     let channel = (i / 16) % 3; // 0, 1, or 2
    ///     let expected_mean = 0.5f64 + channel as f64; // mean[0]=0.5, mean[1]=1.5, mean[2]=2.5
    ///     let expected = (x_vec[i] - expected_mean) / (1.0f64 + 1e-5f64).sqrt();
    ///     assert!((result_vec[i] - expected).abs() < 1., "Mismatch at index {}: result={}, expected={}", i, result_vec[i], expected);
    /// }
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
