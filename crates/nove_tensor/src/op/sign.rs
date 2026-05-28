use crate::{Tensor, TensorError, backpropagation::graph::OpKind};

impl Tensor {
    /// Compute the absolute value of each element in the tensor.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor with absolute values.
    /// * `Err(TensorError)` - The error when computing the absolute value.
    ///
    /// # Examples
    /// * Compute absolute value of a 2x3 matrix
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = if cfg!(feature = "candle-cpu") { nove::device::candle::cpu().unwrap() } else { nove::device::native::cpu().unwrap() };
    ///
    /// let t = Tensor::from_data(vec![vec![-1.0, 2.0, -3.0], vec![4.0, -5.0, 6.0]], &device, false).unwrap();
    /// let result = t.abs().unwrap();
    /// // Result should be: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    /// let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected);
    /// assert_eq!(result.shape().unwrap(), (&[2, 3]).into());
    /// ```
    ///
    /// * Backpropagate for absolute value
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = if cfg!(feature = "candle-cpu") { nove::device::candle::cpu().unwrap() } else { nove::device::native::cpu().unwrap() };
    ///
    /// let t = Tensor::from_data(vec![vec![-1.0, 2.0, -3.0], vec![4.0, -5.0, 6.0]], &device, true).unwrap();
    /// let result = t.abs().unwrap();
    /// result.backward().unwrap();
    /// // The gradient of t should be sign(t)
    /// // For t = [-1.0, 2.0, -3.0, 4.0, -5.0, 6.0]
    /// // gradient = sign(t) = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
    /// let t_grad = t.grad().unwrap().unwrap();
    /// let expected = vec![-1.0, 1.0, -1.0, 1.0, -1.0, 1.0];
    /// let actual = t_grad.to_vec::<f64>().unwrap();
    /// for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
    ///     assert!((a - e).abs() < 1e-6, "Mismatch at index {}: {} != {}", i, a, e);
    /// }
    /// assert_eq!(t_grad.shape().unwrap(), (&[2, 3]).into());
    /// ```
    pub fn abs(&self) -> Result<Self, TensorError> {
        let storage = self.backend_storage()?.abs()?;
        Ok(Self::op_result_with_kind(
            storage,
            self.device()?,
            vec![self.copy()],
            OpKind::Abs,
        ))
    }

    /// Compute the negative of each element in the tensor.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor with negated values.
    /// * `Err(TensorError)` - The error when computing the negative.
    ///
    /// # Examples
    /// * Negate a 2x3 matrix
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = if cfg!(feature = "candle-cpu") { nove::device::candle::cpu().unwrap() } else { nove::device::native::cpu().unwrap() };
    /// // Create a 2x3 matrix
    /// let t = Tensor::from_data(vec![vec![1.0, -2.0, 3.0], vec![-4.0, 5.0, -6.0]], &device, false).unwrap();
    ///
    /// // Compute negative element-wise
    /// let result = t.neg().unwrap();
    /// // Result should be: [[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]]
    /// let expected = vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0];
    /// // Compare results
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected);
    /// assert_eq!(result.shape().unwrap(), (&[2, 3]).into());
    /// ```
    ///
    /// * Backpropagate for negation
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = if cfg!(feature = "candle-cpu") { nove::device::candle::cpu().unwrap() } else { nove::device::native::cpu().unwrap() };
    /// // Create a 2x3 matrix with requires_grad=true
    /// let t = Tensor::from_data(vec![vec![1.0, -2.0, 3.0], vec![-4.0, 5.0, -6.0]], &device, true).unwrap();
    /// let result = t.neg().unwrap();
    /// result.backward().unwrap();
    /// // The gradient should be all -1.0 (d(-x)/dx = -1)
    /// let grad = t.grad().unwrap().unwrap();
    /// assert_eq!(grad.to_vec::<f64>().unwrap(), vec![-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]);
    /// assert_eq!(grad.shape().unwrap(), (&[2, 3]).into());
    /// ```
    pub fn neg(&self) -> Result<Self, TensorError> {
        let storage = self.backend_storage()?.neg()?;
        Ok(Self::op_result_with_kind(
            storage,
            self.device()?,
            vec![self.copy()],
            OpKind::Neg,
        ))
    }
}
