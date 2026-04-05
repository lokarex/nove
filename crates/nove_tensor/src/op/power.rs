use std::sync::{Arc, RwLock};

use crate::{
    Tensor, TensorError,
    tensor::{TensorData, TensorInner},
};

impl Tensor {
    /// Compute the power of the tensor element-wise with broadcasting.
    ///
    /// # Notes
    /// * The exponent tensor must have the same [`crate::DType`] as the base tensor.
    ///
    /// # Arguments
    /// * `exponent` - The exponent tensor.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor after power operation.
    /// * `Err(TensorError)` - The error when computing the power.
    ///
    /// # Examples
    /// * Compute power element-wise without broadcasting
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    ///
    /// let t = Tensor::from_data(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device, false).unwrap();
    /// let exponent = Tensor::from_data(vec![vec![2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0]], &device, false).unwrap();
    /// let result = t.pow(&exponent).unwrap();
    /// // Result should be: [[1.0, 8.0, 81.0], [1024.0, 15625.0, 279936.0]]
    /// let expected = vec![1.0, 8.0, 81.0, 1024.0, 15625.0, 279936.0];
    /// for (lhs, rhs) in result.to_vec::<f64>().unwrap().iter().zip(expected.iter()) {
    ///     assert!((lhs - rhs).abs() < 1e-1);
    /// }
    /// assert_eq!(result.shape().unwrap(), (&[2, 3]).into());
    /// ```
    ///
    /// * Compute power element-wise with broadcasting
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    ///
    /// let t = Tensor::from_data(vec![vec![2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0]], &device, false).unwrap();
    /// let exponent = Tensor::from_data(vec![vec![2.0, 3.0, 4.0]], &device, false).unwrap();
    /// let result = t.pow(&exponent).unwrap();
    /// // Result should be: [[4.0, 27.0, 256.0], [25.0, 216.0, 2401.0]]
    /// let expected = vec![4.0, 27.0, 256.0, 25.0, 216.0, 2401.0];
    /// for (lhs, rhs) in result.to_vec::<f64>().unwrap().iter().zip(expected.iter()) {
    ///     assert!((lhs - rhs).abs() < 1e-1);
    /// }
    /// assert_eq!(result.shape().unwrap(), (&[2, 3]).into());
    /// ```
    ///
    /// * Backpropagate for power operation without broadcasting
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    ///
    /// let t = Tensor::from_data(vec![vec![2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0]], &device, true).unwrap();
    /// let exponent = Tensor::from_data(vec![vec![2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0]], &device, true).unwrap();
    /// let result = t.pow(&exponent).unwrap();
    /// result.backward().unwrap();
    /// // Gradient of t: exponent * t^(exponent-1)
    /// let t_grad = t.grad().unwrap().unwrap();
    /// assert_eq!(t_grad.shape().unwrap(), (&[2, 3]).into());
    /// // Gradient of exponent: t^exponent * ln(t)
    /// let exponent_grad = exponent.grad().unwrap().unwrap();
    /// assert_eq!(exponent_grad.shape().unwrap(), (&[2, 3]).into());
    /// ```
    ///
    /// * Backpropagate for power operation with broadcasting
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    ///
    /// let t = Tensor::from_data(vec![vec![2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0]], &device, true).unwrap();
    /// let exponent = Tensor::from_data(vec![vec![2.0, 3.0, 4.0]], &device, true).unwrap();
    /// let result = t.pow(&exponent).unwrap();
    /// result.backward().unwrap();
    /// // Gradient of t: exponent * t^(exponent-1) (broadcast)
    /// let t_grad = t.grad().unwrap().unwrap();
    /// assert_eq!(t_grad.shape().unwrap(), (&[2, 3]).into());
    /// // Gradient of exponent: sum of t^exponent * ln(t) over broadcast dimensions
    /// let exponent_grad = exponent.grad().unwrap().unwrap();
    /// assert_eq!(exponent_grad.shape().unwrap(), (&[1, 3]).into());
    /// ```
    pub fn pow(&self, exponent: &Tensor) -> Result<Self, TensorError> {
        let inner = self.data.read()?;
        let inner_tensor = match &inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };
        let exponent_inner = exponent.data.read()?;
        let exponent_inner_tensor = match &exponent_inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = TensorInner::Tensor(inner_tensor.broadcast_pow(exponent_inner_tensor)?);

        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                device: self.data.read()?.device.clone(),
                parents: vec![self.copy(), exponent.copy()],
                grad: None,
                name: None,
            })),
        })
    }

    /// Compute the square root of the tensor element-wise.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor after square root.
    /// * `Err(TensorError)` - The error when computing the square root.
    ///
    /// # Examples
    /// * Compute square root of a 2x3 matrix
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    ///
    /// let t = Tensor::from_data(vec![vec![1.0, 4.0, 9.0], vec![16.0, 25.0, 36.0]], &device, false).unwrap();
    /// let result = t.sqrt().unwrap();
    /// // Result should be: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    /// let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected);
    /// assert_eq!(result.shape().unwrap(), (&[2, 3]).into());
    /// ```
    ///
    /// * Backpropagate for square root
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    ///
    /// let t = Tensor::from_data(vec![vec![4.0, 9.0, 16.0], vec![25.0, 36.0, 49.0]], &device, true).unwrap();
    /// let result = t.sqrt().unwrap();
    /// result.backward().unwrap();
    /// // The gradient of t should be 1/(2*sqrt(t))
    /// // For t = [4, 9, 16, 25, 36, 49], sqrt(t) = [2, 3, 4, 5, 6, 7]
    /// // gradient = 1/(2*sqrt(t)) = [1/4, 1/6, 1/8, 1/10, 1/12, 1/14] = [0.25, 0.166666..., 0.125, 0.1, 0.083333..., 0.071428...]
    /// let t_grad = t.grad().unwrap().unwrap();
    /// let expected = vec![0.25, 1.0/6.0, 0.125, 0.1, 1.0/12.0, 1.0/14.0];
    /// let actual = t_grad.to_vec::<f64>().unwrap();
    /// for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
    ///     assert!((a - e).abs() < 1e-6, "Mismatch at index {}: {} != {}", i, a, e);
    /// }
    /// assert_eq!(t_grad.shape().unwrap(), (&[2, 3]).into());
    /// ```
    pub fn sqrt(&self) -> Result<Self, TensorError> {
        let inner = self.data.read()?;
        let inner_tensor = match &inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = TensorInner::Tensor(inner_tensor.sqrt()?);

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

    /// Compute the power of the tensor element-wise.
    ///
    /// # Arguments
    /// * `exponent` - The exponent to raise the tensor to.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor after power operation.
    /// * `Err(TensorError)` - The error when computing the power.
    ///
    /// # Examples
    /// * Compute power element-wise with exponent 2.0
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    ///
    /// let t = Tensor::from_data(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device, false).unwrap();
    /// let result = t.powf(2.0).unwrap();
    /// // Result should be: [[1.0, 4.0, 9.0], [16.0, 25.0, 36.0]]
    /// let expected = vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0];
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected);
    /// assert_eq!(result.shape().unwrap(), (&[2, 3]).into());
    /// ```
    ///
    /// * Backpropagate for power operation
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    ///
    /// let t = Tensor::from_data(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device, true).unwrap();
    /// let result = t.powf(2.0).unwrap();
    /// result.backward().unwrap();
    /// // The gradient of t should be: 2.0 * t^(1.0) = 2.0 * [1, 2, 3, 4, 5, 6]
    /// let t_grad = t.grad().unwrap().unwrap();
    /// assert_eq!(t_grad.to_vec::<f64>().unwrap(), vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
    /// assert_eq!(t_grad.shape().unwrap(), (&[2, 3]).into());
    /// ```
    pub fn powf(&self, exponent: f64) -> Result<Self, TensorError> {
        let inner = self.data.read()?;
        let inner_tensor = match &inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = TensorInner::Tensor(inner_tensor.powf(exponent)?);

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

    /// Compute the reciprocal (1/x) of each element in the tensor.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor with reciprocal values.
    /// * `Err(TensorError)` - The error when computing the reciprocal.
    ///
    /// # Examples
    /// * Compute reciprocal of a 2x3 matrix
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    ///
    /// let t = Tensor::from_data(vec![vec![1.0, 2.0, 4.0], vec![8.0, 10.0, 16.0]], &device, false).unwrap();
    /// let result = t.recip().unwrap();
    /// // Result should be: [[1.0, 0.5, 0.25], [0.125, 0.1, 0.0625]]
    /// let expected = vec![1.0, 0.5, 0.25, 0.125, 0.1, 0.0625];
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected);
    /// assert_eq!(result.shape().unwrap(), (&[2, 3]).into());
    /// ```
    ///
    /// * Backpropagate for reciprocal
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    ///
    /// let t = Tensor::from_data(vec![vec![1.0, 2.0, 4.0], vec![8.0, 10.0, 16.0]], &device, true).unwrap();
    /// let result = t.recip().unwrap();
    /// result.backward().unwrap();
    /// // The gradient of t should be: -1/x^2 = -[1.0, 0.25, 0.0625, 0.015625, 0.01, 0.00390625]
    /// let t_grad = t.grad().unwrap().unwrap();
    /// assert_eq!(t_grad.to_vec::<f64>().unwrap(), vec![-1.0, -0.25, -0.0625, -0.015625, -0.01, -0.00390625]);
    /// assert_eq!(t_grad.shape().unwrap(), (&[2, 3]).into());
    /// ```
    pub fn recip(&self) -> Result<Self, TensorError> {
        let inner = self.data.read()?;
        let inner_tensor = match &inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = TensorInner::Tensor(inner_tensor.recip()?);

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
