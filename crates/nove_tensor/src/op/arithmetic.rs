use std::sync::{Arc, RwLock};

use crate::{
    Tensor, TensorError,
    tensor::{TensorData, TensorInner},
};

impl Tensor {
    /// Add two tensors with broadcasting.
    ///
    /// # Arguments
    /// * `rhs` - The tensor to add.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor after addition.
    /// * `Err(TensorError)` - The error when adding the tensors.
    ///
    /// # Examples
    /// * Add two 2x3 matrices without broadcasting
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    ///
    /// let t1 = Tensor::from_data(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device, false).unwrap();
    /// let t2 = Tensor::from_data(vec![vec![2.0, 4.0, 6.0], vec![8.0, 10.0, 12.0]], &device, false).unwrap();
    /// let result = t1.add(&t2).unwrap();
    /// // Result should be: [[3.0, 6.0, 9.0], [12.0, 15.0, 18.0]]
    /// let expected = vec![3.0, 6.0, 9.0, 12.0, 15.0, 18.0];
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected);
    /// assert_eq!(result.shape().unwrap(), (&[2, 3]).into());
    /// ```
    /// * Add a 2x3 matrix and a 1x3 vector with broadcasting
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    ///
    /// let t1 = Tensor::from_data(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device, false).unwrap();
    /// let t2 = Tensor::from_data(vec![vec![2.0, 4.0, 6.0]], &device, false).unwrap();
    /// let result = t1.add(&t2).unwrap();
    /// // Result should be: [[3.0, 6.0, 9.0], [6.0, 9.0, 12.0]]
    /// let expected = vec![3.0, 6.0, 9.0, 6.0, 9.0, 12.0];
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected);
    /// assert_eq!(result.shape().unwrap(), (&[2, 3]).into());
    /// ```
    ///
    /// * Backpropagate for addition without broadcasting
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    ///
    /// let t1 = Tensor::from_data(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device, true).unwrap();
    /// let t2 = Tensor::from_data(vec![vec![2.0, 4.0, 6.0], vec![8.0, 10.0, 12.0]], &device, true).unwrap();
    /// let result = t1.add(&t2).unwrap();
    /// result.backward().unwrap();
    /// // The gradient of t1 and t2 should be all ones
    /// let t1_grad = t1.grad().unwrap().unwrap();
    /// assert_eq!(t1_grad.to_vec::<f64>().unwrap(), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    /// assert_eq!(t1_grad.shape().unwrap(), (&[2, 3]).into());
    /// let t2_grad = t2.grad().unwrap().unwrap();
    /// assert_eq!(t2_grad.to_vec::<f64>().unwrap(), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    /// assert_eq!(t2_grad.shape().unwrap(), (&[2, 3]).into());
    /// ```
    ///
    /// * Backpropagate for addition with broadcasting
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    ///
    /// let t1 = Tensor::from_data(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device, true).unwrap();
    /// let t2 = Tensor::from_data(vec![vec![2.0, 4.0, 6.0]], &device, true).unwrap();
    /// let result = t1.add(&t2).unwrap();
    /// result.backward().unwrap();
    /// // The gradient of t1 should be all ones, and gradient of t2 should be [2.0, 2.0, 2.0] due to broadcasting
    /// let t1_grad = t1.grad().unwrap().unwrap();
    /// assert_eq!(t1_grad.to_vec::<f64>().unwrap(), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    /// assert_eq!(t1_grad.shape().unwrap(), (&[2, 3]).into());
    /// let t2_grad = t2.grad().unwrap().unwrap();
    /// assert_eq!(t2_grad.to_vec::<f64>().unwrap(), vec![2.0, 2.0, 2.0]);
    /// assert_eq!(t2_grad.shape().unwrap(), (&[1, 3]).into());
    /// ```
    pub fn add(&self, rhs: &Self) -> Result<Self, TensorError> {
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

        // Get the device from the first tensor
        let device = self.data.read()?.device.clone();

        let new_inner = TensorInner::Tensor(inner1_tensor.broadcast_add(inner2_tensor)?);

        let parents = vec![self.copy(), rhs.copy()];

        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                device,
                parents,
                grad: None,
                name: None,
            })),
        })
    }

    /// Multiply two tensors with broadcasting.
    ///
    /// # Arguments
    /// * `rhs` - The tensor to multiply.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor after multiplication.
    /// * `Err(TensorError)` - The error when multiplying the tensors.
    ///
    /// # Examples
    /// * Multiply two 2x3 matrices without broadcasting
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    ///
    /// let t1 = Tensor::from_data(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device, false).unwrap();
    /// let t2 = Tensor::from_data(vec![vec![2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0]], &device, false).unwrap();
    /// let result = t1.mul(&t2).unwrap();
    /// // Result should be: [[2.0, 6.0, 12.0], [20.0, 30.0, 42.0]]
    /// let expected = vec![2.0, 6.0, 12.0, 20.0, 30.0, 42.0];
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected);
    /// assert_eq!(result.shape().unwrap(), (&[2, 3]).into());
    /// ```
    /// * Multiply a 2x3 matrix and a 1x3 vector with broadcasting
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    ///
    /// let t1 = Tensor::from_data(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device, false).unwrap();
    /// let t2 = Tensor::from_data(vec![vec![2.0, 3.0, 4.0]], &device, false).unwrap();
    /// let result = t1.mul(&t2).unwrap();
    /// // Result should be: [[2.0, 6.0, 12.0], [8.0, 15.0, 24.0]]
    /// let expected = vec![2.0, 6.0, 12.0, 8.0, 15.0, 24.0];
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected);
    /// assert_eq!(result.shape().unwrap(), (&[2, 3]).into());
    /// ```
    ///
    /// * Backpropagate for multiplication without broadcasting
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    ///
    /// let t1 = Tensor::from_data(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device, true).unwrap();
    /// let t2 = Tensor::from_data(vec![vec![2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0]], &device, true).unwrap();
    /// let result = t1.mul(&t2).unwrap();
    /// result.backward().unwrap();
    /// // The gradient of t1 should be the values of t2
    /// let t1_grad = t1.grad().unwrap().unwrap();
    /// assert_eq!(t1_grad.to_vec::<f64>().unwrap(), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    /// assert_eq!(t1_grad.shape().unwrap(), (&[2, 3]).into());
    /// // The gradient of t2 should be the values of t1
    /// let t2_grad = t2.grad().unwrap().unwrap();
    /// assert_eq!(t2_grad.to_vec::<f64>().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// assert_eq!(t2_grad.shape().unwrap(), (&[2, 3]).into());
    /// ```
    ///
    /// * Backpropagate for multiplication with broadcasting
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    ///
    /// let t1 = Tensor::from_data(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device, true).unwrap();
    /// let t2 = Tensor::from_data(vec![vec![2.0, 3.0, 4.0]], &device, true).unwrap();
    /// let result = t1.mul(&t2).unwrap();
    /// result.backward().unwrap();
    /// // The gradient of t1 should be the values of t2 (broadcasted)
    /// let t1_grad = t1.grad().unwrap().unwrap();
    /// assert_eq!(t1_grad.to_vec::<f64>().unwrap(), vec![2.0, 3.0, 4.0, 2.0, 3.0, 4.0]);
    /// assert_eq!(t1_grad.shape().unwrap(), (&[2, 3]).into());
    /// // The gradient of t2 should be the sum along the broadcasted dimension
    /// // t2 gradient shape is [1, 3], values: [1+4, 2+5, 3+6] = [5.0, 7.0, 9.0]
    /// let t2_grad = t2.grad().unwrap().unwrap();
    /// assert_eq!(t2_grad.to_vec::<f64>().unwrap(), vec![5.0, 7.0, 9.0]);
    /// assert_eq!(t2_grad.shape().unwrap(), (&[1, 3]).into());
    /// ```
    pub fn mul(&self, rhs: &Self) -> Result<Self, TensorError> {
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

        let new_inner = TensorInner::Tensor(inner1_tensor.broadcast_mul(inner2_tensor)?);

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

    /// Divide two tensors with broadcasting.
    ///
    /// # Arguments
    /// * `rhs` - The tensor to divide by.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor after division.
    /// * `Err(TensorError)` - The error when dividing the tensors.
    ///
    /// # Examples
    /// * Divide two 2x2 matrices without broadcasting
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    ///
    /// let t1 = Tensor::from_data(vec![vec![1.0, 2.0], vec![3.0, 4.0]], &device, false).unwrap();
    /// let t2 = Tensor::from_data(vec![vec![2.0, 2.0], vec![2.0, 2.0]], &device, false).unwrap();
    /// let result = t1.div(&t2).unwrap();
    /// // Result should be: [[0.5, 1.0], [1.5, 2.0]]
    /// let expected = vec![0.5, 1.0, 1.5, 2.0];
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected);
    /// assert_eq!(result.shape().unwrap(), (&[2, 2]).into());
    /// ```
    /// * Divide a 2x2 matrix by a 1x2 vector with broadcasting
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    ///
    /// let t1 = Tensor::from_data(vec![vec![1.0, 2.0], vec![3.0, 4.0]], &device, false).unwrap();
    /// let t2 = Tensor::from_data(vec![vec![2.0, 2.0]], &device, false).unwrap();
    /// let result = t1.div(&t2).unwrap();
    /// // Result should be: [[0.5, 1.0], [1.5, 2.0]]
    /// let expected = vec![0.5, 1.0, 1.5, 2.0];
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected);
    /// assert_eq!(result.shape().unwrap(), (&[2, 2]).into());
    /// ```
    ///
    /// * Backpropagate for division without broadcasting
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    ///
    /// let t1 = Tensor::from_data(vec![vec![1.0, 2.0], vec![3.0, 4.0]], &device, true).unwrap();
    /// let t2 = Tensor::from_data(vec![vec![2.0, 2.0], vec![2.0, 2.0]], &device, true).unwrap();
    /// let result = t1.div(&t2).unwrap();
    /// result.backward().unwrap();
    /// // The gradient of t1 should be 1/t2 = [[0.5, 0.5], [0.5, 0.5]]
    /// let t1_grad = t1.grad().unwrap().unwrap();
    /// assert_eq!(t1_grad.to_vec::<f64>().unwrap(), vec![0.5, 0.5, 0.5, 0.5]);
    /// assert_eq!(t1_grad.shape().unwrap(), (&[2, 2]).into());
    /// // The gradient of t2 should be -t1/(t2^2) = [[-0.25, -0.5], [-0.75, -1.0]]
    /// let t2_grad = t2.grad().unwrap().unwrap();
    /// let expected = vec![-0.25, -0.5, -0.75, -1.0];
    /// let actual = t2_grad.to_vec::<f64>().unwrap();
    /// for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
    ///     assert!((a - e).abs() < 1e-6, "Mismatch at index {}: {} != {}", i, a, e);
    /// }
    /// assert_eq!(t2_grad.shape().unwrap(), (&[2, 2]).into());
    /// ```
    ///
    /// * Backpropagate for division with broadcasting
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    ///
    /// let t1 = Tensor::from_data(vec![vec![1.0, 2.0], vec![3.0, 4.0]], &device, true).unwrap();
    /// let t2 = Tensor::from_data(vec![vec![2.0, 2.0]], &device, true).unwrap();
    /// let result = t1.div(&t2).unwrap();
    /// result.backward().unwrap();
    /// // The gradient of t1 should be 1/t2 (broadcasted) = [[0.5, 0.5], [0.5, 0.5]]
    /// let t1_grad = t1.grad().unwrap().unwrap();
    /// assert_eq!(t1_grad.to_vec::<f64>().unwrap(), vec![0.5, 0.5, 0.5, 0.5]);
    /// assert_eq!(t1_grad.shape().unwrap(), (&[2, 2]).into());
    /// // The gradient of t2 should be sum along broadcasted dimension
    /// // t2 gradient shape is [1, 2], values: [(-1-3)/4, (-2-4)/4] = [-1.0, -1.5]
    /// let t2_grad = t2.grad().unwrap().unwrap();
    /// let expected = vec![-1.0, -1.5];
    /// let actual = t2_grad.to_vec::<f64>().unwrap();
    /// for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
    ///     assert!((a - e).abs() < 1e-6, "Mismatch at index {}: {} != {}", i, a, e);
    /// }
    /// assert_eq!(t2_grad.shape().unwrap(), (&[1, 2]).into());
    /// ```
    pub fn div(&self, rhs: &Self) -> Result<Self, TensorError> {
        let inner = self.data.read()?;
        let inner_tensor = match &inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let rhs_inner = rhs.data.read()?;
        let rhs_inner_tensor = match &rhs_inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = TensorInner::Tensor(inner_tensor.broadcast_div(rhs_inner_tensor)?);

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

    /// Subtract two tensors with broadcasting.
    ///
    /// # Arguments
    /// * `rhs` - The tensor to subtract.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor after subtraction.
    /// * `Err(TensorError)` - The error when subtracting the tensors.
    ///
    /// # Examples
    /// * Subtract two 2x3 matrices without broadcasting
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    ///
    /// let t1 = Tensor::from_data(vec![vec![10.0, 20.0, 30.0], vec![40.0, 50.0, 60.0]], &device, false).unwrap();
    /// let t2 = Tensor::from_data(vec![vec![1.0, 3.0, 5.0], vec![7.0, 9.0, 11.0]], &device, false).unwrap();
    /// let result = t1.sub(&t2).unwrap();
    /// // Result should be: [[9.0, 17.0, 25.0], [33.0, 41.0, 49.0]]
    /// let expected = vec![9.0, 17.0, 25.0, 33.0, 41.0, 49.0];
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected);
    /// assert_eq!(result.shape().unwrap(), (&[2, 3]).into());
    /// ```
    /// * Subtract a 2x3 matrix and a 1x3 vector with broadcasting
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    ///
    /// let t1 = Tensor::from_data(vec![vec![10.0, 20.0, 30.0], vec![40.0, 50.0, 60.0]], &device, false).unwrap();
    /// let t2 = Tensor::from_data(vec![vec![1.0, 3.0, 5.0]], &device, false).unwrap();
    /// let result = t1.sub(&t2).unwrap();
    /// // Result should be: [[9.0, 17.0, 25.0], [39.0, 47.0, 55.0]]
    /// let expected = vec![9.0, 17.0, 25.0, 39.0, 47.0, 55.0];
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected);
    /// assert_eq!(result.shape().unwrap(), (&[2, 3]).into());
    /// ```
    ///
    /// * Backpropagate for subtraction without broadcasting
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    ///
    /// let t1 = Tensor::from_data(vec![vec![10.0, 20.0, 30.0], vec![40.0, 50.0, 60.0]], &device, true).unwrap();
    /// let t2 = Tensor::from_data(vec![vec![1.0, 3.0, 5.0], vec![7.0, 9.0, 11.0]], &device, true).unwrap();
    /// let result = t1.sub(&t2).unwrap();
    /// result.backward().unwrap();
    /// // The gradient of t1 should be all ones
    /// let t1_grad = t1.grad().unwrap().unwrap();
    /// assert_eq!(t1_grad.to_vec::<f64>().unwrap(), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    /// assert_eq!(t1_grad.shape().unwrap(), (&[2, 3]).into());
    /// // The gradient of t2 should be all negative ones
    /// let t2_grad = t2.grad().unwrap().unwrap();
    /// assert_eq!(t2_grad.to_vec::<f64>().unwrap(), vec![-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]);
    /// assert_eq!(t2_grad.shape().unwrap(), (&[2, 3]).into());
    /// ```
    ///
    /// * Backpropagate for subtraction with broadcasting
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    ///
    /// let t1 = Tensor::from_data(vec![vec![10.0, 20.0, 30.0], vec![40.0, 50.0, 60.0]], &device, true).unwrap();
    /// let t2 = Tensor::from_data(vec![vec![1.0, 3.0, 5.0]], &device, true).unwrap();
    /// let result = t1.sub(&t2).unwrap();
    /// result.backward().unwrap();
    /// // The gradient of t1 should be all ones
    /// let t1_grad = t1.grad().unwrap().unwrap();
    /// assert_eq!(t1_grad.to_vec::<f64>().unwrap(), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    /// assert_eq!(t1_grad.shape().unwrap(), (&[2, 3]).into());
    /// // The gradient of t2 should be sum along the broadcasted dimension
    /// // t2 gradient shape is [1, 3], values: [-1-1, -1-1, -1-1] = [-2.0, -2.0, -2.0]
    /// let t2_grad = t2.grad().unwrap().unwrap();
    /// assert_eq!(t2_grad.to_vec::<f64>().unwrap(), vec![-2.0, -2.0, -2.0]);
    /// assert_eq!(t2_grad.shape().unwrap(), (&[1, 3]).into());
    /// ```
    pub fn sub(&self, rhs: &Self) -> Result<Self, TensorError> {
        let inner = self.data.read()?;
        let inner_tensor = match &inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let rhs_inner = rhs.data.read()?;
        let rhs_inner_tensor = match &rhs_inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = TensorInner::Tensor(inner_tensor.broadcast_sub(rhs_inner_tensor)?);

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
}
