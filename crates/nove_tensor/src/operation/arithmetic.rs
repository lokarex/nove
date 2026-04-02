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
    /// let device = Device::cpu();
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
    /// let device = Device::cpu();
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
        let inner = self.data.read()?;
        let inner_tensor = match &inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = TensorInner::Tensor(inner_tensor.abs()?);

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
    /// let device = Device::cpu();
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
    /// let device = Device::cpu();
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
        let inner = self.data.read()?;
        let inner_tensor = match &inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = TensorInner::Tensor(inner_tensor.neg()?);

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
