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
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    ///
    /// // Create two 2x3 matrices
    /// let t1 = Tensor::from_data(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device, false).unwrap();
    /// let t2 = Tensor::from_data(vec![vec![2.0, 4.0, 6.0], vec![8.0, 10.0, 12.0]], &device, false).unwrap();
    /// // Add the tensors element-wise without broadcasting
    /// let result = t1.add(&t2).unwrap();
    /// // Result should be: [[3.0, 6.0, 9.0], [12.0, 15.0, 18.0]]
    /// let expected = vec![3.0, 6.0, 9.0, 12.0, 15.0, 18.0];
    /// // Compare results
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected);
    /// assert_eq!(result.shape().unwrap(), (&[2, 3]).into());
    ///
    /// // Create a 2x3 matrix and a 1x3 vector
    /// let t1 = Tensor::from_data(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device, false).unwrap();
    /// let t2 = Tensor::from_data(vec![vec![2.0, 4.0, 6.0]], &device, false).unwrap();
    /// // Add the tensors element-wise with broadcasting
    /// let result = t1.add(&t2).unwrap();
    /// // Result should be: [[3.0, 6.0, 9.0], [6.0, 9.0, 12.0]]
    /// let expected = vec![3.0, 6.0, 9.0, 6.0, 9.0, 12.0];
    /// // Compare results
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected);
    /// assert_eq!(result.shape().unwrap(), (&[2, 3]).into());
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
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    ///
    /// // Create two 2x3 matrices
    /// let t1 = Tensor::from_data(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device, false).unwrap();
    /// let t2 = Tensor::from_data(vec![vec![2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0]], &device, false).unwrap();
    /// // Multiply the tensors element-wise without broadcasting
    /// let result = t1.mul(&t2).unwrap();
    /// // Result should be: [[2.0, 6.0, 12.0], [20.0, 30.0, 42.0]]
    /// let expected = vec![2.0, 6.0, 12.0, 20.0, 30.0, 42.0];
    /// // Compare results
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected);
    /// assert_eq!(result.shape().unwrap(), (&[2, 3]).into());
    ///
    /// // Create a 2x3 matrix and a 1x3 vector with broadcasting
    /// let t1 = Tensor::from_data(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device, false).unwrap();
    /// let t2 = Tensor::from_data(vec![vec![2.0, 3.0, 4.0]], &device, false).unwrap();
    /// // Multiply the tensors element-wise with broadcasting
    /// let result = t1.mul(&t2).unwrap();
    /// // Result should be: [[2.0, 6.0, 12.0], [8.0, 15.0, 24.0]]
    /// let expected = vec![2.0, 6.0, 12.0, 8.0, 15.0, 24.0];
    /// // Compare results
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected);
    /// assert_eq!(result.shape().unwrap(), (&[2, 3]).into());
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
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    ///
    /// // Create two 2x3 matrices
    /// let t1 = Tensor::from_data(vec![vec![12.0, 24.0, 36.0], vec![48.0, 60.0, 72.0]], &device, false).unwrap();
    /// let t2 = Tensor::from_data(vec![vec![2.0, 3.0, 4.0], vec![6.0, 5.0, 8.0]], &device, false).unwrap();
    /// // Divide the tensors element-wise without broadcasting
    /// let result = t1.div(&t2).unwrap();
    /// // Result should be: [[6.0, 8.0, 9.0], [8.0, 12.0, 9.0]]
    /// let expected = vec![6.0, 8.0, 9.0, 8.0, 12.0, 9.0];
    /// // Compare results
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected);
    /// assert_eq!(result.shape().unwrap(), (&[2, 3]).into());
    ///
    /// // Create a 2x3 matrix and a 1x3 vector with broadcasting
    /// let t1 = Tensor::from_data(vec![vec![12.0, 24.0, 36.0], vec![48.0, 60.0, 72.0]], &device, false).unwrap();
    /// let t2 = Tensor::from_data(vec![vec![2.0, 3.0, 4.0]], &device, false).unwrap();
    /// // Divide the tensors element-wise with broadcasting
    /// let result = t1.div(&t2).unwrap();
    /// // Result should be: [[6.0, 8.0, 9.0], [24.0, 20.0, 18.0]]
    /// let expected = vec![6.0, 8.0, 9.0, 24.0, 20.0, 18.0];
    /// // Compare results
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected);
    /// assert_eq!(result.shape().unwrap(), (&[2, 3]).into());
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
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    ///
    /// // Create two 2x3 matrices
    /// let t1 = Tensor::from_data(vec![vec![10.0, 20.0, 30.0], vec![40.0, 50.0, 60.0]], &device, false).unwrap();
    /// let t2 = Tensor::from_data(vec![vec![1.0, 3.0, 5.0], vec![7.0, 9.0, 11.0]], &device, false).unwrap();
    /// // Subtract the tensors element-wise without broadcasting
    /// let result = t1.sub(&t2).unwrap();
    /// // Result should be: [[9.0, 17.0, 25.0], [33.0, 41.0, 49.0]]
    /// let expected = vec![9.0, 17.0, 25.0, 33.0, 41.0, 49.0];
    /// // Compare results
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected);
    /// assert_eq!(result.shape().unwrap(), (&[2, 3]).into());
    ///
    /// // Create a 2x3 matrix and a 1x3 vector with broadcasting
    /// let t1 = Tensor::from_data(vec![vec![10.0, 20.0, 30.0], vec![40.0, 50.0, 60.0]], &device, false).unwrap();
    /// let t2 = Tensor::from_data(vec![vec![1.0, 3.0, 5.0]], &device, false).unwrap();
    /// // Subtract the tensors element-wise with broadcasting
    /// let result = t1.sub(&t2).unwrap();
    /// // Result should be: [[9.0, 17.0, 25.0], [39.0, 47.0, 55.0]]
    /// let expected = vec![9.0, 17.0, 25.0, 39.0, 47.0, 55.0];
    /// // Compare results
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected);
    /// assert_eq!(result.shape().unwrap(), (&[2, 3]).into());
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
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    /// // Create a 2x3 matrix with perfect squares
    /// let t = Tensor::from_data(vec![vec![1.0, 4.0, 9.0], vec![16.0, 25.0, 36.0]], &device, false).unwrap();
    ///
    /// // Compute square root element-wise
    /// let result = t.sqrt().unwrap();
    /// // Result should be: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    /// let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// // Compare results
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected);
    /// assert_eq!(result.shape().unwrap(), (&[2, 3]).into());
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
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    /// // Create a 2x3 matrix
    /// let t = Tensor::from_data(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device, false).unwrap();
    ///
    /// // Compute power element-wise with exponent 2.0
    /// let result = t.powf(2.0).unwrap();
    /// // Result should be: [[1.0, 4.0, 9.0], [16.0, 25.0, 36.0]]
    /// let expected = vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0];
    /// // Compare results
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected);
    /// assert_eq!(result.shape().unwrap(), (&[2, 3]).into());
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
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    ///
    /// // Create a 2x3 matrix
    /// let t = Tensor::from_data(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device, false).unwrap();
    /// let exponent = Tensor::from_data(vec![vec![2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0]], &device, false).unwrap();
    /// // Compute power element-wise without broadcasting
    /// let result = t.pow(&exponent).unwrap();
    /// // Result should be: [[1.0, 8.0, 81.0], [1024.0, 15625.0, 279936.0]]
    /// let expected = vec![1.0, 8.0, 81.0, 1024.0, 15625.0, 279936.0];
    /// // Compare results
    /// for (lhs, rhs) in result.to_vec::<f64>().unwrap().iter().zip(expected.iter()) {
    ///     assert!((lhs - rhs).abs() < 1e-1);
    /// }
    /// assert_eq!(result.shape().unwrap(), (&[2, 3]).into());
    ///
    /// // Create a 2x3 matrix and a 1x3 vector with broadcasting
    /// let t = Tensor::from_data(vec![vec![2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0]], &device, false).unwrap();
    /// let exponent = Tensor::from_data(vec![vec![2.0, 3.0, 4.0]], &device, false).unwrap();
    /// // Compute power element-wise with broadcasting
    /// let result = t.pow(&exponent).unwrap();
    /// // Result should be: [[4.0, 27.0, 256.0], [25.0, 216.0, 2401.0]]
    /// let expected = vec![4.0, 27.0, 256.0, 25.0, 216.0, 2401.0];
    /// // Compare results
    /// for (lhs, rhs) in result.to_vec::<f64>().unwrap().iter().zip(expected.iter()) {
    ///     assert!((lhs - rhs).abs() < 1e-1);
    /// }
    /// assert_eq!(result.shape().unwrap(), (&[2, 3]).into());
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
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    /// // Create a 2x3 matrix with mixed positive and negative values
    /// let t = Tensor::from_data(vec![vec![-1.0, 2.0, -3.0], vec![4.0, -5.0, 6.0]], &device, false).unwrap();
    ///
    /// // Compute absolute value element-wise
    /// let result = t.abs().unwrap();
    /// // Result should be: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    /// let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// // Compare results
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected);
    /// assert_eq!(result.shape().unwrap(), (&[2, 3]).into());
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
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    /// // Create a 2x3 matrix with non-zero values
    /// let t = Tensor::from_data(vec![vec![1.0, 2.0, 4.0], vec![8.0, 10.0, 16.0]], &device, false).unwrap();
    ///
    /// // Compute reciprocal element-wise
    /// let result = t.recip().unwrap();
    /// // Result should be: [[1.0, 0.5, 0.25], [0.125, 0.1, 0.0625]]
    /// let expected = vec![1.0, 0.5, 0.25, 0.125, 0.1, 0.0625];
    /// // Compare results
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected);
    /// assert_eq!(result.shape().unwrap(), (&[2, 3]).into());
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
