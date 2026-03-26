use std::sync::{Arc, RwLock};

use crate::{
    Tensor, TensorError,
    tensor::{TensorData, TensorInner},
};

impl Tensor {
    /// Apply affine transformation to the tensor: weight * x + bias.
    ///
    /// # Arguments
    /// * `weight` - The weight to multiply the tensor by.
    /// * `bias` - The bias to add to the tensor.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor after applying the affine transformation.
    /// * `Err(TensorError)` - The error when applying the affine transformation.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    /// // Create a 2x3 matrix with typical data distribution
    /// let t = Tensor::from_data(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]], &device, false).unwrap();
    ///
    /// // Apply affine transformation: 2.0 * x + 1.0
    /// let result = t.affine(2.0, 1.0).unwrap();
    /// // Result should be: [[3.0, 5.0, 7.0], [9.0, 11.0, 13.0]]
    /// let expected = vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0];
    /// // Compare results
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected);
    /// assert_eq!(result.shape().unwrap(), (&[3, 2]).into())
    /// ```
    pub fn affine(&self, weight: f64, bias: f64) -> Result<Self, TensorError> {
        let inner = self.data.read()?;
        let inner_tensor = match &inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = TensorInner::Tensor(inner_tensor.affine(weight, bias)?);

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

    /// Compute the exponential (e^x) of each element in the tensor.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor with exponential values.
    /// * `Err(TensorError)` - The error when computing the exponential.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    /// // Create a 2x3 matrix with typical data distribution including negative, zero, and positive values
    /// let t = Tensor::from_data(vec![vec![-1.0, 0.0], vec![1.0, 0.5], vec![2.0, -0.5]], &device, false).unwrap();
    ///
    /// // Compute exponential element-wise
    /// let result = t.exp().unwrap();
    /// // Result should be: [[0.36787944117144233, 1.0, 2.718281828459045], [1.6487212707001282, 7.38905609893065, 0.6065306597126334]]
    /// let expected = vec![0.36787944117144233, 1.0, 2.718281828459045, 1.6487212707001282, 7.38905609893065, 0.6065306597126334];
    /// // Compare results
    /// // Use relative tolerance for floating point comparison
    /// for (i, (r, e)) in result.to_vec::<f64>().unwrap().iter().zip(expected.iter()).enumerate() {
    ///     let diff = (r - e).abs();
    ///     let tolerance = e.abs() * 1e-10;
    ///     assert!(diff < tolerance, "Element {}: result {} != expected {} (diff: {}, tolerance: {})", i, r, e, diff, tolerance);
    /// }
    /// ```
    pub fn exp(&self) -> Result<Self, TensorError> {
        let inner = self.data.read()?;
        let inner_tensor = match &inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = TensorInner::Tensor(inner_tensor.exp()?);

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

    /// Compute the natural logarithm (ln) of each element in the tensor.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor with logarithm values.
    /// * `Err(TensorError)` - The error when computing the logarithm.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    /// // Create a 2x3 matrix with positive values (log domain requires x > 0)
    /// let t = Tensor::from_data(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]], &device, false).unwrap();
    ///
    /// // Compute natural logarithm element-wise
    /// let result = t.log().unwrap();
    /// // Result should be: [[0.0, 0.6931471805599453, 1.0986122886681098], [1.3862943611198906, 1.6094379124341003, 1.791759469228055]]
    /// let expected = vec![0.0, 0.6931471805599453, 1.0986122886681098, 1.3862943611198906, 1.6094379124341003, 1.791759469228055];
    /// // Compare results
    /// // Use relative tolerance for floating point comparison
    /// for (i, (r, e)) in result.to_vec::<f64>().unwrap().iter().zip(expected.iter()).enumerate() {
    ///     let diff = (r - e).abs();
    ///     let tolerance = e.abs() * 1e-10 + 1e-12; // Add small epsilon for zero case
    ///     assert!(diff < tolerance, "Element {}: result {} != expected {} (diff: {}, tolerance: {})", i, r, e, diff, tolerance);
    /// }
    /// ```
    pub fn log(&self) -> Result<Self, TensorError> {
        let inner = self.data.read()?;
        let inner_tensor = match &inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = TensorInner::Tensor(inner_tensor.log()?);

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

    /// Clip (clamp) the tensor values to be within the specified range.
    ///
    /// # Arguments
    /// * `min` - The minimum value. Values below this will be set to `min`.
    /// * `max` - The maximum value. Values above this will be set to `max`.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The clipped tensor.
    /// * `Err(TensorError)` - The error when clipping the tensor.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    /// // Create a 2x3 matrix with values below, within, and above the clipping range
    /// let t = Tensor::from_data(vec![vec![0.5, 1.5], vec![2.5, 3.5], vec![2.0, 0.0]], &device, false).unwrap();
    ///
    /// // Clip values to range [1.0, 3.0]
    /// let result = t.clip(1.0, 3.0).unwrap();
    /// // Expected: values below 1.0 become 1.0, above 3.0 become 3.0, within range stay unchanged
    /// // Original: [[0.5, 1.5, 2.5], [3.5, 2.0, 0.0]]
    /// // Result: [[1.0, 1.5, 2.5], [3.0, 2.0, 1.0]]
    /// let expected = vec![1.0, 1.5, 2.5, 3.0, 2.0, 1.0];
    /// // Compare results
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected);
    /// ```
    pub fn clip(&self, min: f64, max: f64) -> Result<Self, TensorError> {
        let inner = self.data.read()?;
        let inner_tensor = match &inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = TensorInner::Tensor(inner_tensor.clamp(min, max)?);

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

    /// Apply the where operation: `condition ? true_value : false_value` element-wise.
    ///
    /// # Arguments
    /// * `condition` - The boolean tensor (dtype U8) where non-zero values indicate true.
    /// * `true_value` - The tensor to select where condition is true.
    /// * `false_value` - The tensor to select where condition is false.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor with selected values.
    /// * `Err(TensorError)` - The error when applying the where operation.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    /// let shape = Shape::from(&[2, 3]);
    /// // Create condition tensor with alternating true/false pattern
    /// let condition = Tensor::from_data(vec![1u8, 0u8, 1u8, 0u8, 1u8, 0u8], &device, false).unwrap()
    ///     .reshape(&shape).unwrap();
    /// // Create true and false value tensors with different values for better visualization
    /// let true_val = Tensor::from_data(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &device, false).unwrap()
    ///     .reshape(&shape).unwrap();
    /// let false_val = Tensor::from_data(vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0], &device, false).unwrap()
    ///     .reshape(&shape).unwrap();
    ///
    /// // Apply where operation: selects from true_val where condition != 0, otherwise from false_val
    /// let result = Tensor::where_cond(&condition, &true_val, &false_val).unwrap();
    /// // Expected result: [[10.0, 200.0, 30.0], [400.0, 50.0, 600.0]]
    /// // Condition: [[true, false, true], [false, true, false]]
    /// let expected_data = vec![10.0, 200.0, 30.0, 400.0, 50.0, 600.0];
    /// let expected = Tensor::from_data(expected_data.clone(), &device, false).unwrap()
    ///     .reshape(&shape).unwrap();
    /// // Compare results
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected.to_vec::<f64>().unwrap());
    /// assert_eq!(result.shape().unwrap(), expected.shape().unwrap());
    /// ```
    pub fn where_cond(
        condition: &Self,
        true_value: &Self,
        false_value: &Self,
    ) -> Result<Self, TensorError> {
        let cond_inner = condition.data.read()?;
        let cond_tensor = match &cond_inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };
        let true_inner = true_value.data.read()?;
        let true_tensor = match &true_inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };
        let false_inner = false_value.data.read()?;
        let false_tensor = match &false_inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner_tensor = cond_tensor.where_cond(true_tensor, false_tensor)?;
        let new_inner = TensorInner::Tensor(new_inner_tensor);

        let device = condition.data.read()?.device.clone();

        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                device,
                parents: vec![condition.copy(), true_value.copy(), false_value.copy()],
                grad: None,
                name: None,
            })),
        })
    }
}
