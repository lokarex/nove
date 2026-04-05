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
    /// * Apply affine transformation: 2.0 * x + 1.0 on a 2x3 matrix
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    ///
    /// let t = Tensor::from_data(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device, false).unwrap();
    /// let result = t.affine(2.0, 1.0).unwrap();
    /// // Result should be: weight * x + bias = [[3.0, 5.0, 7.0], [9.0, 11.0, 13.0]]
    /// let expected = vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0];
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected);
    /// assert_eq!(result.shape().unwrap(), (&[2, 3]).into());
    /// ```
    ///
    /// * Backpropagate for affine transformation: weight * x + bias
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    ///
    /// let t = Tensor::from_data(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device, true).unwrap();
    /// let result = t.affine(3.0, 2.0).unwrap();
    /// result.backward().unwrap();
    /// // The gradient of input tensor should be the weight (3.0)
    /// let t_grad = t.grad().unwrap().unwrap();
    /// assert_eq!(t_grad.to_vec::<f64>().unwrap(), vec![3.0, 3.0, 3.0, 3.0, 3.0, 3.0]);
    /// assert_eq!(t_grad.shape().unwrap(), (&[2, 3]).into());
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
    /// * Compute exponential of a 2x3 matrix
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    ///
    /// let t = Tensor::from_data(vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0]], &device, false).unwrap();
    /// let result = t.exp().unwrap();
    /// // Result should be: [[1.0, 2.718281828459045, 7.38905609893065], [20.085536923187668, 54.598150033144236, 148.4131591025766]]
    /// let expected = vec![1.0, 2.718281828459045, 7.38905609893065, 20.085536923187668, 54.598150033144236, 148.4131591025766];
    /// // Compare results with relative tolerance
    /// for (i, (r, e)) in result.to_vec::<f64>().unwrap().iter().zip(expected.iter()).enumerate() {
    ///     let diff = (r - e).abs();
    ///     let tolerance = e.abs() * 1e-10;
    ///     assert!(diff < tolerance, "Element {}: result {} != expected {} (diff: {}, tolerance: {})", i, r, e, diff, tolerance);
    /// }
    /// assert_eq!(result.shape().unwrap(), (&[2, 3]).into());
    /// ```
    ///
    /// * Backpropagate for exponential
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    ///
    /// let t = Tensor::from_data(vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0]], &device, true).unwrap();
    /// let result = t.exp().unwrap();
    /// result.backward().unwrap();
    /// // The gradient of input tensor should be exp(x) = the result values
    /// let t_grad = t.grad().unwrap().unwrap();
    /// let expected_grad = vec![1.0, 2.718281828459045, 7.38905609893065, 20.085536923187668, 54.598150033144236, 148.4131591025766];
    /// let actual = t_grad.to_vec::<f64>().unwrap();
    /// for (i, (a, e)) in actual.iter().zip(expected_grad.iter()).enumerate() {
    ///     let diff = (a - e).abs();
    ///     let tolerance = e.abs() * 1e-10;
    ///     assert!(diff < tolerance, "Element {}: gradient {} != expected {} (diff: {}, tolerance: {})", i, a, e, diff, tolerance);
    /// }
    /// assert_eq!(t_grad.shape().unwrap(), (&[2, 3]).into());
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
    /// * Compute natural logarithm of a 2x3 matrix
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    ///
    /// let t = Tensor::from_data(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device, false).unwrap();
    /// let result = t.log().unwrap();
    /// // Result should be: [[0.0, 0.6931471805599453, 1.0986122886681098], [1.3862943611198906, 1.6094379124341003, 1.791759469228055]]
    /// let expected = vec![0.0, 0.6931471805599453, 1.0986122886681098, 1.3862943611198906, 1.6094379124341003, 1.791759469228055];
    /// // Compare results with relative tolerance
    /// for (i, (r, e)) in result.to_vec::<f64>().unwrap().iter().zip(expected.iter()).enumerate() {
    ///     let diff = (r - e).abs();
    ///     let tolerance = e.abs() * 1e-10 + 1e-12; // Add small epsilon for zero case
    ///     assert!(diff < tolerance, "Element {}: result {} != expected {} (diff: {}, tolerance: {})", i, r, e, diff, tolerance);
    /// }
    /// assert_eq!(result.shape().unwrap(), (&[2, 3]).into());
    /// ```
    ///
    /// * Backpropagate for natural logarithm
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    ///
    /// let t = Tensor::from_data(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], &device, true).unwrap();
    /// let result = t.log().unwrap();
    /// result.backward().unwrap();
    /// // The gradient of input tensor should be 1/x
    /// let t_grad = t.grad().unwrap().unwrap();
    /// let expected_grad = vec![1.0, 0.5, 0.3333333333333333, 0.25, 0.2, 0.16666666666666666];
    /// let actual = t_grad.to_vec::<f64>().unwrap();
    /// for (i, (a, e)) in actual.iter().zip(expected_grad.iter()).enumerate() {
    ///     let diff = (a - e).abs();
    ///     let tolerance = e.abs() * 1e-10;
    ///     assert!(diff < tolerance, "Element {}: gradient {} != expected {} (diff: {}, tolerance: {})", i, a, e, diff, tolerance);
    /// }
    /// assert_eq!(t_grad.shape().unwrap(), (&[2, 3]).into());
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
    /// * Clip a 2x3 matrix to range [1.0, 3.0]
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    ///
    /// let t = Tensor::from_data(vec![vec![0.5, 1.5, 2.5], vec![3.5, 2.0, 0.0]], &device, false).unwrap();
    /// let result = t.clip(1.0, 3.0).unwrap();
    /// // Values below 1.0 become 1.0, above 3.0 become 3.0, within range stay unchanged
    /// // Result should be: [[1.0, 1.5, 2.5], [3.0, 2.0, 1.0]]
    /// let expected = vec![1.0, 1.5, 2.5, 3.0, 2.0, 1.0];
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected);
    /// assert_eq!(result.shape().unwrap(), (&[2, 3]).into());
    /// ```
    ///
    /// * Backpropagate for clipping to range [1.0, 3.0]
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    ///
    /// let device = Device::cpu();
    ///
    /// let t = Tensor::from_data(vec![vec![0.5, 1.5, 2.5], vec![3.5, 2.0, 0.0]], &device, true).unwrap();
    /// let result = t.clip(1.0, 3.0).unwrap();
    /// result.backward().unwrap();
    /// // The gradient of input tensor should be 1.0 for values within [1.0, 3.0], 0.0 otherwise
    /// let t_grad = t.grad().unwrap().unwrap();
    /// // Values within range: 1.5, 2.5, 2.0 -> gradient = 1.0
    /// // Values outside range: 0.5, 3.5, 0.0 -> gradient = 0.0
    /// let expected_grad = vec![0.0, 1.0, 1.0, 0.0, 1.0, 0.0];
    /// assert_eq!(t_grad.to_vec::<f64>().unwrap(), expected_grad);
    /// assert_eq!(t_grad.shape().unwrap(), (&[2, 3]).into());
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
}
