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
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::cpu();
    /// let t = Tensor::from_data(vec![1.0, 2.0, 3.0], &device, false).unwrap();
    ///
    /// let result = t.affine(2.0, 1.0).unwrap();
    /// println!("{:?}", result);
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
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::cpu();
    /// let t = Tensor::from_data(vec![0.0, 1.0, 2.0], &device, false).unwrap();
    ///
    /// let exp = t.exp().unwrap();
    /// println!("{:?}", exp);
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
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::cpu();
    /// let t = Tensor::from_data(vec![1.0, 2.0, 3.0], &device, false).unwrap();
    ///
    /// let log = t.log().unwrap();
    /// println!("{:?}", log);
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
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::cpu();
    /// let t = Tensor::from_data(vec![0.5, 1.5, 2.5, 3.5], &device, false).unwrap();
    ///
    /// let clipped = t.clip(1.0, 3.0).unwrap();
    /// println!("{:?}", clipped);
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
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::cpu();
    /// let condition = Tensor::from_data(vec![1u8, 0u8, 1u8, 0u8], &device, false).unwrap();
    /// let true_val = Tensor::from_data(vec![10.0, 10.0, 10.0, 10.0], &device, false).unwrap();
    /// let false_val = Tensor::from_data(vec![20.0, 20.0, 20.0, 20.0], &device, false).unwrap();
    ///
    /// let result = Tensor::where_cond(&condition, &true_val, &false_val).unwrap();
    /// println!("{:?}", result);
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
