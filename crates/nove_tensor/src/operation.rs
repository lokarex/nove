use std::sync::{Arc, RwLock};

use crate::{
    Tensor, TensorError,
    tensor::{TensorData, TensorInner},
};

impl Tensor {
    /// Add two tensors element-wise.
    ///
    /// # Arguments
    /// * `other` - The tensor to add.
    ///
    /// # Returns
    /// * `Ok(Self)` - The result tensor after addition.
    /// * `Err(TensorError)` - The error when adding the tensors.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::cpu();
    /// let t1 = Tensor::from_data(vec![1.0, 2.0], &device, false).unwrap();
    /// let t2 = Tensor::from_data(vec![3.0, 4.0], &device, false).unwrap();
    ///
    /// let t3 = t1.add(&t2).unwrap();
    /// println!("{:?}", t3);
    /// ```
    pub fn add(&self, other: &Self) -> Result<Self, TensorError> {
        // Get the inner tensors
        let inner1 = self.data.inner.read()?;
        let inner1_tensor = match &*inner1 {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };
        let inner2 = other.data.inner.read()?;
        let inner2_tensor = match &*inner2 {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        // Get the device from the first tensor
        let device = self.data.device.read()?.clone();

        // Create the inner tensor
        let new_inner = TensorInner::Tensor(inner1_tensor.add(inner2_tensor)?);

        // Set the parents
        let parents = vec![self.clone(), other.clone()];

        Ok(Self {
            data: Arc::new(TensorData {
                inner: RwLock::new(new_inner),
                device: RwLock::new(device),
                parents: RwLock::new(parents),
                grad: RwLock::new(None),
            }),
        })
    }

    /// Multiply two tensors element-wise.
    ///
    /// # Arguments
    /// * `other` - The tensor to multiply.
    ///
    /// # Returns
    /// * `Ok(Self)` - The result tensor after multiplication.
    /// * `Err(TensorError)` - The error when multiplying the tensors.
    pub fn mul(&self, other: &Self) -> Result<Self, TensorError> {
        // Get the inner tensors
        let inner1 = self.data.inner.read()?;
        let inner1_tensor = match &*inner1 {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };
        let inner2 = other.data.inner.read()?;
        let inner2_tensor = match &*inner2 {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        // Create the inner
        let new_inner = TensorInner::Tensor(inner1_tensor.mul(inner2_tensor)?);

        Ok(Self {
            data: Arc::new(TensorData {
                inner: RwLock::new(new_inner),
                device: RwLock::new(self.data.device.read()?.clone()),
                parents: RwLock::new(vec![self.clone(), other.clone()]),
                grad: RwLock::new(None),
            }),
        })
    }

    /// Stack a list of tensors along a new dimension.
    ///
    /// # Arguments
    /// * `tensors` - The list of tensors to stack.
    /// * `dim` - The dimension along which to stack the tensors.
    ///
    /// # Returns
    /// * `Ok(Self)` - The result tensor after stacking.
    /// * `Err(TensorError)` - The error when stacking the tensors.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::cpu();
    /// let t1 = Tensor::from_data(vec![1.0, 2.0], &device, false).unwrap();
    /// let t2 = Tensor::from_data(vec![3.0, 4.0], &device, false).unwrap();
    /// let t3 = Tensor::from_data(vec![5.0, 6.0], &device, false).unwrap();
    ///
    /// let t4 = Tensor::stack(&[t1, t2, t3], 0).unwrap();
    /// println!("{:?}", t4);
    /// ```
    pub fn stack<A, D>(tensors: &[A], dim: D) -> Result<Self, TensorError>
    where
        A: AsRef<Tensor> + std::clone::Clone,
        D: candle_core::shape::Dim,
    {
        // Get the inner tensors
        let inner_tensors = tensors
            .iter()
            .map(|tensor| {
                let inner = tensor.as_ref().data.inner.read()?;
                match &*inner {
                    TensorInner::Tensor(tensor) => Ok(tensor.clone()),
                    TensorInner::Var(var) => Ok(var.as_tensor().clone()),
                }
            })
            .collect::<Result<Vec<_>, TensorError>>()?;
        // Stack the inner tensors
        let new_inner_tensor = candle_core::Tensor::stack(&inner_tensors, dim)?;

        // Get the device from the first tensor
        let device = tensors
            .first()
            .map(|t| t.as_ref().data.device.read().unwrap().clone())
            .unwrap();

        // Create the new inner
        let new_inner = TensorInner::Tensor(new_inner_tensor);

        //  Set the parents
        let parents = tensors
            .iter()
            .map(|tensor| tensor.as_ref().clone())
            .collect::<Vec<_>>();

        Ok(Self {
            data: Arc::new(TensorData {
                inner: RwLock::new(new_inner),
                device: RwLock::new(device),
                parents: RwLock::new(parents),
                grad: RwLock::new(None),
            }),
        })
    }
}
