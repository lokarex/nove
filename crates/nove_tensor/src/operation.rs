use std::sync::{Arc, RwLock};

use crate::{
    Tensor, TensorError,
    tensor::{TensorData, TensorInner},
};

impl Tensor {
    /// Add two tensors.
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
        let inner1 = self.data.read()?;
        let inner1_tensor = match &inner1.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };
        let inner2 = other.data.read()?;
        let inner2_tensor = match &inner2.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        // Get the device from the first tensor
        let device = self.data.read()?.device.clone();

        let new_inner = TensorInner::Tensor(inner1_tensor.broadcast_add(inner2_tensor)?);

        let parents = vec![self.clone(), other.clone()];

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

    /// Multiply two tensors.
    ///
    /// # Arguments
    /// * `other` - The tensor to multiply.
    ///
    /// # Returns
    /// * `Ok(Self)` - The result tensor after multiplication.
    /// * `Err(TensorError)` - The error when multiplying the tensors.
    pub fn mul(&self, other: &Self) -> Result<Self, TensorError> {
        let inner1 = self.data.read()?;
        let inner1_tensor = match &inner1.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };
        let inner2 = other.data.read()?;
        let inner2_tensor = match &inner2.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = TensorInner::Tensor(inner1_tensor.broadcast_mul(inner2_tensor)?);

        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                device: self.data.read()?.device.clone(),
                parents: vec![self.clone(), other.clone()],
                grad: None,
                name: None,
            })),
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
        let inner_tensors = tensors
            .iter()
            .map(|tensor| {
                let data = tensor.as_ref().data.read()?;
                match &data.inner {
                    TensorInner::Tensor(tensor) => Ok(tensor.clone()),
                    TensorInner::Var(var) => Ok(var.as_tensor().clone()),
                }
            })
            .collect::<Result<Vec<_>, TensorError>>()?;
        // Stack the tensors
        let new_inner_tensor = candle_core::Tensor::stack(&inner_tensors, dim)?;

        // Get the device from the first tensor
        let device = tensors
            .first()
            .map(|t| t.as_ref().data.read().unwrap().device.clone())
            .unwrap();

        let new_inner = TensorInner::Tensor(new_inner_tensor);

        //  Set the parents
        let parents = tensors
            .iter()
            .map(|tensor| tensor.as_ref().clone())
            .collect::<Vec<_>>();

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

    /// Matrix multiplication between two tensors.
    ///
    /// # Arguments
    /// * `other` - The tensor to multiply.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor after matrix multiplication.
    /// * `Err(TensorError)` - The error when multiplying the tensors.
    pub fn matmul(&self, other: &Self) -> Result<Self, TensorError> {
        let inner1 = self.data.read()?;
        let inner1_tensor = match &inner1.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };
        let inner2 = other.data.read()?;
        let inner2_tensor = match &inner2.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = TensorInner::Tensor(inner1_tensor.broadcast_matmul(inner2_tensor)?);

        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                device: self.data.read()?.device.clone(),
                parents: vec![self.clone(), other.clone()],
                grad: None,
                name: None,
            })),
        })
    }
}
