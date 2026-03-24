use std::sync::{Arc, RwLock};

use crate::{
    Tensor, TensorError,
    tensor::{TensorData, TensorInner},
};

impl Tensor {
    /// Stack a list of tensors along a new dimension.
    ///
    /// # Arguments
    /// * `tensors` - The list of tensors to stack.
    /// * `dim` - The dimension along which to stack the tensors.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor after stacking.
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
            .ok_or(TensorError::CandleError(candle_core::Error::Msg(
                "empty tensor slice".to_string(),
            )))?
            .as_ref()
            .data
            .read()?
            .device
            .clone();

        let new_inner = TensorInner::Tensor(new_inner_tensor);

        //  Set the parents
        let parents = tensors
            .iter()
            .map(|tensor| tensor.as_ref().copy())
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

    /// Concatenate a sequence of tensors along the specified dimension.
    ///
    /// # Arguments
    /// * `tensors` - The tensors to concatenate.
    /// * `dim` - The dimension along which to concatenate.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The concatenated tensor.
    /// * `Err(TensorError)` - The error when concatenating the tensors.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::cpu();
    /// let t1 = Tensor::from_data(&[[1.0, 2.0]], &device, false).unwrap();
    /// let t2 = Tensor::from_data(&[[3.0, 4.0]], &device, false).unwrap();
    ///
    /// let concatenated = Tensor::concat(&[t1, t2], 0).unwrap();
    /// println!("{:?}", concatenated);
    /// ```
    pub fn concat<A>(tensors: &[A], dim: usize) -> Result<Self, TensorError>
    where
        A: AsRef<Tensor> + std::clone::Clone,
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

        let new_inner_tensor = candle_core::Tensor::cat(&inner_tensors, dim)?;

        let device = tensors
            .first()
            .ok_or(TensorError::CandleError(candle_core::Error::Msg(
                "empty tensor slice".to_string(),
            )))?
            .as_ref()
            .data
            .read()?
            .device
            .clone();

        let new_inner = TensorInner::Tensor(new_inner_tensor);

        let parents = tensors
            .iter()
            .map(|tensor| tensor.as_ref().copy())
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

    /// Concatenate a sequence of tensors along the specified dimension.
    ///
    /// # Arguments
    /// * `tensors` - The tensors to concatenate.
    /// * `dim` - The dimension along which to concatenate.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The concatenated tensor.
    /// * `Err(TensorError)` - The error when concatenating the tensors.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::cpu();
    /// let t1 = Tensor::from_data(&[[1.0, 2.0]], &device, false).unwrap();
    /// let t2 = Tensor::from_data(&[[3.0, 4.0]], &device, false).unwrap();
    ///
    /// let concatenated = Tensor::cat(&[t1, t2], 0).unwrap();
    /// println!("{:?}", concatenated);
    /// ```
    pub fn cat<A>(tensors: &[A], dim: usize) -> Result<Self, TensorError>
    where
        A: AsRef<Tensor> + std::clone::Clone,
    {
        Self::concat(tensors, dim)
    }

    /// Gather values from the tensor along the specified dimension using the provided indices.
    ///
    /// # Notes
    /// * The data type(`DType`) of the indices tensor must be i64(`DType::I64`).
    ///
    /// # Arguments
    /// * `indices` - The tensor containing the indices to gather.
    /// * `dim` - The dimension along which to gather values.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor with gathered values.
    /// * `Err(TensorError)` - The error when gathering values.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::cpu();
    /// let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &device, false).unwrap();
    /// let indices = Tensor::from_data(vec![0i64, 2i64], &device, false).unwrap();
    ///
    /// let result = t.gather(&indices, 0).unwrap();
    /// println!("{:?}", result);
    /// ```
    pub fn gather(&self, indices: &Self, dim: usize) -> Result<Self, TensorError> {
        let inner = self.data.read()?;
        let inner_tensor = match &inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let indices_inner = indices.data.read()?;
        let indices_inner_tensor = match &indices_inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = TensorInner::Tensor(inner_tensor.gather(indices_inner_tensor, dim)?);

        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                device: self.data.read()?.device.clone(),
                parents: vec![self.copy(), indices.copy()],
                grad: None,
                name: None,
            })),
        })
    }
}
