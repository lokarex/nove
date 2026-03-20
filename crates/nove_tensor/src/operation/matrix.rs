use std::sync::{Arc, RwLock};

use crate::{
    Tensor, TensorError,
    tensor::{TensorData, TensorInner},
};

impl Tensor {
    /// Matrix multiplication between two tensors with broadcasting.
    ///
    /// # Arguments
    /// * `rhs` - The tensor to multiply.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor after matrix multiplication.
    /// * `Err(TensorError)` - The error when multiplying the tensors.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::cpu();
    /// let t1 = Tensor::from_data(&[[1.0, 2.0]], &device, false).unwrap();
    /// let t2 = Tensor::from_data(&[[5.0, 6.0], [7.0, 8.0]], &device, false).unwrap();
    ///
    /// let t3 = t1.matmul(&t2).unwrap();
    /// println!("{:?}", t3);
    /// ```
    pub fn matmul(&self, rhs: &Self) -> Result<Self, TensorError> {
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

        let new_inner = TensorInner::Tensor(inner1_tensor.broadcast_matmul(inner2_tensor)?);

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

    /// Apply batch normalization to the tensor.
    ///
    /// # Arguments
    /// * `mean` - The mean tensor.
    /// * `var` - The variance tensor.
    /// * `epsilon` - A small value added to variance for numerical stability.
    /// * `gamma` - The scale tensor for scaling.
    /// * `beta` - The shift tensor for shifting.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor after batch normalization.
    /// * `Err(TensorError)` - The error when applying batch normalization.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::tensor::{Device, Tensor, DType, Shape};
    /// let device = Device::cpu();
    /// let x = Tensor::rand(0.0f32, 1.0f32, &Shape::from_dims(&[2, 3, 4, 4]), &device, false).unwrap();
    /// let mean = Tensor::zeros(&Shape::from_dims(&[1, 3, 1, 1]), &DType::F32, &device, false).unwrap();
    /// let var = Tensor::ones(&Shape::from_dims(&[1, 3, 1, 1]), &DType::F32, &device, false).unwrap();
    /// let gamma = Tensor::ones(&Shape::from_dims(&[1, 3, 1, 1]), &DType::F32, &device, false).unwrap();
    /// let beta = Tensor::zeros(&Shape::from_dims(&[1, 3, 1, 1]), &DType::F32, &device, false).unwrap();
    ///
    /// let result = x.batch_norm(&mean, &var, 1e-5, &gamma, &beta).unwrap();
    /// println!("{:?}", result);
    /// ```
    pub fn batch_norm(
        &self,
        mean: &Self,
        var: &Self,
        epsilon: f64,
        gamma: &Self,
        beta: &Self,
    ) -> Result<Self, TensorError> {
        let normalized = Tensor::div(
            &self.sub(mean)?,
            &Tensor::sqrt(&var.affine(1f64, epsilon)?)?,
        )?;

        let affined = normalized.mul(gamma)?.add(beta)?;

        Ok(affined)
    }
}
