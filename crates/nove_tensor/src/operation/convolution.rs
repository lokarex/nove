use std::sync::{Arc, RwLock};

use crate::{
    Tensor, TensorError,
    tensor::{TensorData, TensorInner},
};

impl Tensor {
    /// Apply the 2D convolutional operation.
    ///
    /// # Parameters
    /// * `kernel` - The kernel tensor.
    /// * `padding` - The padding size.
    /// * `stride` - The stride size.
    /// * `dilation` - The dilation size.
    /// * `groups` - The number of groups.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The tensor after applying the convolutional operation.
    /// * `Err(TensorError)` - The error when applying the convolutional operation.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    /// let t = Tensor::rand(0.0f32, 1.0f32, &Shape::from_dims(&[1, 3, 5, 5]), &device, false).unwrap();
    /// let kernel = Tensor::rand(0.0f32, 1.0f32, &Shape::from_dims(&[7, 3, 3, 3]), &device, false).unwrap();
    /// let result = t.conv2d(&kernel, 1, 1, 1, 1).unwrap();
    /// println!("{:?}", result);
    /// ```
    pub fn conv2d(
        &self,
        kernel: &Self,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self, TensorError> {
        let inner = self.data.read()?;
        let inner_tensor = match &inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let kernel_inner = kernel.data.read()?;
        let kernel_inner_tensor = match &kernel_inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = TensorInner::Tensor(inner_tensor.conv2d(
            kernel_inner_tensor,
            padding,
            stride,
            dilation,
            groups,
        )?);

        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                device: self.data.read()?.device.clone(),
                parents: vec![self.copy(), kernel.copy()],
                grad: None,
                name: None,
            })),
        })
    }

    /// Apply the 2D max pooling operation.
    ///
    /// # Parameters
    /// * `kernel_size` - The kernel size.
    /// * `stride` - The stride size.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The tensor after applying the max pooling operation.
    /// * `Err(TensorError)` - The error when applying the max pooling operation.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    /// let t = Tensor::rand(0.0f32, 1.0f32, &Shape::from_dims(&[1, 3, 5, 5]), &device, false).unwrap();
    /// let result = t.max_pool2d((2, 2), (2, 2)).unwrap();
    /// println!("{:?}", result);
    /// ```
    pub fn max_pool2d(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self, TensorError> {
        let inner = self.data.read()?;
        let inner_tensor = match &inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner =
            TensorInner::Tensor(inner_tensor.max_pool2d_with_stride(kernel_size, stride)?);

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

    /// Apply the 2D average pooling operation.
    ///
    /// # Parameters
    /// * `kernel_size` - The kernel size.
    /// * `stride` - The stride size.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The tensor after applying the average pooling operation.
    /// * `Err(TensorError)` - The error when applying the average pooling operation.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    /// let t = Tensor::rand(0.0f32, 1.0f32, &Shape::from_dims(&[1, 3, 5, 5]), &device, false).unwrap();
    /// let result = t.avg_pool2d((2, 2), (2, 2)).unwrap();
    /// println!("{:?}", result);
    /// ```
    pub fn avg_pool2d(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self, TensorError> {
        let inner = self.data.read()?;
        let inner_tensor = match &inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner =
            TensorInner::Tensor(inner_tensor.avg_pool2d_with_stride(kernel_size, stride)?);

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