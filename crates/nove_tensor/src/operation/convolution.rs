use std::sync::{Arc, RwLock};

use crate::{
    Tensor, TensorError,
    tensor::{TensorData, TensorInner},
};

impl Tensor {
    /// Apply the 1D convolutional operation.
    ///
    /// # Notes
    /// * Output length formula: `output_length = floor((input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)`
    /// * When `groups > 1`, input channels and output channels must be divisible by `groups`.
    ///
    /// # Arguments
    /// * `kernel` - The convolution kernel tensor with shape `[out_channels, in_channels/groups, kernel_size]`.
    /// * `padding` - Number of zero padding elements added to both sides of the input.
    /// * `stride` - Number of elements to step the kernel across the input at each computation.
    /// * `dilation` - Number of elements inserted between kernel elements.
    /// * `groups` - Number of groups for grouped convolution. Input and kernel are split into this many groups.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The tensor after applying the convolutional operation.
    /// * `Err(TensorError)` - The error when applying the convolutional operation.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    ///
    /// // Create input tensor with shape [batch=1, channels=3, length=10]
    /// let t = Tensor::rand(0.0f32, 1.0f32, &Shape::from_dims(&[1, 3, 10]), &device, false).unwrap();
    /// // Create kernel with shape [out_channels=7, in_channels=3, kernel_size=3]
    /// let kernel = Tensor::rand(0.0f32, 1.0f32, &Shape::from_dims(&[7, 3, 3]), &device, false).unwrap();
    ///
    /// // Apply 1D convolution with padding=1, stride=1, dilation=1, groups=1
    /// let result = t.conv1d(&kernel, 1, 1, 1, 1).unwrap();
    /// // Output shape should be [batch=1, out_channels=7, length=10] (padding preserves length)
    /// let expected_shape = Shape::from_dims(&[1, 7, 10]);
    /// assert_eq!(result.shape().unwrap(), expected_shape);
    ///
    /// // Example with stride=2: output length = floor((10 + 2*1 - 1*(3-1) - 1) / 2 + 1) = 5
    /// let result_stride2 = t.conv1d(&kernel, 1, 2, 1, 1).unwrap();
    /// let expected_shape_stride2 = Shape::from_dims(&[1, 7, 5]);
    /// assert_eq!(result_stride2.shape().unwrap(), expected_shape_stride2);
    /// ```
    pub fn conv1d(
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

        let new_inner = TensorInner::Tensor(inner_tensor.conv1d(
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

    /// Apply the 2D convolutional operation.
    ///    
    /// # Notes
    /// * Output spatial dimensions formula: `output_size = floor((input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)`
    /// * When `groups > 1`, input channels and output channels must be divisible by `groups`.
    ///
    /// # Arguments
    /// * `kernel` - The convolution kernel tensor with shape `[out_channels, in_channels/groups, kernel_h, kernel_w]`.
    /// * `padding` - Number of zero padding elements added to each side of the input (height and width).
    /// * `stride` - Number of elements to step the kernel across the input in both dimensions (height and width).
    /// * `dilation` - Number of elements inserted between kernel elements in both dimensions.
    /// * `groups` - Number of groups for grouped convolution. Input and kernel are split into this many groups.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The tensor after applying the convolutional operation.
    /// * `Err(TensorError)` - The error when applying the convolutional operation.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    ///
    /// // Create input tensor with shape [batch=1, channels=3, height=5, width=5]
    /// let t = Tensor::rand(0.0f32, 1.0f32, &Shape::from_dims(&[1, 3, 5, 5]), &device, false).unwrap();
    /// // Create kernel with shape [out_channels=7, in_channels=3, kernel_h=3, kernel_w=3]
    /// let kernel = Tensor::rand(0.0f32, 1.0f32, &Shape::from_dims(&[7, 3, 3, 3]), &device, false).unwrap();
    ///
    /// // Apply 2D convolution with padding=1, stride=1, dilation=1, groups=1
    /// let result = t.conv2d(&kernel, 1, 1, 1, 1).unwrap();
    /// // Output shape should be [batch=1, out_channels=7, height=5, width=5]
    /// let expected_shape = Shape::from_dims(&[1, 7, 5, 5]);
    /// assert_eq!(result.shape().unwrap(), expected_shape);
    ///
    /// // Example with stride=2: output height/width = floor((5 + 2*1 - 1*(3-1) - 1) / 2 + 1) = 3
    /// let result_stride2 = t.conv2d(&kernel, 1, 2, 1, 1).unwrap();
    /// let expected_shape_stride2 = Shape::from_dims(&[1, 7, 3, 3]);
    /// assert_eq!(result_stride2.shape().unwrap(), expected_shape_stride2);
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
    /// # Notes
    /// * Max pooling selects the maximum value within each kernel window.
    /// * Output spatial dimensions formula: `output_size = floor((input_size - kernel_size) / stride + 1)`
    ///
    /// # Arguments
    /// * `kernel_size` - A tuple `(kernel_h, kernel_w)` specifying the height and width of the pooling window.
    /// * `stride` - A tuple `(stride_h, stride_w)` specifying the stride for pooling in height and width dimensions.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The tensor after applying the max pooling operation.
    /// * `Err(TensorError)` - The error when applying the max pooling operation.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    ///
    /// // Create input tensor with shape [batch=1, channels=3, height=5, width=5]
    /// let t = Tensor::rand(0.0f32, 1.0f32, &Shape::from_dims(&[1, 3, 5, 5]), &device, false).unwrap();
    ///
    /// // Apply 2D max pooling with kernel_size=(2,2), stride=(2,2)
    /// // Output height/width = floor((5 - 2) / 2 + 1) = 2
    /// let result = t.max_pool2d((2, 2), (2, 2)).unwrap();
    /// let expected_shape = Shape::from_dims(&[1, 3, 2, 2]);
    /// assert_eq!(result.shape().unwrap(), expected_shape);
    ///
    /// // Apply 2D max pooling with kernel_size=(3,3), stride=(1,1)
    /// // Output height/width = floor((5 - 3) / 1 + 1) = 3
    /// let result2 = t.max_pool2d((3, 3), (1, 1)).unwrap();
    /// let expected_shape2 = Shape::from_dims(&[1, 3, 3, 3]);
    /// assert_eq!(result2.shape().unwrap(), expected_shape2);
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
    /// # Notes
    /// * Average pooling computes the mean value within each kernel window.
    /// * Output spatial dimensions formula: `output_size = floor((input_size - kernel_size) / stride + 1)`
    ///
    ///
    /// # Arguments
    /// * `kernel_size` - A tuple `(kernel_h, kernel_w)` specifying the height and width of the pooling window.
    /// * `stride` - A tuple `(stride_h, stride_w)` specifying the stride for pooling in height and width dimensions.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The tensor after applying the average pooling operation.
    /// * `Err(TensorError)` - The error when applying the average pooling operation.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    ///
    /// // Create input tensor with shape [batch=1, channels=3, height=5, width=5]
    /// let t = Tensor::rand(0.0f32, 1.0f32, &Shape::from_dims(&[1, 3, 5, 5]), &device, false).unwrap();
    ///
    /// // Apply 2D average pooling with kernel_size=(2,2), stride=(2,2)
    /// // Output height/width = floor((5 - 2) / 2 + 1) = 2
    /// let result = t.avg_pool2d((2, 2), (2, 2)).unwrap();
    /// let expected_shape = Shape::from_dims(&[1, 3, 2, 2]);
    /// assert_eq!(result.shape().unwrap(), expected_shape);
    ///
    /// // Apply 2D average pooling with kernel_size=(3,3), stride=(1,1)
    /// // Output height/width = floor((5 - 3) / 1 + 1) = 3
    /// let result2 = t.avg_pool2d((3, 3), (1, 1)).unwrap();
    /// let expected_shape2 = Shape::from_dims(&[1, 3, 3, 3]);
    /// assert_eq!(result2.shape().unwrap(), expected_shape2);
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

    /// Apply the 1D max pooling operation.
    ///
    /// # Notes
    /// * Output length formula: `output_length = floor((input_length - kernel_size) / stride + 1)`
    ///
    /// # Arguments
    /// * `kernel_size` - The size of the pooling window along the sequence dimension.
    /// * `stride` - The stride for pooling along the sequence dimension.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The tensor after applying the max pooling operation.
    /// * `Err(TensorError)` - The error when applying the max pooling operation.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    ///
    /// // Create input tensor with shape [batch=1, channels=3, length=10]
    /// let t = Tensor::rand(0.0f32, 1.0f32, &Shape::from_dims(&[1, 3, 10]), &device, false).unwrap();
    ///
    /// // Apply 1D max pooling with kernel_size=2, stride=2
    /// // Output length = floor((10 - 2) / 2 + 1) = 5
    /// let result = t.max_pool1d(2, 2).unwrap();
    /// let expected_shape = Shape::from_dims(&[1, 3, 5]);
    /// assert_eq!(result.shape().unwrap(), expected_shape);
    ///
    /// // Apply 1D max pooling with kernel_size=3, stride=1
    /// // Output length = floor((10 - 3) / 1 + 1) = 8
    /// let result2 = t.max_pool1d(3, 1).unwrap();
    /// let expected_shape2 = Shape::from_dims(&[1, 3, 8]);
    /// assert_eq!(result2.shape().unwrap(), expected_shape2);
    /// ```
    pub fn max_pool1d(&self, kernel_size: usize, stride: usize) -> Result<Self, TensorError> {
        let inner = self.data.read()?;
        let inner_tensor = match &inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let dims_len = inner_tensor.shape().dims().len();
        let insert_dim = dims_len;
        let unsqueezed = inner_tensor.unsqueeze(insert_dim)?;
        let pooled = unsqueezed.max_pool2d_with_stride((kernel_size, 1), (stride, 1))?;
        let squeezed = pooled.squeeze(insert_dim)?;
        let new_inner = TensorInner::Tensor(squeezed);

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

    /// Apply the 1D average pooling operation.
    ///
    /// # Notes
    /// * Output length formula: `output_length = floor((input_length - kernel_size) / stride + 1)`
    ///
    /// # Arguments
    /// * `kernel_size` - The size of the pooling window along the sequence dimension.
    /// * `stride` - The stride for pooling along the sequence dimension.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The tensor after applying the average pooling operation.
    /// * `Err(TensorError)` - The error when applying the average pooling operation.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    ///
    /// // Create input tensor with shape [batch=1, channels=3, length=10]
    /// let t = Tensor::rand(0.0f32, 1.0f32, &Shape::from_dims(&[1, 3, 10]), &device, false).unwrap();
    ///
    /// // Apply 1D average pooling with kernel_size=2, stride=2
    /// // Output length = floor((10 - 2) / 2 + 1) = 5
    /// let result = t.avg_pool1d(2, 2).unwrap();
    /// let expected_shape = Shape::from_dims(&[1, 3, 5]);
    /// assert_eq!(result.shape().unwrap(), expected_shape);
    ///
    /// // Apply 1D average pooling with kernel_size=3, stride=1
    /// // Output length = floor((10 - 3) / 1 + 1) = 8
    /// let result2 = t.avg_pool1d(3, 1).unwrap();
    /// let expected_shape2 = Shape::from_dims(&[1, 3, 8]);
    /// assert_eq!(result2.shape().unwrap(), expected_shape2);
    /// ```
    pub fn avg_pool1d(&self, kernel_size: usize, stride: usize) -> Result<Self, TensorError> {
        let inner = self.data.read()?;
        let inner_tensor = match &inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let dims_len = inner_tensor.shape().dims().len();
        let insert_dim = dims_len;
        let unsqueezed = inner_tensor.unsqueeze(insert_dim)?;
        let pooled = unsqueezed.avg_pool2d_with_stride((kernel_size, 1), (stride, 1))?;
        let squeezed = pooled.squeeze(insert_dim)?;
        let new_inner = TensorInner::Tensor(squeezed);

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
