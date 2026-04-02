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
    /// * Forward computation with standard parameters
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    ///
    /// // Create input tensor with shape [batch=1, channels=1, length=4]
    /// let t = Tensor::from_data(vec![vec![vec![1.0, 2.0, 3.0, 4.0]]], &device, false).unwrap();
    /// // Create kernel with shape [out_channels=1, in_channels=1, kernel_size=2]
    /// let kernel = Tensor::from_data(vec![vec![vec![1.0, 2.0]]], &device, false).unwrap();
    ///
    /// // Apply 1D convolution with padding=0, stride=1, dilation=1, groups=1
    /// // Expected output: length = floor((4 + 2*0 - 1*(2-1) - 1) / 1 + 1) = 3
    /// // Computation: [1*1+2*2=5, 2*1+3*2=8, 3*1+4*2=11]
    /// let result = t.conv1d(&kernel, 0, 1, 1, 1).unwrap();
    /// let expected_shape = Shape::from_dims(&[1, 1, 3]);
    /// assert_eq!(result.shape().unwrap(), expected_shape);
    /// let expected_data = vec![5.0, 8.0, 11.0];
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected_data);
    /// ```
    ///
    /// * Backpropagation with requires_grad=true
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    ///
    /// // Create input tensor with requires_grad=true
    /// let t = Tensor::from_data(vec![vec![vec![1.0, 2.0, 3.0, 4.0]]], &device, true).unwrap();
    /// // Create kernel with requires_grad=true
    /// let kernel = Tensor::from_data(vec![vec![vec![1.0, 2.0]]], &device, true).unwrap();
    ///
    /// // Apply 1D convolution
    /// let result = t.conv1d(&kernel, 0, 1, 1, 1).unwrap();
    /// result.backward().unwrap();
    ///
    /// // Verify gradient shapes
    /// let t_grad = t.grad().unwrap().unwrap();
    /// let kernel_grad = kernel.grad().unwrap().unwrap();
    /// assert_eq!(t_grad.shape().unwrap(), Shape::from_dims(&[1, 1, 4]));
    /// assert_eq!(kernel_grad.shape().unwrap(), Shape::from_dims(&[1, 1, 2]));
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
    /// * Forward computation with standard parameters
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    ///
    /// // Create input tensor with shape [batch=1, channels=1, height=2, width=2]
    /// let t = Tensor::from_data(vec![vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]]], &device, false).unwrap();
    /// // Create kernel with shape [out_channels=1, in_channels=1, kernel_h=2, kernel_w=2]
    /// let kernel = Tensor::from_data(vec![vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]]], &device, false).unwrap();
    ///
    /// // Apply 2D convolution with padding=0, stride=1, dilation=1, groups=1
    /// // Expected output: height = floor((2 + 2*0 - 1*(2-1) - 1) / 1 + 1) = 1
    /// // Expected output: width = floor((2 + 2*0 - 1*(2-1) - 1) / 1 + 1) = 1
    /// // Computation: 1*1 + 2*2 + 3*3 + 4*4 = 1 + 4 + 9 + 16 = 30
    /// let result = t.conv2d(&kernel, 0, 1, 1, 1).unwrap();
    /// let expected_shape = Shape::from_dims(&[1, 1, 1, 1]);
    /// assert_eq!(result.shape().unwrap(), expected_shape);
    /// let expected_data = vec![30.0];
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected_data);
    /// ```
    ///
    /// * Backpropagation with requires_grad=true
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    ///
    /// // Create input tensor with requires_grad=true
    /// let t = Tensor::from_data(vec![vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]]], &device, true).unwrap();
    /// // Create kernel with requires_grad=true
    /// let kernel = Tensor::from_data(vec![vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]]], &device, true).unwrap();
    ///
    /// // Apply 2D convolution
    /// let result = t.conv2d(&kernel, 0, 1, 1, 1).unwrap();
    /// result.backward().unwrap();
    ///
    /// // Verify gradient shapes
    /// let t_grad = t.grad().unwrap().unwrap();
    /// let kernel_grad = kernel.grad().unwrap().unwrap();
    /// assert_eq!(t_grad.shape().unwrap(), Shape::from_dims(&[1, 1, 2, 2]));
    /// assert_eq!(kernel_grad.shape().unwrap(), Shape::from_dims(&[1, 1, 2, 2]));
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
    /// * Forward computation with standard parameters
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    ///
    /// // Create input tensor with shape [batch=1, channels=1, height=2, width=2]
    /// let t = Tensor::from_data(vec![vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]]], &device, false).unwrap();
    ///
    /// // Apply 2D max pooling with kernel_size=(2,2), stride=(2,2)
    /// // Expected output: height = floor((2 - 2) / 2 + 1) = 1
    /// // Expected output: width = floor((2 - 2) / 2 + 1) = 1
    /// // Max value in 2x2 window is 4.0
    /// let result = t.max_pool2d((2, 2), (2, 2)).unwrap();
    /// let expected_shape = Shape::from_dims(&[1, 1, 1, 1]);
    /// assert_eq!(result.shape().unwrap(), expected_shape);
    /// let expected_data = vec![4.0];
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected_data);
    /// ```
    ///
    /// * Backpropagation with requires_grad=true
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    ///
    /// // Create input tensor with requires_grad=true
    /// let t = Tensor::from_data(vec![vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]]], &device, true).unwrap();
    ///
    /// // Apply 2D max pooling (kernel_size must equal stride for backward support)
    /// let result = t.max_pool2d((2, 2), (2, 2)).unwrap();
    /// result.backward().unwrap();
    ///
    /// // Verify gradient shapes
    /// let t_grad = t.grad().unwrap().unwrap();
    /// assert_eq!(t_grad.shape().unwrap(), Shape::from_dims(&[1, 1, 2, 2]));
    /// // For max pooling with kernel_size=stride, gradient is distributed to max element
    /// // In input [[1,2],[3,4]], max element is at position (1,1) (0-indexed)
    /// let expected_grad = vec![0.0, 0.0, 0.0, 0.25]; // Notes: gradient is normalized by kernel area (4)
    /// assert_eq!(t_grad.to_vec::<f64>().unwrap(), expected_grad);
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
    /// * Forward computation with standard parameters
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    ///
    /// // Create input tensor with shape [batch=1, channels=1, height=2, width=2]
    /// let t = Tensor::from_data(vec![vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]]], &device, false).unwrap();
    ///
    /// // Apply 2D average pooling with kernel_size=(2,2), stride=(2,2)
    /// // Expected output: height = floor((2 - 2) / 2 + 1) = 1
    /// // Expected output: width = floor((2 - 2) / 2 + 1) = 1
    /// // Average value in 2x2 window: (1.0 + 2.0 + 3.0 + 4.0) / 4 = 2.5
    /// let result = t.avg_pool2d((2, 2), (2, 2)).unwrap();
    /// let expected_shape = Shape::from_dims(&[1, 1, 1, 1]);
    /// assert_eq!(result.shape().unwrap(), expected_shape);
    /// let expected_data = vec![2.5];
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected_data);
    /// ```
    ///
    /// * Backpropagation with requires_grad=true
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    ///
    /// // Create input tensor with requires_grad=true
    /// let t = Tensor::from_data(vec![vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]]], &device, true).unwrap();
    ///
    /// // Apply 2D average pooling
    /// let result = t.avg_pool2d((2, 2), (2, 2)).unwrap();
    /// result.backward().unwrap();
    ///
    /// // Verify gradient shapes and uniform gradient distribution
    /// let t_grad = t.grad().unwrap().unwrap();
    /// assert_eq!(t_grad.shape().unwrap(), Shape::from_dims(&[1, 1, 2, 2]));
    /// // For average pooling, gradient should be uniformly distributed: each input gets 1/(kernel_size) gradient
    /// // With 2x2 kernel, each input element gets 1/4 = 0.25 gradient
    /// let expected_grad = vec![0.25, 0.25, 0.25, 0.25];
    /// assert_eq!(t_grad.to_vec::<f64>().unwrap(), expected_grad);
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
    /// * Forward computation with standard parameters
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    ///
    /// // Create input tensor with shape [batch=1, channels=1, length=4]
    /// let t = Tensor::from_data(vec![vec![vec![1.0, 4.0, 3.0, 2.0]]], &device, false).unwrap();
    ///
    /// // Apply 1D max pooling with kernel_size=2, stride=2
    /// // Expected output: length = floor((4 - 2) / 2 + 1) = 2 (since there are two non-overlapping windows)
    /// // Max values: window 0-1: max(1.0, 4.0) = 4.0, window 2-3: max(3.0, 2.0) = 3.0
    /// let result = t.max_pool1d(2, 2).unwrap();
    /// let expected_shape = Shape::from_dims(&[1, 1, 2]);
    /// assert_eq!(result.shape().unwrap(), expected_shape);
    /// let expected_data = vec![4.0, 3.0];
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected_data);
    /// ```
    ///
    /// * Backpropagation with requires_grad=true
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    ///
    /// // Create input tensor with requires_grad=true
    /// let t = Tensor::from_data(vec![vec![vec![1.0, 4.0, 3.0, 2.0]]], &device, true).unwrap();
    ///
    /// // Apply 1D max pooling (kernel_size must equal stride for backward support)
    /// let result = t.max_pool1d(2, 2).unwrap();
    /// result.backward().unwrap();
    ///
    /// // Verify gradient shapes and sparse gradient characteristic
    /// let t_grad = t.grad().unwrap().unwrap();
    /// assert_eq!(t_grad.shape().unwrap(), Shape::from_dims(&[1, 1, 4]));
    /// // For max pooling, gradient is distributed among max elements
    /// // In input [1.0, 4.0, 3.0, 2.0], max element in window 0-1 is at position 1 (4.0)
    /// // and max element in window 2-3 is at position 2 (3.0)
    /// // Each max element receives gradient 0.5 (distributed equally among output elements)
    /// let expected_grad = vec![0.0, 0.5, 0.5, 0.0];
    /// assert_eq!(t_grad.to_vec::<f64>().unwrap(), expected_grad);
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
    /// * Forward computation with standard parameters
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    ///
    /// // Create input tensor with shape [batch=1, channels=1, length=4]
    /// let t = Tensor::from_data(vec![vec![vec![1.0, 2.0, 3.0, 4.0]]], &device, false).unwrap();
    ///
    /// // Apply 1D average pooling with kernel_size=2, stride=2
    /// // Expected output: length = floor((4 - 2) / 2 + 1) = 2
    /// // Average values: window 0-1: (1.0+2.0)/2=1.5, window 2-3: (3.0+4.0)/2=3.5
    /// let result = t.avg_pool1d(2, 2).unwrap();
    /// let expected_shape = Shape::from_dims(&[1, 1, 2]);
    /// assert_eq!(result.shape().unwrap(), expected_shape);
    /// let expected_data = vec![1.5, 3.5];
    /// assert_eq!(result.to_vec::<f64>().unwrap(), expected_data);
    /// ```
    ///
    /// * Backpropagation with requires_grad=true
    /// ```
    /// use nove::tensor::{Device, Shape, Tensor};
    /// let device = Device::cpu();
    ///
    /// // Create input tensor with requires_grad=true
    /// let t = Tensor::from_data(vec![vec![vec![1.0, 2.0, 3.0, 4.0]]], &device, true).unwrap();
    ///
    /// // Apply 1D average pooling (kernel_size must equal stride for backward support)
    /// let result = t.avg_pool1d(2, 2).unwrap();
    /// result.backward().unwrap();
    ///
    /// // Verify gradient shapes and uniform gradient distribution
    /// let t_grad = t.grad().unwrap().unwrap();
    /// assert_eq!(t_grad.shape().unwrap(), Shape::from_dims(&[1, 1, 4]));
    /// // For average pooling, gradient should be uniformly distributed: each input gets 1/(kernel_size) gradient
    /// // With kernel_size=2 and stride=2, positions 0-1 get 0.5 gradient, positions 2-3 get 0.5 gradient
    /// let expected_grad = vec![0.5, 0.5, 0.5, 0.5];
    /// assert_eq!(t_grad.to_vec::<f64>().unwrap(), expected_grad);
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
