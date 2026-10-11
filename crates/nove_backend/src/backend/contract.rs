use crate::{BackendError, DType, Device, Shape, TensorBuffer, TensorPayload};

/// Internal trait defining the contract every backend must implement.
///
/// Methods are dispatched through this trait so the compiler guarantees every
/// backend exposes the same set of operations. Adding a new operation requires
/// implementations for both `CandleStorage` and `NativeStorage`.
#[doc(hidden)]
pub(crate) trait Backend: Clone + std::fmt::Debug + Sized {
    // -- properties --
    fn dtype(&self) -> DType;
    fn shape(&self) -> Shape;
    fn to_tensor_buffer(&self) -> Result<TensorBuffer, BackendError>;
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;

    // -- lifecycle --
    fn zero_set(&mut self) -> Result<(), BackendError>;

    // -- factories --
    fn from_payload(payload: TensorPayload, device: &Device) -> Result<Self, BackendError>;
    fn zeros(shape: &Shape, dtype: DType, device: &Device) -> Result<Self, BackendError>;
    fn ones(shape: &Shape, dtype: DType, device: &Device) -> Result<Self, BackendError>;
    fn rand(
        dtype: DType,
        low: f64,
        high: f64,
        shape: &Shape,
        device: &Device,
    ) -> Result<Self, BackendError>;
    fn randn(
        dtype: DType,
        mean: f64,
        std: f64,
        shape: &Shape,
        device: &Device,
    ) -> Result<Self, BackendError>;

    // -- collection --
    fn stack(tensors: &[Self], dim: usize) -> Result<Self, BackendError>;
    fn cat(tensors: &[Self], dim: usize) -> Result<Self, BackendError>;

    // -- conversion --
    fn to_dtype(&self, dtype: DType) -> Result<Self, BackendError>;
    fn to_device(&self, _device: &Device) -> Result<Self, BackendError> {
        Ok(self.clone())
    }

    // -- unary (17) --
    fn zeros_like(&self) -> Result<Self, BackendError>;
    fn ones_like(&self) -> Result<Self, BackendError>;
    fn relu(&self) -> Result<Self, BackendError>;
    fn silu(&self) -> Result<Self, BackendError>;
    fn gelu(&self) -> Result<Self, BackendError>;
    fn tanh(&self) -> Result<Self, BackendError>;
    fn exp(&self) -> Result<Self, BackendError>;
    fn log(&self) -> Result<Self, BackendError>;
    fn sqrt(&self) -> Result<Self, BackendError>;
    fn recip(&self) -> Result<Self, BackendError>;
    fn abs(&self) -> Result<Self, BackendError>;
    fn neg(&self) -> Result<Self, BackendError>;
    fn sum_all(&self) -> Result<Self, BackendError>;
    fn max_all(&self) -> Result<Self, BackendError>;
    fn min_all(&self) -> Result<Self, BackendError>;
    fn mean_all(&self) -> Result<Self, BackendError>;
    fn flatten_all(&self) -> Result<Self, BackendError>;

    // -- binary (13) --
    fn broadcast_add(&self, rhs: &Self) -> Result<Self, BackendError>;
    fn broadcast_mul(&self, rhs: &Self) -> Result<Self, BackendError>;
    fn broadcast_div(&self, rhs: &Self) -> Result<Self, BackendError>;
    fn broadcast_sub(&self, rhs: &Self) -> Result<Self, BackendError>;
    fn broadcast_eq(&self, rhs: &Self) -> Result<Self, BackendError>;
    fn broadcast_ne(&self, rhs: &Self) -> Result<Self, BackendError>;
    fn broadcast_gt(&self, rhs: &Self) -> Result<Self, BackendError>;
    fn broadcast_lt(&self, rhs: &Self) -> Result<Self, BackendError>;
    fn broadcast_ge(&self, rhs: &Self) -> Result<Self, BackendError>;
    fn broadcast_le(&self, rhs: &Self) -> Result<Self, BackendError>;
    fn broadcast_matmul(&self, rhs: &Self) -> Result<Self, BackendError>;
    fn broadcast_pow(&self, rhs: &Self) -> Result<Self, BackendError>;
    fn embedding(&self, rhs: &Self) -> Result<Self, BackendError>;

    // -- shape ops (11) --
    fn reshape(&self, shape: &Shape) -> Result<Self, BackendError>;
    fn broadcast_as(&self, shape: &Shape) -> Result<Self, BackendError>;
    fn flatten_from(&self, dim: usize) -> Result<Self, BackendError>;
    fn flatten_to(&self, dim: usize) -> Result<Self, BackendError>;
    fn flatten(&self, start: usize, end: usize) -> Result<Self, BackendError>;
    fn squeeze(&self, dim: usize) -> Result<Self, BackendError>;
    fn unsqueeze(&self, dim: usize) -> Result<Self, BackendError>;
    fn transpose(&self, dim0: usize, dim1: usize) -> Result<Self, BackendError>;
    fn contiguous(&self) -> Result<Self, BackendError> {
        Ok(self.clone())
    }
    fn permute(&self, dims: &[usize]) -> Result<Self, BackendError>;
    fn narrow(&self, dim: usize, start: usize, length: usize) -> Result<Self, BackendError>;

    // -- math (3) --
    fn affine(&self, weight: f64, bias: f64) -> Result<Self, BackendError>;
    fn clamp(&self, min: f64, max: f64) -> Result<Self, BackendError>;
    fn powf(&self, exponent: f64) -> Result<Self, BackendError>;

    // -- reduction (10) --
    fn sum(&self, dim: usize) -> Result<Self, BackendError>;
    fn sum_keepdim(&self, dim: usize) -> Result<Self, BackendError>;
    fn max(&self, dim: usize) -> Result<Self, BackendError>;
    fn max_keepdim(&self, dim: usize) -> Result<Self, BackendError>;
    fn min(&self, dim: usize) -> Result<Self, BackendError>;
    fn min_keepdim(&self, dim: usize) -> Result<Self, BackendError>;
    fn mean(&self, dim: usize) -> Result<Self, BackendError>;
    fn mean_keepdim(&self, dim: usize) -> Result<Self, BackendError>;
    fn var(&self, dim: usize) -> Result<Self, BackendError>;
    fn var_keepdim(&self, dim: usize) -> Result<Self, BackendError>;

    // -- arg ops (4) --
    fn argmax(&self, dim: usize) -> Result<Self, BackendError>;
    fn argmax_keepdim(&self, dim: usize) -> Result<Self, BackendError>;
    fn argmin(&self, dim: usize) -> Result<Self, BackendError>;
    fn argmin_keepdim(&self, dim: usize) -> Result<Self, BackendError>;

    // -- convolution (4) --
    fn conv1d(
        &self,
        kernel: &Self,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self, BackendError>;
    fn conv2d(
        &self,
        kernel: &Self,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self, BackendError>;
    fn conv_transpose1d(
        &self,
        kernel: &Self,
        padding: usize,
        output_padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self, BackendError>;
    fn conv_transpose2d(
        &self,
        kernel: &Self,
        padding: usize,
        output_padding: (usize, usize),
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self, BackendError>;

    // -- pooling (2) --
    fn max_pool2d_with_stride(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self, BackendError>;
    fn avg_pool2d_with_stride(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self, BackendError>;

    // -- indexing (5) --
    fn gather(&self, indexes: &Self, dim: usize) -> Result<Self, BackendError>;
    fn scatter_add(&self, indexes: &Self, source: &Self, dim: usize) -> Result<Self, BackendError>;
    fn index_select(&self, indexes: &Self, dim: usize) -> Result<Self, BackendError>;
    fn index_add(&self, indexes: &Self, source: &Self, dim: usize) -> Result<Self, BackendError>;
    fn where_cond(
        condition: &Self,
        true_value: &Self,
        false_value: &Self,
    ) -> Result<Self, BackendError>;
}
