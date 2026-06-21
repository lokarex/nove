use super::{BackendError, BackendKind, TensorBuffer, TensorPayload};
use crate::{DType, Device, Shape};
use std::fmt::Display;

fn backend_error(error: nove_native::cpu::CpuBackendError) -> BackendError {
    match error {
        nove_native::cpu::CpuBackendError::UnsupportedDType(dtype) => {
            BackendError::UnsupportedDType {
                backend: BackendKind::Native,
                dtype: from_cpu_dtype(dtype),
            }
        }
        nove_native::cpu::CpuBackendError::UnsupportedOperation(operation) => {
            BackendError::UnsupportedOperation {
                backend: BackendKind::Native,
                operation,
            }
        }
        other => BackendError::BackendImplementation {
            backend: BackendKind::Native,
            message: other.to_string(),
        },
    }
}

fn to_cpu_dtype(dtype: DType) -> Result<nove_native::cpu::CpuDType, BackendError> {
    match dtype {
        DType::U8 => Ok(nove_native::cpu::CpuDType::U8),
        DType::U32 => Ok(nove_native::cpu::CpuDType::U32),
        DType::I64 => Ok(nove_native::cpu::CpuDType::I64),
        DType::F32 => Ok(nove_native::cpu::CpuDType::F32),
        DType::F64 => Ok(nove_native::cpu::CpuDType::F64),
        DType::BF16 | DType::F16 => Err(BackendError::UnsupportedDType {
            backend: BackendKind::Native,
            dtype,
        }),
    }
}

fn from_cpu_dtype(dtype: nove_native::cpu::CpuDType) -> DType {
    match dtype {
        nove_native::cpu::CpuDType::U8 => DType::U8,
        nove_native::cpu::CpuDType::U32 => DType::U32,
        nove_native::cpu::CpuDType::I64 => DType::I64,
        nove_native::cpu::CpuDType::F32 => DType::F32,
        nove_native::cpu::CpuDType::F64 => DType::F64,
    }
}

fn to_cpu_buffer(buffer: &TensorBuffer) -> nove_native::cpu::CpuBuffer {
    match buffer {
        TensorBuffer::U8(data) => nove_native::cpu::CpuBuffer::U8(data.clone()),
        TensorBuffer::U32(data) => nove_native::cpu::CpuBuffer::U32(data.clone()),
        TensorBuffer::I64(data) => nove_native::cpu::CpuBuffer::I64(data.clone()),
        TensorBuffer::F32(data) => nove_native::cpu::CpuBuffer::F32(data.clone()),
        TensorBuffer::F64(data) => nove_native::cpu::CpuBuffer::F64(data.clone()),
    }
}

fn from_cpu_buffer(buffer: &nove_native::cpu::CpuBuffer) -> TensorBuffer {
    match buffer {
        nove_native::cpu::CpuBuffer::U8(data) => TensorBuffer::U8(data.clone()),
        nove_native::cpu::CpuBuffer::U32(data) => TensorBuffer::U32(data.clone()),
        nove_native::cpu::CpuBuffer::I64(data) => TensorBuffer::I64(data.clone()),
        nove_native::cpu::CpuBuffer::F32(data) => TensorBuffer::F32(data.clone()),
        nove_native::cpu::CpuBuffer::F64(data) => TensorBuffer::F64(data.clone()),
    }
}

/// Native CPU backend storage adapter.
#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct NativeStorage(nove_native::cpu::CpuStorage);

impl NativeStorage {
    fn new(storage: nove_native::cpu::CpuStorage) -> Self {
        Self(storage)
    }
}

// -- Backend trait implementation -------------------------------------------------

macro_rules! cpu_unary_methods {
    ($($method:ident),+ $(,)?) => {
        $(
            fn $method(&self) -> Result<Self, BackendError> {
                self.0.$method().map(Self::new).map_err(backend_error)
            }
        )+
    };
}

macro_rules! cpu_binary_methods {
    ($($method:ident),+ $(,)?) => {
        $(
            fn $method(&self, rhs: &Self) -> Result<Self, BackendError> {
                self.0.$method(&rhs.0).map(Self::new).map_err(backend_error)
            }
        )+
    };
}

impl super::Backend for NativeStorage {
    cpu_unary_methods!(
        zeros_like,
        ones_like,
        relu,
        silu,
        gelu,
        tanh,
        exp,
        log,
        sqrt,
        recip,
        abs,
        neg,
        sum_all,
        max_all,
        min_all,
        mean_all,
        flatten_all,
    );

    cpu_binary_methods!(
        broadcast_add,
        broadcast_mul,
        broadcast_div,
        broadcast_sub,
        broadcast_eq,
        broadcast_ne,
        broadcast_gt,
        broadcast_lt,
        broadcast_ge,
        broadcast_le,
        broadcast_matmul,
        broadcast_pow,
        embedding,
    );
    fn from_payload(payload: &TensorPayload, _device: &Device) -> Result<Self, BackendError> {
        nove_native::cpu::CpuStorage::from_buffer(
            to_cpu_buffer(payload.buffer()),
            payload.shape().dims(),
        )
        .map(Self::new)
        .map_err(backend_error)
    }

    fn rand(
        dtype: DType,
        low: f64,
        high: f64,
        shape: &Shape,
        _device: &Device,
    ) -> Result<Self, BackendError> {
        Self::rand_impl(dtype, low, high, shape)
    }

    fn randn(
        dtype: DType,
        mean: f64,
        std: f64,
        shape: &Shape,
        _device: &Device,
    ) -> Result<Self, BackendError> {
        Self::randn_impl(dtype, mean, std, shape)
    }

    fn zeros(shape: &Shape, dtype: DType, _device: &Device) -> Result<Self, BackendError> {
        nove_native::cpu::CpuStorage::zeros(shape.dims(), to_cpu_dtype(dtype)?)
            .map(Self::new)
            .map_err(backend_error)
    }

    fn ones(shape: &Shape, dtype: DType, _device: &Device) -> Result<Self, BackendError> {
        nove_native::cpu::CpuStorage::ones(shape.dims(), to_cpu_dtype(dtype)?)
            .map(Self::new)
            .map_err(backend_error)
    }

    fn dtype(&self) -> DType {
        from_cpu_dtype(self.0.dtype())
    }

    fn shape(&self) -> Shape {
        Shape::from(self.0.shape())
    }

    fn to_tensor_buffer(&self) -> Result<TensorBuffer, BackendError> {
        Ok(from_cpu_buffer(self.0.buffer()))
    }

    fn zero_set(&mut self) -> Result<(), BackendError> {
        self.0.zero_set().map_err(backend_error)
    }

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }

    fn stack(tensors: &[Self], dim: usize) -> Result<Self, BackendError> {
        let tensors = tensors.iter().map(|s| s.0.clone()).collect::<Vec<_>>();
        nove_native::cpu::CpuStorage::stack(&tensors, dim)
            .map(Self::new)
            .map_err(backend_error)
    }

    fn cat(tensors: &[Self], dim: usize) -> Result<Self, BackendError> {
        let tensors = tensors.iter().map(|s| s.0.clone()).collect::<Vec<_>>();
        nove_native::cpu::CpuStorage::cat(&tensors, dim)
            .map(Self::new)
            .map_err(backend_error)
    }

    fn to_dtype(&self, dtype: DType) -> Result<Self, BackendError> {
        self.0
            .to_dtype(to_cpu_dtype(dtype)?)
            .map(Self::new)
            .map_err(backend_error)
    }

    fn reshape(&self, shape: &Shape) -> Result<Self, BackendError> {
        self.0
            .reshape(shape.dims())
            .map(Self::new)
            .map_err(backend_error)
    }

    fn broadcast_as(&self, shape: &Shape) -> Result<Self, BackendError> {
        self.0
            .broadcast_as(shape.dims())
            .map(Self::new)
            .map_err(backend_error)
    }

    fn flatten_from(&self, dim: usize) -> Result<Self, BackendError> {
        self.0
            .flatten_from(dim)
            .map(Self::new)
            .map_err(backend_error)
    }

    fn flatten_to(&self, dim: usize) -> Result<Self, BackendError> {
        self.0.flatten_to(dim).map(Self::new).map_err(backend_error)
    }

    fn flatten(&self, start: usize, end: usize) -> Result<Self, BackendError> {
        self.0
            .flatten(start, end)
            .map(Self::new)
            .map_err(backend_error)
    }

    fn squeeze(&self, dim: usize) -> Result<Self, BackendError> {
        self.0.squeeze(dim).map(Self::new).map_err(backend_error)
    }

    fn unsqueeze(&self, dim: usize) -> Result<Self, BackendError> {
        self.0.unsqueeze(dim).map(Self::new).map_err(backend_error)
    }

    fn transpose(&self, dim0: usize, dim1: usize) -> Result<Self, BackendError> {
        self.0
            .transpose(dim0, dim1)
            .map(Self::new)
            .map_err(backend_error)
    }

    fn contiguous(&self) -> Result<Self, BackendError> {
        Ok(self.clone())
    }

    fn permute(&self, dims: &[usize]) -> Result<Self, BackendError> {
        self.0.permute(dims).map(Self::new).map_err(backend_error)
    }

    fn narrow(&self, dim: usize, start: usize, length: usize) -> Result<Self, BackendError> {
        self.0
            .narrow(dim, start, length)
            .map(Self::new)
            .map_err(backend_error)
    }

    fn affine(&self, weight: f64, bias: f64) -> Result<Self, BackendError> {
        self.0
            .affine(weight, bias)
            .map(Self::new)
            .map_err(backend_error)
    }

    fn clamp(&self, min: f64, max: f64) -> Result<Self, BackendError> {
        self.0.clamp(min, max).map(Self::new).map_err(backend_error)
    }

    fn powf(&self, exponent: f64) -> Result<Self, BackendError> {
        self.0.powf(exponent).map(Self::new).map_err(backend_error)
    }

    fn sum(&self, dim: usize) -> Result<Self, BackendError> {
        self.0.sum(dim).map(Self::new).map_err(backend_error)
    }

    fn sum_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        self.0
            .sum_keepdim(dim)
            .map(Self::new)
            .map_err(backend_error)
    }

    fn max(&self, dim: usize) -> Result<Self, BackendError> {
        self.0.max(dim).map(Self::new).map_err(backend_error)
    }

    fn max_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        self.0
            .max_keepdim(dim)
            .map(Self::new)
            .map_err(backend_error)
    }

    fn min(&self, dim: usize) -> Result<Self, BackendError> {
        self.0.min(dim).map(Self::new).map_err(backend_error)
    }

    fn min_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        self.0
            .min_keepdim(dim)
            .map(Self::new)
            .map_err(backend_error)
    }

    fn mean(&self, dim: usize) -> Result<Self, BackendError> {
        self.0.mean(dim).map(Self::new).map_err(backend_error)
    }

    fn mean_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        self.0
            .mean_keepdim(dim)
            .map(Self::new)
            .map_err(backend_error)
    }

    fn var(&self, dim: usize) -> Result<Self, BackendError> {
        self.0.var(dim).map(Self::new).map_err(backend_error)
    }

    fn var_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        self.0
            .var_keepdim(dim)
            .map(Self::new)
            .map_err(backend_error)
    }

    fn argmax(&self, dim: usize) -> Result<Self, BackendError> {
        self.0.argmax(dim).map(Self::new).map_err(backend_error)
    }

    fn argmax_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        self.0
            .argmax_keepdim(dim)
            .map(Self::new)
            .map_err(backend_error)
    }

    fn argmin(&self, dim: usize) -> Result<Self, BackendError> {
        self.0.argmin(dim).map(Self::new).map_err(backend_error)
    }

    fn argmin_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        self.0
            .argmin_keepdim(dim)
            .map(Self::new)
            .map_err(backend_error)
    }

    fn conv1d(
        &self,
        kernel: &Self,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self, BackendError> {
        self.0
            .conv1d(&kernel.0, padding, stride, dilation, groups)
            .map(Self::new)
            .map_err(backend_error)
    }

    fn conv2d(
        &self,
        kernel: &Self,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self, BackendError> {
        self.0
            .conv2d(&kernel.0, padding, stride, dilation, groups)
            .map(Self::new)
            .map_err(backend_error)
    }

    fn conv_transpose1d(
        &self,
        kernel: &Self,
        padding: usize,
        output_padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self, BackendError> {
        self.0
            .conv_transpose1d(&kernel.0, padding, output_padding, stride, dilation, groups)
            .map(Self::new)
            .map_err(backend_error)
    }

    fn conv_transpose2d(
        &self,
        kernel: &Self,
        padding: usize,
        output_padding: (usize, usize),
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self, BackendError> {
        self.0
            .conv_transpose2d(&kernel.0, padding, output_padding, stride, dilation, groups)
            .map(Self::new)
            .map_err(backend_error)
    }

    fn max_pool2d_with_stride(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self, BackendError> {
        self.0
            .max_pool2d_with_stride(kernel_size, stride)
            .map(Self::new)
            .map_err(backend_error)
    }

    fn avg_pool2d_with_stride(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self, BackendError> {
        self.0
            .avg_pool2d_with_stride(kernel_size, stride)
            .map(Self::new)
            .map_err(backend_error)
    }

    fn gather(&self, indexes: &Self, dim: usize) -> Result<Self, BackendError> {
        self.0
            .gather(&indexes.0, dim)
            .map(Self::new)
            .map_err(backend_error)
    }

    fn scatter_add(&self, indexes: &Self, source: &Self, dim: usize) -> Result<Self, BackendError> {
        self.0
            .scatter_add(&indexes.0, &source.0, dim)
            .map(Self::new)
            .map_err(backend_error)
    }

    fn index_select(&self, indexes: &Self, dim: usize) -> Result<Self, BackendError> {
        self.0
            .index_select(&indexes.0, dim)
            .map(Self::new)
            .map_err(backend_error)
    }

    fn index_add(&self, indexes: &Self, source: &Self, dim: usize) -> Result<Self, BackendError> {
        self.0
            .index_add(&indexes.0, &source.0, dim)
            .map(Self::new)
            .map_err(backend_error)
    }

    fn where_cond(
        condition: &Self,
        true_value: &Self,
        false_value: &Self,
    ) -> Result<Self, BackendError> {
        condition
            .0
            .where_cond(&true_value.0, &false_value.0)
            .map(Self::new)
            .map_err(backend_error)
    }
}

// -- Private helpers ---

impl NativeStorage {
    fn rand_impl(dtype: DType, low: f64, high: f64, shape: &Shape) -> Result<Self, BackendError> {
        let count = shape.elem_count();
        let mut state = 0x1234_5678_9abc_def0u64 ^ count as u64;
        let span = high - low;
        let values = (0..count)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let unit = ((state >> 11) as f64) / ((1u64 << 53) as f64);
                low + unit * span
            })
            .collect::<Vec<_>>();
        nove_native::cpu::CpuStorage::from_f64_values(values, shape.dims(), to_cpu_dtype(dtype)?)
            .map(Self::new)
            .map_err(backend_error)
    }

    fn randn_impl(dtype: DType, mean: f64, std: f64, shape: &Shape) -> Result<Self, BackendError> {
        let count = shape.elem_count();
        let mut state = 0x0fed_cba9_8765_4321u64 ^ count as u64;
        let mut values = Vec::with_capacity(count);
        while values.len() < count {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u1 = (((state >> 11) as f64) / ((1u64 << 53) as f64)).max(f64::MIN_POSITIVE);
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u2 = ((state >> 11) as f64) / ((1u64 << 53) as f64);
            let z0 = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
            values.push(mean + std * z0);
        }
        nove_native::cpu::CpuStorage::from_f64_values(values, shape.dims(), to_cpu_dtype(dtype)?)
            .map(Self::new)
            .map_err(backend_error)
    }
}

/// Synchronizes all queued work for a Native-backed device.
///
/// Currently a no-op for the CPU-only native backend.
pub(crate) fn synchronize_device(_device: &Device) -> Result<(), BackendError> {
    Ok(())
}
