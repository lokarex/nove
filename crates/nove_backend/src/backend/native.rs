use super::{BackendError, BackendKind, TensorBuffer, TensorPayload};
use crate::{Device, DType, Shape};
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

    pub(crate) fn from_payload(payload: &TensorPayload) -> Result<Self, BackendError> {
        nove_native::cpu::CpuStorage::from_buffer(
            to_cpu_buffer(payload.buffer()),
            payload.shape().dims(),
        )
        .map(Self::new)
        .map_err(backend_error)
    }

    pub(crate) fn from_f64_values(
        values: Vec<f64>,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self, BackendError> {
        nove_native::cpu::CpuStorage::from_f64_values(values, shape.dims(), to_cpu_dtype(dtype)?)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn zeros(shape: &Shape, dtype: DType) -> Result<Self, BackendError> {
        nove_native::cpu::CpuStorage::zeros(shape.dims(), to_cpu_dtype(dtype)?)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn ones(shape: &Shape, dtype: DType) -> Result<Self, BackendError> {
        nove_native::cpu::CpuStorage::ones(shape.dims(), to_cpu_dtype(dtype)?)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn to_tensor_buffer(&self) -> Result<TensorBuffer, BackendError> {
        Ok(from_cpu_buffer(self.0.buffer()))
    }

    pub(crate) fn shape(&self) -> Shape {
        Shape::from(self.0.shape())
    }

    pub(crate) fn dtype(&self) -> DType {
        from_cpu_dtype(self.0.dtype())
    }

    pub(crate) fn copy_detached(&self) -> Result<Self, BackendError> {
        self.0.copy_detached().map(Self::new).map_err(backend_error)
    }

    pub(crate) fn detach(&self) -> Result<Self, BackendError> {
        self.0.detach().map(Self::new).map_err(backend_error)
    }

    pub(crate) fn assign_from(&mut self, other: &Self) -> Result<(), BackendError> {
        self.0.assign_from(&other.0).map_err(backend_error)
    }

    pub(crate) fn zero_set(&mut self) -> Result<(), BackendError> {
        self.0.zero_set().map_err(backend_error)
    }

    pub(crate) fn to_dtype(&self, dtype: DType) -> Result<Self, BackendError> {
        self.0
            .to_dtype(to_cpu_dtype(dtype)?)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }

    pub(crate) fn stack(tensors: &[Self], dim: usize) -> Result<Self, BackendError> {
        let tensors = tensors
            .iter()
            .map(|storage| storage.0.clone())
            .collect::<Vec<_>>();
        nove_native::cpu::CpuStorage::stack(&tensors, dim)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn cat(tensors: &[Self], dim: usize) -> Result<Self, BackendError> {
        let tensors = tensors
            .iter()
            .map(|storage| storage.0.clone())
            .collect::<Vec<_>>();
        nove_native::cpu::CpuStorage::cat(&tensors, dim)
            .map(Self::new)
            .map_err(backend_error)
    }
}

macro_rules! cpu_unary_methods {
    ($($method:ident),+ $(,)?) => {
        impl NativeStorage {
            $(
                pub(crate) fn $method(&self) -> Result<Self, BackendError> {
                    self.0.$method().map(Self::new).map_err(backend_error)
                }
            )+
        }
    };
}

macro_rules! cpu_binary_methods {
    ($($method:ident),+ $(,)?) => {
        impl NativeStorage {
            $(
                pub(crate) fn $method(&self, rhs: &Self) -> Result<Self, BackendError> {
                    self.0.$method(&rhs.0).map(Self::new).map_err(backend_error)
                }
            )+
        }
    };
}

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

impl NativeStorage {
    pub(crate) fn reshape(&self, shape: &Shape) -> Result<Self, BackendError> {
        self.0
            .reshape(shape.dims())
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn broadcast_as(&self, shape: &Shape) -> Result<Self, BackendError> {
        self.0
            .broadcast_as(shape.dims())
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn flatten_from(&self, dim: usize) -> Result<Self, BackendError> {
        self.0
            .flatten_from(dim)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn flatten_to(&self, dim: usize) -> Result<Self, BackendError> {
        self.0.flatten_to(dim).map(Self::new).map_err(backend_error)
    }

    pub(crate) fn flatten(&self, start: usize, end: usize) -> Result<Self, BackendError> {
        self.0
            .flatten(start, end)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn squeeze(&self, dim: usize) -> Result<Self, BackendError> {
        self.0.squeeze(dim).map(Self::new).map_err(backend_error)
    }

    pub(crate) fn unsqueeze(&self, dim: usize) -> Result<Self, BackendError> {
        self.0.unsqueeze(dim).map(Self::new).map_err(backend_error)
    }

    pub(crate) fn transpose(&self, dim0: usize, dim1: usize) -> Result<Self, BackendError> {
        self.0
            .transpose(dim0, dim1)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn permute(&self, dims: &[usize]) -> Result<Self, BackendError> {
        self.0.permute(dims).map(Self::new).map_err(backend_error)
    }

    pub(crate) fn narrow(
        &self,
        dim: usize,
        start: usize,
        length: usize,
    ) -> Result<Self, BackendError> {
        self.0
            .narrow(dim, start, length)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn affine(&self, weight: f64, bias: f64) -> Result<Self, BackendError> {
        self.0
            .affine(weight, bias)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn clamp(&self, min: f64, max: f64) -> Result<Self, BackendError> {
        self.0.clamp(min, max).map(Self::new).map_err(backend_error)
    }

    pub(crate) fn powf(&self, exponent: f64) -> Result<Self, BackendError> {
        self.0.powf(exponent).map(Self::new).map_err(backend_error)
    }

    pub(crate) fn sum(&self, dim: usize) -> Result<Self, BackendError> {
        self.0.sum(dim).map(Self::new).map_err(backend_error)
    }

    pub(crate) fn sum_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        self.0
            .sum_keepdim(dim)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn max(&self, dim: usize) -> Result<Self, BackendError> {
        self.0.max(dim).map(Self::new).map_err(backend_error)
    }

    pub(crate) fn max_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        self.0
            .max_keepdim(dim)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn min(&self, dim: usize) -> Result<Self, BackendError> {
        self.0.min(dim).map(Self::new).map_err(backend_error)
    }

    pub(crate) fn min_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        self.0
            .min_keepdim(dim)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn mean(&self, dim: usize) -> Result<Self, BackendError> {
        self.0.mean(dim).map(Self::new).map_err(backend_error)
    }

    pub(crate) fn mean_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        self.0
            .mean_keepdim(dim)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn var(&self, dim: usize) -> Result<Self, BackendError> {
        self.0.var(dim).map(Self::new).map_err(backend_error)
    }

    pub(crate) fn var_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        self.0
            .var_keepdim(dim)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn argmax(&self, dim: usize) -> Result<Self, BackendError> {
        self.0.argmax(dim).map(Self::new).map_err(backend_error)
    }

    pub(crate) fn argmax_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        self.0
            .argmax_keepdim(dim)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn argmin(&self, dim: usize) -> Result<Self, BackendError> {
        self.0.argmin(dim).map(Self::new).map_err(backend_error)
    }

    pub(crate) fn argmin_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        self.0
            .argmin_keepdim(dim)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn conv1d(
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

    pub(crate) fn conv2d(
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

    pub(crate) fn max_pool2d_with_stride(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self, BackendError> {
        self.0
            .max_pool2d_with_stride(kernel_size, stride)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn avg_pool2d_with_stride(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self, BackendError> {
        self.0
            .avg_pool2d_with_stride(kernel_size, stride)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn gather(&self, indexes: &Self, dim: usize) -> Result<Self, BackendError> {
        self.0
            .gather(&indexes.0, dim)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn index_select(&self, indexes: &Self, dim: usize) -> Result<Self, BackendError> {
        self.0
            .index_select(&indexes.0, dim)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn where_cond(
        &self,
        true_value: &Self,
        false_value: &Self,
    ) -> Result<Self, BackendError> {
        self.0
            .where_cond(&true_value.0, &false_value.0)
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
