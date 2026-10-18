#[cfg(feature = "native")]
use super::Backend;
use super::{BackendError, BackendKind, BackendStorage, TensorBuffer, TensorPayload};
use crate::{
    DType, Device, Shape,
    device::{BackendDeviceHandle, DeviceKind},
};
use std::collections::HashMap;

fn backend_error(error: nove_candle::CandleBackendError) -> BackendError {
    BackendError::BackendImplementation {
        backend: BackendKind::Candle,
        message: error.to_string(),
    }
}

fn candle_op_error(error: impl std::fmt::Display) -> BackendError {
    BackendError::BackendImplementation {
        backend: BackendKind::Candle,
        message: error.to_string(),
    }
}

fn to_candle_dtype(dtype: DType) -> nove_candle::CandleDType {
    match dtype {
        DType::U8 => nove_candle::CandleDType::U8,
        DType::U32 => nove_candle::CandleDType::U32,
        DType::I64 => nove_candle::CandleDType::I64,
        DType::BF16 => nove_candle::CandleDType::BF16,
        DType::F16 => nove_candle::CandleDType::F16,
        DType::F32 => nove_candle::CandleDType::F32,
        DType::F64 => nove_candle::CandleDType::F64,
    }
}

fn from_candle_dtype(dtype: nove_candle::CandleDType) -> DType {
    match dtype {
        nove_candle::CandleDType::U8 => DType::U8,
        nove_candle::CandleDType::U32 => DType::U32,
        nove_candle::CandleDType::I64 => DType::I64,
        nove_candle::CandleDType::BF16 => DType::BF16,
        nove_candle::CandleDType::F16 => DType::F16,
        nove_candle::CandleDType::F32 => DType::F32,
        nove_candle::CandleDType::F64 => DType::F64,
    }
}

fn to_candle_device_kind(kind: DeviceKind) -> nove_candle::CandleDeviceKind {
    match kind {
        DeviceKind::Cpu => nove_candle::CandleDeviceKind::Cpu,
        DeviceKind::Cuda => nove_candle::CandleDeviceKind::Cuda,
        DeviceKind::Metal => nove_candle::CandleDeviceKind::Metal,
    }
}

pub(crate) fn to_candle_device(device: &Device) -> Result<nove_candle::CandleDevice, BackendError> {
    match device.handle() {
        BackendDeviceHandle::Candle(device) => Ok(device.clone()),
        #[cfg(feature = "native")]
        BackendDeviceHandle::Native(_) => Err(BackendError::UnsupportedBackend(device.backend())),
    }
}

/// Candle-backed tensor storage adapter.
#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct CandleStorage(nove_candle::CandleStorage);

impl CandleStorage {
    pub(crate) fn new(storage: nove_candle::CandleStorage) -> Self {
        Self(storage)
    }

    pub(crate) fn inner(&self) -> &nove_candle::CandleStorage {
        &self.0
    }

    pub(crate) fn into_inner(self) -> nove_candle::CandleStorage {
        self.0
    }

    pub(crate) fn to_candle_tensor(&self) -> Result<nove_candle::CandleTensor, BackendError> {
        self.inner().to_candle_tensor().map_err(backend_error)
    }
}

// -- Backend trait implementation -------------------------------------------------

impl super::Backend for CandleStorage {
    // -- properties --
    fn dtype(&self) -> DType {
        from_candle_dtype(self.inner().dtype())
    }

    fn shape(&self) -> Shape {
        Shape::from(self.inner().shape())
    }

    fn to_tensor_buffer(&self) -> Result<TensorBuffer, BackendError> {
        let tensor = self
            .inner()
            .as_tensor()
            .flatten_all()
            .map_err(candle_op_error)?;
        match self.dtype() {
            DType::U8 => tensor
                .to_vec1::<u8>()
                .map(TensorBuffer::U8)
                .map_err(candle_op_error),
            DType::U32 => tensor
                .to_vec1::<u32>()
                .map(TensorBuffer::U32)
                .map_err(candle_op_error),
            DType::I64 => tensor
                .to_vec1::<i64>()
                .map(TensorBuffer::I64)
                .map_err(candle_op_error),
            DType::F32 => tensor
                .to_vec1::<f32>()
                .map(TensorBuffer::F32)
                .map_err(candle_op_error),
            DType::F64 => tensor
                .to_vec1::<f64>()
                .map(TensorBuffer::F64)
                .map_err(candle_op_error),
            dtype => Err(BackendError::UnsupportedDType {
                backend: BackendKind::Candle,
                dtype,
            }),
        }
    }

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner().fmt(f)
    }

    // -- lifecycle --
    fn try_clone(&self) -> Result<Self, BackendError> {
        let tensor = self.to_candle_tensor()?;
        nove_candle::CandleStorage::from_candle_tensor(tensor, self.inner().as_tensor().device())
            .map(Self::new)
            .map_err(backend_error)
    }

    fn same_device(&self, other: &Self) -> bool {
        self.inner()
            .as_tensor()
            .device()
            .same_device(other.inner().as_tensor().device())
    }

    fn zero_set(&mut self) -> Result<(), BackendError> {
        self.0.zero_set().map_err(backend_error)
    }

    // -- factories --
    fn from_payload(payload: TensorPayload, device: &Device) -> Result<Self, BackendError> {
        let (buffer, shape) = payload.into_parts();
        match buffer {
            TensorBuffer::U8(data) => Self::from_vec(data, &shape, device),
            TensorBuffer::U32(data) => Self::from_vec(data, &shape, device),
            TensorBuffer::I64(data) => Self::from_vec(data, &shape, device),
            TensorBuffer::F32(data) => Self::from_vec(data, &shape, device),
            TensorBuffer::F64(data) => Self::from_vec(data, &shape, device),
        }
    }

    fn zeros(shape: &Shape, dtype: DType, device: &Device) -> Result<Self, BackendError> {
        let device = to_candle_device(device)?;
        nove_candle::CandleStorage::zeros(shape.dims(), to_candle_dtype(dtype), &device)
            .map(Self::new)
            .map_err(backend_error)
    }

    fn ones(shape: &Shape, dtype: DType, device: &Device) -> Result<Self, BackendError> {
        let device = to_candle_device(device)?;
        nove_candle::CandleStorage::ones(shape.dims(), to_candle_dtype(dtype), &device)
            .map(Self::new)
            .map_err(backend_error)
    }

    fn rand(
        dtype: DType,
        low: f64,
        high: f64,
        shape: &Shape,
        device: &Device,
    ) -> Result<Self, BackendError> {
        match dtype {
            DType::F32 => Self::rand_typed(low as f32, high as f32, shape, device),
            DType::F64 => Self::rand_typed(low, high, shape, device),
            dtype => Err(BackendError::UnsupportedDType {
                backend: BackendKind::Candle,
                dtype,
            }),
        }
    }

    fn randn(
        dtype: DType,
        mean: f64,
        std: f64,
        shape: &Shape,
        device: &Device,
    ) -> Result<Self, BackendError> {
        match dtype {
            DType::F32 => Self::randn_typed(mean as f32, std as f32, shape, device),
            DType::F64 => Self::randn_typed(mean, std, shape, device),
            dtype => Err(BackendError::UnsupportedDType {
                backend: BackendKind::Candle,
                dtype,
            }),
        }
    }

    // -- collection --
    fn stack(tensors: &[Self], dim: usize) -> Result<Self, BackendError> {
        let tensors = tensors
            .iter()
            .map(|s| s.inner().clone())
            .collect::<Vec<_>>();
        nove_candle::CandleStorage::stack(&tensors, dim)
            .map(Self::new)
            .map_err(backend_error)
    }

    fn cat(tensors: &[Self], dim: usize) -> Result<Self, BackendError> {
        let tensors = tensors
            .iter()
            .map(|s| s.inner().clone())
            .collect::<Vec<_>>();
        nove_candle::CandleStorage::cat(&tensors, dim)
            .map(Self::new)
            .map_err(backend_error)
    }

    // -- conversion --
    fn to_device(&self, device: &Device) -> Result<Self, BackendError> {
        let device = to_candle_device(device)?;
        self.inner()
            .to_device(&device)
            .map(Self::new)
            .map_err(backend_error)
    }

    fn to_dtype(&self, dtype: DType) -> Result<Self, BackendError> {
        self.inner()
            .to_dtype(to_candle_dtype(dtype))
            .map(Self::new)
            .map_err(backend_error)
    }

    // -- unary (17) --
    fn zeros_like(&self) -> Result<Self, BackendError> {
        self.inner()
            .zeros_like()
            .map(Self::new)
            .map_err(backend_error)
    }
    fn ones_like(&self) -> Result<Self, BackendError> {
        self.inner()
            .ones_like()
            .map(Self::new)
            .map_err(backend_error)
    }
    fn relu(&self) -> Result<Self, BackendError> {
        self.inner().relu().map(Self::new).map_err(backend_error)
    }
    fn silu(&self) -> Result<Self, BackendError> {
        self.inner().silu().map(Self::new).map_err(backend_error)
    }
    fn gelu(&self) -> Result<Self, BackendError> {
        self.inner().gelu().map(Self::new).map_err(backend_error)
    }
    fn tanh(&self) -> Result<Self, BackendError> {
        self.inner().tanh().map(Self::new).map_err(backend_error)
    }
    fn exp(&self) -> Result<Self, BackendError> {
        self.inner().exp().map(Self::new).map_err(backend_error)
    }
    fn log(&self) -> Result<Self, BackendError> {
        self.inner().log().map(Self::new).map_err(backend_error)
    }
    fn sqrt(&self) -> Result<Self, BackendError> {
        self.inner().sqrt().map(Self::new).map_err(backend_error)
    }
    fn recip(&self) -> Result<Self, BackendError> {
        self.inner().recip().map(Self::new).map_err(backend_error)
    }
    fn abs(&self) -> Result<Self, BackendError> {
        self.inner().abs().map(Self::new).map_err(backend_error)
    }
    fn neg(&self) -> Result<Self, BackendError> {
        self.inner().neg().map(Self::new).map_err(backend_error)
    }
    fn sum_all(&self) -> Result<Self, BackendError> {
        self.inner().sum_all().map(Self::new).map_err(backend_error)
    }
    fn max_all(&self) -> Result<Self, BackendError> {
        self.inner().max_all().map(Self::new).map_err(backend_error)
    }
    fn min_all(&self) -> Result<Self, BackendError> {
        self.inner().min_all().map(Self::new).map_err(backend_error)
    }
    fn mean_all(&self) -> Result<Self, BackendError> {
        self.inner()
            .mean_all()
            .map(Self::new)
            .map_err(backend_error)
    }
    fn flatten_all(&self) -> Result<Self, BackendError> {
        self.inner()
            .flatten_all()
            .map(Self::new)
            .map_err(backend_error)
    }

    // -- binary (13) --
    fn broadcast_add(&self, rhs: &Self) -> Result<Self, BackendError> {
        self.inner()
            .broadcast_add(rhs.inner())
            .map(Self::new)
            .map_err(backend_error)
    }
    fn broadcast_mul(&self, rhs: &Self) -> Result<Self, BackendError> {
        self.inner()
            .broadcast_mul(rhs.inner())
            .map(Self::new)
            .map_err(backend_error)
    }
    fn broadcast_div(&self, rhs: &Self) -> Result<Self, BackendError> {
        self.inner()
            .broadcast_div(rhs.inner())
            .map(Self::new)
            .map_err(backend_error)
    }
    fn broadcast_sub(&self, rhs: &Self) -> Result<Self, BackendError> {
        self.inner()
            .broadcast_sub(rhs.inner())
            .map(Self::new)
            .map_err(backend_error)
    }
    fn broadcast_eq(&self, rhs: &Self) -> Result<Self, BackendError> {
        self.inner()
            .broadcast_eq(rhs.inner())
            .map(Self::new)
            .map_err(backend_error)
    }
    fn broadcast_ne(&self, rhs: &Self) -> Result<Self, BackendError> {
        self.inner()
            .broadcast_ne(rhs.inner())
            .map(Self::new)
            .map_err(backend_error)
    }
    fn broadcast_gt(&self, rhs: &Self) -> Result<Self, BackendError> {
        self.inner()
            .broadcast_gt(rhs.inner())
            .map(Self::new)
            .map_err(backend_error)
    }
    fn broadcast_lt(&self, rhs: &Self) -> Result<Self, BackendError> {
        self.inner()
            .broadcast_lt(rhs.inner())
            .map(Self::new)
            .map_err(backend_error)
    }
    fn broadcast_ge(&self, rhs: &Self) -> Result<Self, BackendError> {
        self.inner()
            .broadcast_ge(rhs.inner())
            .map(Self::new)
            .map_err(backend_error)
    }
    fn broadcast_le(&self, rhs: &Self) -> Result<Self, BackendError> {
        self.inner()
            .broadcast_le(rhs.inner())
            .map(Self::new)
            .map_err(backend_error)
    }
    fn broadcast_matmul(&self, rhs: &Self) -> Result<Self, BackendError> {
        self.inner()
            .broadcast_matmul(rhs.inner())
            .map(Self::new)
            .map_err(backend_error)
    }
    fn broadcast_pow(&self, rhs: &Self) -> Result<Self, BackendError> {
        self.inner()
            .broadcast_pow(rhs.inner())
            .map(Self::new)
            .map_err(backend_error)
    }
    fn embedding(&self, rhs: &Self) -> Result<Self, BackendError> {
        self.inner()
            .embedding(rhs.inner())
            .map(Self::new)
            .map_err(backend_error)
    }

    // -- shape ops (11) --
    fn reshape(&self, shape: &Shape) -> Result<Self, BackendError> {
        self.inner()
            .reshape(shape.dims())
            .map(Self::new)
            .map_err(backend_error)
    }
    fn broadcast_as(&self, shape: &Shape) -> Result<Self, BackendError> {
        self.inner()
            .broadcast_as(shape.dims())
            .map(Self::new)
            .map_err(backend_error)
    }
    fn flatten_from(&self, dim: usize) -> Result<Self, BackendError> {
        self.inner()
            .flatten_from(dim)
            .map(Self::new)
            .map_err(backend_error)
    }
    fn flatten_to(&self, dim: usize) -> Result<Self, BackendError> {
        self.inner()
            .flatten_to(dim)
            .map(Self::new)
            .map_err(backend_error)
    }
    fn flatten(&self, start: usize, end: usize) -> Result<Self, BackendError> {
        self.inner()
            .flatten(start, end)
            .map(Self::new)
            .map_err(backend_error)
    }
    fn squeeze(&self, dim: usize) -> Result<Self, BackendError> {
        self.inner()
            .squeeze(dim)
            .map(Self::new)
            .map_err(backend_error)
    }
    fn unsqueeze(&self, dim: usize) -> Result<Self, BackendError> {
        self.inner()
            .unsqueeze(dim)
            .map(Self::new)
            .map_err(backend_error)
    }
    fn transpose(&self, dim0: usize, dim1: usize) -> Result<Self, BackendError> {
        self.inner()
            .transpose(dim0, dim1)
            .map(Self::new)
            .map_err(backend_error)
    }
    fn contiguous(&self) -> Result<Self, BackendError> {
        self.inner()
            .contiguous()
            .map(Self::new)
            .map_err(backend_error)
    }
    fn permute(&self, dims: &[usize]) -> Result<Self, BackendError> {
        self.inner()
            .permute(dims)
            .map(Self::new)
            .map_err(backend_error)
    }
    fn narrow(&self, dim: usize, start: usize, length: usize) -> Result<Self, BackendError> {
        self.inner()
            .narrow(dim, start, length)
            .map(Self::new)
            .map_err(backend_error)
    }

    // -- math (3) --
    fn affine(&self, weight: f64, bias: f64) -> Result<Self, BackendError> {
        self.inner()
            .affine(weight, bias)
            .map(Self::new)
            .map_err(backend_error)
    }
    fn clamp(&self, min: f64, max: f64) -> Result<Self, BackendError> {
        self.inner()
            .clamp(min, max)
            .map(Self::new)
            .map_err(backend_error)
    }
    fn powf(&self, exponent: f64) -> Result<Self, BackendError> {
        self.inner()
            .powf(exponent)
            .map(Self::new)
            .map_err(backend_error)
    }

    // -- reduction (10) --
    fn sum(&self, dim: usize) -> Result<Self, BackendError> {
        self.inner().sum(dim).map(Self::new).map_err(backend_error)
    }
    fn sum_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        self.inner()
            .sum_keepdim(dim)
            .map(Self::new)
            .map_err(backend_error)
    }
    fn max(&self, dim: usize) -> Result<Self, BackendError> {
        self.inner().max(dim).map(Self::new).map_err(backend_error)
    }
    fn max_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        self.inner()
            .max_keepdim(dim)
            .map(Self::new)
            .map_err(backend_error)
    }
    fn min(&self, dim: usize) -> Result<Self, BackendError> {
        self.inner().min(dim).map(Self::new).map_err(backend_error)
    }
    fn min_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        self.inner()
            .min_keepdim(dim)
            .map(Self::new)
            .map_err(backend_error)
    }
    fn mean(&self, dim: usize) -> Result<Self, BackendError> {
        self.inner().mean(dim).map(Self::new).map_err(backend_error)
    }
    fn mean_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        self.inner()
            .mean_keepdim(dim)
            .map(Self::new)
            .map_err(backend_error)
    }
    fn var(&self, dim: usize) -> Result<Self, BackendError> {
        self.inner().var(dim).map(Self::new).map_err(backend_error)
    }
    fn var_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        self.inner()
            .var_keepdim(dim)
            .map(Self::new)
            .map_err(backend_error)
    }

    // -- arg ops (4) --
    fn argmax(&self, dim: usize) -> Result<Self, BackendError> {
        self.inner()
            .argmax(dim)
            .map(Self::new)
            .map_err(backend_error)
    }
    fn argmax_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        self.inner()
            .argmax_keepdim(dim)
            .map(Self::new)
            .map_err(backend_error)
    }
    fn argmin(&self, dim: usize) -> Result<Self, BackendError> {
        self.inner()
            .argmin(dim)
            .map(Self::new)
            .map_err(backend_error)
    }
    fn argmin_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        self.inner()
            .argmin_keepdim(dim)
            .map(Self::new)
            .map_err(backend_error)
    }

    // -- convolution (2) --
    fn conv1d(
        &self,
        kernel: &Self,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self, BackendError> {
        self.inner()
            .conv1d(kernel.inner(), padding, stride, dilation, groups)
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
        self.inner()
            .conv2d(kernel.inner(), padding, stride, dilation, groups)
            .map(Self::new)
            .map_err(backend_error)
    }

    // -- pooling (2) --
    fn max_pool2d_with_stride(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self, BackendError> {
        self.inner()
            .max_pool2d_with_stride(kernel_size, stride)
            .map(Self::new)
            .map_err(backend_error)
    }
    fn avg_pool2d_with_stride(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self, BackendError> {
        self.inner()
            .avg_pool2d_with_stride(kernel_size, stride)
            .map(Self::new)
            .map_err(backend_error)
    }

    // -- indexing (5) --
    fn gather(&self, indexes: &Self, dim: usize) -> Result<Self, BackendError> {
        self.inner()
            .gather(indexes.inner(), dim)
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
        self.inner()
            .conv_transpose1d(
                kernel.inner(),
                padding,
                output_padding,
                stride,
                dilation,
                groups,
            )
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
        self.inner()
            .conv_transpose2d(
                kernel.inner(),
                padding,
                output_padding,
                stride,
                dilation,
                groups,
            )
            .map(Self::new)
            .map_err(backend_error)
    }
    fn scatter_add(&self, indexes: &Self, source: &Self, dim: usize) -> Result<Self, BackendError> {
        self.inner()
            .scatter_add(indexes.inner(), source.inner(), dim)
            .map(Self::new)
            .map_err(backend_error)
    }
    fn index_select(&self, indexes: &Self, dim: usize) -> Result<Self, BackendError> {
        self.inner()
            .index_select(indexes.inner(), dim)
            .map(Self::new)
            .map_err(backend_error)
    }
    fn index_add(&self, indexes: &Self, source: &Self, dim: usize) -> Result<Self, BackendError> {
        self.inner()
            .index_add(indexes.inner(), source.inner(), dim)
            .map(Self::new)
            .map_err(backend_error)
    }
    fn where_cond(
        condition: &Self,
        true_value: &Self,
        false_value: &Self,
    ) -> Result<Self, BackendError> {
        condition
            .inner()
            .where_cond(true_value.inner(), false_value.inner())
            .map(Self::new)
            .map_err(backend_error)
    }
}

// -- Private helpers (generic over Candle types) ---

impl CandleStorage {
    fn from_vec<D>(data: Vec<D>, shape: &Shape, device: &Device) -> Result<Self, BackendError>
    where
        D: nove_candle::CandleWithDType,
    {
        let device = to_candle_device(device)?;
        nove_candle::CandleStorage::from_vec(data, shape.dims(), &device)
            .map(Self::new)
            .map_err(backend_error)
    }

    fn rand_typed<T>(low: T, high: T, shape: &Shape, device: &Device) -> Result<Self, BackendError>
    where
        T: nove_candle::CandleFloatDType,
    {
        let device = to_candle_device(device)?;
        nove_candle::CandleStorage::rand(low, high, shape.dims(), &device)
            .map(Self::new)
            .map_err(backend_error)
    }

    fn randn_typed<T>(mean: T, std: T, shape: &Shape, device: &Device) -> Result<Self, BackendError>
    where
        T: nove_candle::CandleFloatDType,
    {
        let device = to_candle_device(device)?;
        nove_candle::CandleStorage::randn(mean, std, shape.dims(), &device)
            .map(Self::new)
            .map_err(backend_error)
    }
}

pub(crate) fn validate_device(kind: DeviceKind, index: usize) -> Result<(), BackendError> {
    nove_candle::validate_device(to_candle_device_kind(kind), index).map_err(backend_error)
}

pub(crate) fn create_candle_device(
    kind: DeviceKind,
    index: usize,
) -> Result<nove_candle::CandleDevice, BackendError> {
    nove_candle::to_candle_device(to_candle_device_kind(kind), index).map_err(backend_error)
}

pub(crate) fn synchronize_device(device: &Device) -> Result<(), BackendError> {
    let device = to_candle_device(device)?;
    nove_candle::synchronize_device(&device).map_err(backend_error)
}

pub(crate) fn save_safetensors(
    file_path: &str,
    tensors: HashMap<String, BackendStorage>,
) -> Result<(), BackendError> {
    let tensors = tensors
        .into_iter()
        .map(|(name, storage)| match storage {
            BackendStorage::Candle(storage) => Ok((name, storage.into_inner())),
            #[cfg(feature = "native")]
            BackendStorage::Native(storage) => {
                let payload = TensorPayload::new(storage.to_tensor_buffer()?, storage.shape())?;
                let handle = create_candle_device(DeviceKind::Cpu, 0)?;
                let device =
                    Device::from_handle(BackendDeviceHandle::Candle(handle), DeviceKind::Cpu, 0);
                let storage = CandleStorage::from_payload(payload, &device)?;
                Ok((name, storage.into_inner()))
            }
        })
        .collect::<Result<HashMap<_, _>, BackendError>>()?;
    nove_candle::save_safetensors(file_path, tensors).map_err(backend_error)
}

pub(crate) fn load_safetensors(
    file_path: &str,
    device: &Device,
) -> Result<HashMap<String, BackendStorage>, BackendError> {
    let candle_device = to_candle_device(device)?;
    nove_candle::load_safetensors(file_path, &candle_device)
        .map(|tensors| {
            tensors
                .into_iter()
                .map(|(name, storage)| (name, BackendStorage::Candle(CandleStorage::new(storage))))
                .collect()
        })
        .map_err(backend_error)
}

#[cfg(test)]
mod tests {
    use super::{CandleStorage, from_candle_dtype, to_candle_dtype};
    use crate::{DType, backend::Backend};

    #[test]
    fn dtype_conversions_preserve_all_supported_types() {
        use nove_candle::CandleDType;
        for (dtype, candle_dtype) in [
            (DType::U8, CandleDType::U8),
            (DType::U32, CandleDType::U32),
            (DType::I64, CandleDType::I64),
            (DType::BF16, CandleDType::BF16),
            (DType::F16, CandleDType::F16),
            (DType::F32, CandleDType::F32),
            (DType::F64, CandleDType::F64),
        ] {
            assert_eq!(to_candle_dtype(dtype), candle_dtype);
            assert_eq!(from_candle_dtype(candle_dtype), dtype);
        }
    }

    #[cfg(any(
        feature = "candle-cpu",
        feature = "candle-cuda",
        feature = "candle-metal"
    ))]
    fn assert_storage_device(
        storage: &crate::backend::BackendStorage,
        device: &crate::Device,
    ) -> Result<(), crate::BackendError> {
        let expected = super::to_candle_device(device)?;
        match storage {
            crate::backend::BackendStorage::Candle(storage) => {
                assert!(storage.inner().as_tensor().device().same_device(&expected));
            }
            #[cfg(feature = "native")]
            crate::backend::BackendStorage::Native(_) => panic!("expected Candle storage"),
        }
        Ok(())
    }

    #[cfg(feature = "candle-cpu")]
    #[test]
    fn candle_storage_uses_requested_cpu_device() -> Result<(), Box<dyn std::error::Error>> {
        let device = crate::device::candle::cpu()?;
        let source = crate::backend::BackendStorage::from_data(vec![1.0f32, 2.0], &device)?;
        assert_storage_device(&source, &device)?;
        assert_storage_device(&source.try_clone()?, &device)?;
        Ok(())
    }

    #[cfg(any(feature = "candle-cuda", feature = "candle-metal"))]
    fn check_gpu_device_handles(
        first: &crate::Device,
        second: &crate::Device,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let handle = super::to_candle_device(first)?;
        assert!(handle.same_device(&super::to_candle_device(first)?));
        assert!(handle.same_device(&super::to_candle_device(&first.clone())?));
        assert!(!handle.same_device(&super::to_candle_device(second)?));
        let source = crate::backend::BackendStorage::from_data(vec![1.0f32, 2.0], first)?;
        let other = crate::backend::BackendStorage::from_data(vec![3.0f32, 4.0], second)?;
        assert_storage_device(&source, first)?;
        assert_storage_device(&source.try_clone()?, first)?;
        assert_storage_device(&other, second)?;
        assert!(!source.same_device(&other));
        Ok(())
    }

    #[cfg(feature = "candle-cuda")]
    #[test]
    #[ignore = "requires a CUDA device"]
    fn candle_cuda_device_conversions_reuse_the_same_handle()
    -> Result<(), Box<dyn std::error::Error>> {
        let first = crate::device::candle::cuda(0)?;
        let second = crate::device::candle::cuda(0)?;
        check_gpu_device_handles(&first, &second)?;
        let source = crate::backend::BackendStorage::from_data(vec![1.0f32, 2.0], &first)?;
        let moved = source.to_device(&second)?;
        assert_storage_device(&moved, &second)?;
        assert_eq!(moved.to_vec::<f32>()?, vec![1.0, 2.0]);
        Ok(())
    }

    #[cfg(feature = "candle-metal")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn candle_metal_device_conversions_reuse_the_same_handle()
    -> Result<(), Box<dyn std::error::Error>> {
        let first = crate::device::candle::metal(0)?;
        let second = crate::device::candle::metal(0)?;
        check_gpu_device_handles(&first, &second)
    }

    #[test]
    fn candle_clone_has_independent_buffer() {
        let device = nove_candle::CandleDevice::Cpu;
        let raw = nove_candle::CandleTensor::from_slice(&[1.0f32, 2.0], &[2], &device).unwrap();
        let source = CandleStorage::new(
            nove_candle::CandleStorage::from_candle_tensor(raw.clone(), &device).unwrap(),
        );
        let copy = source.try_clone().unwrap();
        raw.zero_set().unwrap();
        assert_eq!(
            source.inner().as_tensor().to_vec1::<f32>().unwrap(),
            [0.0, 0.0]
        );
        assert_eq!(
            copy.inner().as_tensor().to_vec1::<f32>().unwrap(),
            [1.0, 2.0]
        );
        assert!(copy.same_device(&source));
    }

    #[test]
    fn clone_preserves_noncontiguous_values() {
        let device = nove_candle::CandleDevice::Cpu;
        let raw = nove_candle::CandleTensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device)
            .unwrap()
            .transpose(0, 1)
            .unwrap();
        assert!(!raw.is_contiguous());
        let source = CandleStorage::new(
            nove_candle::CandleStorage::from_candle_tensor(raw.clone(), &device).unwrap(),
        );
        let copy = source.try_clone().unwrap();
        raw.zero_set().unwrap();
        assert_eq!(
            copy.inner().as_tensor().to_vec2::<f32>().unwrap(),
            vec![vec![1.0, 3.0], vec![2.0, 4.0]]
        );
        assert_eq!(copy.shape().dims(), &[2, 2]);
        assert!(copy.same_device(&source));
    }

    #[test]
    fn clone_preserves_half_dtype_and_independence() {
        let device = nove_candle::CandleDevice::Cpu;
        for dtype in [DType::F16, DType::BF16] {
            let raw = nove_candle::CandleTensor::from_slice(&[1.0f32, 2.0], &[2], &device)
                .unwrap()
                .to_dtype(to_candle_dtype(dtype))
                .unwrap();
            let source = CandleStorage::new(
                nove_candle::CandleStorage::from_candle_tensor(raw.clone(), &device).unwrap(),
            );
            let copy = source.try_clone().unwrap();
            raw.zero_set().unwrap();
            assert_eq!(copy.dtype(), dtype);
            assert_eq!(copy.shape().dims(), &[2]);
            assert!(copy.same_device(&source));
            assert_eq!(
                copy.inner()
                    .as_tensor()
                    .to_dtype(nove_candle::CandleDType::F32)
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
                [1.0, 2.0]
            );
        }
    }
}
