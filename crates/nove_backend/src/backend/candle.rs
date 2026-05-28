use super::{BackendError, BackendKind, BackendStorage, TensorBuffer, TensorPayload};
use crate::{DType, Device, Shape, device::DeviceKind};
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
    dtype.into()
}

fn from_candle_dtype(dtype: nove_candle::CandleDType) -> DType {
    dtype.into()
}

fn to_candle_device_kind(kind: DeviceKind) -> nove_candle::CandleDeviceKind {
    match kind {
        DeviceKind::Cpu => nove_candle::CandleDeviceKind::Cpu,
        DeviceKind::Cuda => nove_candle::CandleDeviceKind::Cuda,
        DeviceKind::Metal => nove_candle::CandleDeviceKind::Metal,
    }
}

pub(crate) fn to_candle_device(device: &Device) -> Result<nove_candle::CandleDevice, BackendError> {
    nove_candle::to_candle_device(to_candle_device_kind(device.kind()), device.index())
        .map_err(backend_error)
}

/// Candle-backed tensor storage adapter.
#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct CandleStorage(nove_candle::CandleStorage);

impl CandleStorage {
    fn new(storage: nove_candle::CandleStorage) -> Self {
        Self(storage)
    }

    fn inner(&self) -> &nove_candle::CandleStorage {
        &self.0
    }

    fn into_inner(self) -> nove_candle::CandleStorage {
        self.0
    }

    pub(crate) fn from_payload(
        payload: &TensorPayload,
        device: &Device,
        requires_grad: bool,
    ) -> Result<Self, BackendError> {
        match payload.buffer() {
            TensorBuffer::U8(data) => {
                Self::from_slice(data, payload.shape(), device, requires_grad)
            }
            TensorBuffer::U32(data) => {
                Self::from_slice(data, payload.shape(), device, requires_grad)
            }
            TensorBuffer::I64(data) => {
                Self::from_slice(data, payload.shape(), device, requires_grad)
            }
            TensorBuffer::F32(data) => {
                Self::from_slice(data, payload.shape(), device, requires_grad)
            }
            TensorBuffer::F64(data) => {
                Self::from_slice(data, payload.shape(), device, requires_grad)
            }
        }
    }

    pub(crate) fn rand(
        dtype: DType,
        low: f64,
        high: f64,
        shape: &Shape,
        device: &Device,
        requires_grad: bool,
    ) -> Result<Self, BackendError> {
        match dtype {
            DType::F32 => Self::rand_typed(low as f32, high as f32, shape, device, requires_grad),
            DType::F64 => Self::rand_typed(low, high, shape, device, requires_grad),
            dtype => Err(BackendError::UnsupportedDType {
                backend: BackendKind::Candle,
                dtype,
            }),
        }
    }

    pub(crate) fn randn(
        dtype: DType,
        mean: f64,
        std: f64,
        shape: &Shape,
        device: &Device,
        requires_grad: bool,
    ) -> Result<Self, BackendError> {
        match dtype {
            DType::F32 => Self::randn_typed(mean as f32, std as f32, shape, device, requires_grad),
            DType::F64 => Self::randn_typed(mean, std, shape, device, requires_grad),
            dtype => Err(BackendError::UnsupportedDType {
                backend: BackendKind::Candle,
                dtype,
            }),
        }
    }

    fn from_slice<D>(
        data: &[D],
        shape: &Shape,
        device: &Device,
        requires_grad: bool,
    ) -> Result<Self, BackendError>
    where
        D: nove_candle::CandleWithDType,
    {
        let device = to_candle_device(device)?;
        nove_candle::CandleStorage::from_slice(data, shape.dims(), &device, requires_grad)
            .map(Self::new)
            .map_err(backend_error)
    }

    fn rand_typed<T>(
        low: T,
        high: T,
        shape: &Shape,
        device: &Device,
        requires_grad: bool,
    ) -> Result<Self, BackendError>
    where
        T: nove_candle::CandleFloatDType,
    {
        let device = to_candle_device(device)?;
        nove_candle::CandleStorage::rand(low, high, shape.dims(), &device, requires_grad)
            .map(Self::new)
            .map_err(backend_error)
    }

    fn randn_typed<T>(
        mean: T,
        std: T,
        shape: &Shape,
        device: &Device,
        requires_grad: bool,
    ) -> Result<Self, BackendError>
    where
        T: nove_candle::CandleFloatDType,
    {
        let device = to_candle_device(device)?;
        nove_candle::CandleStorage::randn(mean, std, shape.dims(), &device, requires_grad)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn zeros(
        shape: &Shape,
        dtype: DType,
        device: &Device,
        requires_grad: bool,
    ) -> Result<Self, BackendError> {
        let device = to_candle_device(device)?;
        nove_candle::CandleStorage::zeros(
            shape.dims(),
            to_candle_dtype(dtype),
            &device,
            requires_grad,
        )
        .map(Self::new)
        .map_err(backend_error)
    }

    pub(crate) fn ones(
        shape: &Shape,
        dtype: DType,
        device: &Device,
        requires_grad: bool,
    ) -> Result<Self, BackendError> {
        let device = to_candle_device(device)?;
        nove_candle::CandleStorage::ones(
            shape.dims(),
            to_candle_dtype(dtype),
            &device,
            requires_grad,
        )
        .map(Self::new)
        .map_err(backend_error)
    }

    pub(crate) fn from_candle_tensor(
        tensor: nove_candle::CandleTensor,
        device: &Device,
        requires_grad: bool,
    ) -> Result<Self, BackendError> {
        let device = to_candle_device(device)?;
        nove_candle::CandleStorage::from_candle_tensor(tensor, &device, requires_grad)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn to_candle_tensor(&self) -> Result<nove_candle::CandleTensor, BackendError> {
        self.inner().to_candle_tensor().map_err(backend_error)
    }

    pub(crate) fn copy_detached(&self) -> Result<Self, BackendError> {
        self.inner()
            .copy_detached()
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn detach(&self) -> Result<Self, BackendError> {
        self.inner().detach().map(Self::new).map_err(backend_error)
    }

    pub(crate) fn with_requires_grad(&self, requires_grad: bool) -> Result<Self, BackendError> {
        self.inner()
            .with_requires_grad(requires_grad)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn assign_from(
        &mut self,
        other: &Self,
        requires_grad: bool,
    ) -> Result<(), BackendError> {
        self.0
            .assign_from(other.inner(), requires_grad)
            .map_err(backend_error)
    }

    pub(crate) fn dtype(&self) -> DType {
        from_candle_dtype(self.inner().dtype())
    }

    pub(crate) fn to_tensor_buffer(&self) -> Result<TensorBuffer, BackendError> {
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

    pub(crate) fn shape(&self) -> Shape {
        Shape::from(self.inner().shape())
    }

    pub(crate) fn zero_set(&mut self) -> Result<(), BackendError> {
        self.0.zero_set().map_err(backend_error)
    }

    pub(crate) fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner().fmt(f)
    }

    pub(crate) fn stack(tensors: &[Self], dim: usize) -> Result<Self, BackendError> {
        let tensors = tensors
            .iter()
            .map(|storage| storage.inner().clone())
            .collect::<Vec<_>>();
        nove_candle::CandleStorage::stack(&tensors, dim)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn cat(tensors: &[Self], dim: usize) -> Result<Self, BackendError> {
        let tensors = tensors
            .iter()
            .map(|storage| storage.inner().clone())
            .collect::<Vec<_>>();
        nove_candle::CandleStorage::cat(&tensors, dim)
            .map(Self::new)
            .map_err(backend_error)
    }
}

macro_rules! candle_unary_methods {
    ($($method:ident),+ $(,)?) => {
        impl CandleStorage {
            $(
                pub(crate) fn $method(&self) -> Result<Self, BackendError> {
                    self.inner().$method().map(Self::new).map_err(backend_error)
                }
            )+
        }
    };
}

macro_rules! candle_binary_methods {
    ($($method:ident),+ $(,)?) => {
        impl CandleStorage {
            $(
                pub(crate) fn $method(&self, rhs: &Self) -> Result<Self, BackendError> {
                    self.inner()
                        .$method(rhs.inner())
                        .map(Self::new)
                        .map_err(backend_error)
                }
            )+
        }
    };
}

candle_unary_methods!(
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

candle_binary_methods!(
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

impl CandleStorage {
    pub(crate) fn to_device(&self, device: &Device) -> Result<Self, BackendError> {
        let device = to_candle_device(device)?;
        self.inner()
            .to_device(&device)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn to_dtype(&self, dtype: DType) -> Result<Self, BackendError> {
        self.inner()
            .to_dtype(to_candle_dtype(dtype))
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn reshape(&self, shape: &Shape) -> Result<Self, BackendError> {
        self.inner()
            .reshape(shape.dims())
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn broadcast_as(&self, shape: &Shape) -> Result<Self, BackendError> {
        self.inner()
            .broadcast_as(shape.dims())
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn flatten_from(&self, dim: usize) -> Result<Self, BackendError> {
        self.inner()
            .flatten_from(dim)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn flatten_to(&self, dim: usize) -> Result<Self, BackendError> {
        self.inner()
            .flatten_to(dim)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn flatten(&self, start: usize, end: usize) -> Result<Self, BackendError> {
        self.inner()
            .flatten(start, end)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn squeeze(&self, dim: usize) -> Result<Self, BackendError> {
        self.inner()
            .squeeze(dim)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn unsqueeze(&self, dim: usize) -> Result<Self, BackendError> {
        self.inner()
            .unsqueeze(dim)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn transpose(&self, dim0: usize, dim1: usize) -> Result<Self, BackendError> {
        self.inner()
            .transpose(dim0, dim1)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn permute(&self, dims: &[usize]) -> Result<Self, BackendError> {
        self.inner()
            .permute(dims)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn narrow(
        &self,
        dim: usize,
        start: usize,
        length: usize,
    ) -> Result<Self, BackendError> {
        self.inner()
            .narrow(dim, start, length)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn affine(&self, weight: f64, bias: f64) -> Result<Self, BackendError> {
        self.inner()
            .affine(weight, bias)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn clamp(&self, min: f64, max: f64) -> Result<Self, BackendError> {
        self.inner()
            .clamp(min, max)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn powf(&self, exponent: f64) -> Result<Self, BackendError> {
        self.inner()
            .powf(exponent)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn sum(&self, dim: usize) -> Result<Self, BackendError> {
        self.inner().sum(dim).map(Self::new).map_err(backend_error)
    }

    pub(crate) fn sum_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        self.inner()
            .sum_keepdim(dim)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn max(&self, dim: usize) -> Result<Self, BackendError> {
        self.inner().max(dim).map(Self::new).map_err(backend_error)
    }

    pub(crate) fn max_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        self.inner()
            .max_keepdim(dim)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn min(&self, dim: usize) -> Result<Self, BackendError> {
        self.inner().min(dim).map(Self::new).map_err(backend_error)
    }

    pub(crate) fn min_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        self.inner()
            .min_keepdim(dim)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn mean(&self, dim: usize) -> Result<Self, BackendError> {
        self.inner().mean(dim).map(Self::new).map_err(backend_error)
    }

    pub(crate) fn mean_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        self.inner()
            .mean_keepdim(dim)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn var(&self, dim: usize) -> Result<Self, BackendError> {
        self.inner().var(dim).map(Self::new).map_err(backend_error)
    }

    pub(crate) fn var_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        self.inner()
            .var_keepdim(dim)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn argmax(&self, dim: usize) -> Result<Self, BackendError> {
        self.inner()
            .argmax(dim)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn argmax_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        self.inner()
            .argmax_keepdim(dim)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn argmin(&self, dim: usize) -> Result<Self, BackendError> {
        self.inner()
            .argmin(dim)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn argmin_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        self.inner()
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
        self.inner()
            .conv1d(kernel.inner(), padding, stride, dilation, groups)
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
        self.inner()
            .conv2d(kernel.inner(), padding, stride, dilation, groups)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn max_pool2d_with_stride(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self, BackendError> {
        self.inner()
            .max_pool2d_with_stride(kernel_size, stride)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn avg_pool2d_with_stride(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self, BackendError> {
        self.inner()
            .avg_pool2d_with_stride(kernel_size, stride)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn where_cond(
        &self,
        true_value: &Self,
        false_value: &Self,
    ) -> Result<Self, BackendError> {
        self.inner()
            .where_cond(true_value.inner(), false_value.inner())
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn gather(&self, indexes: &Self, dim: usize) -> Result<Self, BackendError> {
        self.inner()
            .gather(indexes.inner(), dim)
            .map(Self::new)
            .map_err(backend_error)
    }

    pub(crate) fn index_select(&self, indexes: &Self, dim: usize) -> Result<Self, BackendError> {
        self.inner()
            .index_select(indexes.inner(), dim)
            .map(Self::new)
            .map_err(backend_error)
    }
}

impl From<DType> for nove_candle::CandleDType {
    fn from(dtype: DType) -> Self {
        match dtype {
            DType::U8 => Self::U8,
            DType::U32 => Self::U32,
            DType::I64 => Self::I64,
            DType::BF16 => Self::BF16,
            DType::F16 => Self::F16,
            DType::F32 => Self::F32,
            DType::F64 => Self::F64,
        }
    }
}

impl From<nove_candle::CandleDType> for DType {
    fn from(dtype: nove_candle::CandleDType) -> Self {
        match dtype {
            nove_candle::CandleDType::U8 => Self::U8,
            nove_candle::CandleDType::U32 => Self::U32,
            nove_candle::CandleDType::I64 => Self::I64,
            nove_candle::CandleDType::BF16 => Self::BF16,
            nove_candle::CandleDType::F16 => Self::F16,
            nove_candle::CandleDType::F32 => Self::F32,
            nove_candle::CandleDType::F64 => Self::F64,
        }
    }
}

pub(crate) fn validate_device(kind: DeviceKind, index: usize) -> Result<(), BackendError> {
    nove_candle::validate_device(to_candle_device_kind(kind), index).map_err(backend_error)
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
            #[cfg(feature = "native-cpu")]
            BackendStorage::Native(storage) => {
                let payload = TensorPayload::new(storage.to_tensor_buffer()?, storage.shape())?;
                let device = Device::new(BackendKind::Candle, DeviceKind::Cpu, 0);
                let storage = CandleStorage::from_payload(&payload, &device, false)?;
                Ok((name, storage.into_inner()))
            }
            BackendStorage::Unavailable => Err(BackendError::NoBackendEnabled),
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

/// Creates backend storage from a detached Candle tensor.
///
/// # Notes
/// This is the lower-level Candle interoperability boundary used by
/// `nove_tensor`. The returned storage participates in Nove graph metadata, not
/// Candle autograd state.
///
/// # Arguments
/// * `tensor` - Candle tensor to copy into backend storage.
/// * `device` - Target Nove device descriptor.
/// * `requires_grad` - Whether Nove should mark the created tensor as requiring gradients.
///
/// # Returns
/// * `Ok(BackendStorage)` - Candle-backed storage wrapping a detached Candle tensor.
/// * `Err(BackendError)` - The error returned by Candle conversion.
///
/// # Examples
/// ```
/// use nove_backend::{backend::candle, device};
/// use nove_candle::CandleTensor;
///
/// let device = device::candle::cpu().unwrap();
/// let candle_device = device.to_candle_device().unwrap();
/// let tensor = CandleTensor::from_slice(&[1.0f32, 2.0], &[2], &candle_device).unwrap();
///
/// let storage = candle::storage_from_candle_tensor(tensor, &device, false).unwrap();
/// let roundtrip = candle::storage_to_candle_tensor(&storage).unwrap();
///
/// assert_eq!(roundtrip.dims(), &[2]);
/// ```
pub fn storage_from_candle_tensor(
    tensor: nove_candle::CandleTensor,
    device: &Device,
    requires_grad: bool,
) -> Result<BackendStorage, BackendError> {
    Ok(BackendStorage::Candle(CandleStorage::from_candle_tensor(
        tensor,
        device,
        requires_grad,
    )?))
}

/// Converts backend storage into a detached Candle tensor.
///
/// # Notes
/// Native CPU storage is materialized through a backend-independent payload
/// before the Candle tensor is created. No Candle variable or gradient store is
/// produced.
///
/// # Arguments
/// * `storage` - Backend storage to convert.
///
/// # Returns
/// * `Ok(CandleTensor)` - A detached Candle tensor.
/// * `Err(BackendError)` - The error returned by conversion.
///
/// # Examples
/// ```
/// use nove_backend::{DType, Shape, backend::BackendStorage, device};
///
/// let device = device::candle::cpu().unwrap();
/// let storage = BackendStorage::ones(&Shape::from_dims(&[2]), DType::F32, &device, false).unwrap();
/// let candle_tensor = nove_backend::backend::candle::storage_to_candle_tensor(&storage).unwrap();
///
/// assert_eq!(candle_tensor.dims(), &[2]);
/// ```
pub fn storage_to_candle_tensor(
    storage: &BackendStorage,
) -> Result<nove_candle::CandleTensor, BackendError> {
    match storage {
        BackendStorage::Candle(storage) => storage.to_candle_tensor(),
        #[cfg(feature = "native-cpu")]
        BackendStorage::Native(storage) => {
            let payload = TensorPayload::new(storage.to_tensor_buffer()?, storage.shape())?;
            let device = Device::new(BackendKind::Candle, DeviceKind::Cpu, 0);
            CandleStorage::from_payload(&payload, &device, false)?.to_candle_tensor()
        }
        BackendStorage::Unavailable => Err(BackendError::NoBackendEnabled),
    }
}

impl Device {
    /// Converts this Nove device descriptor into a Candle device.
    ///
    /// # Notes
    /// This method is available only when the Candle backend feature is
    /// enabled. It is intended for explicit Candle interoperability.
    ///
    /// # Returns
    /// * `Ok(CandleDevice)` - The Candle device if the selected device is available.
    /// * `Err(DeviceError)` - The error when the Candle device cannot be created.
    ///
    /// # Examples
    /// ```
    /// use nove_backend::device;
    /// use nove_candle::CandleDevice;
    ///
    /// let device = device::candle::cpu().unwrap();
    /// let candle_device = device.to_candle_device().unwrap();
    ///
    /// assert!(matches!(candle_device, CandleDevice::Cpu));
    /// ```
    pub fn to_candle_device(&self) -> Result<nove_candle::CandleDevice, crate::DeviceError> {
        Ok(to_candle_device(self)?)
    }
}
