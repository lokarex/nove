//! Candle-backed implementation crate for Nove.
//!
//! This crate keeps direct `candle_core` integration behind a narrow adapter
//! boundary. Nove autograd is not represented with Candle variables here;
//! tensors crossing this boundary are treated as detached Candle tensors.
//!
//! # Notes
//! `nove_backend` wraps this crate so high-level tensor APIs do not depend on
//! Candle types except at explicit Candle interoperability points.
//!
//! # Examples
//! ```
//! use nove_candle::{CandleDType, CandleDevice, CandleStorage};
//!
//! let device = CandleDevice::Cpu;
//! let storage = CandleStorage::ones(&[2, 2], CandleDType::F32, &device, false).unwrap();
//!
//! assert_eq!(storage.shape(), vec![2, 2]);
//! assert_eq!(storage.dtype(), CandleDType::F32);
//! assert_eq!(
//!     storage.to_candle_tensor().unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap(),
//!     vec![1.0, 1.0, 1.0, 1.0]
//! );
//! ```

use std::{collections::HashMap, fmt::Display};

pub use candle_core::{
    DType as CandleDType, Device as CandleDevice, FloatDType as CandleFloatDType,
    NdArray as CandleNdArray, Tensor as CandleTensor, WithDType as CandleWithDType,
};

/// Error returned by the Candle backend implementation crate.
///
/// # Notes
/// This type wraps Candle errors as strings so the backend facade can map them
/// into Nove's public backend error categories.
#[derive(Debug, thiserror::Error)]
#[error("candle backend error: {message}")]
pub struct CandleBackendError {
    message: String,
}

impl From<candle_core::Error> for CandleBackendError {
    fn from(error: candle_core::Error) -> Self {
        Self {
            message: error.to_string(),
        }
    }
}

/// Result type returned by the Candle backend implementation crate.
pub type CandleResult<T> = std::result::Result<T, CandleBackendError>;

/// Device kinds understood by the Candle backend implementation crate.
///
/// # Examples
/// ```
/// use nove_candle::CandleDeviceKind;
///
/// assert_eq!(CandleDeviceKind::Cpu, CandleDeviceKind::Cpu);
/// assert_ne!(CandleDeviceKind::Cpu, CandleDeviceKind::Cuda);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CandleDeviceKind {
    /// Candle CPU device.
    Cpu,
    /// Candle CUDA device.
    Cuda,
    /// Candle Metal device.
    Metal,
}

/// Candle-backed tensor storage.
///
/// # Notes
/// Storage values are detached Candle tensors. The `requires_grad` arguments on
/// constructors are accepted for adapter compatibility, but Nove's graph
/// metadata owns gradient tracking.
#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct CandleStorage(candle_core::Tensor);

impl CandleStorage {
    pub fn from_data<A>(data: A, device: &CandleDevice, _requires_grad: bool) -> CandleResult<Self>
    where
        A: CandleNdArray,
    {
        Ok(Self(candle_core::Tensor::new(data, device)?))
    }

    pub fn from_vec<D>(
        data: Vec<D>,
        shape: &[usize],
        device: &CandleDevice,
        _requires_grad: bool,
    ) -> CandleResult<Self>
    where
        D: CandleWithDType,
    {
        Ok(Self(candle_core::Tensor::from_vec(data, shape, device)?))
    }

    pub fn from_slice<D>(
        data: &[D],
        shape: &[usize],
        device: &CandleDevice,
        _requires_grad: bool,
    ) -> CandleResult<Self>
    where
        D: CandleWithDType,
    {
        Ok(Self(candle_core::Tensor::from_slice(data, shape, device)?))
    }

    pub fn rand<T>(
        low: T,
        high: T,
        shape: &[usize],
        device: &CandleDevice,
        _requires_grad: bool,
    ) -> CandleResult<Self>
    where
        T: CandleFloatDType,
    {
        Ok(Self(candle_core::Tensor::rand(low, high, shape, device)?))
    }

    pub fn randn<T>(
        mean: T,
        std: T,
        shape: &[usize],
        device: &CandleDevice,
        _requires_grad: bool,
    ) -> CandleResult<Self>
    where
        T: CandleFloatDType,
    {
        Ok(Self(candle_core::Tensor::randn(mean, std, shape, device)?))
    }

    pub fn zeros(
        shape: &[usize],
        dtype: CandleDType,
        device: &CandleDevice,
        _requires_grad: bool,
    ) -> CandleResult<Self> {
        Ok(Self(candle_core::Tensor::zeros(shape, dtype, device)?))
    }

    pub fn ones(
        shape: &[usize],
        dtype: CandleDType,
        device: &CandleDevice,
        _requires_grad: bool,
    ) -> CandleResult<Self> {
        Ok(Self(candle_core::Tensor::ones(shape, dtype, device)?))
    }

    pub fn from_candle_tensor(
        tensor: candle_core::Tensor,
        device: &CandleDevice,
        _requires_grad: bool,
    ) -> CandleResult<Self> {
        let tensor = tensor.copy()?.to_device(device)?.detach();
        Ok(Self(tensor))
    }

    pub fn to_candle_tensor(&self) -> CandleResult<candle_core::Tensor> {
        Ok(self.as_tensor().copy()?.detach())
    }

    pub fn as_tensor(&self) -> &candle_core::Tensor {
        &self.0
    }

    pub fn copy_detached(&self) -> CandleResult<Self> {
        Ok(Self(self.as_tensor().copy()?.detach()))
    }

    pub fn detach(&self) -> CandleResult<Self> {
        Ok(Self(self.as_tensor().detach()))
    }

    pub fn with_requires_grad(&self, _requires_grad: bool) -> CandleResult<Self> {
        Ok(Self(self.as_tensor().copy()?.detach()))
    }

    pub fn assign_from(&mut self, other: &Self, _requires_grad: bool) -> CandleResult<()> {
        *self = Self(other.as_tensor().copy()?.detach());
        Ok(())
    }

    pub fn dtype(&self) -> CandleDType {
        self.as_tensor().dtype()
    }

    pub fn shape(&self) -> Vec<usize> {
        self.as_tensor().shape().dims().to_vec()
    }

    pub fn zero_set(&mut self) -> CandleResult<()> {
        self.0.zero_set()?;
        Ok(())
    }

    pub fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_tensor().fmt(f)
    }

    pub fn stack(tensors: &[Self], dim: usize) -> CandleResult<Self> {
        let tensors = tensors
            .iter()
            .map(|storage| storage.as_tensor().clone())
            .collect::<Vec<_>>();
        Ok(Self(candle_core::Tensor::stack(&tensors, dim)?))
    }

    pub fn cat(tensors: &[Self], dim: usize) -> CandleResult<Self> {
        let tensors = tensors
            .iter()
            .map(|storage| storage.as_tensor().clone())
            .collect::<Vec<_>>();
        Ok(Self(candle_core::Tensor::cat(&tensors, dim)?))
    }
}

macro_rules! candle_unary_methods {
    ($($method:ident),+ $(,)?) => {
        impl CandleStorage {
            $(
                pub fn $method(&self) -> CandleResult<Self> {
                    Ok(Self(self.as_tensor().$method()?))
                }
            )+
        }
    };
}

macro_rules! candle_binary_methods {
    ($($method:ident),+ $(,)?) => {
        impl CandleStorage {
            $(
                pub fn $method(&self, rhs: &Self) -> CandleResult<Self> {
                    Ok(Self(self.as_tensor().$method(rhs.as_tensor())?))
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
    contiguous,
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
    broadcast_pow,
    embedding,
);

impl CandleStorage {
    pub fn to_device(&self, device: &CandleDevice) -> CandleResult<Self> {
        Ok(Self(self.as_tensor().to_device(device)?))
    }

    pub fn to_dtype(&self, dtype: CandleDType) -> CandleResult<Self> {
        Ok(Self(self.as_tensor().to_dtype(dtype)?))
    }

    pub fn reshape(&self, shape: &[usize]) -> CandleResult<Self> {
        Ok(Self(self.as_tensor().reshape(shape)?))
    }

    pub fn broadcast_as(&self, shape: &[usize]) -> CandleResult<Self> {
        Ok(Self(self.as_tensor().broadcast_as(shape)?))
    }

    /// Broadcast-aware matrix multiplication with contiguity guarantee.
    ///
    /// Candle\'s underlying matmul requires contiguous tensors. When
    /// a broadcasted view (e.g. a gradient expanded from a scalar via
    /// broadcast_as) flows through matmul backward, its zero-stride
    /// layout causes a non-contiguous error. We make both operands
    /// contiguous before calling the candle routine.
    pub fn broadcast_matmul(&self, rhs: &Self) -> CandleResult<Self> {
        Ok(Self(
            self.as_tensor()
                .contiguous()?
                .broadcast_matmul(&rhs.as_tensor().contiguous()?)?,
        ))
    }

    pub fn flatten_from(&self, dim: usize) -> CandleResult<Self> {
        Ok(Self(self.as_tensor().flatten_from(dim)?))
    }

    pub fn flatten_to(&self, dim: usize) -> CandleResult<Self> {
        Ok(Self(self.as_tensor().flatten_to(dim)?))
    }

    pub fn flatten(&self, start: usize, end: usize) -> CandleResult<Self> {
        Ok(Self(self.as_tensor().flatten(start, end)?))
    }

    pub fn squeeze(&self, dim: usize) -> CandleResult<Self> {
        Ok(Self(self.as_tensor().squeeze(dim)?))
    }

    pub fn unsqueeze(&self, dim: usize) -> CandleResult<Self> {
        Ok(Self(self.as_tensor().unsqueeze(dim)?))
    }

    pub fn transpose(&self, dim0: usize, dim1: usize) -> CandleResult<Self> {
        Ok(Self(self.as_tensor().transpose(dim0, dim1)?))
    }

    pub fn permute(&self, dims: &[usize]) -> CandleResult<Self> {
        Ok(Self(self.as_tensor().permute(dims)?))
    }

    pub fn narrow(&self, dim: usize, start: usize, length: usize) -> CandleResult<Self> {
        Ok(Self(self.as_tensor().narrow(dim, start, length)?))
    }

    pub fn affine(&self, weight: f64, bias: f64) -> CandleResult<Self> {
        Ok(Self(self.as_tensor().affine(weight, bias)?))
    }

    pub fn clamp(&self, min: f64, max: f64) -> CandleResult<Self> {
        Ok(Self(self.as_tensor().clamp(min, max)?))
    }

    pub fn powf(&self, exponent: f64) -> CandleResult<Self> {
        Ok(Self(self.as_tensor().powf(exponent)?))
    }

    pub fn sum(&self, dim: usize) -> CandleResult<Self> {
        Ok(Self(self.as_tensor().sum(dim)?))
    }

    pub fn sum_keepdim(&self, dim: usize) -> CandleResult<Self> {
        Ok(Self(self.as_tensor().sum_keepdim(dim)?))
    }

    pub fn max(&self, dim: usize) -> CandleResult<Self> {
        Ok(Self(self.as_tensor().max(dim)?))
    }

    pub fn max_keepdim(&self, dim: usize) -> CandleResult<Self> {
        Ok(Self(self.as_tensor().max_keepdim(dim)?))
    }

    pub fn min(&self, dim: usize) -> CandleResult<Self> {
        Ok(Self(self.as_tensor().min(dim)?))
    }

    pub fn min_keepdim(&self, dim: usize) -> CandleResult<Self> {
        Ok(Self(self.as_tensor().min_keepdim(dim)?))
    }

    pub fn mean(&self, dim: usize) -> CandleResult<Self> {
        Ok(Self(self.as_tensor().mean(dim)?))
    }

    pub fn mean_keepdim(&self, dim: usize) -> CandleResult<Self> {
        Ok(Self(self.as_tensor().mean_keepdim(dim)?))
    }

    pub fn var(&self, dim: usize) -> CandleResult<Self> {
        Ok(Self(self.as_tensor().var(dim)?))
    }

    pub fn var_keepdim(&self, dim: usize) -> CandleResult<Self> {
        Ok(Self(self.as_tensor().var_keepdim(dim)?))
    }

    pub fn argmax(&self, dim: usize) -> CandleResult<Self> {
        Ok(Self(self.as_tensor().argmax(dim)?))
    }

    pub fn argmax_keepdim(&self, dim: usize) -> CandleResult<Self> {
        Ok(Self(self.as_tensor().argmax_keepdim(dim)?))
    }

    pub fn argmin(&self, dim: usize) -> CandleResult<Self> {
        Ok(Self(self.as_tensor().argmin(dim)?))
    }

    pub fn argmin_keepdim(&self, dim: usize) -> CandleResult<Self> {
        Ok(Self(self.as_tensor().argmin_keepdim(dim)?))
    }

    pub fn conv1d(
        &self,
        kernel: &Self,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> CandleResult<Self> {
        Ok(Self(self.as_tensor().conv1d(
            kernel.as_tensor(),
            padding,
            stride,
            dilation,
            groups,
        )?))
    }

    pub fn conv2d(
        &self,
        kernel: &Self,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> CandleResult<Self> {
        Ok(Self(self.as_tensor().conv2d(
            kernel.as_tensor(),
            padding,
            stride,
            dilation,
            groups,
        )?))
    }

    pub fn max_pool2d_with_stride(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> CandleResult<Self> {
        Ok(Self(
            self.as_tensor()
                .max_pool2d_with_stride(kernel_size, stride)?,
        ))
    }

    pub fn avg_pool2d_with_stride(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> CandleResult<Self> {
        Ok(Self(
            self.as_tensor()
                .avg_pool2d_with_stride(kernel_size, stride)?,
        ))
    }

    pub fn where_cond(&self, true_value: &Self, false_value: &Self) -> CandleResult<Self> {
        Ok(Self(self.as_tensor().where_cond(
            true_value.as_tensor(),
            false_value.as_tensor(),
        )?))
    }

    pub fn gather(&self, indexes: &Self, dim: usize) -> CandleResult<Self> {
        Ok(Self(self.as_tensor().gather(indexes.as_tensor(), dim)?))
    }

    pub fn index_select(&self, indexes: &Self, dim: usize) -> CandleResult<Self> {
        // Candle CUDA does not support index_select with empty index tensors
        // (returns CUDA_ERROR_INVALID_VALUE). Handle empty case by constructing
        // a correctly-shaped empty tensor directly.
        let idx_elem_count: usize = indexes.as_tensor().dims().iter().product();
        if idx_elem_count == 0 {
            let mut output_dims = self.as_tensor().dims().to_vec();
            output_dims[dim] = 0;
            let empty = candle_core::Tensor::zeros(
                output_dims.as_slice(),
                self.as_tensor().dtype(),
                self.as_tensor().device(),
            )?;
            return Ok(Self(empty));
        }

        // Validate indices are within bounds to prevent device-side asserts on CUDA.
        let dim_size = self.as_tensor().dims()[dim];
        let idx_vec = indexes
            .as_tensor()
            .flatten_all()
            .map_err(|e| CandleBackendError {
                message: e.to_string(),
            })?
            .to_vec1::<i64>()
            .map_err(|e| CandleBackendError {
                message: e.to_string(),
            })?;
        for (i, &idx) in idx_vec.iter().enumerate() {
            if idx < 0 || idx as usize >= dim_size {
                return Err(CandleBackendError {
                    message: format!(
                        "index_select: index {} at position {} is out of range for dimension {} with size {}",
                        idx, i, dim, dim_size
                    ),
                });
            }
        }

        Ok(Self(
            self.as_tensor().index_select(indexes.as_tensor(), dim)?,
        ))
    }
}

/// Validates that Candle can construct a device of the requested kind.
///
/// # Arguments
/// * `kind` - Candle device kind.
/// * `index` - CUDA or Metal device index. CPU ignores this value.
///
/// # Returns
/// * `Ok(())` - Candle accepted the requested device.
/// * `Err(CandleBackendError)` - Candle could not construct the device.
///
/// # Examples
/// ```
/// use nove_candle::{CandleDeviceKind, validate_device};
///
/// validate_device(CandleDeviceKind::Cpu, 0).unwrap();
/// ```
pub fn validate_device(kind: CandleDeviceKind, index: usize) -> CandleResult<()> {
    match kind {
        CandleDeviceKind::Cpu => Ok(()),
        CandleDeviceKind::Cuda => {
            let _ = candle_core::Device::new_cuda(index)?;
            Ok(())
        }
        CandleDeviceKind::Metal => {
            let _ = candle_core::Device::new_metal(index)?;
            Ok(())
        }
    }
}

/// Synchronizes a Candle device.
///
/// # Arguments
/// * `device` - Candle device to synchronize.
///
/// # Returns
/// * `Ok(())` - Synchronization completed successfully.
/// * `Err(CandleBackendError)` - Candle returned an error.
///
/// # Examples
/// ```
/// use nove_candle::{CandleDevice, synchronize_device};
///
/// synchronize_device(&CandleDevice::Cpu).unwrap();
/// ```
pub fn synchronize_device(device: &CandleDevice) -> CandleResult<()> {
    device.synchronize()?;
    Ok(())
}

/// Creates a Candle device from a Nove-facing Candle device kind.
///
/// # Arguments
/// * `kind` - Candle device kind.
/// * `index` - CUDA or Metal device index. CPU ignores this value.
///
/// # Returns
/// * `Ok(CandleDevice)` - The Candle device.
/// * `Err(CandleBackendError)` - Candle could not construct the device.
///
/// # Examples
/// ```
/// use nove_candle::{CandleDevice, CandleDeviceKind, to_candle_device};
///
/// let device = to_candle_device(CandleDeviceKind::Cpu, 0).unwrap();
/// assert!(matches!(device, CandleDevice::Cpu));
/// ```
pub fn to_candle_device(kind: CandleDeviceKind, index: usize) -> CandleResult<CandleDevice> {
    match kind {
        CandleDeviceKind::Cpu => Ok(candle_core::Device::Cpu),
        CandleDeviceKind::Cuda => Ok(candle_core::Device::new_cuda(index)?),
        CandleDeviceKind::Metal => Ok(candle_core::Device::new_metal(index)?),
    }
}

/// Saves Candle-backed storage values to a safetensors file.
///
/// # Arguments
/// * `file_path` - Destination file path.
/// * `tensors` - Named Candle-backed storage values to save.
///
/// # Returns
/// * `Ok(())` - The file was written successfully.
/// * `Err(CandleBackendError)` - Candle failed to serialize one or more tensors.
///
/// # Examples
/// ```no_run
/// use std::collections::HashMap;
/// use nove_candle::save_safetensors;
///
/// save_safetensors("model.safetensors", HashMap::new()).unwrap();
/// ```
pub fn save_safetensors(
    file_path: &str,
    tensors: HashMap<String, CandleStorage>,
) -> CandleResult<()> {
    let tensors = tensors
        .into_iter()
        .map(|(name, storage)| Ok((name, storage.to_candle_tensor()?)))
        .collect::<CandleResult<HashMap<_, _>>>()?;
    candle_core::safetensors::save(&tensors, file_path)?;
    Ok(())
}

/// Loads Candle-backed storage values from a safetensors file.
///
/// # Arguments
/// * `file_path` - Source file path.
/// * `device` - Candle device used to load tensor values.
///
/// # Returns
/// * `Ok(HashMap<String, CandleStorage>)` - Named Candle-backed storage values.
/// * `Err(CandleBackendError)` - Candle failed to load the file.
///
/// # Examples
/// ```no_run
/// use nove_candle::{CandleDevice, load_safetensors};
///
/// let tensors = load_safetensors("model.safetensors", &CandleDevice::Cpu).unwrap();
/// assert!(tensors.is_empty() || !tensors.is_empty());
/// ```
pub fn load_safetensors(
    file_path: &str,
    device: &CandleDevice,
) -> CandleResult<HashMap<String, CandleStorage>> {
    let tensors = candle_core::safetensors::load(file_path, device)?;
    tensors
        .into_iter()
        .map(|(name, tensor)| {
            Ok((
                name,
                CandleStorage::from_candle_tensor(tensor, device, false)?,
            ))
        })
        .collect::<CandleResult<HashMap<_, _>>>()
}
