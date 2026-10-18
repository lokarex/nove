//! Native backend family storage and dispatch.

#[cfg(feature = "native-cpu")]
mod cpu;

use super::{BackendError, TensorBuffer, TensorPayload};
use crate::device::{BackendDeviceHandle, NativeDeviceHandle};
use crate::{DType, Device, Shape};
#[cfg(feature = "native-cpu")]
use std::fmt::Display;

/// Native backend storage selected by device capability.
#[derive(Clone, Debug)]
#[doc(hidden)]
pub enum NativeStorage {
    /// Storage for the native CPU implementation.
    #[cfg(feature = "native-cpu")]
    Cpu(nove_native::cpu::CpuStorage),
}

macro_rules! dispatch_native {
    ($input:ident, $storage:ident, $body:expr) => {
        match *$input {
            #[cfg(feature = "native-cpu")]
            Self::Cpu(ref $storage) => $body,
        }
    };
}

macro_rules! native_unary_methods {
    ($($method:ident),+ $(,)?) => {
        $(
            fn $method(&self) -> Result<Self, BackendError> {
                dispatch_native!(self, storage, storage.$method().map(Self::Cpu).map_err(cpu::backend_error))
            }
        )+
    };
}

macro_rules! native_binary_methods {
    ($($method:ident),+ $(,)?) => {
        $(
            fn $method(&self, rhs: &Self) -> Result<Self, BackendError> {
                dispatch_native!(self, lhs, dispatch_native!(rhs, rhs, lhs.$method(rhs).map(Self::Cpu).map_err(cpu::backend_error)))
            }
        )+
    };
}

fn native_handle(device: &Device) -> Result<&NativeDeviceHandle, BackendError> {
    match device.handle() {
        BackendDeviceHandle::Native(handle) => Ok(handle),
        #[cfg(feature = "candle")]
        BackendDeviceHandle::Candle(_) => Err(BackendError::UnsupportedBackend(device.backend())),
    }
}

// -- Backend trait implementation -------------------------------------------------

impl super::Backend for NativeStorage {
    native_unary_methods!(
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

    native_binary_methods!(
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

    fn try_clone(&self) -> Result<Self, BackendError> {
        Ok(self.clone())
    }

    fn same_device(&self, other: &Self) -> bool {
        dispatch_native!(self, _storage, dispatch_native!(other, _other, true))
    }

    fn from_payload(payload: TensorPayload, device: &Device) -> Result<Self, BackendError> {
        match *native_handle(device)? {
            #[cfg(feature = "native-cpu")]
            NativeDeviceHandle::Cpu => cpu::from_payload(payload).map(Self::Cpu),
        }
    }

    fn rand(
        dtype: DType,
        low: f64,
        high: f64,
        shape: &Shape,
        device: &Device,
    ) -> Result<Self, BackendError> {
        match *native_handle(device)? {
            #[cfg(feature = "native-cpu")]
            NativeDeviceHandle::Cpu => cpu::rand(dtype, low, high, shape).map(Self::Cpu),
        }
    }

    fn randn(
        dtype: DType,
        mean: f64,
        std: f64,
        shape: &Shape,
        device: &Device,
    ) -> Result<Self, BackendError> {
        match *native_handle(device)? {
            #[cfg(feature = "native-cpu")]
            NativeDeviceHandle::Cpu => cpu::randn(dtype, mean, std, shape).map(Self::Cpu),
        }
    }

    fn zeros(shape: &Shape, dtype: DType, device: &Device) -> Result<Self, BackendError> {
        match *native_handle(device)? {
            #[cfg(feature = "native-cpu")]
            NativeDeviceHandle::Cpu => cpu::zeros(shape, dtype).map(Self::Cpu),
        }
    }

    fn ones(shape: &Shape, dtype: DType, device: &Device) -> Result<Self, BackendError> {
        match *native_handle(device)? {
            #[cfg(feature = "native-cpu")]
            NativeDeviceHandle::Cpu => cpu::ones(shape, dtype).map(Self::Cpu),
        }
    }

    fn dtype(&self) -> DType {
        dispatch_native!(self, storage, { cpu::from_cpu_dtype(storage.dtype()) })
    }

    fn shape(&self) -> Shape {
        dispatch_native!(self, storage, { Shape::from(storage.shape()) })
    }

    fn to_tensor_buffer(&self) -> Result<TensorBuffer, BackendError> {
        dispatch_native!(self, storage, {
            Ok(cpu::from_cpu_buffer(storage.buffer()))
        })
    }

    fn zero_set(&mut self) -> Result<(), BackendError> {
        match *self {
            #[cfg(feature = "native-cpu")]
            Self::Cpu(ref mut storage) => storage.zero_set().map_err(cpu::backend_error),
        }
    }

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        dispatch_native!(self, storage, { storage.fmt(f) })
    }

    fn stack(tensors: &[Self], dim: usize) -> Result<Self, BackendError> {
        let first = tensors
            .first()
            .ok_or(BackendError::EmptyInputs { operation: "stack" })?;
        dispatch_native!(first, _storage, {
            let tensors = tensors
                .iter()
                .map(|storage| dispatch_native!(storage, storage, storage.clone()))
                .collect::<Vec<_>>();
            nove_native::cpu::CpuStorage::stack(&tensors, dim)
                .map(Self::Cpu)
                .map_err(cpu::backend_error)
        })
    }

    fn cat(tensors: &[Self], dim: usize) -> Result<Self, BackendError> {
        let first = tensors
            .first()
            .ok_or(BackendError::EmptyInputs { operation: "cat" })?;
        dispatch_native!(first, _storage, {
            let tensors = tensors
                .iter()
                .map(|storage| dispatch_native!(storage, storage, storage.clone()))
                .collect::<Vec<_>>();
            nove_native::cpu::CpuStorage::cat(&tensors, dim)
                .map(Self::Cpu)
                .map_err(cpu::backend_error)
        })
    }

    fn to_dtype(&self, dtype: DType) -> Result<Self, BackendError> {
        dispatch_native!(self, storage, {
            storage
                .to_dtype(cpu::to_cpu_dtype(dtype)?)
                .map(Self::Cpu)
                .map_err(cpu::backend_error)
        })
    }

    fn reshape(&self, shape: &Shape) -> Result<Self, BackendError> {
        dispatch_native!(self, storage, {
            storage
                .reshape(shape.dims())
                .map(Self::Cpu)
                .map_err(cpu::backend_error)
        })
    }

    fn broadcast_as(&self, shape: &Shape) -> Result<Self, BackendError> {
        dispatch_native!(self, storage, {
            storage
                .broadcast_as(shape.dims())
                .map(Self::Cpu)
                .map_err(cpu::backend_error)
        })
    }

    fn flatten_from(&self, dim: usize) -> Result<Self, BackendError> {
        dispatch_native!(self, storage, {
            storage
                .flatten_from(dim)
                .map(Self::Cpu)
                .map_err(cpu::backend_error)
        })
    }

    fn flatten_to(&self, dim: usize) -> Result<Self, BackendError> {
        dispatch_native!(self, storage, {
            storage
                .flatten_to(dim)
                .map(Self::Cpu)
                .map_err(cpu::backend_error)
        })
    }

    fn flatten(&self, start: usize, end: usize) -> Result<Self, BackendError> {
        dispatch_native!(self, storage, {
            storage
                .flatten(start, end)
                .map(Self::Cpu)
                .map_err(cpu::backend_error)
        })
    }

    fn squeeze(&self, dim: usize) -> Result<Self, BackendError> {
        dispatch_native!(self, storage, {
            storage
                .squeeze(dim)
                .map(Self::Cpu)
                .map_err(cpu::backend_error)
        })
    }

    fn unsqueeze(&self, dim: usize) -> Result<Self, BackendError> {
        dispatch_native!(self, storage, {
            storage
                .unsqueeze(dim)
                .map(Self::Cpu)
                .map_err(cpu::backend_error)
        })
    }

    fn transpose(&self, dim0: usize, dim1: usize) -> Result<Self, BackendError> {
        dispatch_native!(self, storage, {
            storage
                .transpose(dim0, dim1)
                .map(Self::Cpu)
                .map_err(cpu::backend_error)
        })
    }

    fn contiguous(&self) -> Result<Self, BackendError> {
        Ok(self.clone())
    }

    fn permute(&self, dims: &[usize]) -> Result<Self, BackendError> {
        dispatch_native!(self, storage, {
            storage
                .permute(dims)
                .map(Self::Cpu)
                .map_err(cpu::backend_error)
        })
    }

    fn narrow(&self, dim: usize, start: usize, length: usize) -> Result<Self, BackendError> {
        dispatch_native!(self, storage, {
            storage
                .narrow(dim, start, length)
                .map(Self::Cpu)
                .map_err(cpu::backend_error)
        })
    }

    fn affine(&self, weight: f64, bias: f64) -> Result<Self, BackendError> {
        dispatch_native!(self, storage, {
            storage
                .affine(weight, bias)
                .map(Self::Cpu)
                .map_err(cpu::backend_error)
        })
    }

    fn clamp(&self, min: f64, max: f64) -> Result<Self, BackendError> {
        dispatch_native!(self, storage, {
            storage
                .clamp(min, max)
                .map(Self::Cpu)
                .map_err(cpu::backend_error)
        })
    }

    fn powf(&self, exponent: f64) -> Result<Self, BackendError> {
        dispatch_native!(self, storage, {
            storage
                .powf(exponent)
                .map(Self::Cpu)
                .map_err(cpu::backend_error)
        })
    }

    fn sum(&self, dim: usize) -> Result<Self, BackendError> {
        dispatch_native!(self, storage, {
            storage.sum(dim).map(Self::Cpu).map_err(cpu::backend_error)
        })
    }

    fn sum_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        dispatch_native!(self, storage, {
            storage
                .sum_keepdim(dim)
                .map(Self::Cpu)
                .map_err(cpu::backend_error)
        })
    }

    fn max(&self, dim: usize) -> Result<Self, BackendError> {
        dispatch_native!(self, storage, {
            storage.max(dim).map(Self::Cpu).map_err(cpu::backend_error)
        })
    }

    fn max_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        dispatch_native!(self, storage, {
            storage
                .max_keepdim(dim)
                .map(Self::Cpu)
                .map_err(cpu::backend_error)
        })
    }

    fn min(&self, dim: usize) -> Result<Self, BackendError> {
        dispatch_native!(self, storage, {
            storage.min(dim).map(Self::Cpu).map_err(cpu::backend_error)
        })
    }

    fn min_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        dispatch_native!(self, storage, {
            storage
                .min_keepdim(dim)
                .map(Self::Cpu)
                .map_err(cpu::backend_error)
        })
    }

    fn mean(&self, dim: usize) -> Result<Self, BackendError> {
        dispatch_native!(self, storage, {
            storage.mean(dim).map(Self::Cpu).map_err(cpu::backend_error)
        })
    }

    fn mean_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        dispatch_native!(self, storage, {
            storage
                .mean_keepdim(dim)
                .map(Self::Cpu)
                .map_err(cpu::backend_error)
        })
    }

    fn var(&self, dim: usize) -> Result<Self, BackendError> {
        dispatch_native!(self, storage, {
            storage.var(dim).map(Self::Cpu).map_err(cpu::backend_error)
        })
    }

    fn var_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        dispatch_native!(self, storage, {
            storage
                .var_keepdim(dim)
                .map(Self::Cpu)
                .map_err(cpu::backend_error)
        })
    }

    fn argmax(&self, dim: usize) -> Result<Self, BackendError> {
        dispatch_native!(self, storage, {
            storage
                .argmax(dim)
                .map(Self::Cpu)
                .map_err(cpu::backend_error)
        })
    }

    fn argmax_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        dispatch_native!(self, storage, {
            storage
                .argmax_keepdim(dim)
                .map(Self::Cpu)
                .map_err(cpu::backend_error)
        })
    }

    fn argmin(&self, dim: usize) -> Result<Self, BackendError> {
        dispatch_native!(self, storage, {
            storage
                .argmin(dim)
                .map(Self::Cpu)
                .map_err(cpu::backend_error)
        })
    }

    fn argmin_keepdim(&self, dim: usize) -> Result<Self, BackendError> {
        dispatch_native!(self, storage, {
            storage
                .argmin_keepdim(dim)
                .map(Self::Cpu)
                .map_err(cpu::backend_error)
        })
    }

    fn conv1d(
        &self,
        kernel: &Self,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self, BackendError> {
        dispatch_native!(self, storage, {
            dispatch_native!(kernel, kernel, {
                storage
                    .conv1d(kernel, padding, stride, dilation, groups)
                    .map(Self::Cpu)
                    .map_err(cpu::backend_error)
            })
        })
    }

    fn conv2d(
        &self,
        kernel: &Self,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self, BackendError> {
        dispatch_native!(self, storage, {
            dispatch_native!(kernel, kernel, {
                storage
                    .conv2d(kernel, padding, stride, dilation, groups)
                    .map(Self::Cpu)
                    .map_err(cpu::backend_error)
            })
        })
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
        dispatch_native!(self, storage, {
            dispatch_native!(kernel, kernel, {
                storage
                    .conv_transpose1d(kernel, padding, output_padding, stride, dilation, groups)
                    .map(Self::Cpu)
                    .map_err(cpu::backend_error)
            })
        })
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
        dispatch_native!(self, storage, {
            dispatch_native!(kernel, kernel, {
                storage
                    .conv_transpose2d(kernel, padding, output_padding, stride, dilation, groups)
                    .map(Self::Cpu)
                    .map_err(cpu::backend_error)
            })
        })
    }

    fn max_pool2d_with_stride(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self, BackendError> {
        dispatch_native!(self, storage, {
            storage
                .max_pool2d_with_stride(kernel_size, stride)
                .map(Self::Cpu)
                .map_err(cpu::backend_error)
        })
    }

    fn avg_pool2d_with_stride(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self, BackendError> {
        dispatch_native!(self, storage, {
            storage
                .avg_pool2d_with_stride(kernel_size, stride)
                .map(Self::Cpu)
                .map_err(cpu::backend_error)
        })
    }

    fn gather(&self, indexes: &Self, dim: usize) -> Result<Self, BackendError> {
        dispatch_native!(self, storage, {
            dispatch_native!(indexes, indexes, {
                storage
                    .gather(indexes, dim)
                    .map(Self::Cpu)
                    .map_err(cpu::backend_error)
            })
        })
    }

    fn scatter_add(&self, indexes: &Self, source: &Self, dim: usize) -> Result<Self, BackendError> {
        dispatch_native!(self, storage, {
            dispatch_native!(indexes, indexes, {
                dispatch_native!(source, source, {
                    storage
                        .scatter_add(indexes, source, dim)
                        .map(Self::Cpu)
                        .map_err(cpu::backend_error)
                })
            })
        })
    }

    fn index_select(&self, indexes: &Self, dim: usize) -> Result<Self, BackendError> {
        dispatch_native!(self, storage, {
            dispatch_native!(indexes, indexes, {
                storage
                    .index_select(indexes, dim)
                    .map(Self::Cpu)
                    .map_err(cpu::backend_error)
            })
        })
    }

    fn index_add(&self, indexes: &Self, source: &Self, dim: usize) -> Result<Self, BackendError> {
        dispatch_native!(self, storage, {
            dispatch_native!(indexes, indexes, {
                dispatch_native!(source, source, {
                    storage
                        .index_add(indexes, source, dim)
                        .map(Self::Cpu)
                        .map_err(cpu::backend_error)
                })
            })
        })
    }

    fn where_cond(
        condition: &Self,
        true_value: &Self,
        false_value: &Self,
    ) -> Result<Self, BackendError> {
        dispatch_native!(condition, storage, {
            dispatch_native!(true_value, true_value, {
                dispatch_native!(false_value, false_value, {
                    storage
                        .where_cond(true_value, false_value)
                        .map(Self::Cpu)
                        .map_err(cpu::backend_error)
                })
            })
        })
    }

    fn to_device(&self, device: &Device) -> Result<Self, BackendError> {
        match *native_handle(device)? {
            #[cfg(feature = "native-cpu")]
            NativeDeviceHandle::Cpu => dispatch_native!(self, _storage, Ok(self.clone())),
        }
    }
}

/// Synchronizes all queued work for a Native-backed device.
pub(crate) fn synchronize_device(device: &Device) -> Result<(), BackendError> {
    match *native_handle(device)? {
        #[cfg(feature = "native-cpu")]
        NativeDeviceHandle::Cpu => Ok(()),
    }
}

#[cfg(all(test, feature = "native-cpu"))]
mod tests {
    use super::NativeStorage;
    use crate::backend::Backend;
    use crate::{Shape, backend::BackendStorage, device};

    #[test]
    fn owned_payload_moves_native_buffer() {
        let data = vec![1.0f32, 2.0];
        let original_ptr = data.as_ptr();
        let device = device::native::cpu().unwrap();
        let storage = BackendStorage::from_vec(data, &Shape::from_dims(&[2]), &device).unwrap();
        match storage {
            BackendStorage::Native(NativeStorage::Cpu(storage)) => {
                match storage.buffer() {
                    nove_native::cpu::CpuBuffer::F32(values) => {
                        assert_eq!(values.as_ptr(), original_ptr);
                        assert_eq!(values, &[1.0, 2.0]);
                    }
                    other => panic!("expected F32 buffer, got {other:?}"),
                }
                assert_eq!(storage.shape(), &[2]);
            }
            #[cfg(feature = "candle")]
            other => panic!("expected Native storage, got {other:?}"),
        }
    }

    #[test]
    fn native_clone_has_independent_typed_buffer() {
        let values = vec![9_007_199_254_740_993i64, -9_007_199_254_740_993];
        let source = NativeStorage::Cpu(
            nove_native::cpu::CpuStorage::from_buffer(
                nove_native::cpu::CpuBuffer::I64(values.clone()),
                &[2],
            )
            .unwrap(),
        );
        let copy = source.try_clone().unwrap();
        let NativeStorage::Cpu(source) = source;
        let NativeStorage::Cpu(copy) = copy;
        match (source.buffer(), copy.buffer()) {
            (nove_native::cpu::CpuBuffer::I64(a), nove_native::cpu::CpuBuffer::I64(b)) => {
                assert_ne!(a.as_ptr(), b.as_ptr());
                assert_eq!(b.as_slice(), values.as_slice());
            }
            other => panic!("expected I64 buffers, got {other:?}"),
        }
    }
}
