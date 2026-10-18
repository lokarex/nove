//! Native CPU conversions, factories and backend error mapping.

use crate::{BackendError, BackendKind, DType, Shape, TensorBuffer, TensorPayload};
use nove_native::cpu::CpuStorage;

pub(super) fn backend_error(error: nove_native::cpu::CpuBackendError) -> BackendError {
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
        nove_native::cpu::CpuBackendError::InvalidOperation(message) => {
            BackendError::InvalidOperation(message)
        }
        other => BackendError::BackendImplementation {
            backend: BackendKind::Native,
            message: other.to_string(),
        },
    }
}

pub(super) fn to_cpu_dtype(dtype: DType) -> Result<nove_native::cpu::CpuDType, BackendError> {
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

pub(super) fn from_cpu_dtype(dtype: nove_native::cpu::CpuDType) -> DType {
    match dtype {
        nove_native::cpu::CpuDType::U8 => DType::U8,
        nove_native::cpu::CpuDType::U32 => DType::U32,
        nove_native::cpu::CpuDType::I64 => DType::I64,
        nove_native::cpu::CpuDType::F32 => DType::F32,
        nove_native::cpu::CpuDType::F64 => DType::F64,
    }
}

pub(super) fn to_cpu_buffer(buffer: TensorBuffer) -> nove_native::cpu::CpuBuffer {
    match buffer {
        TensorBuffer::U8(data) => nove_native::cpu::CpuBuffer::U8(data),
        TensorBuffer::U32(data) => nove_native::cpu::CpuBuffer::U32(data),
        TensorBuffer::I64(data) => nove_native::cpu::CpuBuffer::I64(data),
        TensorBuffer::F32(data) => nove_native::cpu::CpuBuffer::F32(data),
        TensorBuffer::F64(data) => nove_native::cpu::CpuBuffer::F64(data),
    }
}

pub(super) fn from_cpu_buffer(buffer: &nove_native::cpu::CpuBuffer) -> TensorBuffer {
    match buffer {
        nove_native::cpu::CpuBuffer::U8(data) => TensorBuffer::U8(data.clone()),
        nove_native::cpu::CpuBuffer::U32(data) => TensorBuffer::U32(data.clone()),
        nove_native::cpu::CpuBuffer::I64(data) => TensorBuffer::I64(data.clone()),
        nove_native::cpu::CpuBuffer::F32(data) => TensorBuffer::F32(data.clone()),
        nove_native::cpu::CpuBuffer::F64(data) => TensorBuffer::F64(data.clone()),
    }
}

pub(super) fn from_payload(payload: TensorPayload) -> Result<CpuStorage, BackendError> {
    let (buffer, shape) = payload.into_parts();
    CpuStorage::from_buffer(to_cpu_buffer(buffer), shape.dims()).map_err(backend_error)
}

pub(super) fn zeros(shape: &Shape, dtype: DType) -> Result<CpuStorage, BackendError> {
    CpuStorage::zeros(shape.dims(), to_cpu_dtype(dtype)?).map_err(backend_error)
}

pub(super) fn ones(shape: &Shape, dtype: DType) -> Result<CpuStorage, BackendError> {
    CpuStorage::ones(shape.dims(), to_cpu_dtype(dtype)?).map_err(backend_error)
}

pub(super) fn rand(
    dtype: DType,
    low: f64,
    high: f64,
    shape: &Shape,
) -> Result<CpuStorage, BackendError> {
    let count = shape.checked_elem_count()?;
    let mut state = 0x1234_5678_9abc_def0u64 ^ count as u64;
    let span = high - low;
    let values = (0..count)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let unit = ((state >> 11) as f64) / ((1u64 << 53) as f64);
            low + unit * span
        })
        .collect::<Vec<_>>();
    CpuStorage::from_f64_values(values, shape.dims(), to_cpu_dtype(dtype)?).map_err(backend_error)
}

pub(super) fn randn(
    dtype: DType,
    mean: f64,
    std: f64,
    shape: &Shape,
) -> Result<CpuStorage, BackendError> {
    let count = shape.checked_elem_count()?;
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
    CpuStorage::from_f64_values(values, shape.dims(), to_cpu_dtype(dtype)?).map_err(backend_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nove_native::cpu::{CpuBackendError, CpuDType};

    #[test]
    fn shape_and_dtype_errors_keep_backend_context() {
        for (error, expected, found) in [
            (
                CpuBackendError::ShapeMismatch {
                    expected: vec![6],
                    found: vec![5],
                },
                "[6]",
                "[5]",
            ),
            (
                CpuBackendError::DTypeMismatch {
                    expected: CpuDType::I64,
                    found: CpuDType::F32,
                },
                "I64",
                "F32",
            ),
        ] {
            assert!(matches!(backend_error(error),
                BackendError::BackendImplementation { backend: BackendKind::Native, message }
                if message.contains(expected) && message.contains(found)));
        }
    }
}
