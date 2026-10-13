//! Backend-owned storage, metadata and formatting.

mod combination;
mod conversion;
mod convolution;
mod dispatch;
mod factory;
mod index;
mod lifecycle;
mod ops;

use super::Backend;
#[cfg(feature = "candle")]
use super::candle;
#[cfg(feature = "native")]
use super::native;
use crate::{BackendError, BackendKind, DType, Shape};
use std::fmt::Display;

pub(super) fn ensure_compatible_devices(
    operation: &'static str,
    storages: &[&BackendStorage],
) -> Result<(), BackendError> {
    let Some((first, others)) = storages.split_first() else {
        return Err(BackendError::EmptyInputs { operation });
    };
    let expected = first.backend_kind()?;
    for other in others {
        let found = other.backend_kind()?;
        if found != expected {
            return Err(BackendError::BackendMismatch { expected, found });
        }
    }
    for (index, other) in others.iter().enumerate() {
        if !first.same_device(other) {
            return Err(BackendError::DeviceMismatch {
                operation,
                input_index: index + 1,
            });
        }
    }
    Ok(())
}

/// Backend-owned tensor storage.
#[derive(Clone, Debug)]
#[doc(hidden)]
pub enum BackendStorage {
    #[cfg(feature = "candle")]
    Candle(candle::CandleStorage),
    #[cfg(feature = "native")]
    Native(native::NativeStorage),
}

impl BackendStorage {
    pub fn backend_kind(&self) -> Result<BackendKind, BackendError> {
        match self {
            #[cfg(feature = "candle")]
            Self::Candle(_) => Ok(BackendKind::Candle),
            #[cfg(feature = "native")]
            Self::Native(_) => Ok(BackendKind::Native),
        }
    }

    pub fn dtype(&self) -> Result<DType, BackendError> {
        match self {
            #[cfg(feature = "candle")]
            Self::Candle(storage) => Ok(storage.dtype()),
            #[cfg(feature = "native")]
            Self::Native(storage) => Ok(storage.dtype()),
        }
    }

    pub fn shape(&self) -> Result<Shape, BackendError> {
        match self {
            #[cfg(feature = "candle")]
            Self::Candle(storage) => Ok(storage.shape()),
            #[cfg(feature = "native")]
            Self::Native(storage) => Ok(storage.shape()),
        }
    }

    pub fn fmt_backend(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(feature = "candle")]
            Self::Candle(storage) => storage.fmt(f),
            #[cfg(feature = "native")]
            Self::Native(storage) => storage.fmt(f),
        }
    }
}

impl Display for BackendStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.fmt_backend(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_devices() -> Vec<crate::Device> {
        vec![
            #[cfg(feature = "candle-cpu")]
            crate::device::candle::cpu().unwrap(),
            #[cfg(feature = "native-cpu")]
            crate::device::native::cpu().unwrap(),
        ]
    }

    #[test]
    fn compatibility_rejects_empty_inputs() {
        assert!(matches!(
            ensure_compatible_devices("test", &[]),
            Err(BackendError::EmptyInputs { operation: "test" })
        ));
    }

    #[test]
    fn compatibility_accepts_same_device() {
        for device in cpu_devices() {
            let a = BackendStorage::from_data(vec![1.0f32], &device).unwrap();
            let b = BackendStorage::from_data(vec![2.0f32], &device.clone()).unwrap();
            ensure_compatible_devices("test", &[&a, &b]).unwrap();
        }
    }

    #[cfg(all(feature = "candle-cpu", feature = "native-cpu"))]
    #[test]
    fn compatibility_reports_first_backend_mismatch() {
        let candle =
            BackendStorage::from_data(vec![1.0f32], &crate::device::candle::cpu().unwrap())
                .unwrap();
        let native =
            BackendStorage::from_data(vec![2.0f32], &crate::device::native::cpu().unwrap())
                .unwrap();
        for (a, b) in [(&candle, &native), (&native, &candle)] {
            match ensure_compatible_devices("test", &[a, a, b]) {
                Err(BackendError::BackendMismatch { expected, found }) => {
                    assert_eq!(expected, a.backend_kind().unwrap());
                    assert_eq!(found, b.backend_kind().unwrap());
                }
                result => panic!("unexpected result: {result:?}"),
            }
        }
    }
}
