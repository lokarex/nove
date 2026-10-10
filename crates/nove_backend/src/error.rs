use crate::{BackendKind, DType, device::DeviceKind};
use thiserror::Error;

/// Errors returned by backend implementations and dispatch helpers.
///
/// # Notes
/// Backend adapters should preserve stable high-level error categories, such as
/// unsupported dtype, unsupported operation, and backend mismatch, instead of
/// exposing backend-specific error types through public tensor APIs.
#[derive(Error, Debug)]
pub enum BackendError {
    /// A backend implementation returned an error message.
    #[error("{backend:?} backend error: {message}")]
    BackendImplementation {
        /// Backend that produced the error.
        backend: BackendKind,
        /// Backend-provided message.
        message: String,
    },

    /// The selected backend is not available in the compiled feature set.
    #[error("Unsupported backend: {0:?}")]
    UnsupportedBackend(BackendKind),

    /// The selected device kind is not supported by the backend.
    #[error("Unsupported device: {0:?}")]
    UnsupportedDevice(DeviceKind),

    /// The selected dtype is not supported by the backend.
    #[error("Unsupported dtype for {backend:?}: {dtype:?}")]
    UnsupportedDType { backend: BackendKind, dtype: DType },

    /// The selected operation is not supported by the backend.
    #[error("Unsupported operation for {backend:?}: {operation}")]
    UnsupportedOperation {
        /// Backend that rejected the operation.
        backend: BackendKind,
        /// Operation name.
        operation: String,
    },

    /// Two tensors or storages belong to different backends.
    #[error("Backend mismatch: expected {expected:?}, found {found:?}")]
    BackendMismatch {
        /// Backend required by the current operation.
        expected: BackendKind,
        /// Backend found on the input storage.
        found: BackendKind,
    },

    /// The operation request is invalid before reaching a backend kernel.
    #[error("Invalid backend operation: {0}")]
    InvalidOperation(String),
}
