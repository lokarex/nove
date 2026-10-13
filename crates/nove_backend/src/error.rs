use crate::{BackendKind, DType, Shape, device::DeviceKind};
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

    /// The operation requires at least one input.
    #[error("{operation}: requires at least one input")]
    EmptyInputs { operation: &'static str },

    /// An input uses a different actual device from input 0.
    #[error("{operation}: input {input_index} uses a different device from input 0")]
    DeviceMismatch {
        operation: &'static str,
        input_index: usize,
    },

    /// An axis is outside the input's dimensions.
    #[error("{operation}: axis {axis} is out of range for rank {rank}")]
    AxisOutOfBounds {
        operation: &'static str,
        axis: usize,
        rank: usize,
    },

    /// An insertion axis exceeds the input rank.
    #[error("{operation}: insertion axis {axis} exceeds rank {rank}")]
    InsertionAxisOutOfBounds {
        operation: &'static str,
        axis: usize,
        rank: usize,
    },

    /// An input does not have the required rank.
    #[error("{operation}: input {input_index} requires rank {expected}, found {found}")]
    RankMismatch {
        operation: &'static str,
        input_index: usize,
        expected: usize,
        found: usize,
    },

    /// An input rank is below the operation's minimum.
    #[error("{operation}: input {input_index} requires rank >= {minimum}, found {found}")]
    RankTooSmall {
        operation: &'static str,
        input_index: usize,
        minimum: usize,
        found: usize,
    },

    /// An input dtype does not match its role in the operation.
    #[error("{operation}: input {input_index} requires dtype {expected:?}, found {found:?}")]
    DTypeMismatch {
        operation: &'static str,
        input_index: usize,
        expected: DType,
        found: DType,
    },

    /// Two shapes do not satisfy the operation's compatibility rules.
    #[error("{operation}: incompatible shapes {lhs:?} and {rhs:?}, axis {axis:?}")]
    ShapeMismatch {
        operation: &'static str,
        lhs: Shape,
        rhs: Shape,
        axis: Option<usize>,
    },

    /// A permutation has an incorrect length or repeats an axis.
    #[error("{operation}: invalid permutation {permutation:?} for rank {rank}")]
    InvalidPermutation {
        operation: &'static str,
        permutation: Vec<usize>,
        rank: usize,
    },

    /// An inclusive flatten axis range is reversed or out of bounds.
    #[error("{operation}: invalid axis range {start}..={end} for rank {rank}")]
    InvalidAxisRange {
        operation: &'static str,
        start: usize,
        end: usize,
        rank: usize,
    },

    /// A slice exceeds the selected axis length.
    #[error(
        "{operation}: axis {axis} has size {axis_size}, cannot slice start {start}, length {length}"
    )]
    SliceOutOfBounds {
        operation: &'static str,
        axis: usize,
        start: usize,
        length: usize,
        axis_size: usize,
    },

    /// An output or intermediate shape's element count overflows usize.
    #[error("{operation}: element count overflows usize for shape {shape:?}")]
    ShapeOverflow {
        operation: &'static str,
        shape: Shape,
    },

    /// Computing a slice end or concatenated axis length overflows usize.
    #[error("{operation}: {parameter} overflows usize when adding {lhs} and {rhs}")]
    SizeOverflow {
        operation: &'static str,
        parameter: &'static str,
        lhs: usize,
        rhs: usize,
    },

    /// Clamp or uniform distribution bounds violate the operation's rules.
    #[error("{operation}: invalid bounds, lower={lower}, upper={upper}")]
    InvalidBounds {
        operation: &'static str,
        lower: f64,
        upper: f64,
    },

    /// Normal distribution parameters are nonfinite or std is negative.
    #[error(
        "{operation}: expected finite mean and nonnegative finite std, got mean={mean}, std={std}"
    )]
    InvalidNormalParameters {
        operation: &'static str,
        mean: f64,
        std: f64,
    },

    /// An integer parameter that must be nonzero is zero.
    #[error("{operation}: {parameter} must be nonzero, got 0")]
    ZeroParameter {
        operation: &'static str,
        parameter: &'static str,
    },

    /// A required spatial dimension is zero.
    #[error("{operation}: input {input_index} has an empty spatial dimension at axis {axis}")]
    EmptySpatialDimension {
        operation: &'static str,
        input_index: usize,
        axis: usize,
    },

    /// A pooling kernel exceeds the input's spatial dimensions.
    #[error("{operation}: kernel {kernel:?} exceeds input spatial dimensions {spatial:?}")]
    KernelTooLarge {
        operation: &'static str,
        kernel: (usize, usize),
        spatial: (usize, usize),
    },

    /// The operation request is invalid before reaching a backend kernel.
    #[error("Invalid backend operation: {0}")]
    InvalidOperation(String),
}
