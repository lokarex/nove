//! Native backend family facade for Nove.
//!
//! This crate groups Nove-owned backend implementations behind a single
//! dependency boundary. Device capability features, such as `cpu`, enable the
//! concrete implementation crates that provide storage and kernels.
//!
//! # Notes
//! Enabling this crate without a device capability feature only compiles the
//! native facade. It does not make any native device constructor available.
//!
//! # Examples
//! ```
//! # #[cfg(feature = "cpu")]
//! # {
//! use nove_native::cpu::{CpuDType, CpuStorage};
//!
//! let storage = CpuStorage::ones(&[2, 2], CpuDType::F32).unwrap();
//!
//! assert_eq!(storage.shape(), &[2, 2]);
//! assert_eq!(storage.dtype(), CpuDType::F32);
//! # }
//! ```

/// Native CPU backend implementation re-exports.
///
/// # Notes
/// This module is available when the `cpu` feature is enabled. It re-exports
/// the concrete CPU backend crate so higher-level crates can depend on the
/// native backend family rather than each native device implementation.
#[cfg(feature = "cpu")]
pub mod cpu {
    pub use nove_cpu::{CpuBackendError, CpuBuffer, CpuDType, CpuResult, CpuStorage};
}
