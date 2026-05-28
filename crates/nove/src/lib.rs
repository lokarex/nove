pub mod dataloader {
    pub use nove_dataloader::*;
}

pub mod dataset {
    pub use nove_dataset::*;
}

pub mod tensor {
    //! Tensor API re-exports.
    //!
    //! Device constructors are intentionally exposed through [`crate::device`]
    //! at the umbrella crate level. Use `nove::device::candle::*` or
    //! `nove::device::native::*` to construct devices.
    //!
    //! # Examples
    //! ```
    //! use nove::tensor::{DType, Shape, Tensor};
    //!
    //! let _shape = Shape::from_dims(&[2, 2]);
    //! let _dtype = DType::F32;
    //! let _ = core::mem::size_of::<Tensor>();
    //! ```
    //!
    //! ```compile_fail
    //! let _ = nove::tensor::
    //!     device::candle::cpu();
    //! ```

    pub use nove_tensor::{
        BackendError, BackendKind, DType, Device, DeviceError, DeviceKind, FloatTensorElement, Shape,
        Tensor, TensorBuffer, IntoTensorPayload, TensorElement, TensorError, TensorPayload, backend,
        safetensor,
    };
}

pub mod device {
    pub use nove_tensor::device::*;
}

pub mod model {
    pub use nove_model::*;
}

pub mod lossfn {
    pub use nove_lossfn::*;
}

pub mod optimizer {
    pub use nove_optimizer::*;
}

pub mod r#macro {
    pub use nove_macro::*;
}

pub mod learner {
    pub use nove_learner::*;
}

pub mod metric {
    pub use nove_metric::*;
}
