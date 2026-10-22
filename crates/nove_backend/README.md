# nove_backend

Backend-independent types, input conversion, validation and storage dispatch for
Nove tensors. The default feature is `candle-cpu`.

`nove_tensor` owns the computation graph and gradients. This crate describes and
creates storage, checks selected input contracts, and forwards operations to
Candle or Native. Its storage adapters do not manage Nove autograd.

## Responsibilities

| Component | Responsibility |
|---|---|
| `DType`, `Shape`, `BackendKind`, `BackendError` | Nove types and errors at the backend boundary |
| `payload` | Typed host data, shape metadata and input container conversion |
| `device` | Backend-qualified devices, identity and location |
| `backend::BackendStorage` | Storage creation, conversion and operation dispatch |
| Internal validation | Shape, axis, dtype, parameter and compatibility checks before selected operations |
| Candle and Native adapters | Conversion to lower implementations and error mapping |
| Serialization routing | Existing safetensors save/load capabilities |

`BackendStorage` and adapter types remain publicly accessible for integration,
although hidden from the generated API listing. The `Backend` trait is internal;
it is not a public plugin interface. Root payload/error exports and their
`backend` re-exports refer to the same types.

## Inputs and safe shape counts

`TensorPayload` carries a `TensorBuffer` and a `Shape` before creating backend
storage. Supported buffer element types are U8, U32, I64, F32 and F64.
Nested inputs must be rectangular. Creation checks that the element count fits
in `usize` and matches the supplied data length.

```rust
use nove_backend::{BackendError, Shape, TensorBuffer, TensorPayload};

let payload = TensorPayload::new(
    TensorBuffer::I64(vec![9_007_199_254_740_993]),
    Shape::from_dims(&[1]),
)?;
let (buffer, shape) = payload.into_parts();
assert_eq!(buffer, TensorBuffer::I64(vec![9_007_199_254_740_993]));
assert_eq!(shape.dims(), &[1]);

assert_eq!(Shape::from(()).checked_elem_count()?, 1);
assert_eq!(Shape::from_dims(&[usize::MAX, 2, 0]).checked_elem_count()?, 0);
assert!(matches!(
    Shape::from_dims(&[usize::MAX, 2]).checked_elem_count(),
    Err(BackendError::InvalidOperation(_))
));
# Ok::<(), BackendError>(())
```

`checked_elem_count()` scans for zero dimensions before multiplying. Ordinary
`elem_count()` retains unchecked multiplication: overflow can panic with overflow
checks enabled or wrap when disabled. A zero element count at the Shape/payload
layer does not guarantee lower storage supports every enormous zero-sized shape.

Owned payloads are consumed, and `into_parts()` moves both fields. The Native CPU
input path moves the existing typed vector into its buffer; Candle receives owned
vectors through its existing factory. Borrowed slices are copied into independent
data. This is an input ownership contract, not a guarantee of zero copies for GPU
transfers, readback, layout operations or every kernel. Random/zero/one factories
also check shape counts before calling the lower implementation.

## Storage and Tensor lifecycle

| Operation | Data and graph behavior |
|---|---|
| `BackendStorage::try_clone()` | Independent data; preserves backend, actual device, dtype and shape |
| `BackendStorage::clone()` | Backend-dependent storage clone; does not guarantee independent data across backends |
| `BackendStorage::zero_set()` | Sets values to zero; aliases of the same buffer may observe the change |
| `Tensor::copy()` | Shares the same Tensor data, graph and gradient |
| `Tensor::try_clone()` / `Tensor::clone()` | Independent values and any stored gradient; disconnected from parent graph; `clone()` panics on failure |
| `Tensor::detach()` | Disconnects graph relationships; the data buffer may still be shared |

Storage does not have `detach` or `assign_from` methods. Tensor detaching remains
an upper-layer operation. `Tensor::update_from_tensor()` requires compatible
backend, actual device, dtype and shape before replacing storage, preserving the
target's identity, name, device and gradient setting. A target with gradients
enabled retains its graph relationships and stored gradient; a target with
gradients disabled clears its graph parents and stored gradient. The source's
graph and metadata settings are not copied.

```rust
use nove_backend::{BackendError, Device, backend::BackendStorage};

# #[cfg(any(feature = "candle-cpu", feature = "native-cpu"))] {
let device = Device::default();
let mut source = BackendStorage::from_data(vec![1.0f32, 2.0], &device)?;
let independent = source.try_clone()?;
source.zero_set()?;
assert_eq!(source.to_vec::<f32>()?, vec![0.0, 0.0]);
assert_eq!(independent.to_vec::<f32>()?, vec![1.0, 2.0]);
assert!(source.same_device(&independent));
# }
# Ok::<(), BackendError>(())
```

## Features and devices

| Feature | Enables |
|---|---|
| `candle` | Candle family adapter and storage variant |
| `candle-cpu` (default) | Candle family and public CPU device construction |
| `candle-cuda` / `candle-metal` | Candle family and the corresponding GPU capability; requires the platform/toolkit and hardware |
| `native` | Native family adapter and storage variant |
| `native-cpu` | Native family and its CPU implementation |

At least one family (`candle` or `native`) must be enabled. Pure family
configurations compile as libraries but have no usable device without a device
capability. `NativeStorage::Cpu` requires `native-cpu`; enabling lower
`nove_native/cpu` through another package does not enable this capability.
Device constructor paths remain available and return
`DeviceError::BackendFeatureNotEnabled` for disabled capabilities.

`Device::default()` selects by enabled feature priority:
CUDA > Metal > Candle CPU > Native CPU. It panics if no device capability is
enabled or its selected GPU cannot initialize. It does not search for another
working device. With their GPU feature enabled, `cuda_if_available` and
`metal_if_available` attempt a fallback only to enabled Candle CPU when GPU
construction returns an error; they never switch to Native. With the GPU feature
disabled, these helpers report the missing GPU feature.

`Device ==` and `BackendStorage::same_device()` use backend device identity.
Independent CPU devices of the same backend are equal. Independent Candle GPU
instances can have the same index but be unequal; cloning a Device preserves its
identity. `DeviceLocation` compares backend, hardware kind and constructor index.

```rust
use nove_backend::{DeviceKind, device};

# #[cfg(feature = "candle-cpu")] {
let first = device::candle::cpu()?;
let second = device::candle::cpu()?;
assert_eq!(first, second);
assert_eq!(first, first.clone());
assert_eq!(first.location(), second.location());
assert_eq!(first.location().kind(), DeviceKind::Cpu);
assert_eq!(first.location().index(), 0);
# }
# #[cfg(feature = "native-cpu")] {
let first = device::native::cpu()?;
assert_eq!(first, first.clone());
assert_eq!(first.location().kind(), DeviceKind::Cpu);
# }
# Ok::<(), nove_backend::DeviceError>(())
```

Ordinary multi-input computation requires compatible backends and actual
devices; it never silently moves an input. Use `to_device()` explicitly.
Successful transfers preserve values and dtype where the target supports them.
CUDA instance transfers use existing lower transfer paths and may stage through
CPU; distinct Metal instance transfer retains its existing capability error.

```rust
use nove_backend::{BackendError, BackendKind, backend::BackendStorage, device};

# #[cfg(all(feature = "candle-cpu", feature = "native-cpu"))] {
let native = device::native::cpu()?;
let candle = device::candle::cpu()?;
let source = BackendStorage::from_data(vec![1.0f32, 2.0], &native)?;
let moved = source.to_device(&candle)?;
assert_eq!(moved.to_vec::<f32>()?, vec![1.0, 2.0]);
assert_eq!(moved.backend_kind()?, BackendKind::Candle);
assert!(matches!(
    source.broadcast_add(&moved),
    Err(BackendError::BackendMismatch {
        expected: BackendKind::Native,
        found: BackendKind::Candle,
    })
));
# }
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Validation and errors

Selected operation inputs are checked before dispatch: shape relationships,
axes, output element counts, index metadata, collection inputs and basic
convolution/pooling/random parameters. For multi-input computation, empty
collections are rejected first, then all backend comparisons precede actual
device checks, followed by operation-specific checks. Shape/type rules differ
by operation; elementwise arithmetic does not impose a global same-dtype rule.

These checks use structured errors such as `AxisOutOfBounds`, `ShapeMismatch`,
`DTypeMismatch`, `DeviceMismatch` and `ShapeOverflow`. Creation and payload count
errors retain `InvalidOperation`. Native CPU maps its lower `InvalidOperation`
directly, keeps `UnsupportedDType` / `UnsupportedOperation`, and wraps lower shape
and dtype errors in `BackendImplementation { backend: Native, message }`.
Candle errors retain Candle `BackendImplementation`; their text is not parsed
to guess a category. These categories do not guarantee that every invalid kernel
request is checked before dispatch.

Intentional input restrictions include I64 indices, U8 `where_cond` conditions
with nonzero meaning true, equal value dtypes for `where_cond`, non-NaN ordered
clamp bounds, finite valid uniform bounds and finite normal parameters with
nonnegative standard deviation. Existing random algorithms and backend type
promotion differences remain unchanged.

## Capabilities by operation

| Capability | Candle | Native CPU |
|---|---|---|
| Typed payload creation/readback | U8, U32, I64, F32, F64 | U8, U32, I64, F32, F64 |
| F16/BF16 storage | Existing factories/conversion and supported lower operations | Unsupported |
| F16/BF16 Rust value readback | Convert to F32 before the five-type buffer path | Unsupported |
| Matrix multiplication | Existing lower support, including batch broadcasting | Rank-2 only; valid batched requests return `UnsupportedOperation` |
| Safetensors saving | Existing implementation, including half dtypes | Internal Candle CPU staging if Candle is compiled; otherwise unsupported |
| Safetensors loading | Into an available Candle device | Unsupported |

This is not a promise that every operation or gradient supports every listed
storage dtype. In particular, half materialization and half gradient support are
separate concerns. Some legal operation capabilities still differ: Candle where
broadcasting, scatter subranges, scalar flatten and non-unit squeeze are not
unified here. Native random generation remains deterministic for its current
fixed-seed scheme; no new RNG state or performance guarantee is introduced.

## Safetensors routing

`backend::save_safetensors` and `backend::load_safetensors` keep their existing
signatures. With Candle enabled, Candle storage is saved directly and Native
storage is copied into temporary Candle CPU storage for writing. Mixed
collections are allowed, even though mixed-backend computation is rejected.
Source values, backends and devices are not changed. Internal CPU staging also
works without the public `candle-cpu` capability.

Without Candle, saving any collection, including an empty one, returns
`UnsupportedOperation { backend: Native, operation: "save_safetensors" }` before
file operations. Loading into Native always returns the corresponding
`load_safetensors` error before reading. Saving Native successfully through
Candle does not imply that loading into Native is supported.

```rust
use std::collections::HashMap;
use nove_backend::{backend, device};

# #[cfg(feature = "candle-cpu")] {
# let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target");
# std::fs::create_dir_all(&root)?;
# let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos();
# let directory = root.join(format!("backend-readme-{}-{nonce}", std::process::id()));
# std::fs::create_dir(&directory)?;
# let file = directory.join("model.safetensors");
# let path = file.to_str().ok_or("non-UTF8 test path")?;
let device = device::candle::cpu()?;
let storage = backend::BackendStorage::from_data(vec![1.0f32, 2.0], &device)?;
backend::save_safetensors(path, HashMap::from([("weights".into(), storage)]))?;
let loaded = backend::load_safetensors(path, &device)?;
assert_eq!(loaded["weights"].to_vec::<f32>()?, vec![1.0, 2.0]);
# std::fs::remove_file(file)?;
# std::fs::remove_dir(directory)?;
# }
# Ok::<(), Box<dyn std::error::Error>>(())
```

Candle supports empty collections. Supported-path file errors retain the Candle
implementation error. Ordinary writes are not guaranteed to be atomic, and no
ordering is specified between multiple conversion failures in a map. File
protection for rejected operations applies to these backend entry points; upper
builders may already have created directories.

## API migration and verification scope

The refactor preserves root/backend type paths, adapter paths and
`BackendStorage` variants. Additions include `Shape::checked_elem_count`,
`TensorPayload::into_parts`, storage `try_clone` / `same_device`, structured
validation errors, `DeviceLocation` and `Device::location`. Internal
`Backend::from_payload` consumes its payload.

Removed Nove-facing APIs are storage `detach` / `assign_from`,
`Tensor::from_candle_tensor` / `to_candle_tensor`, backend raw Candle storage
import/export, `Device::to_candle_device` and global Nove/Candle dtype `From`
implementations. Internal Candle helpers and the lower `nove_candle` crate remain.
Device equality now uses identity; compare `location()` when only location matters.
External exhaustive error matches must account for added validation variants.

Tests cover boundary contracts and regressions, not full mathematical
correctness. Deferred work includes complete convolution checks, empty reduction
semantics, upper Tensor permutation validation, variance/where gradients, Native
integer precision in layout kernels, old-graph mutation, no-gradient pruning and
half gradients. Some existing lower paths can still panic, including Metal
out-of-range device selection. No benchmark establishes a general speedup.

CI retains three-platform workspace tests and adds independent library profiles,
four CPU contract configurations, family-only contracts, external feature merge,
dependency isolation and model loading checks. GPU identity/transfer tests are
ignored until explicitly run with their feature and hardware; skipped tests do
not establish GPU support.
