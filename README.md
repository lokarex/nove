# nove

[![License](https://img.shields.io/github/license/lokarex/nove?color=blue)](https://github.com/lokarex/nove/blob/main/LICENSE-MIT)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square)](https://github.com/lokarex/nove/blob/main/LICENSE-APACHE)

**nove** is an easy-to-use, lightweight deep learning library for Rust, built on top of [Candle](https://github.com/huggingface/candle). It provides a high-level, PyTorch‑like API for training and evaluating neural networks, with built‑in support for common tasks like image classification, data loading, model building, optimization, and metrics tracking.

## Features

- **Tensor operations** backed by Candle (CPU/CUDA/Metal)
- **Dataset & DataLoader** with built‑in datasets (CIFAR‑10, CIFAR‑100, MNIST, IMDb) and customizable pipelines
- **Model building** via composable layers (convolution, pooling, batch normalization, dropout, linear, activations)
- **Loss functions** (cross‑entropy, MSE, L1, BCE, NLL)
- **Optimizers** (SGD, Adam, AdamW, RMSProp, AdaGrad)
- **Training loop** with a unified `Learner` abstraction
- **Metrics** for accuracy, loss, CPU usage, and more
- **Macro‑based model definition** (`#[derive(Model)]`) for easy struct‑to‑model conversion
- **Prefetching & parallel data loading** for efficient I/O
- **Extensible architecture** – add custom datasets, layers, loss functions, and metrics

## Installation

Add `nove` as a dependency in your `Cargo.toml`. You can enable CUDA or Metal backends via features:

Using `cargo add` (recommended):

```bash
cargo add nove
```

Or manually edit your `Cargo.toml`:

```toml
[dependencies]
nove = { git = "https://github.com/lokarex/nove" }
```

Or, if you prefer to work locally with the workspace:

```toml
[dependencies]
nove = { path = "crates/nove" }
```

**Available features:**
- `cuda` – enable CUDA acceleration (requires CUDA toolkit)
- `metal` – enable Metal acceleration (macOS)

## Examples

The repository includes two complete examples:

- **`examples/cifar10_cnn`** – CNN for CIFAR‑10 classification
- **`examples/mnist_cnn`** – CNN for MNIST digit recognition

Run them like this:

```bash
cargo run --example train_cifar10_cnn --release
```

## Project Structure

nove is organized as a Cargo workspace with the following crates:

| Crate | Purpose |
|-------|---------|
| `nove` | Main library, re‑exports all components |
| `nove_tensor` | Tensor operations (wraps Candle) |
| `nove_dataset` | Datasets (CIFAR‑10, MNIST, IMDb, …) and dataset utilities |
| `nove_dataloader` | Data loading, batching, shuffling, prefetching |
| `nove_model` | Neural network layers and model definition |
| `nove_lossfn` | Loss functions |
| `nove_optimizer` | Optimizers |
| `nove_learner` | Training loop and learner abstraction |
| `nove_metric` | Metrics (accuracy, CPU usage, etc.) |
| `nove_macro` | Procedural macros for `#[derive(Model)]` |

## Documentation

API documentation is available via `cargo doc`:

```bash
cargo doc --open --no-deps
```

The documentation covers all public modules, structs, and functions.

## Contributing

Contributions are welcome! Please open an issue or a pull request on [GitHub](https://github.com/lokarex/nove).

## License

This project is dual‑licensed under either:

- **MIT License** – see [LICENSE-MIT](LICENSE-MIT)
- **Apache License 2.0** – see [LICENSE-APACHE](LICENSE-APACHE)

at your option.