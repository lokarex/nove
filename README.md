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

## Quick Start

This guide will walk you through creating a simple MNIST CNN model, printing its structure, and running a forward pass that displays the input and output tensors.

### 1. Create a new project

```bash
cargo new mnist_cnn
cd mnist_cnn
```

### 2. Add nove dependency

Using `cargo add` (recommended):

```bash
cargo add nove
```

If you need CUDA or Metal support, add the corresponding feature:

```bash
cargo add nove --features cuda
```

```bash
cargo add nove --features metal
```

### 3. Design your model

Replace the contents of `src/main.rs` with the following code:

```rust
use nove::r#macro::Model;
use nove::model::layer::{Conv2dBlock, Conv2dBlockBuilder, LinearBlock, LinearBlockBuilder};
use nove::model::{Model, ModelError};
use nove::tensor::{Device, Shape, Tensor};

#[derive(Debug, Clone, Model)]
#[model(input = "(Tensor, bool)", output = "Tensor")]
pub struct MnistCNN {
    conv1: Conv2dBlock,
    conv2: Conv2dBlock,
    linear1: LinearBlock,
    linear2: LinearBlock,
}

impl MnistCNN {
    fn new(device: Device) -> Result<Self, ModelError> {
        let conv1 = Conv2dBlockBuilder::new(3, 32, (3, 3), 1, 1)
            .with_relu()
            .with_max_pool((2, 2), (2, 2))
            .device(device.clone())
            .build()?;
        let conv2 = Conv2dBlockBuilder::new(32, 64, (3, 3), 1, 1)
            .with_relu()
            .with_max_pool((2, 2), (2, 2))
            .device(device.clone())
            .build()?;
        let linear1 = LinearBlockBuilder::new(3136, 128)
            .with_relu()
            .device(device.clone())
            .build()?;
        let linear2 = LinearBlockBuilder::new(128, 10)
            .device(device.clone())
            .build()?;
        Ok(Self {
            conv1,
            conv2,
            linear1,
            linear2,
        })
    }

    fn forward(&mut self, input: (Tensor, bool)) -> Result<Tensor, ModelError> {
        let (x, training) = input;
        let mut x = self.conv1.forward((x, training))?;
        x = self.conv2.forward((x, training))?;

        x = x.flatten(Some(1), None)?;

        x = self.linear1.forward((x, training))?;
        x = self.linear2.forward((x, training))?;
        Ok(x)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::cpu();
    let mut model = MnistCNN::new(device.clone())?;
    println!("Model structure:\n{}\n", model);

    // Create a dummy input tensor (batch size = 1, channels = 3, height = 28, width = 28)
    let input = Tensor::rand(
        -1f32,
        1f32,
        &Shape::from_dims(&[1, 3, 28, 28]),
        &device.clone(),
        false,
    )?;
    println!("Input tensor:\n{}\n", input);
    let output = model.forward((input, true))?;
    println!("Output tensor:\n{}", output);

    Ok(())
}

```

**Note:** The model uses **3 input channels** to match the output of `ImageClassificationDataloader` (which converts MNIST grayscale images to RGB by default). The flatten dimension (`3136`) is calculated based on the output size after two pooling layers: 64 channels × 7×7 spatial dimensions.

### 4. Run the project

```bash
cargo run --release
```

You should see:
1. The model structure
2. The input tensor shape `[1, 3, 28, 28]` (batch size 1, 3 channels, 28×28 pixels) and its values
3. The output tensor shape `[1, 10]` (batch size 1, 10 classes) and its values

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