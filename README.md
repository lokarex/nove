# nove

[![CI](https://github.com/lokarex/nove/actions/workflows/test.yml/badge.svg)](https://github.com/lokarex/nove/actions/workflows/test.yml)
[![Crates.io](https://img.shields.io/crates/v/nove?style=flat-square)](https://crates.io/crates/nove)
[![Docs.rs](https://img.shields.io/badge/docs-latest-blue)](https://docs.rs/nove/latest/nove/)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue?style=flat-square)](https://github.com/lokarex/nove#license)

**nove** is a beginner-friendly deep learning library for Rust.It provides a PyTorch-like
API for building, training, and evaluating neural networks.

- **Dual backends** — leverage Candle for GPU acceleration (CUDA/Metal) or use
  the pure Rust CPU backend when you need zero native dependencies
- **Batteries included** — datasets, data loaders, layers, loss functions,
  optimizers, metrics, and a unified training loop — everything you need in
  one crate
- **`#[derive(Model)]`** — annotate struct and get `forward()`,
  `parameters()`, `save()`, `load()`, and more, generated automatically

## Features

### Tensors

Create tensors on CPU, CUDA, or Metal. Every arithmetic, shape, and reduction
operation is tracked in nove's computational graph for automatic
differentiation. Save and load tensors in the safetensors format.

```rust
let x = Tensor::randn(0f32, 1.0, &Shape::from_dims(&[2, 3]), &device, true)?;
let y = x.add(&x)?.mean(None)?;
y.backward()?;
```

### Model

20+ built-in layers — convolution, pooling, batch/layer normalization,
dropout, RNN, GRU, LSTM, embedding, and more. Compose them into blocks
(`Conv2dBlock`, `LinearBlock`) that bundle activation and pooling.

Use `#[derive(Model)]` to turn struct into a trainable model:

```rust
#[derive(Debug, Clone, Model)]
#[model(input = "Tensor", output = "Tensor")]
pub struct MyCNN {
    conv1: Conv2dBlock,
    linear: LinearBlock,
}
```

### Loss Functions

Six loss functions with a common `LossFn` trait: cross-entropy, NLL, BCE,
BCE-with-logits, MSE, and L1.

```rust
let loss = CrossEntropyLoss::new();
let value = loss.loss((logits, targets))?;
```

### Optimizers

SGD, Adam, AdamW, RMSProp, and AdaGrad — each with a builder API.

```rust
let mut opt = AdamBuilder::new(model.parameters()?, 1e-3).build()?;
opt.step()?;
opt.zero_grad()?;
```

### Training Loop

The `Learner` trait unifies training, validation, and testing. `EpochLearner`
gives you fine-grained control; `ImageClassificationLearner` works out of
the box for classification tasks.

```rust
let mut learner = ImageClassificationLearnerBuilder::new()
    .train_dataloader(train_dl)
    .validate_dataloader(val_dl)
    .model(model)
    .optimizer(opt)
    .lossfn(loss)
    .epoch(5)
    .build()?;
learner.train()?;
```

### Data

Built-in datasets (MNIST, CIFAR-10, CIFAR-100, IMDb) with automatic download
and caching. `PrefetchDataloader` loads the next batch in parallel while the
model trains on the current one.

```rust
let dataset = Cifar10::new(&data_dir)?.train()?;
let dl = ImageClassificationDataloaderBuilder::default()
    .dataset(dataset)
    .batch_size(64)
    .shuffle_seed(Some(32))
    .build()?;
```

## Installation

Add `nove` to your `Cargo.toml`:

```bash
cargo add nove
```

The default backend is Candle CPU — no extra setup required. To switch
backends, enable one of the following features:

| Feature | Backend | When to use |
|---------|---------|--------------|
| *(default)* `candle-cpu` | Candle CPU | Works everywhere, zero configuration |
| `candle-cuda` | Candle CUDA | NVIDIA GPU with CUDA toolkit installed |
| `candle-metal` | Candle Metal | macOS with Apple Silicon or AMD GPU |
| `native-cpu` | Pure Rust CPU | Environments where Candle cannot be compiled |

Example with CUDA:

```bash
cargo add nove --features candle-cuda
```

Or manually in `Cargo.toml`:

```toml
[dependencies]
nove = { version = "0.1", features = ["candle-cuda"] }
```

> **Note:** To use the
> pure Rust backend without any Candle dependency, disable default features:
>
> ```toml
> [dependencies]
> nove = { version = "0.1", default-features = false, features = ["native-cpu"] }
> ```

## Quick Start

This guide creates a CNN for MNIST digit recognition, prints the model
structure, and runs a single forward pass.

### 1. Create a project

```bash
cargo new mnist_cnn
cd mnist_cnn
cargo add nove
```

### 2. Write the model

Replace `src/main.rs` with:

```rust
use nove::r#macro::Model;
use nove::model::nn::{
    Activation, Conv2dBlock, Conv2dBlockBuilder, LinearBlock, LinearBlockBuilder, Pool2d,
};
use nove::model::{Model, ModelError};
use nove::tensor::{Device, Shape, Tensor};

#[derive(Debug, Clone, Model)]
#[model(input = "Tensor", output = "Tensor")]
pub struct MnistCNN {
    conv1: Conv2dBlock,
    conv2: Conv2dBlock,
    linear1: LinearBlock,
    linear2: LinearBlock,
}

impl MnistCNN {
    fn new(device: Device) -> Result<Self, ModelError> {
        let conv1 = Conv2dBlockBuilder::new(3, 32, (3, 3), 1, 1)
            .with_activation(Activation::relu())
            .with_pool2d(Pool2d::max_pool2d((2, 2), Some((2, 2)))?)
            .device(device.clone())
            .build()?;
        let conv2 = Conv2dBlockBuilder::new(32, 64, (3, 3), 1, 1)
            .with_activation(Activation::relu())
            .with_pool2d(Pool2d::max_pool2d((2, 2), Some((2, 2)))?)
            .device(device.clone())
            .build()?;
        // After two (2,2) max-pooling layers with stride (2,2) on a 28x28 input:
        // 28 -> 14 -> 7.  64 channels x 7 x 7 = 3136.
        let linear1 = LinearBlockBuilder::new(3136, 128)
            .with_activation(Activation::relu())
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

    fn forward(&mut self, input: Tensor) -> Result<Tensor, ModelError> {
        let mut x = self.conv1.forward(input)?;
        x = self.conv2.forward(x)?;
        x = x.flatten(Some(1), None)?;
        x = self.linear1.forward(x)?;
        x = self.linear2.forward(x)?;
        Ok(x)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::default();
    let mut model = MnistCNN::new(device.clone())?;

    // The #[derive(Model)] macro also generates Display —
    // you can print the full model structure:
    println!("{}", model);

    let input = Tensor::rand(
        -1f32,
        1f32,
        &Shape::from_dims(&[1, 3, 28, 28]),
        &device,
        false,
    )?;
    let output = model.forward(input)?;
    println!("Output shape: {:?}", output.shape());

    Ok(())
}
```

### 3. Run it

```bash
cargo run --release
```

You will see the model structure printed, followed by the output tensor shape
`[1, 10]` (batch size 1, 10 digit classes).

> **Why 3 input channels?** MNIST images are grayscale, but
> `ImageClassificationDataloader` converts them to RGB by default.

### Next steps

Ready to train? See the complete examples:

```bash
# Train a CNN on MNIST
cargo run --example train_mnist_cnn --release

# Train a CNN on CIFAR-10
cargo run --example train_cifar10_cnn --release
```

## Project Structure

nove is a Cargo workspace of 14 crates, organized in four layers:

**Core & Backend**

| Crate | Purpose |
|-------|---------|
| `nove` | Umbrella crate — re-exports everything. This is the only dependency you need. |
| `nove_backend` | Backend-agnostic types: `Device`, `DType`, `Shape`, `TensorPayload`. |
| `nove_tensor` | Tensor operations, computational graph, and automatic differentiation. |
| `nove_candle` | Candle backend — delegates ops to `candle_core` for CPU, CUDA, and Metal. |
| `nove_cpu` | Pure Rust CPU backend — no Candle dependency, always compiles. |
| `nove_native` | Native backend facade — groups pure Rust backends under one feature flag. |

**Model & Training**

| Crate | Purpose |
|-------|---------|
| `nove_model` | `Model` trait + 20+ layers (conv, pool, norm, dropout, RNN, GRU, LSTM, …). |
| `nove_lossfn` | `LossFn` trait + six loss functions (cross-entropy, MSE, L1, BCE, NLL). |
| `nove_optimizer` | `Optimizer` trait + SGD, Adam, AdamW, RMSProp, AdaGrad with builder APIs. |
| `nove_learner` | `Learner` trait + `EpochLearner` and `ImageClassificationLearner`. |

**Data & Metrics**

| Crate | Purpose |
|-------|---------|
| `nove_dataset` | `Dataset` trait + built-in datasets: MNIST, CIFAR-10, CIFAR-100, IMDb. |
| `nove_dataloader` | `Dataloader` trait + batching, shuffling, and parallel prefetching. |
| `nove_metric` | `Metric` trait + accuracy, loss, and CPU usage tracking. |

**Utilities**

| Crate | Purpose |
|-------|---------|
| `nove_macro` | Procedural macros — `#[derive(Model)]` generates forward, parameters, save/load, and Display. |

## Documentation

API documentation is available on [docs.rs](https://docs.rs/nove/latest/nove/)
or build it locally:

```bash
cargo doc --open --no-deps
```

## Contributing

Contributions are welcome! Please open an issue or pull request on
[GitHub](https://github.com/lokarex/nove).

Before submitting, make sure your changes pass:

```bash
cargo fmt --all --check
cargo clippy --all-targets --workspace -- -D warnings
cargo test --workspace
```

## License

Licensed under either of:

- [MIT License](LICENSE-MIT)
- [Apache License 2.0](LICENSE-APACHE)

at your option.
