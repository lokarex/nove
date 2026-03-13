//! The `resource` module provides real-world datasets.

mod cifar10;
mod cifar100;
mod mnist;

pub use cifar10::{Cifar10, Cifar10Dataset};
pub use cifar100::{Cifar100, Cifar100Dataset};
pub use mnist::{Mnist, MnistDataset};
