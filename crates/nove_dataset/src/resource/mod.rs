//! The `resource` module provides real-world datasets.

mod acl_imdb10;
mod acl_imdb2;
mod acl_imdb_unsup;
mod cifar10;
mod cifar100;
mod mnist;

pub use acl_imdb_unsup::{AclImdbUnsup, AclImdbUnsupDataset};
pub use acl_imdb2::{AclImdb2, AclImdb2Dataset};
pub use acl_imdb10::{AclImdb10, AclImdb10Dataset};
pub use cifar10::{Cifar10, Cifar10Dataset};
pub use cifar100::{Cifar100, Cifar100Dataset};
pub use mnist::{Mnist, MnistDataset};
