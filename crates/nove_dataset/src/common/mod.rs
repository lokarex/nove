//! The `util` module provides some practical `Dataset` structs to handle datasets.

mod shufflable_dataset;
pub use shufflable_dataset::ShufflableDataset;

mod persistent_dataset;
pub use persistent_dataset::PersistentDataset;

mod iterable_dataset;
pub use iterable_dataset::IterableDataset;

mod vec_dataset;
pub use vec_dataset::VecDataset;
