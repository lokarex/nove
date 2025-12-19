//! The `util` module provides some practical `Dataset` structs to handle datasets.

mod shuffled_dataset;
pub use shuffled_dataset::ShuffledDataset;

mod persisted_dataset;
pub use persisted_dataset::PersistedDataset;

mod iterable_dataset;
pub use iterable_dataset::IterableDataset;
