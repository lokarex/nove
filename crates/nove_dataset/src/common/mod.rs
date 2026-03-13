//! The `common` module provides some common utilities for datasets.

mod downloader;
pub use downloader::{ChecksumType, Downloader, download, download_and_verify};

mod extractor;
pub use extractor::{ArchiveFormat, Extractor, extract_archive};

mod shufflable_dataset;
pub use shufflable_dataset::ShufflableDataset;

mod persistent_dataset;
pub use persistent_dataset::PersistentDataset;

mod iterable_dataset;
pub use iterable_dataset::IterableDataset;

mod vec_dataset;
pub use vec_dataset::VecDataset;
