//! The `dataset` module defines a generic trait `Dataset` for datasets,
//! provides some real datasets and achieves a few practical `Dataset` structs
//! to handle datasets.

use thiserror::Error;

pub mod common;

#[derive(Debug, Error)]
pub enum DatasetError {
    /// The index is out of bounds.
    #[error("Index out of bounds: {0} (len: {1})")]
    IndexOutOfBounds(usize, usize),

    /// I/O errors from the standard library.
    #[error(transparent)]
    IoError(#[from] std::io::Error),

    /// Encoding errors from `bincode` library.
    #[error(transparent)]
    EncodingError(#[from] bincode::error::EncodeError),

    /// Decoding errors from `bincode` library.
    #[error(transparent)]
    DecodingError(#[from] bincode::error::DecodeError),

    /// Other errors.
    #[error("{0}")]
    OtherError(String),
}

/// The `Dataset` trait defines a generic interface for datasets.
/// Every dataset should implement this trait.
///
/// # Required Type Parameters
/// * `Item` - The type of items in the dataset.
///
/// # Required Methods
/// * `get(index: usize) -> Self::Item` - Get the item at the given index.
/// * `len() -> usize` - Get the number of items in the dataset.
///
/// # Provided Methods
/// * `is_empty() -> bool` - Check if the dataset is empty.
pub trait Dataset {
    /// The type of items in the dataset.
    type Item;

    /// Get the item at the given index.
    ///
    /// # Arguments
    /// * `index` - The index of the item to get.
    ///
    /// # Returns
    /// * `Ok(Self::Item)` - The item at the given index.
    /// * `Err(DatasetError)` - The error when getting the item at the given index.
    fn get(&self, index: usize) -> Result<Self::Item, DatasetError>;

    /// Get the number of items in the dataset.
    ///
    /// # Returns
    /// * `Ok(usize)` - The number of items in the dataset.
    /// * `Err(DatasetError)` - The error when getting the number of items in the dataset.
    fn len(&self) -> Result<usize, DatasetError>;

    /// Check if the dataset is empty.
    ///
    /// # Returns
    /// * `Ok(bool)` - `true` if the dataset is empty, `false` otherwise.
    /// * `Err(DatasetError)` - The error when checking if the dataset is empty.
    fn is_empty(&self) -> Result<bool, DatasetError> {
        Ok(self.len()? == 0)
    }
}
