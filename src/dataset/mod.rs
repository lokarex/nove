//! The `dataset` module defines a generic trait `Dataset` for datasets,
//! provides some real datasets and achieves a few practical `Dataset` structs
//! to handle datasets.

pub mod util;

/// The `Dataset` trait defines a generic interface for datasets.
/// Every dataset should implement this trait.
///
/// # Required Type Parameters
/// * `Item` - The type of items in the dataset.
///
/// # Required Methods
/// * `get(index: u64) -> Self::Item` - Get the item at the given index.
/// * `len() -> u64` - Get the number of items in the dataset.
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
    /// The item at the given index.
    fn get(&self, index: u64) -> Self::Item;

    /// Get the number of items in the dataset.
    ///
    /// # Returns
    /// The number of items in the dataset.
    fn len(&self) -> u64;

    /// Check if the dataset is empty.
    ///
    /// # Returns
    /// `true` if the dataset is empty, `false` otherwise.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
