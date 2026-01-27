use crate::{Dataset, DatasetError};

/// A dataset that stores items in a vector.
///
/// # Notes
/// * The `T` type needs to implement the `Clone` trait.
///
/// # Generic Type Parameters
/// * `T` - The type of items in the dataset.
///
/// # Fields
/// * `items` - The vector of items in the dataset.
///
/// # Examples
/// ```rust
/// use nove::dataset::common::VecDataset;
///
/// let dataset = VecDataset::from_vec(vec![1usize, 2usize, 3usize]);
/// ```
#[derive(Clone)]
pub struct VecDataset<T> {
    items: Vec<T>,
}

impl<T> VecDataset<T> {
    /// Create a new `VecDataset` from a vector of items.
    ///
    /// # Arguments
    /// * `items` - The vector of items to create the dataset from.
    ///
    /// # Returns
    /// * `Self` - The created `VecDataset`.
    ///
    /// # Examples
    /// ```rust
    /// use nove::dataset::common::VecDataset;
    ///
    /// let dataset = VecDataset::from_vec(vec![1usize, 2usize, 3usize]);
    /// ```
    pub fn from_vec(items: Vec<T>) -> Self {
        Self { items }
    }
}

impl<T: Clone> Dataset for VecDataset<T> {
    type Item = T;

    fn get(&self, index: usize) -> Result<Self::Item, DatasetError> {
        self.items
            .get(index)
            .ok_or_else(|| DatasetError::IndexOutOfBounds(index, self.items.len()))
            .cloned()
    }

    fn len(&self) -> Result<usize, DatasetError> {
        Ok(self.items.len())
    }
}
