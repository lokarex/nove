use crate::{Dataset, DatasetError};
use std::ops::Fn;

/// A dataset that wraps another dataset and applies a transformation function to each item.
///
/// # Notes
/// * The dataset that is wrapped by `AugmentedDataset` must implement `Dataset` trait.
/// * The transformation function is applied lazily when items are accessed.
/// * This is useful for data augmentation, preprocessing, or any item-wise transformation.
///
/// # Lifetime Parameters
/// * `'a` - The lifetime of the inner dataset.
///
/// # Generic Type Parameters
/// * `D` - The type of the inner dataset.
/// * `F` - The type of the transformation function.
/// * `T` - The type of the transformed item.
///
/// # Fields
/// * `inner` - The inner dataset.
/// * `transform` - The transformation function to apply to each item.
///
/// # Examples
/// ```no_run
/// use nove::dataset::common::{AugmentedDataset, VecDataset};
///
/// let dataset = VecDataset::from_vec(vec![1usize, 2usize, 3usize]);
/// let augmented = AugmentedDataset::from_dataset(&dataset, |x| x * 2).unwrap();
/// ```
pub struct AugmentedDataset<'a, D: Dataset, F: Fn(D::Item) -> T, T> {
    inner: &'a D,
    transform: F,
    _phantom: std::marker::PhantomData<T>,
}

impl<'a, D: Dataset, F: Fn(D::Item) -> T, T> AugmentedDataset<'a, D, F, T> {
    /// Create a new `AugmentedDataset` from the given dataset and transformation function.
    ///
    /// # Arguments
    /// * `dataset` - The dataset to wrap.
    /// * `transform` - The transformation function to apply to each item.
    ///
    /// # Returns
    /// * `Ok(Self)` - A new `AugmentedDataset` instance.
    /// * `Err(DatasetError)` - The error when creating the `AugmentedDataset` instance.
    ///
    /// # Examples
    /// ```
    /// use nove::dataset::common::{AugmentedDataset, VecDataset};
    /// use nove::dataset::Dataset;
    ///
    /// let dataset = VecDataset::from_vec(vec![1usize, 2usize, 3usize]);
    /// let augmented = AugmentedDataset::from_dataset(&dataset, |x| x * 2).unwrap();
    ///
    /// assert_eq!(augmented.get(0).unwrap(), 2);
    /// assert_eq!(augmented.get(1).unwrap(), 4);
    /// assert_eq!(augmented.get(2).unwrap(), 6);
    /// ```
    pub fn from_dataset(dataset: &'a D, transform: F) -> Result<Self, DatasetError> {
        if dataset.len()? == 0 {
            return Err(DatasetError::EmptyDataset);
        }

        Ok(Self {
            inner: dataset,
            transform,
            _phantom: std::marker::PhantomData,
        })
    }
}

impl<'a, D: Dataset, F: Fn(D::Item) -> T, T> Dataset for AugmentedDataset<'a, D, F, T> {
    type Item = T;

    fn get(&self, index: usize) -> Result<Self::Item, DatasetError> {
        let item = self.inner.get(index)?;
        Ok((self.transform)(item))
    }

    fn len(&self) -> Result<usize, DatasetError> {
        self.inner.len()
    }
}
