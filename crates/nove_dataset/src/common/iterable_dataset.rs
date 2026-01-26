use crate::{Dataset, DatasetError};

/// IterableDataset could wrap a dataset and offer a way to iterate over it.
///
/// # Note
/// * The dataset that is wrapped by `IterableDataset` must implement `Dataset` trait.
///
/// # Lifetime Parameters
/// * `'a` - The lifetime of the inner dataset.
///
/// # Generic Type Parameters
/// * `D` - The type of the inner dataset.
///
/// # Fields
/// * `inner` - The inner dataset.
/// * `index` - The current index of the iterator.
pub struct IterableDataset<'a, D: Dataset> {
    inner: &'a dyn Dataset<Item = D::Item>,
    index: usize,
}

impl<'a, D: Dataset> IterableDataset<'a, D> {
    /// Create a new `IterableDataset` from a dataset.
    ///
    /// # Arguments
    /// * `dataset` - The dataset to wrap.
    ///
    /// # Returns
    /// * `Ok(Self)` - A new `IterableDataset` instance.
    /// * `Err(DatasetError)` - The error when creating the `IterableDataset` instance.
    pub fn from_dataset(dataset: &'a D) -> Result<Self, DatasetError> {
        Ok(Self {
            inner: dataset as &'a dyn Dataset<Item = D::Item>,
            index: 0,
        })
    }
}

impl<'a, D: Dataset> Dataset for IterableDataset<'a, D> {
    type Item = D::Item;

    fn get(&self, index: usize) -> Result<Self::Item, DatasetError> {
        self.inner.get(index)
    }

    fn len(&self) -> Result<usize, DatasetError> {
        self.inner.len()
    }
}

impl<'a, D: Dataset> Iterator for IterableDataset<'a, D> {
    type Item = Result<D::Item, DatasetError>;

    fn next(&mut self) -> Option<Self::Item> {
        let len = match self.len() {
            Ok(len) => len,
            Err(err) => {
                return Some(Err(err));
            }
        };

        if self.index >= len {
            return None;
        }
        let item = self.inner.get(self.index);
        self.index += 1;
        Some(item)
    }
}

impl<'a, D: Dataset> IntoIterator for &'a IterableDataset<'a, D> {
    type Item = Result<D::Item, DatasetError>;
    type IntoIter = IterableDataset<'a, D>;

    fn into_iter(self) -> Self::IntoIter {
        IterableDataset {
            inner: self.inner,
            index: self.index,
        }
    }
}
