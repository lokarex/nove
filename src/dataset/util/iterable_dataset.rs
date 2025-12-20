use crate::dataset::Dataset;

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
/// * `len` - The number of items in the dataset.
pub struct IterableDataset<'a, D: Dataset> {
    inner: &'a dyn Dataset<Item = D::Item>,
    index: usize,
    len: usize,
}

impl<'a, D: Dataset> IterableDataset<'a, D> {
    /// Create a new `IterableDataset` from a dataset.
    ///
    /// # Arguments
    /// * `dataset` - The dataset to wrap.
    ///
    /// # Returns
    /// A new `IterableDataset` instance.
    pub fn from_dataset(dataset: &'a D) -> Self {
        Self {
            inner: dataset as &'a dyn Dataset<Item = D::Item>,
            index: 0,
            len: dataset.len(),
        }
    }
}

impl<'a, D: Dataset> Dataset for IterableDataset<'a, D> {
    type Item = D::Item;

    fn get(&self, index: usize) -> Self::Item {
        self.inner.get(index)
    }

    fn len(&self) -> usize {
        self.len
    }
}

impl<'a, D: Dataset> Iterator for IterableDataset<'a, D> {
    type Item = D::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.len {
            return None;
        }
        let item = self.inner.get(self.index);
        self.index += 1;
        Some(item)
    }
}

impl<'a, D: Dataset> IntoIterator for &'a IterableDataset<'a, D> {
    type Item = D::Item;
    type IntoIter = IterableDataset<'a, D>;

    fn into_iter(self) -> Self::IntoIter {
        IterableDataset {
            inner: self.inner,
            index: self.index,
            len: self.len,
        }
    }
}
