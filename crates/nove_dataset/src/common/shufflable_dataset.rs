use crate::{Dataset, DatasetError};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};

/// A dataset that wraps another dataset and shuffles its indices.
///
/// # Note
/// * The dataset that is wrapped by `ShufflableDataset` must implement `Dataset` trait.
///
/// # Lifetime Parameters
/// * `'a` - The lifetime of the inner dataset.
///
/// # Generic Type Parameters
/// * `D` - The type of the inner dataset.
///
/// # Fields
/// * `inner` - The inner dataset.
/// * `indices` - The shuffled indices of the inner dataset.
///
/// # Examples
/// ```rust
/// use nove::dataset::common::{ShufflableDataset, VecDataset};
///
/// let dataset = VecDataset::from_vec(vec![1usize, 2usize, 3usize]);
/// let mut shufflable_dataset = ShufflableDataset::from_dataset(&dataset).unwrap();
///
/// shufflable_dataset.shuffle(42);
/// ```
pub struct ShufflableDataset<'a, D: Dataset> {
    inner: &'a dyn Dataset<Item = D::Item>,
    indices: Vec<usize>,
}

impl<'a, D: Dataset> ShufflableDataset<'a, D> {
    /// Create a new `ShufflableDataset` from the given dataset.
    ///
    /// # Arguments
    /// * `dataset` - The dataset to wrap.
    ///
    /// # Returns
    /// * `Ok(Self)` - The new `ShufflableDataset` instance.
    /// * `Err(DatasetError)` - If the inner dataset is empty.
    ///
    /// # Examples
    /// ```rust
    /// use nove::dataset::common::{ShufflableDataset, VecDataset};
    ///
    /// let dataset = VecDataset::from_vec(vec![1usize, 2usize, 3usize]);
    /// let mut shufflable_dataset = ShufflableDataset::from_dataset(&dataset).unwrap();
    /// ```
    pub fn from_dataset(dataset: &'a D) -> Result<Self, DatasetError> {
        let len = dataset.len()?;
        if len == 0 {
            return Err(DatasetError::EmptyDataset);
        }

        let indices = (0..len).collect::<Vec<_>>();
        Ok(Self {
            inner: dataset as &'a dyn Dataset<Item = D::Item>,
            indices,
        })
    }

    /// Shuffle the indices of the inner dataset.
    ///
    /// # Arguments
    /// * `seed` - The seed to use for shuffling.
    /// # Examples
    /// ```rust
    /// use nove::dataset::common::{ShufflableDataset, VecDataset};
    ///
    /// let dataset = VecDataset::from_vec(vec![1usize, 2usize, 3usize]);
    /// let mut shufflable_dataset = ShufflableDataset::from_dataset(&dataset).unwrap();
    ///
    /// shufflable_dataset.shuffle(42);
    /// ```
    pub fn shuffle(&mut self, seed: usize) {
        self.indices
            .shuffle(&mut StdRng::seed_from_u64(seed as u64));
    }
}

impl<'a, D: Dataset> Dataset for ShufflableDataset<'a, D> {
    type Item = D::Item;

    fn get(&self, index: usize) -> Result<Self::Item, DatasetError> {
        self.inner.get(self.indices[index])
    }

    fn len(&self) -> Result<usize, DatasetError> {
        Ok(self.indices.len())
    }
}
