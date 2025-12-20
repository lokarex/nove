use crate::dataset::Dataset;
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};

/// A dataset that wraps another dataset and shuffles its indices.
///
/// # Note
/// * The dataset that is wrapped by `ShuffledDataset` must implement `Dataset` trait.
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
pub struct ShuffledDataset<'a, D: Dataset> {
    inner: &'a dyn Dataset<Item = D::Item>,
    indices: Vec<usize>,
}

impl<'a, D: Dataset> ShuffledDataset<'a, D> {
    /// Create a new `ShuffledDataset` from the given dataset.
    ///
    /// # Arguments
    /// * `dataset` - The dataset to wrap.
    ///
    /// # Returns
    /// A new `ShuffledDataset` instance.
    pub fn from_dataset(dataset: &'a D) -> Self {
        let len = dataset.len();
        let indices = (0..len).collect::<Vec<_>>();
        Self {
            inner: dataset as &'a dyn Dataset<Item = D::Item>,
            indices,
        }
    }

    /// Shuffle the indices of the inner dataset.
    ///
    /// # Arguments
    /// * `seed` - The seed to use for shuffling.
    pub fn shuffle(&mut self, seed: usize) {
        self.indices
            .shuffle(&mut StdRng::seed_from_u64(seed as u64));
    }
}

impl<'a, D: Dataset> Dataset for ShuffledDataset<'a, D> {
    type Item = D::Item;

    fn get(&self, index: usize) -> Self::Item {
        self.inner.get(self.indices[index as usize])
    }

    fn len(&self) -> usize {
        self.indices.len() as usize
    }
}
