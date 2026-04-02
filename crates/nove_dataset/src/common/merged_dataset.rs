use crate::{Dataset, DatasetError};

/// A dataset that merges multiple datasets into a single view.
///
/// # Notes
/// * All datasets that are merged must implement `Dataset` trait with the same `Item` type.
/// * The merge is a view, not a copy. Changes to the original datasets will be reflected.
/// * Items are accessed in order: first all items from the first dataset, then the second, etc.
///
/// # Lifetime Parameters
/// * `'a` - The lifetime of the merged datasets.
///
/// # Generic Type Parameters
/// * `D` - The type of the merged datasets.
///
/// # Fields
/// * `datasets` - The vector of datasets to merge.
/// * `offsets` - The cumulative offsets for each dataset (precomputed for efficiency).
///
/// # Examples
/// ```no_run
/// use nove::dataset::common::{MergedDataset, VecDataset};
///
/// let dataset1 = VecDataset::from_vec(vec![1usize, 2usize, 3usize]);
/// let dataset2 = VecDataset::from_vec(vec![4usize, 5usize, 6usize]);
///
/// let merged = MergedDataset::from_datasets(&[&dataset1, &dataset2]).unwrap();
/// ```
pub struct MergedDataset<'a, D: Dataset> {
    datasets: Vec<&'a dyn Dataset<Item = D::Item>>,
    offsets: Vec<usize>,
    total_len: usize,
}

impl<'a, D: Dataset> MergedDataset<'a, D> {
    /// Create a new `MergedDataset` from the given datasets.
    ///
    /// # Arguments
    /// * `datasets` - The slice of datasets to merge.
    ///
    /// # Returns
    /// * `Ok(Self)` - A new `MergedDataset` instance.
    /// * `Err(DatasetError)` - The error when creating the `MergedDataset` instance.
    ///
    /// # Examples
    /// ```
    /// use nove::dataset::common::{MergedDataset, VecDataset};
    /// use nove::dataset::Dataset;
    ///
    /// let dataset1 = VecDataset::from_vec(vec![1usize, 2usize, 3usize]);
    /// let dataset2 = VecDataset::from_vec(vec![4usize, 5usize, 6usize]);
    ///
    /// let datasets = [&dataset1, &dataset2];
    /// let merged = MergedDataset::from_datasets(&datasets).unwrap();
    /// assert_eq!(merged.len().unwrap(), 6);
    /// assert_eq!(merged.get(0).unwrap(), 1);
    /// assert_eq!(merged.get(1).unwrap(), 2);
    /// assert_eq!(merged.get(2).unwrap(), 3);
    /// assert_eq!(merged.get(3).unwrap(), 4);
    /// assert_eq!(merged.get(4).unwrap(), 5);
    /// assert_eq!(merged.get(5).unwrap(), 6);
    /// ```
    pub fn from_datasets(datasets: &'a [&D]) -> Result<Self, DatasetError> {
        if datasets.is_empty() {
            return Err(DatasetError::EmptyDataset);
        }

        let mut merged_datasets = Vec::with_capacity(datasets.len());
        let mut offsets = Vec::with_capacity(datasets.len());
        let mut total_len = 0;

        for dataset in datasets {
            let len = dataset.len()?;
            if len == 0 {
                continue;
            }
            offsets.push(total_len);
            total_len += len;
            merged_datasets.push(*dataset as &'a dyn Dataset<Item = D::Item>);
        }

        if merged_datasets.is_empty() {
            return Err(DatasetError::EmptyDataset);
        }

        Ok(Self {
            datasets: merged_datasets,
            offsets,
            total_len,
        })
    }

    /// Create a new `MergedDataset` by merging two datasets.
    ///
    /// # Arguments
    /// * `dataset1` - The first dataset.
    /// * `dataset2` - The second dataset.
    ///
    /// # Returns
    /// * `Ok(Self)` - A new `MergedDataset` instance.
    /// * `Err(DatasetError)` - The error when creating the `MergedDataset` instance.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::dataset::common::{MergedDataset, VecDataset};
    ///
    /// let dataset1 = VecDataset::from_vec(vec![1usize, 2usize, 3usize]);
    /// let dataset2 = VecDataset::from_vec(vec![4usize, 5usize, 6usize]);
    ///
    /// let merged = MergedDataset::from_two_datasets(&dataset1, &dataset2).unwrap();
    /// ```
    pub fn from_two_datasets(dataset1: &'a D, dataset2: &'a D) -> Result<Self, DatasetError> {
        let len1 = dataset1.len()?;
        let len2 = dataset2.len()?;

        if len1 == 0 && len2 == 0 {
            return Err(DatasetError::EmptyDataset);
        }

        let mut datasets = Vec::with_capacity(2);
        let mut offsets = Vec::with_capacity(2);
        let mut total_len = 0;

        if len1 > 0 {
            offsets.push(0);
            total_len = len1;
            datasets.push(dataset1 as &'a dyn Dataset<Item = D::Item>);
        }

        if len2 > 0 {
            offsets.push(total_len);
            total_len += len2;
            datasets.push(dataset2 as &'a dyn Dataset<Item = D::Item>);
        }

        Ok(Self {
            datasets,
            offsets,
            total_len,
        })
    }

    /// Get the number of datasets in this merged dataset.
    ///
    /// # Returns
    /// * `usize` - The number of datasets.
    ///
    /// # Examples
    /// ```rust
    /// use nove::dataset::common::{MergedDataset, VecDataset};
    ///
    /// let dataset1 = VecDataset::from_vec(vec![1usize, 2usize, 3usize]);
    /// let dataset2 = VecDataset::from_vec(vec![4usize, 5usize, 6usize]);
    ///
    /// let datasets = [&dataset1, &dataset2];
    /// let merged = MergedDataset::from_datasets(&datasets).unwrap();
    /// assert_eq!(merged.dataset_count(), 2);
    /// ```
    pub fn dataset_count(&self) -> usize {
        self.datasets.len()
    }

    /// Find which dataset and local index a global index belongs to.
    fn find_dataset_index(&self, global_index: usize) -> Option<(usize, usize)> {
        if global_index >= self.total_len {
            return None;
        }

        let mut left = 0;
        let mut right = self.offsets.len();

        while left < right {
            let mid = left + (right - left) / 2;
            if self.offsets[mid] <= global_index {
                if mid + 1 == self.offsets.len() || self.offsets[mid + 1] > global_index {
                    let local_index = global_index - self.offsets[mid];
                    return Some((mid, local_index));
                }
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        None
    }
}

impl<'a, D: Dataset> Dataset for MergedDataset<'a, D> {
    type Item = D::Item;

    fn get(&self, index: usize) -> Result<Self::Item, DatasetError> {
        if index >= self.total_len {
            return Err(DatasetError::IndexOutOfBounds(index, self.total_len));
        }

        let (dataset_idx, local_idx) = self
            .find_dataset_index(index)
            .ok_or(DatasetError::InvalidIndex(index))?;

        self.datasets[dataset_idx].get(local_idx)
    }

    fn len(&self) -> Result<usize, DatasetError> {
        Ok(self.total_len)
    }
}
