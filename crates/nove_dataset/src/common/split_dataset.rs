use crate::{Dataset, DatasetError};

/// A dataset that wraps another dataset and provides a view into a subset of it.
///
/// # Note
/// * The dataset that is wrapped by `SplitDataset` must implement `Dataset` trait.
/// * The split is a view, not a copy. Changes to the original dataset will be reflected.
///
/// # Lifetime Parameters
/// * `'a` - The lifetime of the inner dataset.
///
/// # Generic Type Parameters
/// * `D` - The type of the inner dataset.
///
/// # Fields
/// * `inner` - The inner dataset.
/// * `start` - The starting index of the split.
/// * `len` - The length of the split.
///
/// # Examples
/// ```rust
/// use nove::dataset::common::{SplitDataset, VecDataset};
/// use nove::dataset::Dataset;
///
/// let dataset = VecDataset::from_vec(vec![1usize, 2usize, 3usize, 4usize, 5usize]);
///
/// // Create a split from index 1 with length 3
/// let split = SplitDataset::from_dataset(&dataset, 1, 3).unwrap();
/// assert_eq!(split.len().unwrap(), 3);
/// assert_eq!(split.get(0).unwrap(), 2);
/// ```
pub struct SplitDataset<'a, D: Dataset> {
    inner: &'a dyn Dataset<Item = D::Item>,
    start: usize,
    len: usize,
}

impl<'a, D: Dataset> SplitDataset<'a, D> {
    /// Create a new `SplitDataset` from the given dataset with specified start index and length.
    ///
    /// # Arguments
    /// * `dataset` - The dataset to wrap.
    /// * `start` - The starting index of the split.
    /// * `len` - The length of the split.
    ///
    /// # Returns
    /// * `Ok(Self)` - A new `SplitDataset` instance.
    /// * `Err(DatasetError)` - The error when creating the `SplitDataset` instance.
    ///
    /// # Examples
    /// ```rust
    /// use nove::dataset::common::{SplitDataset, VecDataset};
    ///
    /// let dataset = VecDataset::from_vec(vec![1usize, 2usize, 3usize, 4usize, 5usize]);
    /// let split = SplitDataset::from_dataset(&dataset, 1, 3).unwrap();
    /// ```
    pub fn from_dataset(dataset: &'a D, start: usize, len: usize) -> Result<Self, DatasetError> {
        let dataset_len = dataset.len()?;
        if dataset_len == 0 {
            return Err(DatasetError::EmptyDataset);
        }
        if start >= dataset_len {
            return Err(DatasetError::IndexOutOfBounds(start, dataset_len));
        }
        let actual_len = len.min(dataset_len - start);
        if actual_len == 0 {
            return Err(DatasetError::OtherError("Split length is zero".to_string()));
        }

        Ok(Self {
            inner: dataset as &'a dyn Dataset<Item = D::Item>,
            start,
            len: actual_len,
        })
    }

    /// Split the dataset into multiple parts according to the given ratios.
    ///
    /// # Arguments
    /// * `dataset` - The dataset to split.
    /// * `ratios` - The ratios for each split. Must sum to 1.0.
    ///
    /// # Returns
    /// * `Ok(Vec<Self>)` - A vector of `SplitDataset` instances.
    /// * `Err(DatasetError)` - The error when splitting the dataset.
    ///
    /// # Examples
    /// ```rust
    /// use nove::dataset::common::{SplitDataset, VecDataset};
    ///
    /// let dataset = VecDataset::from_vec(vec![1usize, 2usize, 3usize, 4usize, 5usize, 6usize]);
    /// let splits = SplitDataset::split(&dataset, &[0.5, 0.3, 0.2]).unwrap();
    /// assert_eq!(splits.len(), 3);
    /// ```
    pub fn split(dataset: &'a D, ratios: &[f64]) -> Result<Vec<Self>, DatasetError> {
        if ratios.is_empty() {
            return Err(DatasetError::OtherError(
                "Ratios cannot be empty".to_string(),
            ));
        }

        let total_ratio: f64 = ratios.iter().sum();
        if (total_ratio - 1.0).abs() > 1e-6 {
            return Err(DatasetError::OtherError(
                "Ratios must sum to 1.0".to_string(),
            ));
        }

        let dataset_len = dataset.len()?;
        if dataset_len == 0 {
            return Err(DatasetError::EmptyDataset);
        }

        let mut splits = Vec::with_capacity(ratios.len());
        let mut current_start = 0;

        for ratio in ratios {
            let split_len = (*ratio * dataset_len as f64).round() as usize;
            if split_len == 0 {
                continue;
            }

            let split = Self::from_dataset(dataset, current_start, split_len)?;
            splits.push(split);
            current_start += split_len;
        }

        Ok(splits)
    }

    /// Split the dataset into training and testing sets.
    ///
    /// # Arguments
    /// * `dataset` - The dataset to split.
    /// * `test_ratio` - The ratio of the testing set. Must be between 0 and 1.
    ///
    /// # Returns
    /// * `Ok((Self, Self))` - A tuple of (training_set, testing_set).
    /// * `Err(DatasetError)` - The error when splitting the dataset.
    ///
    /// # Examples
    /// ```rust
    /// use nove::dataset::common::{SplitDataset, VecDataset};
    ///
    /// let dataset = VecDataset::from_vec(vec![1usize, 2usize, 3usize, 4usize, 5usize, 6usize]);
    /// let (train, test) = SplitDataset::train_test_split(&dataset, 0.3).unwrap();
    /// ```
    pub fn train_test_split(dataset: &'a D, test_ratio: f64) -> Result<(Self, Self), DatasetError> {
        if test_ratio <= 0.0 || test_ratio >= 1.0 {
            return Err(DatasetError::OtherError(
                "Test ratio must be between 0 and 1".to_string(),
            ));
        }

        let splits = Self::split(dataset, &[1.0 - test_ratio, test_ratio])?;
        if splits.len() < 2 {
            return Err(DatasetError::OtherError(
                "Failed to create train/test split".to_string(),
            ));
        }

        let mut iter = splits.into_iter();
        let train = iter.next().unwrap();
        let test = iter.next().unwrap();
        Ok((train, test))
    }

    /// Split the dataset into training, validation, and testing sets.
    ///
    /// # Arguments
    /// * `dataset` - The dataset to split.
    /// * `val_ratio` - The ratio of the validation set. Must be greater than 0.
    /// * `test_ratio` - The ratio of the testing set. Must be greater than 0.
    ///
    /// # Returns
    /// * `Ok((Self, Self, Self))` - A tuple of (training_set, validation_set, testing_set).
    /// * `Err(DatasetError)` - The error when splitting the dataset.
    ///
    /// # Examples
    /// ```rust
    /// use nove::dataset::common::{SplitDataset, VecDataset};
    ///
    /// let dataset = VecDataset::from_vec(vec![1usize, 2usize, 3usize, 4usize, 5usize, 6usize]);
    /// let (train, val, test) = SplitDataset::train_val_test_split(&dataset, 0.2, 0.2).unwrap();
    /// ```
    pub fn train_val_test_split(
        dataset: &'a D,
        val_ratio: f64,
        test_ratio: f64,
    ) -> Result<(Self, Self, Self), DatasetError> {
        if val_ratio <= 0.0 || test_ratio <= 0.0 || val_ratio + test_ratio >= 1.0 {
            return Err(DatasetError::OtherError(
                "Invalid validation/test ratios".to_string(),
            ));
        }

        let train_ratio = 1.0 - val_ratio - test_ratio;
        let splits = Self::split(dataset, &[train_ratio, val_ratio, test_ratio])?;
        if splits.len() < 3 {
            return Err(DatasetError::OtherError(
                "Failed to create train/val/test split".to_string(),
            ));
        }

        let mut iter = splits.into_iter();
        Ok((
            iter.next().unwrap(),
            iter.next().unwrap(),
            iter.next().unwrap(),
        ))
    }
}

impl<'a, D: Dataset> Dataset for SplitDataset<'a, D> {
    type Item = D::Item;

    fn get(&self, index: usize) -> Result<Self::Item, DatasetError> {
        if index >= self.len {
            return Err(DatasetError::IndexOutOfBounds(index, self.len));
        }
        self.inner.get(self.start + index)
    }

    fn len(&self) -> Result<usize, DatasetError> {
        Ok(self.len)
    }
}
