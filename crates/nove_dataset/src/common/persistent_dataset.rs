use std::{
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
};

use crate::{Dataset, DatasetError};
use bincode::{Decode, Encode, config};

/// PersistentDataset could wrap another dataset and persist the items in the inner
/// dataset to a file and load them back when needed.
///
/// # Note
/// * The dataset that is wrapped by `PersistentDataset` must implement `Dataset` trait.
/// * The D::Item must implement Encode + Decode<()>.
/// * When loading the dataset from the file, the type of the inner dataset should be explicitly
///   specified and must be the same as the type when saving.
/// * When saving the dataset to the file, only the items in the inner
///   dataset are persisted. So, after loading the dataset from the file,
///   the other fields of the inner dataset are lost.
///
/// # Lifetime Parameters
/// * `'a` - The lifetime of the inner dataset.
///
/// # Generic Type Parameters
/// * `D` - The type of the inner dataset.
///
/// # Fields
/// * `inner` - The inner dataset.
/// * `dataset_file` - The file to persist the items to.
///
/// # Examples
/// ```rust
/// use tempfile::TempDir;
/// use nove::dataset::common::{PersistentDataset, VecDataset};
///
/// // Create a temporary directory to store the dataset file.
/// let temp_dir = TempDir::new().unwrap();
/// let dataset_file = temp_dir.path().join("dataset.bin");
///
/// let dataset = VecDataset::from_vec(vec![1usize, 2usize, 3usize]);
/// let persistent_dataset = PersistentDataset::from_dataset(&dataset).unwrap();
///
/// // Save the dataset to the file.
/// persistent_dataset.save_to_file(dataset_file.to_str().unwrap()).unwrap();
///
/// // Load the dataset from the file.
/// let loaded_dataset: PersistentDataset<'_, VecDataset<usize>> = PersistentDataset::load_from_file(
///     dataset_file.to_str().unwrap(),
/// ).unwrap();
/// ```
pub struct PersistentDataset<'a, D: Dataset>
where
    D::Item: Encode + Decode<()>,
{
    inner: Option<&'a dyn Dataset<Item = D::Item>>,
    dataset_file: Option<File>,
}

impl<'a, D: Dataset> PersistentDataset<'a, D>
where
    D::Item: Encode + Decode<()>,
{
    /// Create a new `PersistentDataset` from the given dataset.
    ///
    /// # Arguments
    /// * `dataset` - The dataset to wrap.
    ///
    /// # Returns
    /// * `Ok(Self)` - A new `PersistentDataset` instance.
    /// * `Err(DatasetError)` - The error when creating the `PersistentDataset` instance.
    ///
    /// # Examples
    /// ```rust
    /// use nove::dataset::common::{PersistentDataset, VecDataset};
    ///
    /// let dataset = VecDataset::from_vec(vec![1usize, 2usize, 3usize]);
    /// let persistent_dataset = PersistentDataset::from_dataset(&dataset).unwrap();
    /// ```
    pub fn from_dataset(dataset: &'a D) -> Result<Self, DatasetError> {
        if dataset.len()? == 0 {
            return Err(DatasetError::EmptyDataset);
        }

        Ok(Self {
            inner: Some(dataset as &'a dyn Dataset<Item = D::Item>),
            dataset_file: None,
        })
    }

    /// Save the items in the inner dataset to the given file.
    ///
    /// # Arguments
    /// * `file_path` - The path to the file to save to.
    ///
    /// # Returns
    /// * `Ok(())` - The dataset is saved to the file successfully.
    /// * `Err(DatasetError)` - The error when saving the dataset to the file.
    ///
    /// # Examples
    /// ```rust
    /// use tempfile::TempDir;
    /// use nove::dataset::common::{PersistentDataset, VecDataset};
    ///
    /// // Create a temporary directory to store the dataset file.
    /// let temp_dir = TempDir::new().unwrap();
    /// let dataset_file = temp_dir.path().join("dataset.bin");
    ///
    /// let dataset = VecDataset::from_vec(vec![1usize, 2usize, 3usize]);
    /// let persistent_dataset = PersistentDataset::from_dataset(&dataset).unwrap();
    ///
    /// // Save the dataset to the file.
    /// persistent_dataset.save_to_file(dataset_file.to_str().unwrap()).unwrap();
    /// ```
    pub fn save_to_file(&self, file_path: &str) -> Result<(), DatasetError> {
        let mut file = File::create(file_path)?;
        let item_size = std::mem::size_of::<D::Item>();
        for i in 0..self.len()? {
            file.seek(SeekFrom::End(0))?;
            let item = self.get(i)?;
            let mut data = bincode::encode_to_vec(item, config::standard())?;

            // Force encode to fixed size, pad with 0 if necessary.
            if data.len() < item_size {
                data.resize(item_size, 0);
            }
            file.write_all(&data)?;
        }
        Ok(())
    }

    /// Load a new `PersistentDataset` instance from the given file.
    ///
    /// # Arguments
    /// * `file_path` - The path to the file to load from.
    ///
    /// # Returns
    /// * `Ok(Self)` - A new `PersistentDataset` instance.
    /// * `Err(DatasetError)` - The error when loading the `PersistentDataset` instance.
    ///
    /// # Examples
    /// ```rust
    /// use tempfile::TempDir;
    /// use nove::dataset::common::{PersistentDataset, VecDataset};
    ///
    /// // Create a temporary directory to store the dataset file.
    /// let temp_dir = TempDir::new().unwrap();
    /// let dataset_file = temp_dir.path().join("dataset.bin");
    ///
    /// let dataset = VecDataset::from_vec(vec![1usize, 2usize, 3usize]);
    /// let persistent_dataset = PersistentDataset::from_dataset(&dataset).unwrap();
    ///
    /// // Save the dataset to the file.
    /// persistent_dataset.save_to_file(dataset_file.to_str().unwrap()).unwrap();
    ///
    /// // Load the dataset from the file.
    /// let loaded_dataset: PersistentDataset<'_, VecDataset<usize>> = PersistentDataset::load_from_file(
    ///     dataset_file.to_str().unwrap(),
    /// ).unwrap();
    /// ```
    pub fn load_from_file(file_path: &str) -> Result<Self, DatasetError> {
        Ok(Self {
            inner: None,
            dataset_file: Some(File::open(file_path)?),
        })
    }
}

impl<'a, D: Dataset> Dataset for PersistentDataset<'a, D>
where
    D::Item: Encode + Decode<()>,
{
    type Item = D::Item;

    fn get(&self, index: usize) -> Result<Self::Item, DatasetError> {
        // If the PersistentDataset is created from another dataset,
        // we just delegate the get call to the inner dataset.
        if let Some(inner) = self.inner {
            return inner.get(index);
        }

        if self.len()? <= index {
            return Err(DatasetError::IndexOutOfBounds(index, self.len()?));
        }

        // If the PersistentDataset is created from a file,
        // we read the item from the file.
        if let Some(file) = &self.dataset_file {
            let mut file = file.try_clone()?;
            file.seek(SeekFrom::Start(
                index as u64 * std::mem::size_of::<D::Item>() as u64,
            ))?;

            let mut data = vec![0u8; std::mem::size_of::<D::Item>()];
            file.read_exact(&mut data)?;
            let (item, _) = bincode::decode_from_slice(&data, config::standard())?;
            return Ok(item);
        }

        unreachable!(
            "PersistentDataset is not initialized properly: both inner dataset and file are None"
        );
    }

    fn len(&self) -> Result<usize, DatasetError> {
        // If the PersistentDataset is created from another dataset,
        // we just delegate the len call to the inner dataset.
        if let Some(inner) = self.inner {
            return inner.len();
        }

        // If the PersistentDataset is created from a file,
        // we calculate the item count from the file size.
        if let Some(file) = &self.dataset_file {
            let file_size = file.metadata()?.len();
            let item_size = std::mem::size_of::<D::Item>();
            let item_count = file_size / item_size as u64;

            return Ok(item_count as usize);
        }

        unreachable!(
            "PersistentDataset is not initialized properly: both inner dataset and file are None"
        );
    }
}
