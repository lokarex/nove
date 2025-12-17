use std::{
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
};

use crate::dataset::Dataset;
use bincode::{Decode, Encode, config};

/// PersistedDataset could wrap another dataset and persist the items in the inner
/// dataset to a file and load them back when needed.
///
/// # Note
/// * The dataset that is wrapped by `PersistedDataset` must implement `Dataset` trait.
/// * The D::Item must implement Encode + Decode<()>.
/// * When loading the dataset from the file, the type of the inner dataset should be explicitly
///   specified and must be the same as the type when saving.
///
/// # Warning
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
pub struct PersistedDataset<'a, D: Dataset>
where
    D::Item: Encode + Decode<()>,
{
    inner: Option<&'a dyn Dataset<Item = D::Item>>,
    dataset_file: Option<File>,
}

impl<'a, D: Dataset> PersistedDataset<'a, D>
where
    D::Item: Encode + Decode<()>,
{
    /// Create a new `PersistedDataset` from the given dataset.
    ///
    /// # Arguments
    /// * `dataset` - The dataset to wrap.
    ///
    /// # Returns
    /// A new `PersistedDataset` instance.
    pub fn from_dataset(dataset: &'a D) -> Self {
        Self {
            inner: Some(dataset as &'a dyn Dataset<Item = D::Item>),
            dataset_file: None,
        }
    }

    /// Save the items in the inner dataset to the given file.
    ///
    /// # Arguments
    /// * `file_path` - The path to the file to save to.
    pub fn save_as_file(&self, file_path: &str) {
        let mut file = File::create(file_path).unwrap();
        let item_size = std::mem::size_of::<D::Item>();
        for i in 0..self.inner.unwrap().len() {
            file.seek(SeekFrom::End(0)).unwrap();
            let item = self.inner.unwrap().get(i).unwrap();
            let mut data = bincode::encode_to_vec(item, config::standard()).unwrap();

            // Force encode to fixed size, pad with 0 if necessary.
            if data.len() < item_size {
                data.resize(item_size, 0);
            }
            file.write_all(&data).unwrap();
        }
    }

    /// Load a new `PersistedDataset` instance from the given file.
    ///
    /// # Arguments
    /// * `file_path` - The path to the file to load from.
    ///
    /// # Returns
    /// A new `PersistedDataset` instance.
    pub fn load_from_file(file_path: &str) -> Self {
        Self {
            inner: None,
            dataset_file: Some(File::open(file_path).unwrap()),
        }
    }
}

impl<'a, D: Dataset> Dataset for PersistedDataset<'a, D>
where
    D::Item: Encode + Decode<()>,
{
    type Item = D::Item;

    fn get(&self, index: u64) -> Option<Self::Item> {
        if index >= self.len() {
            return None;
        }

        // If the PersistedDataset is created from another dataset,
        // we just delegate the get call to the inner dataset.
        if let Some(inner) = self.inner {
            return inner.get(index);
        }

        // If the PersistedDataset is created from a file,
        // we read the item from the file.
        if let Some(file) = &self.dataset_file {
            let mut file = file.try_clone().unwrap();
            file.seek(SeekFrom::Start(
                index * std::mem::size_of::<D::Item>() as u64,
            ))
            .unwrap();

            let mut data = vec![0u8; std::mem::size_of::<D::Item>()];
            file.read_exact(&mut data).unwrap();
            let (item, _) = bincode::decode_from_slice(&data, config::standard()).unwrap();
            return Some(item);
        }

        return None;
    }

    fn len(&self) -> u64 {
        // If the PersistedDataset is created from another dataset,
        // we just delegate the len call to the inner dataset.
        if let Some(inner) = self.inner {
            return inner.len();
        }

        // If the PersistedDataset is created from a file,
        // we calculate the item count from the file size.
        if let Some(file) = &self.dataset_file {
            let file_size = file.metadata().unwrap().len();
            let item_size = std::mem::size_of::<D::Item>() as u64;
            let item_count = file_size / item_size;

            return item_count;
        }

        return 0;
    }
}
