use nove::dataset::Dataset;
use nove_dataset::DatasetError;

/// A simple dataset for testing.
///
/// The type of item in the dataset is `usize`. It contains
/// 100 items, each item is the index of the item.
#[derive(Clone)]
pub struct SimpleDataset {}
impl Dataset for SimpleDataset {
    type Item = usize;
    fn get(&self, index: usize) -> Result<Self::Item, DatasetError> {
        Ok(index)
    }
    fn len(&self) -> Result<usize, DatasetError> {
        Ok(100)
    }
}
