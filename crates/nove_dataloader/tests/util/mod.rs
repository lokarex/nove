use nove::dataset::Dataset;

/// A simple dataset for testing.
///
/// The type of item in the dataset is `usize`. It contains
/// 100 items, each item is the index of the item.
pub struct SimpleDataset {}
impl Dataset for SimpleDataset {
    type Item = usize;
    fn get(&self, index: usize) -> Self::Item {
        index
    }
    fn len(&self) -> usize {
        100
    }
}
