use nove::dataset::Dataset;

/// A simple dataset for testing.
///
/// The type of item in the dataset is `u64`. It contains
/// 100 items, each item is the index of the item.
pub struct SimpleDataset {}
impl Dataset for SimpleDataset {
    type Item = u64;
    fn get(&self, index: u64) -> Option<Self::Item> {
        Some(index)
    }
    fn len(&self) -> u64 {
        100
    }
}
