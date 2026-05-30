mod util;

use nove::dataset::{Dataset, common::IterableDataset};
use util::SimpleDataset;

#[test]
fn test_immut_iterable_dataset() {
    let dataset = SimpleDataset {};
    let iterable_dataset = IterableDataset::from_dataset(&dataset).unwrap();

    let simulated_items = (0..iterable_dataset.len().unwrap()).collect::<Vec<_>>();

    // First iteration
    for (simulated_index, item) in (&iterable_dataset).into_iter().enumerate() {
        assert_eq!(item.unwrap(), simulated_items[simulated_index]);
    }

    // Second iteration
    for (simulated_index, item) in iterable_dataset.into_iter().enumerate() {
        assert_eq!(item.unwrap(), simulated_items[simulated_index]);
    }
}
