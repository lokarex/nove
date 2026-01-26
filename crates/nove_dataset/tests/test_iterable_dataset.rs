mod util;

use nove::dataset::{Dataset, common::IterableDataset};
use util::SimpleDataset;

#[test]
fn test_immut_iterable_dataset() {
    let dataset = SimpleDataset {};
    let iterable_dataset = IterableDataset::from_dataset(&dataset).unwrap();

    let simulated_items = (0..iterable_dataset.len().unwrap()).collect::<Vec<_>>();

    // First iteration
    let mut simulated_index = 0;
    for item in &iterable_dataset {
        assert_eq!(item.unwrap(), simulated_items[simulated_index]);
        simulated_index += 1;
    }

    // Second iteration
    let mut simulated_index = 0;
    for item in iterable_dataset {
        assert_eq!(item.unwrap(), simulated_items[simulated_index]);
        simulated_index += 1;
    }
}
