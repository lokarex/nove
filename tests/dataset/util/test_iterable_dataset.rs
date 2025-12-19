use crate::dataset::util::simple_dataset;
use nove::dataset::{Dataset, util::IterableDataset};
use simple_dataset::SimpleDataset;

#[test]
fn test_immut_iterable_dataset() {
    let dataset = SimpleDataset {};
    let iterable_dataset = IterableDataset::from_dataset(&dataset);

    let simulated_items = (0..iterable_dataset.len()).collect::<Vec<_>>();

    // First iteration
    let mut simulated_index = 0;
    for item in &iterable_dataset {
        assert_eq!(item, simulated_items[simulated_index]);
        simulated_index += 1;
    }

    // Second iteration
    let mut simulated_index = 0;
    for item in iterable_dataset {
        assert_eq!(item, simulated_items[simulated_index]);
        simulated_index += 1;
    }
}
