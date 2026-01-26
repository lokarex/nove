mod util;

use nove::dataset::{Dataset, common::ShufflableDataset};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use util::SimpleDataset;

#[test]
fn test_shufflable_dataset_with_simple_dataset() {
    // Create a simple dataset.
    let dataset = SimpleDataset {};

    // Create a shufflable dataset from the simple dataset.
    let mut shufflable_dataset = ShufflableDataset::from_dataset(&dataset).unwrap();

    // Check the shufflable dataset without shuffling is the same as the original dataset.
    assert_eq!(shufflable_dataset.len().unwrap(), 100);
    for i in 0..shufflable_dataset.len().unwrap() {
        assert_eq!(shufflable_dataset.get(i).unwrap(), i);
    }

    // Shuffle the shufflable dataset with a seed.
    shufflable_dataset.shuffle(42);

    // Create a simulated dataset.
    let mut dataset: Vec<usize> = (0..shufflable_dataset.len().unwrap()).collect();
    dataset.shuffle(&mut StdRng::seed_from_u64(42));

    // Check the shufflable dataset is the same as the simulated dataset.
    assert_eq!(shufflable_dataset.len().unwrap(), dataset.len() as usize);
    for i in 0..shufflable_dataset.len().unwrap() {
        assert_eq!(shufflable_dataset.get(i).unwrap(), dataset[i as usize]);
    }
}
