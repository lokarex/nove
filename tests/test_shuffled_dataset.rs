mod simple_dataset;

use nove::dataset::{Dataset, util::ShuffledDataset};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use simple_dataset::SimpleDataset;

#[test]
fn test_shuffled_dataset_with_simple_dataset() {
    // Create a simple dataset.
    let dataset = SimpleDataset {};

    // Create a shuffled dataset from the simple dataset.
    let mut shuffled_dataset = ShuffledDataset::from_dataset(&dataset);

    // Check the shuffled dataset without shuffling is the same as the original dataset.
    assert_eq!(shuffled_dataset.len(), 100);
    for i in 0..shuffled_dataset.len() {
        assert_eq!(shuffled_dataset.get(i), i);
    }

    // Shuffle the shuffled dataset with a seed.
    shuffled_dataset.shuffle(42);

    // Create a simulated dataset.
    let mut dataset: Vec<u64> = (0..shuffled_dataset.len()).collect();
    dataset.shuffle(&mut StdRng::seed_from_u64(42));

    // Check the shuffled dataset is the same as the simulated dataset.
    assert_eq!(shuffled_dataset.len(), dataset.len() as u64);
    for i in 0..shuffled_dataset.len() {
        assert_eq!(shuffled_dataset.get(i), dataset[i as usize]);
    }
}
