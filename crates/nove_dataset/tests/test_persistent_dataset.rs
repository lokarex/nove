mod util;

use nove::dataset::{Dataset, common::PersistentDataset};
use tempfile::TempDir;
use util::SimpleDataset;

#[test]
fn test_persistent_dataset_with_simple_dataset() {
    // Create a simple dataset.
    let dataset = SimpleDataset {};

    // Create a persistent dataset from the simple dataset.
    let persistent_dataset = PersistentDataset::from_dataset(&dataset).unwrap();

    // Save the persistent dataset to a file.
    let temp_dir = TempDir::new().unwrap();
    persistent_dataset
        .save_to_file(
            temp_dir
                .path()
                .join("test_persistent_dataset.bin")
                .to_str()
                .unwrap(),
        )
        .unwrap();

    // Load the persistent dataset from the file.
    let loaded_dataset: PersistentDataset<'_, SimpleDataset> = PersistentDataset::load_from_file(
        temp_dir
            .path()
            .join("test_persistent_dataset.bin")
            .to_str()
            .unwrap(),
    )
    .unwrap();

    // Check the loaded dataset is the same as the original dataset.
    assert_eq!(loaded_dataset.len().unwrap(), dataset.len().unwrap());
    for i in 0..loaded_dataset.len().unwrap() {
        assert_eq!(loaded_dataset.get(i).unwrap(), i);
    }
}
