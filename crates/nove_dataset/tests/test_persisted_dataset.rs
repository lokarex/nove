mod util;

use nove::dataset::{Dataset, util::PersistedDataset};
use tempfile::TempDir;
use util::SimpleDataset;

#[test]
fn test_persisted_dataset_with_simple_dataset() {
    // Create a simple dataset.
    let dataset = SimpleDataset {};

    // Create a persisted dataset from the simple dataset.
    let persisted_dataset = PersistedDataset::from_dataset(&dataset);

    // Save the persisted dataset to a file.
    let temp_dir = TempDir::new().unwrap();
    persisted_dataset.save_as_file(
        temp_dir
            .path()
            .join("test_persisted_dataset.bin")
            .to_str()
            .unwrap(),
    );

    // Load the persisted dataset from the file.
    let loaded_dataset: PersistedDataset<'_, SimpleDataset> = PersistedDataset::load_from_file(
        temp_dir
            .path()
            .join("test_persisted_dataset.bin")
            .to_str()
            .unwrap(),
    );

    // Check the loaded dataset is the same as the original dataset.
    assert_eq!(loaded_dataset.len(), dataset.len());
    for i in 0..loaded_dataset.len() {
        assert_eq!(loaded_dataset.get(i), i);
    }
}
