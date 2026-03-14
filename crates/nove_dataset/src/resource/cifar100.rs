use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::common::{ChecksumType, download_and_verify, extract_archive};
use crate::{Dataset, DatasetError};

const CIFAR100_URL: &str = "https://s3.amazonaws.com/fast-ai-imageclas/cifar100.tgz";
const CIFAR100_SHA256: &str = "085ac613ceb0b3659c8072143ae553d5dd146b3c4206c3672a56ed02d0e77d28";

const CIFAR100_FINE_LABELS: &[&str] = &[
    "beaver",
    "dolphin",
    "otter",
    "seal",
    "whale",
    "aquarium_fish",
    "flatfish",
    "ray",
    "shark",
    "trout",
    "orchid",
    "poppy",
    "rose",
    "sunflower",
    "tulip",
    "bottle",
    "bowl",
    "can",
    "cup",
    "plate",
    "apple",
    "mushroom",
    "orange",
    "pear",
    "sweet_pepper",
    "clock",
    "keyboard",
    "lamp",
    "telephone",
    "television",
    "bed",
    "chair",
    "couch",
    "table",
    "wardrobe",
    "bee",
    "beetle",
    "butterfly",
    "caterpillar",
    "cockroach",
    "bear",
    "leopard",
    "lion",
    "tiger",
    "wolf",
    "bridge",
    "castle",
    "house",
    "road",
    "skyscraper",
    "cloud",
    "forest",
    "mountain",
    "plain",
    "sea",
    "camel",
    "cattle",
    "chimpanzee",
    "elephant",
    "kangaroo",
    "fox",
    "porcupine",
    "possum",
    "raccoon",
    "skunk",
    "crab",
    "lobster",
    "snail",
    "spider",
    "worm",
    "baby",
    "boy",
    "girl",
    "man",
    "woman",
    "crocodile",
    "dinosaur",
    "lizard",
    "snake",
    "turtle",
    "hamster",
    "mouse",
    "rabbit",
    "shrew",
    "squirrel",
    "maple_tree",
    "oak_tree",
    "palm_tree",
    "pine_tree",
    "willow_tree",
    "bicycle",
    "bus",
    "motorcycle",
    "pickup_truck",
    "train",
    "lawn_mower",
    "rocket",
    "streetcar",
    "tank",
    "tractor",
];

const CIFAR100_COARSE_LABELS: &[&str] = &[
    "aquatic_mammals",
    "fish",
    "flowers",
    "food_containers",
    "fruit_and_vegetables",
    "household_electrical_devices",
    "household_furniture",
    "insects",
    "large_carnivores",
    "large_man-made_outdoor_things",
    "large_natural_outdoor_scenes",
    "large_omnivores_and_herbivores",
    "medium_mammals",
    "non-insect_invertebrates",
    "people",
    "reptiles",
    "small_mammals",
    "trees",
    "vehicles_1",
    "vehicles_2",
];

/// CIFAR-100 image dataset manager.
///
/// The CIFAR-100 dataset consists of 60,000 32x32 colour images in 100 classes,
/// with 600 images per class. There are 50,000 training images and 10,000 test images.
///
/// The dataset is organized hierarchically with 100 fine-grained classes grouped into
/// 20 coarse-grained superclasses. Each superclass contains 5 fine-grained classes.
///
/// The dataset is downloaded from a remote source and extracted to a local directory.
/// Use [`Cifar100::train()`] or [`Cifar100::test()`] methods to get the specific split of the dataset.
///
/// # Data Source
/// The PNG version of CIFAR-100 is downloaded from:
/// <https://s3.amazonaws.com/fast-ai-imageclas/cifar100.tgz>
///
/// # License
/// The CIFAR-100 dataset is available from the official website:
/// <https://www.cs.toronto.edu/~kriz/cifar.html>
/// If you use this dataset, please cite the tech report.
///
/// # Citation
/// If you use this dataset in your research, please cite:
/// ```text
/// @techreport{krizhevsky2009learning,
///   title={Learning multiple layers of features from tiny images},
///   author={Krizhevsky, Alex and Hinton, Geoffrey},
///   year={2009},
///   institution={University of Toronto}
/// }
/// ```
///
/// # Directory Structure
/// After extraction, the dataset will have the following structure:
/// ```text
/// <root_dir>/
/// ├── cifar100/
/// │   ├── train/
/// │   │   ├── aquatic_mammals/
/// │   │   │   ├── beaver/
/// │   │   │   │   ├── 1.png
/// │   │   │   │   ├── 2.png
/// │   │   │   │   └── ...
/// │   │   │   ├── dolphin/
/// │   │   │   └── ...
/// │   │   ├── fish/
/// │   │   ├── ...
/// │   │   └── vehicles_2/
/// │   └── test/
/// │       ├── aquatic_mammals/
/// │       ├── ...
/// │       └── vehicles_2/
/// ```
///
/// # Fields
/// * `dataset_dir` - The root directory where the dataset is stored.
///
/// # Examples
/// ```rust,no_run
/// use nove::dataset::resource::Cifar100;
/// use nove::dataset::Dataset;
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // Create CIFAR-100 manager (downloads and extracts if needed)
///     let cifar100 = Cifar100::new("path/to/data")?;
///
///     // Get training dataset
///     let train_dataset = cifar100.train()?;
///     let (image_path, fine_label) = train_dataset.get(0)?;
///     println!("Train Image: {:?}, Fine Label: {}",
///              image_path, fine_label);
///
///     // Get testing dataset
///     let test_dataset = cifar100.test()?;
///     println!("Test samples: {}", test_dataset.len()?);
///     Ok(())
/// }
/// ```
///
/// # See Also
/// * [`Cifar10`](crate::resource::Cifar10) - CIFAR dataset with 10 classes
pub struct Cifar100 {
    dataset_dir: PathBuf,
}

/// The split of CIFAR-100 dataset (training or testing).
enum Cifar100Split {
    Train,
    Test,
}

impl Cifar100 {
    /// Creates a new CIFAR-100 manager.
    ///
    /// If the dataset is not present in the specified directory, it will be
    /// downloaded and extracted automatically.
    ///
    /// # Arguments
    /// * `root_dir` - The root directory where the dataset will be stored.
    ///
    /// # Returns
    /// * `Ok(Self)` - A new Cifar100 instance.
    /// * `Err(DatasetError)` - An error occurred during creation.
    ///
    /// # Examples
    /// ```rust,no_run
    /// use nove::dataset::resource::Cifar100;
    ///
    /// fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let cifar100 = Cifar100::new("data/cifar100")?;
    ///     Ok(())
    /// }
    /// ```
    pub fn new<P: AsRef<Path>>(root_dir: P) -> Result<Self, DatasetError> {
        let root_dir = root_dir.as_ref().to_path_buf();
        let dataset_dir = root_dir.join("cifar100");

        Self::download_and_extract(&root_dir)?;

        Ok(Self { dataset_dir })
    }

    /// Returns the training dataset.
    ///
    /// # Returns
    /// * `Ok(Cifar100Dataset)` - The training dataset.
    /// * `Err(DatasetError)` - An error occurred during loading.
    pub fn train(&self) -> Result<Cifar100Dataset, DatasetError> {
        let samples = Self::load_samples(&self.dataset_dir, Cifar100Split::Train)?;
        if samples.is_empty() {
            return Err(DatasetError::EmptyDataset);
        }
        Ok(Cifar100Dataset { samples })
    }

    /// Returns the testing dataset.
    ///
    /// # Returns
    /// * `Ok(Cifar100Dataset)` - The testing dataset.
    /// * `Err(DatasetError)` - An error occurred during loading.
    pub fn test(&self) -> Result<Cifar100Dataset, DatasetError> {
        let samples = Self::load_samples(&self.dataset_dir, Cifar100Split::Test)?;
        if samples.is_empty() {
            return Err(DatasetError::EmptyDataset);
        }
        Ok(Cifar100Dataset { samples })
    }

    /// Returns the dataset directory path.
    ///
    /// # Returns
    /// * `&Path` - The path to the dataset directory.
    pub fn dataset_dir(&self) -> &Path {
        &self.dataset_dir
    }

    /// Downloads and extracts the CIFAR-100 dataset if it is not already present.
    ///
    /// # Arguments
    /// * `root_dir` - The root directory where the dataset will be stored.
    ///
    /// # Returns
    /// * `Ok(())` - The dataset was downloaded and extracted successfully.
    /// * `Err(DatasetError)` - An error occurred during download or extraction.
    fn download_and_extract(root_dir: &Path) -> Result<(), DatasetError> {
        fs::create_dir_all(root_dir)?;

        let archive_path = root_dir.join("cifar100.tgz");
        let dataset_dir = root_dir.join("cifar100");

        if !archive_path.exists() {
            download_and_verify(
                CIFAR100_URL,
                &archive_path,
                ChecksumType::Sha256,
                CIFAR100_SHA256,
                true,
            )?;
        } else {
            println!("Archive already exists, skipping download");
        }

        if !dataset_dir.exists() {
            extract_archive(&archive_path, root_dir, true)?;
        } else {
            println!("Dataset already extracted, skipping extraction");
        }

        Ok(())
    }

    fn load_samples(
        dataset_dir: &Path,
        split: Cifar100Split,
    ) -> Result<Vec<(PathBuf, usize, usize)>, DatasetError> {
        let split_dir = match split {
            Cifar100Split::Train => dataset_dir.join("train"),
            Cifar100Split::Test => dataset_dir.join("test"),
        };

        let mut samples = Vec::new();
        let mut fine_label_counts: HashMap<usize, usize> = HashMap::new();
        let mut coarse_label_counts: HashMap<usize, usize> = HashMap::new();

        for fine_label in 0..100 {
            let coarse_label = Self::fine_to_coarse(fine_label);
            let label_dir = split_dir
                .join(CIFAR100_COARSE_LABELS[coarse_label])
                .join(CIFAR100_FINE_LABELS[fine_label]);
            if !label_dir.exists() {
                return Err(DatasetError::InvalidLabelDir(label_dir));
            }

            let entries = fs::read_dir(&label_dir)?;
            let mut fine_count = 0;

            for entry in entries {
                let entry = entry?;
                let path = entry.path();

                if path.extension().is_some_and(|ext| ext == "png") {
                    samples.push((path, coarse_label, fine_label));
                    fine_count += 1;
                }
            }

            *fine_label_counts.entry(fine_label).or_insert(0) += fine_count;
            *coarse_label_counts.entry(coarse_label).or_insert(0) += fine_count;
        }

        samples.sort_by(|a, b| {
            let coarse_cmp = a.1.cmp(&b.1);
            if coarse_cmp != std::cmp::Ordering::Equal {
                return coarse_cmp;
            }
            let fine_cmp = a.2.cmp(&b.2);
            if fine_cmp != std::cmp::Ordering::Equal {
                return fine_cmp;
            }
            a.0.cmp(&b.0)
        });

        println!(
            "Loaded {} samples for {} split:",
            samples.len(),
            match split {
                Cifar100Split::Train => "train",
                Cifar100Split::Test => "test",
            }
        );
        println!("Fine labels (sample per class):");
        for label in 0usize..100 {
            if let Some(count) = fine_label_counts.get(&label) {
                println!(
                    "  {} ({}): {} samples",
                    label, CIFAR100_FINE_LABELS[label], count
                );
            }
        }
        println!("Coarse labels (superclasses):");
        for label in 0usize..20 {
            if let Some(count) = coarse_label_counts.get(&label) {
                println!(
                    "  {} ({}): {} samples",
                    label, CIFAR100_COARSE_LABELS[label], count
                );
            }
        }

        Ok(samples)
    }

    /// Returns the human-readable fine-grained label name for a given label index.
    ///
    /// # Arguments
    /// * `fine_label` - The fine-grained label index (0-99).
    ///
    /// # Returns
    /// * `Some(&str)` - The fine-grained label name if the index is valid.
    /// * `None` - If the index is out of range.
    ///
    /// # Examples
    /// ```rust
    /// use nove::dataset::resource::Cifar100;
    ///
    /// let label_name = Cifar100::fine_label_name(0);
    /// assert_eq!(label_name, Some("beaver"));
    /// ```
    pub fn fine_label_name(fine_label: usize) -> Option<&'static str> {
        CIFAR100_FINE_LABELS.get(fine_label).copied()
    }

    /// Returns the human-readable coarse-grained label name for a given label index.
    ///
    /// # Arguments
    /// * `coarse_label` - The coarse-grained label index (0-19).
    ///
    /// # Returns
    /// * `Some(&str)` - The coarse-grained label name if the index is valid.
    /// * `None` - If the index is out of range.
    ///
    /// # Examples
    /// ```rust
    /// use nove::dataset::resource::Cifar100;
    ///
    /// let label_name = Cifar100::coarse_label_name(0);
    /// assert_eq!(label_name, Some("aquatic_mammals"));
    /// ```
    pub fn coarse_label_name(coarse_label: usize) -> Option<&'static str> {
        CIFAR100_COARSE_LABELS.get(coarse_label).copied()
    }

    /// Converts a fine-grained label index to its corresponding coarse-grained label index.
    ///
    /// The CIFAR-100 dataset has 100 fine-grained classes grouped into 20 coarse-grained
    /// superclasses. Each superclass contains exactly 5 fine-grained classes.
    ///
    /// # Arguments
    /// * `fine_label` - The fine-grained label index (0-99).
    ///
    /// # Returns
    /// * `usize` - The coarse-grained label index (0-19).
    ///
    /// # Examples
    /// ```rust
    /// use nove::dataset::resource::Cifar100;
    ///
    /// let coarse_label = Cifar100::fine_to_coarse(0);
    /// assert_eq!(coarse_label, 0);
    ///
    /// let coarse_label = Cifar100::fine_to_coarse(5);
    /// assert_eq!(coarse_label, 1);
    /// ```
    pub fn fine_to_coarse(fine_label: usize) -> usize {
        fine_label / 5
    }
}

/// CIFAR-100 dataset containing samples for a specific split (training or testing).
///
/// Each sample is represented as a tuple of (image_path, coarse_label, fine_label).
///
/// This struct cannot be instantiated directly. Use [`Cifar100::train()`] or
/// [`Cifar100::test()`] to obtain a `Cifar100Dataset` instance.
///
/// # Fields
/// * `samples` - A vector of (image_path, coarse_label, fine_label) tuples.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cifar100Dataset {
    samples: Vec<(PathBuf, usize, usize)>,
}

impl Cifar100Dataset {
    /// Returns the number of samples for each fine-grained label.
    ///
    /// # Returns
    /// * `HashMap<usize, usize>` - A map where keys are fine-grained labels (0-99) and values are the number of samples for each label.
    pub fn fine_label_distribution(&self) -> HashMap<usize, usize> {
        let mut distribution = HashMap::new();
        for (_, _, fine_label) in &self.samples {
            *distribution.entry(*fine_label).or_insert(0) += 1;
        }
        distribution
    }

    /// Returns the number of samples for each coarse-grained label.
    ///
    /// # Returns
    /// * `HashMap<usize, usize>` - A map where keys are coarse-grained labels (0-19) and values are the number of samples for each label.
    pub fn coarse_label_distribution(&self) -> HashMap<usize, usize> {
        let mut distribution = HashMap::new();
        for (_, coarse_label, _) in &self.samples {
            *distribution.entry(*coarse_label).or_insert(0) += 1;
        }
        distribution
    }
}

impl Dataset for Cifar100Dataset {
    type Item = (PathBuf, usize);

    fn get(&self, index: usize) -> Result<Self::Item, DatasetError> {
        self.samples
            .get(index)
            .ok_or(DatasetError::IndexOutOfBounds(index, self.samples.len()))
            .map(|item| (item.0.clone(), item.2))
    }

    fn len(&self) -> Result<usize, DatasetError> {
        Ok(self.samples.len())
    }
}
