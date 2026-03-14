use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::common::{ChecksumType, download_and_verify, extract_archive};
use crate::{Dataset, DatasetError};

const CIFAR10_URL: &str = "https://s3.amazonaws.com/fast-ai-sample/cifar10.tgz";
const CIFAR10_SHA256: &str = "c47812bd9e6a0f0a09f56719f20edb3d513e42462ce55e2b45ce7f9be1da5d59";

const CIFAR10_LABELS: &[&str] = &[
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
];

/// CIFAR-10 image dataset manager.
///
/// The CIFAR-10 dataset consists of 60,000 32x32 colour images in 10 classes,
/// with 6,000 images per class. There are 50,000 training images and 10,000 test images.
///
/// The dataset is downloaded from a remote source and extracted to a local directory.
/// Use [`Cifar10::train()`] or [`Cifar10::test()`] methods to get the specific split of the dataset.
///
/// # Data Source
/// The PNG version of CIFAR-10 is downloaded from:
/// <https://s3.amazonaws.com/fast-ai-sample/cifar10.tgz>
///
/// # License
/// The CIFAR-10 dataset is available from the official website:
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
/// ├── cifar10/
/// │   ├── train/
/// │   │   ├── airplane/
/// │   │   │   ├── 1.png
/// │   │   │   ├── 2.png
/// │   │   │   └── ...
/// │   │   ├── automobile/
/// │   │   ├── ...
/// │   │   └── truck/
/// │   └── test/
/// │       ├── airplane/
/// │       ├── ...
/// │       └── truck/
/// ```
///
/// # Fields
/// * `dataset_dir` - The root directory where the dataset is stored.
///
/// # Examples
/// ```rust,no_run
/// use nove::dataset::resource::Cifar10;
/// use nove::dataset::Dataset;
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // Create CIFAR-10 manager (downloads and extracts if needed)
///     let cifar10 = Cifar10::new("path/to/data")?;
///
///     // Get training dataset
///     let train_dataset = cifar10.train()?;
///     let (image_path, label) = train_dataset.get(0)?;
///     println!("Train Image: {:?}, Label: {}", image_path, label);
///
///     // Get testing dataset
///     let test_dataset = cifar10.test()?;
///     println!("Test samples: {}", test_dataset.len()?);
///     Ok(())
/// }
/// ```
///
/// # See Also
/// * [`Cifar100`](crate::resource::Cifar100) - CIFAR dataset with 100 fine-grained classes
pub struct Cifar10 {
    dataset_dir: PathBuf,
}

/// The split of CIFAR-10 dataset (training or testing).
enum Cifar10Split {
    Train,
    Test,
}

impl Cifar10 {
    /// Creates a new CIFAR-10 manager.
    ///
    /// If the dataset is not present in the specified directory, it will be
    /// downloaded and extracted automatically.
    ///
    /// # Arguments
    /// * `root_dir` - The root directory where the dataset will be stored.
    ///
    /// # Returns
    /// * `Ok(Self)` - A new Cifar10 instance.
    /// * `Err(DatasetError)` - An error occurred during creation.
    ///
    /// # Examples
    /// ```rust,no_run
    /// use nove::dataset::resource::Cifar10;
    ///
    /// fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let cifar10 = Cifar10::new("data/cifar10")?;
    ///     Ok(())
    /// }
    /// ```
    pub fn new<P: AsRef<Path>>(root_dir: P) -> Result<Self, DatasetError> {
        let root_dir = root_dir.as_ref().to_path_buf();
        let dataset_dir = root_dir.join("cifar10");

        Self::download_and_extract(&root_dir)?;

        Ok(Self { dataset_dir })
    }

    /// Returns the training dataset.
    ///
    /// # Returns
    /// * `Ok(Cifar10Dataset)` - The training dataset.
    /// * `Err(DatasetError)` - An error occurred during loading.
    pub fn train(&self) -> Result<Cifar10Dataset, DatasetError> {
        let samples = Self::load_samples(&self.dataset_dir, Cifar10Split::Train)?;
        if samples.is_empty() {
            return Err(DatasetError::EmptyDataset);
        }
        Ok(Cifar10Dataset { samples })
    }

    /// Returns the testing dataset.
    ///
    /// # Returns
    /// * `Ok(Cifar10Dataset)` - The testing dataset.
    /// * `Err(DatasetError)` - An error occurred during loading.
    pub fn test(&self) -> Result<Cifar10Dataset, DatasetError> {
        let samples = Self::load_samples(&self.dataset_dir, Cifar10Split::Test)?;
        if samples.is_empty() {
            return Err(DatasetError::EmptyDataset);
        }
        Ok(Cifar10Dataset { samples })
    }

    /// Returns the dataset directory path.
    ///
    /// # Returns
    /// * `&Path` - The path to the dataset directory.
    pub fn dataset_dir(&self) -> &Path {
        &self.dataset_dir
    }

    /// Downloads and extracts the CIFAR-10 dataset if it is not already present.
    ///
    /// # Arguments
    /// * `root_dir` - The root directory where the dataset will be stored.
    ///
    /// # Returns
    /// * `Ok(())` - The dataset was downloaded and extracted successfully.
    /// * `Err(DatasetError)` - An error occurred during download or extraction.
    fn download_and_extract(root_dir: &Path) -> Result<(), DatasetError> {
        fs::create_dir_all(root_dir)?;

        let archive_path = root_dir.join("cifar10.tgz");
        let dataset_dir = root_dir.join("cifar10");

        if !archive_path.exists() {
            download_and_verify(
                CIFAR10_URL,
                &archive_path,
                ChecksumType::Sha256,
                CIFAR10_SHA256,
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
        split: Cifar10Split,
    ) -> Result<Vec<(PathBuf, usize)>, DatasetError> {
        let split_dir = match split {
            Cifar10Split::Train => dataset_dir.join("train"),
            Cifar10Split::Test => dataset_dir.join("test"),
        };

        let mut samples = Vec::new();
        let mut label_counts: HashMap<usize, usize> = HashMap::new();

        for label in 0u8..=9 {
            let label_usize = label as usize;
            let label_dir = split_dir.join(CIFAR10_LABELS[label_usize]);
            if !label_dir.exists() {
                return Err(DatasetError::InvalidLabelDir(label_dir));
            }

            let entries = fs::read_dir(&label_dir)?;
            let mut count = 0;

            for entry in entries {
                let entry = entry?;
                let path = entry.path();

                if path.extension().is_some_and(|ext| ext == "png") {
                    samples.push((path, label_usize));
                    count += 1;
                }
            }

            label_counts.insert(label_usize, count);
        }

        samples.sort_by(|a, b| {
            let label_cmp = a.1.cmp(&b.1);
            if label_cmp != std::cmp::Ordering::Equal {
                return label_cmp;
            }
            a.0.cmp(&b.0)
        });

        println!(
            "Loaded {} samples for {} split:",
            samples.len(),
            match split {
                Cifar10Split::Train => "train",
                Cifar10Split::Test => "test",
            }
        );
        for label in 0usize..=9 {
            if let Some(count) = label_counts.get(&label) {
                println!(
                    "  Label {} ({}): {} samples",
                    label, CIFAR10_LABELS[label], count
                );
            }
        }

        Ok(samples)
    }

    /// Returns the human-readable label name for a given label index.
    ///
    /// # Arguments
    /// * `label` - The label index (0-9).
    ///
    /// # Returns
    /// * `Some(&str)` - The label name if the index is valid.
    /// * `None` - If the index is out of range.
    ///
    /// # Examples
    /// ```rust
    /// use nove::dataset::resource::Cifar10;
    ///
    /// let label_name = Cifar10::label_name(0);
    /// assert_eq!(label_name, Some("airplane"));
    /// ```
    pub fn label_name(label: usize) -> Option<&'static str> {
        CIFAR10_LABELS.get(label).copied()
    }
}

/// CIFAR-10 dataset containing samples for a specific split (training or testing).
///
/// Each sample is represented as a tuple of (image_path, label).
///
/// This struct cannot be instantiated directly. Use [`Cifar10::train()`] or
/// [`Cifar10::test()`] to obtain a `Cifar10Dataset` instance.
///
/// # Fields
/// * `samples` - A vector of (image_path, label) tuples.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cifar10Dataset {
    samples: Vec<(PathBuf, usize)>,
}

impl Cifar10Dataset {
    /// Returns the number of samples for each label.
    ///
    /// # Returns
    /// * `HashMap<usize, usize>` - A map where keys are labels (0-9) and values are the number of samples for each label.
    pub fn label_distribution(&self) -> HashMap<usize, usize> {
        let mut distribution = HashMap::new();
        for (_, label) in &self.samples {
            *distribution.entry(*label).or_insert(0) += 1;
        }
        distribution
    }
}

impl Dataset for Cifar10Dataset {
    type Item = (PathBuf, usize);

    fn get(&self, index: usize) -> Result<Self::Item, DatasetError> {
        self.samples
            .get(index)
            .ok_or(DatasetError::IndexOutOfBounds(index, self.samples.len()))
            .cloned()
    }

    fn len(&self) -> Result<usize, DatasetError> {
        Ok(self.samples.len())
    }
}
