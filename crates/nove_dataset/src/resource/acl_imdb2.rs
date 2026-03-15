use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::common::{ChecksumType, download_and_verify, extract_archive};
use crate::{Dataset, DatasetError};

const ACL_IMDB_URL: &str = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz";
const ACL_IMDB_SHA256: &str = "c40f74a18d3b61f90feba1e17730e0d38e8b97c05fde7008942e91923d1658fe";

const ACL_IMDB2_LABELS: &[&str] = &["neg", "pos"];

/// ACL IMDB sentiment classification dataset manager (binary: positive/negative).
///
/// The ACL IMDB dataset contains 50,000 movie reviews for sentiment analysis.
/// This version provides binary classification (positive vs negative sentiment).
///
/// The dataset is split into 25,000 training and 25,000 test reviews.
/// Each set contains an equal number of positive and negative reviews.
///
/// - Positive reviews have a rating >= 7 out of 10
/// - Negative reviews have a rating <= 4 out of 10
///
/// The dataset is downloaded from Stanford AI Lab and extracted to a local directory.
/// Use [`AclImdb2::train()`] or [`AclImdb2::test()`] methods to get the specific split.
///
/// # Data Source
/// The dataset is downloaded from:
/// <https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz>
///
/// # License
/// The ACL IMDB dataset is available from the official website:
/// <https://ai.stanford.edu/~amaas/data/sentiment/>
/// If you use this dataset, please cite the paper.
///
/// # Citation
/// If you use this dataset in your research, please cite:
/// ```text
/// @InProceedings{maas-EtAl:2011:ACL-HLT2011,
///   author    = {Maas, Andrew L. and Daly, Raymond E. and Pham, Peter T. and Huang, Dan and Ng, Andrew Y. and Potts, Christopher},
///   title     = {Learning Word Vectors for Sentiment Analysis},
///   booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
///   month     = {June},
///   year      = {2011},
///   address   = {Portland, Oregon, USA},
///   publisher = {Association for Computational Linguistics},
///   pages     = {142--150},
///   url       = {http://www.aclweb.org/anthology/P11-1015}
/// }
/// ```
///
/// # Directory Structure
/// After extraction, the dataset will have the following structure:
/// ```text
/// <root_dir>/
/// ├── aclImdb/
/// │   ├── train/
/// │   │   ├── pos/
/// │   │   │   ├── 0_10.txt
/// │   │   │   ├── 1_10.txt
/// │   │   │   └── ...
/// │   │   └── neg/
/// │   │       ├── 0_3.txt
/// │   │       ├── 1_3.txt
/// │   │       └── ...
/// │   └── test/
/// │       ├── pos/
/// │       └── neg/
/// ```
///
/// # Fields
/// * `dataset_dir` - The root directory where the dataset is stored.
///
/// # Examples
/// ```rust,no_run
/// use nove::dataset::resource::AclImdb2;
/// use nove::dataset::Dataset;
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // Create IMDB manager (downloads and extracts if needed)
///     let imdb = AclImdb2::new("path/to/data")?;
///
///     // Get training dataset
///     let train_dataset = imdb.train()?;
///     let (review_text, label) = train_dataset.get(0)?;
///     println!("Review: {}, Label: {}", review_text.chars().take(50).collect::<String>(), label);
///
///     // Get testing dataset
///     let test_dataset = imdb.test()?;
///     println!("Test samples: {}", test_dataset.len()?);
///     Ok(())
/// }
/// ```
///
/// # See Also
/// * [`AclImdb10`](crate::resource::AclImdb10) - IMDB dataset with 10-class rating classification
/// * [`AclImdbUnsup`](crate::resource::AclImdbUnsup) - IMDB dataset for unsupervised learning
pub struct AclImdb2 {
    dataset_dir: PathBuf,
}

/// The split of ACL IMDB dataset (training or testing).
enum AclImdbSplit {
    Train,
    Test,
}

impl AclImdb2 {
    /// Creates a new ACL IMDB binary sentiment manager.
    ///
    /// If the dataset is not present in the specified directory, it will be
    /// downloaded and extracted automatically.
    ///
    /// # Arguments
    /// * `root_dir` - The root directory where the dataset will be stored.
    ///
    /// # Returns
    /// * `Ok(Self)` - A new AclImdb2 instance.
    /// * `Err(DatasetError)` - An error occurred during creation.
    ///
    /// # Examples
    /// ```rust,no_run
    /// use nove::dataset::resource::AclImdb2;
    ///
    /// fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let imdb = AclImdb2::new("data/imdb")?;
    ///     Ok(())
    /// }
    /// ```
    pub fn new<P: AsRef<Path>>(root_dir: P) -> Result<Self, DatasetError> {
        let root_dir = root_dir.as_ref().to_path_buf();
        let dataset_dir = root_dir.join("aclImdb");

        Self::download_and_extract(&root_dir)?;

        Ok(Self { dataset_dir })
    }

    /// Returns the training dataset.
    ///
    /// # Returns
    /// * `Ok(AclImdb2Dataset)` - The training dataset.
    /// * `Err(DatasetError)` - An error occurred during loading.
    pub fn train(&self) -> Result<AclImdb2Dataset, DatasetError> {
        let samples = Self::load_samples(&self.dataset_dir, AclImdbSplit::Train)?;
        if samples.is_empty() {
            return Err(DatasetError::EmptyDataset);
        }
        Ok(AclImdb2Dataset { samples })
    }

    /// Returns the testing dataset.
    ///
    /// # Returns
    /// * `Ok(AclImdb2Dataset)` - The testing dataset.
    /// * `Err(DatasetError)` - An error occurred during loading.
    pub fn test(&self) -> Result<AclImdb2Dataset, DatasetError> {
        let samples = Self::load_samples(&self.dataset_dir, AclImdbSplit::Test)?;
        if samples.is_empty() {
            return Err(DatasetError::EmptyDataset);
        }
        Ok(AclImdb2Dataset { samples })
    }

    /// Returns the dataset directory path.
    ///
    /// # Returns
    /// * `&Path` - The path to the dataset directory.
    pub fn dataset_dir(&self) -> &Path {
        &self.dataset_dir
    }

    fn download_and_extract(root_dir: &Path) -> Result<(), DatasetError> {
        fs::create_dir_all(root_dir)?;

        let archive_path = root_dir.join("aclImdb_v1.tar.gz");
        let dataset_dir = root_dir.join("aclImdb");

        if !archive_path.exists() {
            download_and_verify(
                ACL_IMDB_URL,
                &archive_path,
                ChecksumType::Sha256,
                ACL_IMDB_SHA256,
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
        split: AclImdbSplit,
    ) -> Result<Vec<(PathBuf, usize)>, DatasetError> {
        let split_dir = match split {
            AclImdbSplit::Train => dataset_dir.join("train"),
            AclImdbSplit::Test => dataset_dir.join("test"),
        };

        let mut samples = Vec::new();
        let mut label_counts: HashMap<usize, usize> = HashMap::new();

        for (label_idx, label_name) in ACL_IMDB2_LABELS.iter().enumerate() {
            let label_dir = split_dir.join(label_name);
            if !label_dir.exists() {
                return Err(DatasetError::InvalidLabelDir(label_dir));
            }

            let entries = fs::read_dir(&label_dir)?;
            let mut count = 0;

            for entry in entries {
                let entry = entry?;
                let path = entry.path();

                if path.extension().is_some_and(|ext| ext == "txt") {
                    samples.push((path, label_idx));
                    count += 1;
                }
            }

            label_counts.insert(label_idx, count);
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
                AclImdbSplit::Train => "train",
                AclImdbSplit::Test => "test",
            }
        );
        for (label_idx, label_name) in ACL_IMDB2_LABELS.iter().enumerate() {
            if let Some(count) = label_counts.get(&label_idx) {
                println!("  Label {} ({}): {} samples", label_idx, label_name, count);
            }
        }

        Ok(samples)
    }

    /// Returns the human-readable label name for a given label index.
    ///
    /// # Arguments
    /// * `label` - The label index (0-1).
    ///
    /// # Returns
    /// * `Some(&str)` - The label name if the index is valid.
    /// * `None` - If the index is out of range.
    ///
    /// # Examples
    /// ```rust
    /// use nove::dataset::resource::AclImdb2;
    ///
    /// let label_name = AclImdb2::label_name(0);
    /// assert_eq!(label_name, Some("neg"));
    ///
    /// let label_name = AclImdb2::label_name(1);
    /// assert_eq!(label_name, Some("pos"));
    /// ```
    pub fn label_name(label: usize) -> Option<&'static str> {
        ACL_IMDB2_LABELS.get(label).copied()
    }
}

/// ACL IMDB binary sentiment dataset containing samples for a specific split.
///
/// Each sample is represented as a tuple of (review_text, label) where:
/// - `review_text` is the text content of the movie review (read from file)
/// - `label` is 0 for negative sentiment, 1 for positive sentiment
///
/// This struct cannot be instantiated directly. Use [`AclImdb2::train()`] or
/// [`AclImdb2::test()`] to obtain an `AclImdb2Dataset` instance.
///
/// # Fields
/// * `samples` - A vector of (review_path, label) tuples.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AclImdb2Dataset {
    samples: Vec<(PathBuf, usize)>,
}

impl AclImdb2Dataset {
    /// Returns the number of samples for each label.
    ///
    /// # Returns
    /// * `HashMap<usize, usize>` - A map where keys are labels (0-1) and values are the number of samples for each label.
    pub fn label_distribution(&self) -> HashMap<usize, usize> {
        let mut distribution = HashMap::new();
        for (_, label) in &self.samples {
            *distribution.entry(*label).or_insert(0) += 1;
        }
        distribution
    }
}

impl Dataset for AclImdb2Dataset {
    type Item = (String, usize);

    fn get(&self, index: usize) -> Result<Self::Item, DatasetError> {
        let (path, label) = self
            .samples
            .get(index)
            .ok_or(DatasetError::IndexOutOfBounds(index, self.samples.len()))?;
        let content = fs::read_to_string(path)?;
        Ok((content, *label))
    }

    fn len(&self) -> Result<usize, DatasetError> {
        Ok(self.samples.len())
    }
}
