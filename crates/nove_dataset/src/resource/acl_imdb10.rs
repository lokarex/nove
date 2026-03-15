use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::common::{ChecksumType, download_and_verify, extract_archive};
use crate::{Dataset, DatasetError};

const ACL_IMDB_URL: &str = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz";
const ACL_IMDB_SHA256: &str = "c40f74a18d3b61f90feba1e17730e0d38e8b97c05fde7008942e91923d1658fe";

/// ACL IMDB rating classification dataset manager (10-point scale: rating 1-10).
///
/// The ACL IMDB dataset contains 50,000 movie reviews for sentiment analysis.
/// This version provides rating classification based on the 10-point star rating scale.
///
/// The original dataset only includes reviews with ratings 1-4 (negative)
/// and 7-10 (positive). Reviews with neutral ratings (5-6) are excluded.
/// Therefore, this dataset has 8 effective rating classes (1, 2, 3, 4, 7, 8, 9, 10).
/// Ratings 5 and 6 are NOT present in this dataset.
///
/// The dataset is split into 25,000 training and 25,000 test reviews.
///
/// File naming convention: `[id]_[rating].txt`
/// - `id` is a unique identifier
/// - `rating` is the star rating (1-10, excluding 5 and 6)
///
/// The dataset is downloaded from Stanford AI Lab and extracted to a local directory.
/// Use [`AclImdb10::train()`] or [`AclImdb10::test()`] methods to get the specific split.
///
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
/// │   │   │   ├── 0_10.txt   (rating: 10)
/// │   │   │   ├── 1_9.txt    (rating: 9)
/// │   │   │   └── ...
/// │   │   └── neg/
/// │   │       ├── 0_3.txt    (rating: 3)
/// │   │       ├── 1_4.txt    (rating: 4)
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
/// use nove::dataset::resource::AclImdb10;
/// use nove::dataset::Dataset;
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // Create IMDB manager (downloads and extracts if needed)
///     let imdb = AclImdb10::new("path/to/data")?;
///
///     // Get training dataset
///     let train_dataset = imdb.train()?;
///     let (review_text, rating) = train_dataset.get(0)?;
///     println!("Review: {}, Rating: {}", review_text.chars().take(50).collect::<String>(), rating);
///
///     // Get testing dataset
///     let test_dataset = imdb.test()?;
///     println!("Test samples: {}", test_dataset.len()?);
///     Ok(())
/// }
/// ```
///
/// # See Also
/// * [`AclImdb2`](crate::resource::AclImdb2) - IMDB dataset with binary sentiment classification
/// * [`AclImdbUnsup`](crate::resource::AclImdbUnsup) - IMDB dataset for unsupervised learning
pub struct AclImdb10 {
    dataset_dir: PathBuf,
}

/// The split of ACL IMDB dataset (training or testing).
enum AclImdbSplit {
    Train,
    Test,
}

impl AclImdb10 {
    /// Creates a new ACL IMDB 10-class rating manager.
    ///
    /// If the dataset is not present in the specified directory, it will be
    /// downloaded and extracted automatically.
    ///
    /// # Arguments
    /// * `root_dir` - The root directory where the dataset will be stored.
    ///
    /// # Returns
    /// * `Ok(Self)` - A new AclImdb10 instance.
    /// * `Err(DatasetError)` - An error occurred during creation.
    ///
    /// # Examples
    /// ```rust,no_run
    /// use nove::dataset::resource::AclImdb10;
    ///
    /// fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let imdb = AclImdb10::new("data/imdb")?;
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
    /// * `Ok(AclImdb10Dataset)` - The training dataset.
    /// * `Err(DatasetError)` - An error occurred during loading.
    pub fn train(&self) -> Result<AclImdb10Dataset, DatasetError> {
        let samples = Self::load_samples(&self.dataset_dir, AclImdbSplit::Train)?;
        if samples.is_empty() {
            return Err(DatasetError::EmptyDataset);
        }
        Ok(AclImdb10Dataset { samples })
    }

    /// Returns the testing dataset.
    ///
    /// # Returns
    /// * `Ok(AclImdb10Dataset)` - The testing dataset.
    /// * `Err(DatasetError)` - An error occurred during loading.
    pub fn test(&self) -> Result<AclImdb10Dataset, DatasetError> {
        let samples = Self::load_samples(&self.dataset_dir, AclImdbSplit::Test)?;
        if samples.is_empty() {
            return Err(DatasetError::EmptyDataset);
        }
        Ok(AclImdb10Dataset { samples })
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

    fn parse_rating_from_filename(filename: &str) -> Option<u8> {
        let stem = filename.strip_suffix(".txt")?;
        let parts: Vec<&str> = stem.split('_').collect();
        if parts.len() == 2 {
            parts[1].parse::<u8>().ok()
        } else {
            None
        }
    }

    /// Checks if a rating value is valid for this dataset.
    ///
    /// Valid ratings are: 1, 2, 3, 4, 7, 8, 9, 10.
    /// Ratings 5 and 6 are not present in the dataset.
    ///
    /// # Arguments
    /// * `rating` - The rating value to check.
    ///
    /// # Returns
    /// * `true` - If the rating is valid.
    /// * `false` - If the rating is invalid (5, 6, or out of range).
    fn is_valid_rating(rating: usize) -> bool {
        matches!(rating, 1..=4 | 7..=10)
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
        let mut rating_counts: HashMap<usize, usize> = HashMap::new();

        for sentiment in &["pos", "neg"] {
            let sentiment_dir = split_dir.join(sentiment);
            if !sentiment_dir.exists() {
                continue;
            }

            let entries = fs::read_dir(&sentiment_dir)?;

            for entry in entries {
                let entry = entry?;
                let path = entry.path();

                if path.extension().is_some_and(|ext| ext == "txt") {
                    if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                        if let Some(rating) = Self::parse_rating_from_filename(filename) {
                            let rating = rating as usize;
                            if Self::is_valid_rating(rating) {
                                samples.push((path, rating));
                                *rating_counts.entry(rating).or_insert(0) += 1;
                            }
                        }
                    }
                }
            }
        }

        samples.sort_by(|a, b| {
            let rating_cmp = a.1.cmp(&b.1);
            if rating_cmp != std::cmp::Ordering::Equal {
                return rating_cmp;
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
        for rating in [1usize, 2, 3, 4, 7, 8, 9, 10] {
            if let Some(count) = rating_counts.get(&rating) {
                println!("  Rating {}: {} samples", rating, count);
            }
        }

        Ok(samples)
    }

    /// Returns the human-readable label name for a given rating.
    ///
    /// The label name is the string representation of the rating.
    /// Valid ratings are: 1, 2, 3, 4, 7, 8, 9, 10.
    ///
    /// # Arguments
    /// * `rating` - The rating value.
    ///
    /// # Returns
    /// * `Some(&str)` - The label name if the rating is valid.
    /// * `None` - If the rating is invalid (5, 6, or out of range).
    ///
    /// # Examples
    /// ```rust
    /// use nove::dataset::resource::AclImdb10;
    ///
    /// let label_name = AclImdb10::label_name(1);
    /// assert_eq!(label_name, Some("1"));
    ///
    /// let label_name = AclImdb10::label_name(7);
    /// assert_eq!(label_name, Some("7"));
    ///
    /// let label_name = AclImdb10::label_name(5);
    /// assert_eq!(label_name, None);
    /// ```
    pub fn label_name(rating: usize) -> Option<&'static str> {
        match rating {
            1 => Some("1"),
            2 => Some("2"),
            3 => Some("3"),
            4 => Some("4"),
            7 => Some("7"),
            8 => Some("8"),
            9 => Some("9"),
            10 => Some("10"),
            _ => None,
        }
    }
}

/// ACL IMDB rating classification dataset containing samples for a specific split.
///
/// Each sample is represented as a tuple of (review_text, rating) where:
/// - `review_text` is the text content of the movie review (read from file)
/// - `rating` is the star rating (1, 2, 3, 4, 7, 8, 9, or 10)
///
/// Ratings 5 and 6 are NOT present in this dataset.
///
/// This struct cannot be instantiated directly. Use [`AclImdb10::train()`] or
/// [`AclImdb10::test()`] to obtain an `AclImdb10Dataset` instance.
///
/// # Fields
/// * `samples` - A vector of (review_path, rating) tuples.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AclImdb10Dataset {
    samples: Vec<(PathBuf, usize)>,
}

impl AclImdb10Dataset {
    /// Returns the number of samples for each rating.
    ///
    /// # Returns
    /// * `HashMap<usize, usize>` - A map where keys are ratings (1, 2, 3, 4, 7, 8, 9, 10) and values are the number of samples for each rating.
    pub fn rating_distribution(&self) -> HashMap<usize, usize> {
        let mut distribution = HashMap::new();
        for (_, rating) in &self.samples {
            *distribution.entry(*rating).or_insert(0) += 1;
        }
        distribution
    }
}

impl Dataset for AclImdb10Dataset {
    type Item = (String, usize);

    fn get(&self, index: usize) -> Result<Self::Item, DatasetError> {
        let (path, rating) = self
            .samples
            .get(index)
            .ok_or(DatasetError::IndexOutOfBounds(index, self.samples.len()))?;
        let content = fs::read_to_string(path)?;
        Ok((content, *rating))
    }

    fn len(&self) -> Result<usize, DatasetError> {
        Ok(self.samples.len())
    }
}
