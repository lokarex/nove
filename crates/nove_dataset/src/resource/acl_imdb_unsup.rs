use std::fs;
use std::path::{Path, PathBuf};

use crate::common::{ChecksumType, download_and_verify, extract_archive};
use crate::{Dataset, DatasetError};

const ACL_IMDB_URL: &str = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz";
const ACL_IMDB_SHA256: &str = "c40f74a18d3b61f90feba1e17730e0d38e8b97c05fde7008942e91923d1658fe";

/// ACL IMDB unsupervised dataset manager.
///
/// The ACL IMDB dataset contains 50,000 unlabeled movie reviews for unsupervised learning.
/// These reviews are stored in the `train/unsup/` directory.
///
/// Unlike the supervised versions ([`AclImdb2`](crate::resource::AclImdb2) and
/// [`AclImdb10`](crate::resource::AclImdb10)), this dataset does not provide labels.
/// Each sample is just the path to a review text file.
///
/// The dataset is downloaded from Stanford AI Lab and extracted to a local directory.
/// Use [`AclImdbUnsup::unsup()`] method to get the unsupervised dataset.
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
/// │   └── train/
/// │       └── unsup/
/// │           ├── 0_0.txt
/// │           ├── 1_0.txt
/// │           └── ...
/// ```
///
/// # Fields
/// * `dataset_dir` - The root directory where the dataset is stored.
///
/// # Examples
/// ```rust,no_run
/// use nove::dataset::resource::AclImdbUnsup;
/// use nove::dataset::Dataset;
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // Create IMDB unsupervised manager (downloads and extracts if needed)
///     let imdb = AclImdbUnsup::new("path/to/data")?;
///
///     // Get unsupervised dataset
///     let unsup_dataset = imdb.unsup()?;
///     let review_path = unsup_dataset.get(0)?;
///     println!("Review: {:?}", review_path);
///     println!("Total unsupervised samples: {}", unsup_dataset.len()?);
///     Ok(())
/// }
/// ```
///
/// # See Also
/// * [`AclImdb2`](crate::resource::AclImdb2) - IMDB dataset with binary sentiment classification
/// * [`AclImdb10`](crate::resource::AclImdb10) - IMDB dataset with rating classification
pub struct AclImdbUnsup {
    dataset_dir: PathBuf,
}

impl AclImdbUnsup {
    /// Creates a new ACL IMDB unsupervised manager.
    ///
    /// If the dataset is not present in the specified directory, it will be
    /// downloaded and extracted automatically.
    ///
    /// # Arguments
    /// * `root_dir` - The root directory where the dataset will be stored.
    ///
    /// # Returns
    /// * `Ok(Self)` - A new AclImdbUnsup instance.
    /// * `Err(DatasetError)` - An error occurred during creation.
    ///
    /// # Examples
    /// ```rust,no_run
    /// use nove::dataset::resource::AclImdbUnsup;
    ///
    /// fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let imdb = AclImdbUnsup::new("data/imdb")?;
    ///     Ok(())
    /// }
    /// ```
    pub fn new<P: AsRef<Path>>(root_dir: P) -> Result<Self, DatasetError> {
        let root_dir = root_dir.as_ref().to_path_buf();
        let dataset_dir = root_dir.join("aclImdb");

        Self::download_and_extract(&root_dir)?;

        Ok(Self { dataset_dir })
    }

    /// Returns the unsupervised dataset.
    ///
    /// # Returns
    /// * `Ok(AclImdbUnsupDataset)` - The unsupervised dataset.
    /// * `Err(DatasetError)` - An error occurred during loading.
    pub fn unsup(&self) -> Result<AclImdbUnsupDataset, DatasetError> {
        let samples = Self::load_samples(&self.dataset_dir)?;
        if samples.is_empty() {
            return Err(DatasetError::EmptyDataset);
        }
        Ok(AclImdbUnsupDataset { samples })
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

    fn load_samples(dataset_dir: &Path) -> Result<Vec<PathBuf>, DatasetError> {
        let unsup_dir = dataset_dir.join("train").join("unsup");

        if !unsup_dir.exists() {
            return Err(DatasetError::InvalidLabelDir(unsup_dir));
        }

        let mut samples = Vec::new();

        let entries = fs::read_dir(&unsup_dir)?;
        for entry in entries {
            let entry = entry?;
            let path = entry.path();

            if path.extension().is_some_and(|ext| ext == "txt") {
                samples.push(path);
            }
        }

        samples.sort();

        println!("Loaded {} unsupervised samples", samples.len());

        Ok(samples)
    }
}

/// ACL IMDB unsupervised dataset containing unlabeled reviews.
///
/// Each sample is represented as a `PathBuf` pointing to a review text file.
/// Unlike the supervised datasets, this does not include labels.
///
/// This struct cannot be instantiated directly. Use [`AclImdbUnsup::unsup()`]
/// to obtain an `AclImdbUnsupDataset` instance.
///
/// # Fields
/// * `samples` - A vector of review file paths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AclImdbUnsupDataset {
    samples: Vec<PathBuf>,
}

impl Dataset for AclImdbUnsupDataset {
    type Item = PathBuf;

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
