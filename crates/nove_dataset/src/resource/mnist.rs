use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use flate2::read::GzDecoder;
use indicatif::{ProgressBar, ProgressStyle};
use sha2::{Digest, Sha256};
use tar::Archive;

use crate::{Dataset, DatasetError};

const MNIST_PNG_URL: &str = "https://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz";
const MNIST_PNG_SHA256: &str = "9e18edaa3a08b065d8f80a019ca04329e6d9b3e391363414a9bd1ada30563672";

/// MNIST handwritten digit dataset manager.
///
/// The MNIST database (Modified National Institute of Standards and Technology database)
/// is a large database of handwritten digits commonly used for training various image
/// processing systems. It contains 60,000 training images and 10,000 testing images.
///
/// The dataset is downloaded from a remote source and extracted to a local directory.
/// Use [`Mnist::train()`] or [`Mnist::test()`] methods to get the specific split of the dataset.
///
/// # Data Source
/// The PNG version of MNIST is downloaded from:
/// <https://github.com/myleott/mnist_png>
///
/// # License
/// The original MNIST dataset is made available under the terms of the
/// [Creative Commons Attribution-Share Alike 3.0 license](https://creativecommons.org/licenses/by-sa/3.0/).
/// The `mnist_png` repository converts the original binary format to PNG images.
///
/// # Citation
/// If you use this dataset in your research, please cite:
/// ```text
/// @article{lecun1998mnist,
///   title={Gradient-based learning applied to document recognition},
///   author={LeCun, Yann and Bottou, L{\'e}on and Bengio, Yoshua and Haffner, Patrick},
///   journal={Proceedings of the IEEE},
///   volume={86},
///   number={11},
///   pages={2278--2324},
///   year={1998},
///   publisher={IEEE}
/// }
/// ```
///
/// # Directory Structure
/// After extraction, the dataset will have the following structure:
/// ```text
/// <root_dir>/
/// ├── mnist_png/
/// │   ├── training/
/// │   │   ├── 0/
/// │   │   │   ├── 1.png
/// │   │   │   ├── 2.png
/// │   │   │   └── ...
/// │   │   ├── 1/
/// │   │   ├── ...
/// │   │   └── 9/
/// │   └── testing/
/// │       ├── 0/
/// │       ├── ...
/// │       └── 9/
/// ```
///
/// # Fields
/// * `dataset_dir` - The root directory where the dataset is stored.
///
/// # Examples
/// ```rust,no_run
/// use nove::dataset::resource::Mnist;
/// use nove::dataset::Dataset;
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // Create MNIST manager (downloads and extracts if needed)
///     let mnist = Mnist::new("path/to/data")?;
///
///     // Get training dataset
///     let train_dataset = mnist.train()?;
///     let (image_path, label) = train_dataset.get(0)?;
///     println!("Train Image: {}, Label: {}", image_path, label);
///
///     // Get testing dataset
///     let test_dataset = mnist.test()?;
///     println!("Test samples: {}", test_dataset.len()?);
///     Ok(())
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mnist {
    dataset_dir: PathBuf,
}

/// The split of MNIST dataset (training or testing).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MnistSplit {
    Training,
    Testing,
}

impl Mnist {
    /// Creates a new MNIST manager.
    ///
    /// If the dataset is not present in the specified directory, it will be
    /// downloaded and extracted automatically.
    ///
    /// # Arguments
    /// * `root_dir` - The root directory where the dataset will be stored.
    ///
    /// # Returns
    /// * `Ok(Self)` - A new Mnist instance.
    /// * `Err(DatasetError)` - An error occurred during creation.
    ///
    /// # Examples
    /// ```rust,no_run
    /// use nove::dataset::resource::Mnist;
    ///
    /// fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let mnist = Mnist::new("data/mnist")?;
    ///     Ok(())
    /// }
    /// ```
    pub fn new<P: AsRef<Path>>(root_dir: P) -> Result<Self, DatasetError> {
        let root_dir = root_dir.as_ref().to_path_buf();
        let dataset_dir = root_dir.join("mnist_png");

        Self::download_and_extract(&root_dir)?;

        Ok(Self { dataset_dir })
    }

    /// Returns the training dataset.
    ///
    /// # Returns
    /// * `Ok(MnistDataset)` - The training dataset.
    /// * `Err(DatasetError)` - An error occurred during loading.
    pub fn train(&self) -> Result<MnistDataset, DatasetError> {
        let samples = Self::load_samples(&self.dataset_dir, MnistSplit::Training)?;
        if samples.is_empty() {
            return Err(DatasetError::EmptyDataset);
        }
        Ok(MnistDataset { samples })
    }

    /// Returns the testing dataset.
    ///
    /// # Returns
    /// * `Ok(MnistDataset)` - The testing dataset.
    /// * `Err(DatasetError)` - An error occurred during loading.
    pub fn test(&self) -> Result<MnistDataset, DatasetError> {
        let samples = Self::load_samples(&self.dataset_dir, MnistSplit::Testing)?;
        if samples.is_empty() {
            return Err(DatasetError::EmptyDataset);
        }
        Ok(MnistDataset { samples })
    }

    /// Returns the dataset directory path.
    ///
    /// # Returns
    /// * `&Path` - The path to the dataset directory.
    pub fn dataset_dir(&self) -> &Path {
        &self.dataset_dir
    }

    /// Downloads and extracts the MNIST dataset if it is not already present.
    ///
    /// # Arguments
    /// * `root_dir` - The root directory where the dataset will be stored.
    ///
    /// # Returns
    /// * `Ok(())` - The dataset was downloaded and extracted successfully.
    /// * `Err(DatasetError)` - An error occurred during download or extraction.
    fn download_and_extract(root_dir: &Path) -> Result<(), DatasetError> {
        fs::create_dir_all(root_dir)?;

        let archive_path = root_dir.join("mnist_png.tar.gz");
        let dataset_dir = root_dir.join("mnist_png");

        if !archive_path.exists() {
            Self::download_with_progress(&archive_path)?;
        } else {
            println!("Archive already exists, skipping download");
        }

        if !Self::is_dataset_extracted(&dataset_dir)? {
            Self::verify_checksum(&archive_path)?;
            Self::extract_with_progress(&archive_path, root_dir)?;
        } else {
            println!("Dataset already extracted, skipping extraction");
        }

        Ok(())
    }

    /// Checks if the dataset is already extracted with the expected number of files.
    ///
    /// The MNIST PNG dataset should contain:
    /// - Training set: 60,000 images (per label: 5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949)
    /// - Testing set: 10,000 images (per label: 980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009)
    ///
    /// # Arguments
    /// * `dataset_dir` - The directory containing the extracted dataset.
    ///
    /// # Returns
    /// * `Ok(true)` - The dataset is fully extracted.
    /// * `Ok(false)` - The dataset needs to be extracted.
    /// * `Err(DatasetError)` - An error occurred during checking.
    fn is_dataset_extracted(dataset_dir: &Path) -> Result<bool, DatasetError> {
        if !dataset_dir.exists() {
            return Ok(false);
        }

        const TRAINING_COUNTS: [usize; 10] =
            [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949];
        const TESTING_COUNTS: [usize; 10] = [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009];

        fn count_png_files(dir: &Path) -> usize {
            let entries = match fs::read_dir(dir) {
                Ok(entries) => entries,
                Err(_) => return 0,
            };
            entries
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension().map_or(false, |ext| ext == "png"))
                .count()
        }

        let training_dir = dataset_dir.join("training");
        if !training_dir.exists() {
            return Ok(false);
        }
        for (label, &expected) in TRAINING_COUNTS.iter().enumerate() {
            let label_dir = training_dir.join(label.to_string());
            if !label_dir.exists() {
                return Ok(false);
            }
            let count = count_png_files(&label_dir);
            if count != expected {
                return Ok(false);
            }
        }

        let testing_dir = dataset_dir.join("testing");
        if !testing_dir.exists() {
            return Ok(false);
        }
        for (label, &expected) in TESTING_COUNTS.iter().enumerate() {
            let label_dir = testing_dir.join(label.to_string());
            if !label_dir.exists() {
                return Ok(false);
            }
            let count = count_png_files(&label_dir);
            if count != expected {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Downloads the MNIST archive with a progress bar.
    ///
    /// # Arguments
    /// * `archive_path` - The path where the archive will be saved.
    ///
    /// # Returns
    /// * `Ok(())` - The archive was downloaded successfully.
    /// * `Err(DatasetError)` - An error occurred during download.
    fn download_with_progress(archive_path: &Path) -> Result<(), DatasetError> {
        println!("Downloading MNIST PNG dataset...");

        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| DatasetError::DownloadError(e.to_string()))?;

        rt.block_on(async {
            use futures_util::StreamExt;

            let response = reqwest::Client::new()
                .get(MNIST_PNG_URL)
                .send()
                .await
                .map_err(|e| DatasetError::DownloadError(e.to_string()))?;

            if !response.status().is_success() {
                return Err(DatasetError::DownloadError(format!(
                    "HTTP status: {}",
                    response.status()
                )));
            }

            let total_size = response.content_length().unwrap_or(0);

            let pb = ProgressBar::new(total_size);
            pb.set_style(
                ProgressStyle::with_template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})",
                )
                .unwrap()
                .progress_chars("#>-"),
            );

            let mut file = File::create(archive_path)?;
            let mut downloaded: u64 = 0;
            let mut stream = response.bytes_stream();

            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(|e| DatasetError::DownloadError(e.to_string()))?;
                file.write_all(&chunk)?;
                downloaded += chunk.len() as u64;
                pb.set_position(downloaded);
            }

            pb.finish_with_message("Download completed");

            Ok(())
        })
    }

    /// Verifies the SHA256 checksum of the downloaded archive.
    ///
    /// # Arguments
    /// * `archive_path` - The path to the downloaded archive.
    ///
    /// # Returns
    /// * `Ok(())` - The checksum matches.
    /// * `Err(DatasetError)` - The checksum does not match.
    fn verify_checksum(archive_path: &Path) -> Result<(), DatasetError> {
        println!("Verifying checksum...");

        let mut file = File::open(archive_path)?;
        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; 8192];

        loop {
            let bytes_read = file.read(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }
            hasher.update(&buffer[..bytes_read]);
        }

        let hash = hasher.finalize();
        let hash_hex = format!("{:x}", hash);

        if hash_hex != MNIST_PNG_SHA256 {
            return Err(DatasetError::ChecksumError {
                expected: MNIST_PNG_SHA256.to_string(),
                actual: hash_hex,
            });
        }

        println!("Checksum verified successfully");

        Ok(())
    }

    /// Extracts the MNIST archive with a progress bar.
    ///
    /// # Arguments
    /// * `archive_path` - The path to the downloaded archive.
    /// * `root_dir` - The directory where the archive will be extracted.
    ///
    /// # Returns
    /// * `Ok(())` - The archive was extracted successfully.
    /// * `Err(DatasetError)` - An error occurred during extraction.
    fn extract_with_progress(archive_path: &Path, root_dir: &Path) -> Result<(), DatasetError> {
        println!("Extracting MNIST PNG dataset...");

        let file = File::open(archive_path)?;
        let decoder = GzDecoder::new(file);
        let mut archive = Archive::new(decoder);

        let entries: Vec<_> = archive
            .entries()
            .map_err(|e| DatasetError::ExtractionError(e.to_string()))?
            .collect();

        let total = entries.len();
        let pb = ProgressBar::new(total as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} files ({eta})",
            )
            .unwrap()
            .progress_chars("#>-"),
        );

        let file = File::open(archive_path)?;
        let decoder = GzDecoder::new(file);
        let mut archive = Archive::new(decoder);

        for entry in archive
            .entries()
            .map_err(|e| DatasetError::ExtractionError(e.to_string()))?
        {
            let mut entry = entry.map_err(|e| DatasetError::ExtractionError(e.to_string()))?;
            entry
                .unpack_in(root_dir)
                .map_err(|e| DatasetError::ExtractionError(e.to_string()))?;
            pb.inc(1);
        }

        pb.finish_with_message("Extraction completed");

        Ok(())
    }

    /// Loads sample paths and labels from the dataset directory.
    ///
    /// # Arguments
    /// * `dataset_dir` - The directory containing the extracted dataset.
    /// * `split` - The dataset split to load (training or testing).
    ///
    /// # Returns
    /// * `Ok(Vec<(String, usize)>)` - A vector of (image_path, label) tuples.
    /// * `Err(DatasetError)` - An error occurred during loading.
    fn load_samples(
        dataset_dir: &Path,
        split: MnistSplit,
    ) -> Result<Vec<(String, usize)>, DatasetError> {
        let split_dir = match split {
            MnistSplit::Training => dataset_dir.join("training"),
            MnistSplit::Testing => dataset_dir.join("testing"),
        };

        let mut samples = Vec::new();
        let mut label_counts: HashMap<usize, usize> = HashMap::new();

        for label in 0u8..=9 {
            let label_usize = label as usize;
            let label_dir = split_dir.join(label.to_string());
            if !label_dir.exists() {
                continue;
            }

            let entries = fs::read_dir(&label_dir)?;
            let mut count = 0;

            for entry in entries {
                let entry = entry?;
                let path = entry.path();

                if path.extension().map_or(false, |ext| ext == "png") {
                    if let Some(path_str) = path.to_str() {
                        samples.push((path_str.to_string(), label_usize));
                        count += 1;
                    }
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
                MnistSplit::Training => "training",
                MnistSplit::Testing => "testing",
            }
        );
        for label in 0usize..=9 {
            if let Some(count) = label_counts.get(&label) {
                println!("  Label {}: {} samples", label, count);
            }
        }

        Ok(samples)
    }
}

/// MNIST dataset containing samples for a specific split (training or testing).
///
/// Each sample is represented as a tuple of (image_path, label).
///
/// This struct cannot be instantiated directly. Use [`Mnist::train()`] or
/// [`Mnist::test()`] to obtain a `MnistDataset` instance.
///
/// # Fields
/// * `samples` - A vector of (image_path, label) tuples.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MnistDataset {
    samples: Vec<(String, usize)>,
}

impl MnistDataset {
    /// Returns the number of samples for each label.
    pub fn label_distribution(&self) -> HashMap<usize, usize> {
        let mut distribution = HashMap::new();
        for (_, label) in &self.samples {
            *distribution.entry(*label).or_insert(0) += 1;
        }
        distribution
    }
}

impl Dataset for MnistDataset {
    type Item = (String, usize);

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
