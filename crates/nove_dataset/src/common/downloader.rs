use std::fs::File;
use std::io::Write;
use std::path::Path;

use indicatif::{ProgressBar, ProgressStyle};
use sha2::{Digest, Sha256};

use crate::DatasetError;

/// Checksum type for download verification.
///
/// # Variants
/// * `Md5` - MD5 checksum (128-bit).
/// * `Sha256` - SHA-256 checksum (256-bit).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChecksumType {
    Md5,
    Sha256,
}

/// A configurable file downloader with retry support and optional checksum computation.
///
/// # Notes
/// * The `Downloader` implements the `Default` trait, so you can use `Downloader::default()`
///   to create a downloader with default values.
/// * The downloader supports automatic retry on failure with configurable delay.
/// * Progress bar display can be enabled or disabled.
/// * Checksum (MD5 or SHA-256) can be computed during download without extra file I/O.
///
/// # Fields
/// * `max_retries` - Maximum number of retry attempts on failure.
/// * `retry_delay_ms` - Delay in milliseconds between retry attempts.
/// * `show_progress` - Whether to display a progress bar during download.
/// * `checksum_type` - Optional checksum type to compute during download.
///
/// # Examples
/// ```no_run
/// use nove::dataset::common::{Downloader, ChecksumType};
/// use std::path::Path;
///
/// // Create a downloader with default settings
/// let downloader = Downloader::new();
/// let checksum = downloader.download("https://example.com/file.tar.gz", Path::new("file.tar.gz")).unwrap();
///
/// // Create a downloader with custom settings
/// let checksum = Downloader::new()
///     .with_max_retries(5)
///     .with_retry_delay(2000)
///     .with_show_progress(false)
///     .with_checksum(ChecksumType::Sha256)
///     .download("https://example.com/file.tar.gz", Path::new("file.tar.gz"))
///     .unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct Downloader {
    max_retries: u32,
    retry_delay_ms: u64,
    show_progress: bool,
    checksum_type: Option<ChecksumType>,
}

impl Downloader {
    /// Create a new `Downloader` with default settings.
    ///
    /// # Default Values
    /// * `max_retries` - 3
    /// * `retry_delay_ms` - 1000
    /// * `show_progress` - true
    /// * `checksum_type` - None
    ///
    /// # Returns
    /// * `Self` - A new `Downloader` instance with default settings.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::dataset::common::Downloader;
    ///
    /// let downloader = Downloader::new();
    /// ```
    pub fn new() -> Self {
        Self {
            max_retries: 3,
            retry_delay_ms: 1000,
            show_progress: true,
            checksum_type: None,
        }
    }

    /// Configure the maximum number of retry attempts on failure.
    ///
    /// # Arguments
    /// * `max_retries` - The maximum number of retry attempts.
    ///
    /// # Returns
    /// * `Self` - The downloader with the configured retry count.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::dataset::common::Downloader;
    ///
    /// let downloader = Downloader::new().with_max_retries(5);
    /// ```
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Configure the delay in milliseconds between retry attempts.
    ///
    /// # Arguments
    /// * `delay_ms` - The delay in milliseconds between retries.
    ///
    /// # Returns
    /// * `Self` - The downloader with the configured retry delay.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::dataset::common::Downloader;
    ///
    /// let downloader = Downloader::new().with_retry_delay(2000);
    /// ```
    pub fn with_retry_delay(mut self, delay_ms: u64) -> Self {
        self.retry_delay_ms = delay_ms;
        self
    }

    /// Configure whether to display a progress bar during download.
    ///
    /// # Arguments
    /// * `show_progress` - Whether to display a progress bar.
    ///
    /// # Returns
    /// * `Self` - The downloader with the configured progress bar setting.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::dataset::common::Downloader;
    ///
    /// let downloader = Downloader::new().with_show_progress(false);
    /// ```
    pub fn with_show_progress(mut self, show_progress: bool) -> Self {
        self.show_progress = show_progress;
        self
    }

    /// Configure the checksum type to compute during download.
    ///
    /// The checksum is computed during download without additional file I/O,
    /// making it more efficient than computing after download.
    ///
    /// # Arguments
    /// * `checksum_type` - The type of checksum to compute (`Md5` or `Sha256`).
    ///
    /// # Returns
    /// * `Self` - The downloader with the configured checksum type.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::dataset::common::{Downloader, ChecksumType};
    ///
    /// let downloader = Downloader::new().with_checksum(ChecksumType::Sha256);
    /// ```
    pub fn with_checksum(mut self, checksum_type: ChecksumType) -> Self {
        self.checksum_type = Some(checksum_type);
        self
    }

    /// Disable checksum computation during download.
    ///
    /// # Returns
    /// * `Self` - The downloader with checksum computation disabled.
    ///
    /// # Examples
    /// ```
    /// use nove::dataset::common::{Downloader, ChecksumType};
    ///
    /// let downloader = Downloader::new()
    ///     .with_checksum(ChecksumType::Sha256)
    ///     .without_checksum();
    /// ```no_run
    pub fn without_checksum(mut self) -> Self {
        self.checksum_type = None;
        self
    }

    /// Download a file from the given URL to the specified destination.
    ///
    /// This method will automatically retry on failure up to `max_retries` times,
    /// with `retry_delay_ms` milliseconds between each attempt.
    ///
    /// # Arguments
    /// * `url` - The URL to download from.
    /// * `dest` - The destination file path.
    ///
    /// # Returns
    /// * `Ok(Some(String))` - The download succeeded and the checksum was computed.
    /// * `Ok(None)` - The download succeeded but no checksum was configured.
    /// * `Err(DatasetError)` - The download failed after all retry attempts.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::dataset::common::{Downloader, ChecksumType};
    /// use std::path::Path;
    ///
    /// let downloader = Downloader::new().with_checksum(ChecksumType::Sha256);
    /// let checksum = downloader.download("https://example.com/file.tar.gz", Path::new("file.tar.gz")).unwrap();
    /// if let Some(hash) = checksum {
    ///     println!("SHA-256: {}", hash);
    /// }
    /// ```
    pub fn download(&self, url: &str, dest: &Path) -> Result<Option<String>, DatasetError> {
        let mut last_error = None;

        for attempt in 1..=self.max_retries {
            if attempt > 1 {
                println!(
                    "Retrying download (attempt {}/{}): {}",
                    attempt, self.max_retries, url
                );
                std::thread::sleep(std::time::Duration::from_millis(self.retry_delay_ms));
            }

            match self.try_download(url, dest) {
                Ok(checksum) => return Ok(checksum),
                Err(e) => {
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or(DatasetError::DownloadError(
            "Unknown download error".to_string(),
        )))
    }

    fn try_download(&self, url: &str, dest: &Path) -> Result<Option<String>, DatasetError> {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| DatasetError::DownloadError(e.to_string()))?;

        rt.block_on(async {
            use futures_util::StreamExt;

            let response = reqwest::Client::new()
                .get(url)
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

            let pb = if self.show_progress {
                let pb = ProgressBar::new(total_size);
                pb.set_style(
                    ProgressStyle::with_template(
                        "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})",
                    )
                    .unwrap()
                    .progress_chars("#>-"),
                );
                Some(pb)
            } else {
                None
            };

            let mut file = File::create(dest)?;
            let mut downloaded: u64 = 0;
            let mut stream = response.bytes_stream();

            let mut md5_hasher = md5::Context::new();
            let mut sha256_hasher = Sha256::new();

            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(|e| DatasetError::DownloadError(e.to_string()))?;
                file.write_all(&chunk)?;

                if let Some(checksum_type) = self.checksum_type {
                    match checksum_type {
                        ChecksumType::Md5 => md5_hasher.consume(&chunk),
                        ChecksumType::Sha256 => sha256_hasher.update(&chunk),
                    }
                }

                downloaded += chunk.len() as u64;
                if let Some(ref pb) = pb {
                    pb.set_position(downloaded);
                }
            }

            if let Some(pb) = pb {
                pb.finish_with_message("Download completed");
            }

            let checksum = match self.checksum_type {
                Some(ChecksumType::Md5) => {
                    let hash = md5_hasher.compute();
                    Some(format!("{:x}", hash))
                }
                Some(ChecksumType::Sha256) => {
                    let hash = sha256_hasher.finalize();
                    Some(format!("{:x}", hash))
                }
                None => None,
            };

            Ok(checksum)
        })
    }
}

impl Default for Downloader {
    fn default() -> Self {
        Self::new()
    }
}

/// Download a file and verify its checksum.
///
/// This is a convenience function that wraps the `Downloader` to download a file
/// and verify its checksum in one operation.
///
/// # Arguments
/// * `url` - The URL to download from.
/// * `dest` - The destination file path.
/// * `checksum_type` - The type of checksum to verify (MD5 or SHA-256).
/// * `expected_checksum` - The expected checksum (hex string, case insensitive).
/// * `show_progress` - Whether to display a progress bar during download.
///
/// # Returns
/// * `Ok(())` - The download succeeded and the checksum matches.
/// * `Err(DatasetError)` - The download failed or the checksum doesn't match.
///
/// # Examples
/// ```no_run
/// use nove::dataset::common::{download_and_verify, ChecksumType};
/// use std::path::Path;
///
/// download_and_verify(
///     "https://example.com/file.tar.gz",
///     Path::new("file.tar.gz"),
///     ChecksumType::Sha256,
///     "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
///     true
/// ).unwrap();
/// ```
pub fn download_and_verify(
    url: &str,
    dest: &Path,
    checksum_type: ChecksumType,
    expected_checksum: &str,
    show_progress: bool,
) -> Result<(), DatasetError> {
    let downloader = Downloader::new()
        .with_checksum(checksum_type)
        .with_show_progress(show_progress);

    let checksum = downloader.download(url, dest)?;

    match checksum {
        Some(actual_checksum) => {
            if actual_checksum.to_lowercase() == expected_checksum.to_lowercase() {
                Ok(())
            } else {
                let checksum_name = match checksum_type {
                    ChecksumType::Md5 => "MD5",
                    ChecksumType::Sha256 => "SHA256",
                };
                Err(DatasetError::DownloadError(format!(
                    "{} checksum mismatch: expected {}, got {}",
                    checksum_name, expected_checksum, actual_checksum
                )))
            }
        }
        None => {
            let checksum_name = match checksum_type {
                ChecksumType::Md5 => "MD5",
                ChecksumType::Sha256 => "SHA256",
            };
            Err(DatasetError::DownloadError(format!(
                "Failed to compute {} checksum",
                checksum_name
            )))
        }
    }
}

/// Download a file without verification.
///
/// This is a convenience function that wraps the `Downloader` to download a file
/// without any checksum verification.
///
/// # Arguments
/// * `url` - The URL to download from.
/// * `dest` - The destination file path.
/// * `show_progress` - Whether to display a progress bar during download.
///
/// # Returns
/// * `Ok(())` - The download succeeded.
/// * `Err(DatasetError)` - The download failed.
///
/// # Examples
/// ```rust,no_run
/// use nove::dataset::common::download;
/// use std::path::Path;
///
/// download(
///     "https://example.com/file.tar.gz",
///     Path::new("file.tar.gz"),
///     true
/// ).unwrap();
/// ```
pub fn download(url: &str, dest: &Path, show_progress: bool) -> Result<(), DatasetError> {
    let downloader = Downloader::new()
        .without_checksum()
        .with_show_progress(show_progress);
    downloader.download(url, dest)?;
    Ok(())
}
