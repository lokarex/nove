use std::fs::File;
use std::path::Path;

use flate2::read::GzDecoder;
use indicatif::{ProgressBar, ProgressStyle};
use tar::Archive;

use crate::DatasetError;

/// Archive format type for extraction.
///
/// # Variants
/// * `TarGz` - Tar archive compressed with gzip (.tar.gz or .tgz).
/// * `Tar` - Uncompressed tar archive (.tar).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchiveFormat {
    TarGz,
    Tar,
}

impl ArchiveFormat {
    /// Detect archive format from file extension.
    ///
    /// # Arguments
    /// * `path` - The path to the archive file.
    ///
    /// # Returns
    /// * `Some(ArchiveFormat)` - The detected archive format.
    /// * `None` - The format could not be determined from the extension.
    ///
    /// # Examples
    /// ```
    /// use nove::dataset::common::ArchiveFormat;
    /// use std::path::Path;
    ///
    /// let format = ArchiveFormat::from_path(Path::new("archive.tar.gz")).unwrap();
    /// assert_eq!(format, ArchiveFormat::TarGz);
    /// ```
    pub fn from_path(path: &Path) -> Option<Self> {
        let ext = path.extension()?.to_str()?;
        match ext {
            "tgz" | "gz" => Some(ArchiveFormat::TarGz),
            "tar" => Some(ArchiveFormat::Tar),
            _ => None,
        }
    }
}

/// A configurable archive extractor with progress bar support.
///
/// # Notes
/// * The `Extractor` supports tar and tar.gz archive formats.
/// * Progress bar display can be enabled or disabled.
/// * Archive format can be auto-detected from file extension using [`ArchiveFormat::from_path`].
///
/// # Fields
/// * `format` - The archive format to extract ([`ArchiveFormat::TarGz`] or [`ArchiveFormat::Tar`]).
/// * `show_progress` - Whether to display a progress bar during extraction.
///
/// # Examples
/// ```no_run
/// use nove::dataset::common::{Extractor, ArchiveFormat};
/// use std::path::Path;
///
/// // Create an extractor with default settings
/// let extractor = Extractor::new(ArchiveFormat::TarGz);
/// extractor.extract(Path::new("archive.tar.gz"), Path::new("output")).unwrap();
///
/// // Create an extractor with custom settings
/// let extractor = Extractor::new(ArchiveFormat::TarGz)
///     .with_progress(false);
/// extractor.extract(Path::new("archive.tar.gz"), Path::new("output")).unwrap();
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Extractor {
    format: ArchiveFormat,
    show_progress: bool,
}

impl Extractor {
    /// Create a new `Extractor` with default settings.
    ///
    /// # Default Values
    /// * `format` - The specified archive format.
    /// * `show_progress` - true
    ///
    /// # Arguments
    /// * `format` - The archive format to extract.
    ///
    /// # Returns
    /// * `Self` - A new `Extractor` instance with default settings.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::dataset::common::{Extractor, ArchiveFormat};
    ///
    /// let extractor = Extractor::new(ArchiveFormat::TarGz);
    /// ```
    pub fn new(format: ArchiveFormat) -> Self {
        Self {
            format,
            show_progress: true,
        }
    }

    /// Configure the archive format to extract.
    ///
    /// # Arguments
    /// * `format` - The archive format to extract.
    ///
    /// # Returns
    /// * `Self` - The extractor with the configured format.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::dataset::common::{Extractor, ArchiveFormat};
    ///
    /// let extractor = Extractor::new(ArchiveFormat::Tar)
    ///     .with_format(ArchiveFormat::TarGz);
    /// ```
    pub fn with_format(mut self, format: ArchiveFormat) -> Self {
        self.format = format;
        self
    }

    /// Configure whether to display a progress bar during extraction.
    ///
    /// # Arguments
    /// * `show_progress` - Whether to display a progress bar.
    ///
    /// # Returns
    /// * `Self` - The extractor with the configured progress bar setting.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::dataset::common::{Extractor, ArchiveFormat};
    ///
    /// let extractor = Extractor::new(ArchiveFormat::TarGz)
    ///     .with_progress(false);
    /// ```
    pub fn with_progress(mut self, show_progress: bool) -> Self {
        self.show_progress = show_progress;
        self
    }

    /// Extract an archive to the specified destination.
    ///
    /// # Arguments
    /// * `archive_path` - The path to the archive file to extract.
    /// * `dest` - The destination directory where files will be extracted.
    ///
    /// # Returns
    /// * `Ok(())` - The extraction succeeded.
    /// * `Err(DatasetError)` - The extraction failed.
    ///
    /// # Examples
    /// ```no_run
    /// use nove::dataset::common::{Extractor, ArchiveFormat};
    /// use std::path::Path;
    ///
    /// let extractor = Extractor::new(ArchiveFormat::TarGz);
    /// extractor.extract(Path::new("archive.tar.gz"), Path::new("output")).unwrap();
    /// ```
    pub fn extract(&self, archive_path: &Path, dest: &Path) -> Result<(), DatasetError> {
        self.extract_with_progress(archive_path, dest, self.show_progress)
    }

    /// Internal method to extract an archive with optional progress display.
    ///
    /// # Arguments
    /// * `archive_path` - The path to the archive file to extract.
    /// * `dest` - The destination directory where files will be extracted.
    /// * `show_progress` - Whether to display a progress bar during extraction.
    ///
    /// # Returns
    /// * `Ok(())` - The extraction succeeded.
    /// * `Err(DatasetError)` - The extraction failed.
    fn extract_with_progress(
        &self,
        archive_path: &Path,
        dest: &Path,
        show_progress: bool,
    ) -> Result<(), DatasetError> {
        match self.format {
            ArchiveFormat::TarGz => self.extract_tar_gz(archive_path, dest, show_progress),
            ArchiveFormat::Tar => self.extract_tar(archive_path, dest, show_progress),
        }
    }

    fn extract_tar_gz(
        &self,
        archive_path: &Path,
        dest: &Path,
        show_progress: bool,
    ) -> Result<(), DatasetError> {
        println!("Extracting tar.gz archive...");

        let file = File::open(archive_path)?;
        let decoder = GzDecoder::new(file);
        let mut archive = Archive::new(decoder);

        let entries: Vec<_> = archive
            .entries()
            .map_err(|e| DatasetError::ExtractionError(e.to_string()))?
            .collect();

        let total = entries.len();
        let pb = if show_progress {
            let pb = ProgressBar::new(total as u64);
            pb.set_style(
                ProgressStyle::with_template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} files ({eta})",
                )
                .unwrap()
                .progress_chars("#>-"),
            );
            Some(pb)
        } else {
            None
        };

        let file = File::open(archive_path)?;
        let decoder = GzDecoder::new(file);
        let mut archive = Archive::new(decoder);

        for entry in archive
            .entries()
            .map_err(|e| DatasetError::ExtractionError(e.to_string()))?
        {
            let mut entry = entry.map_err(|e| DatasetError::ExtractionError(e.to_string()))?;
            entry
                .unpack_in(dest)
                .map_err(|e| DatasetError::ExtractionError(e.to_string()))?;
            if let Some(ref pb) = pb {
                pb.inc(1);
            }
        }

        if let Some(pb) = pb {
            pb.finish_with_message("Extraction completed");
        }

        Ok(())
    }

    fn extract_tar(
        &self,
        archive_path: &Path,
        dest: &Path,
        show_progress: bool,
    ) -> Result<(), DatasetError> {
        println!("Extracting tar archive...");

        let file = File::open(archive_path)?;
        let mut archive = Archive::new(file);

        let entries: Vec<_> = archive
            .entries()
            .map_err(|e| DatasetError::ExtractionError(e.to_string()))?
            .collect();

        let total = entries.len();
        let pb = if show_progress {
            let pb = ProgressBar::new(total as u64);
            pb.set_style(
                ProgressStyle::with_template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} files ({eta})",
                )
                .unwrap()
                .progress_chars("#>-"),
            );
            Some(pb)
        } else {
            None
        };

        let file = File::open(archive_path)?;
        let mut archive = Archive::new(file);

        for entry in archive
            .entries()
            .map_err(|e| DatasetError::ExtractionError(e.to_string()))?
        {
            let mut entry = entry.map_err(|e| DatasetError::ExtractionError(e.to_string()))?;
            entry
                .unpack_in(dest)
                .map_err(|e| DatasetError::ExtractionError(e.to_string()))?;
            if let Some(ref pb) = pb {
                pb.inc(1);
            }
        }

        if let Some(pb) = pb {
            pb.finish_with_message("Extraction completed");
        }

        Ok(())
    }
}

/// Extract an archive with automatic format detection.
///
/// This is a convenience function that automatically detects the archive format
/// from the file extension and extracts it to the specified destination.
///
/// # Arguments
/// * `archive_path` - The path to the archive file to extract.
/// * `dest` - The destination directory where files will be extracted.
/// * `show_progress` - Whether to display a progress bar during extraction.
///
/// # Returns
/// * `Ok(())` - The extraction succeeded.
/// * `Err(DatasetError)` - The extraction failed or the format could not be detected.
///
/// # Examples
/// ```no_run
/// use nove::dataset::common::extract_archive;
/// use std::path::Path;
///
/// extract_archive(
///     Path::new("archive.tar.gz"),
///     Path::new("output"),
///     true
/// ).unwrap();
/// ```
pub fn extract_archive(
    archive_path: &Path,
    dest: &Path,
    show_progress: bool,
) -> Result<(), DatasetError> {
    let format = ArchiveFormat::from_path(archive_path).ok_or_else(|| {
        DatasetError::ExtractionError(format!("Unknown archive format: {:?}", archive_path))
    })?;
    let extractor = Extractor::new(format).with_progress(show_progress);
    extractor.extract(archive_path, dest)
}
