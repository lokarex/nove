//! The `dataloader` module defines the `Dataloader` trait and provides some
//! implemented data loader.

use nove_dataset::DatasetError;
use thiserror::Error;

pub mod common;
pub mod resource;

#[derive(Debug, Error)]
pub enum DataloaderError {
    /// The dataset is empty.
    #[error("The dataset is empty.")]
    EmptyDataset,

    /// Dataset errors from the `nove_dataset` crate.
    #[error(transparent)]
    DatasetError(#[from] DatasetError),

    /// Missing argument.
    #[error("Missing argument: {0}")]
    MissingArgument(String),

    /// Other errors.
    #[error("Other errors: {0}")]
    OtherError(String),
}

/// The `Dataloader` trait defines the interface for a data loader.
/// Every data loader should implement this trait.
///
/// # Required Type Parameters:
/// * `Output` - The type of the output data.
///
/// # Required Methods:
/// * `next` - Returns the next data.
/// * `reset` - Resets the dataloader to the initial state.
pub trait Dataloader {
    /// The type of the output data.
    type Output;

    /// Returns the next data.
    ///
    /// Returns:
    /// * `Ok(Some(Self::Output))` - The next data.
    /// * `Ok(None)` - There is no more data.
    /// * `Err(DataloaderError)` - Some errors occur.
    fn next(&mut self) -> Result<Option<Self::Output>, DataloaderError>;

    /// Resets the dataloader to the initial state.
    ///
    /// Returns:
    /// * `Ok(())` - The dataloader is reset successfully.
    /// * `Err(DataloaderError)` - Some errors occur.
    fn reset(&mut self) -> Result<(), DataloaderError>;
}
