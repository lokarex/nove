//! The `dataloader` module defines the `Dataloader` trait and provides some
//! implemented data loader.

pub mod util;

/// The `Dataloader` trait defines the interface for a data loader.
/// Every data loader should implement this trait.
///
/// Required Type Parameters:
/// * `Output` - The type of the output data.
///
/// Required Methods:
/// * `next` - Returns the next data.
pub trait Dataloader {
    /// The type of the output data.
    type Output;

    /// Returns the next data.
    ///
    /// Returns:
    /// * `Some(data)` - The next data.
    /// * `None` - There is no more data.
    fn next(&mut self) -> Option<Self::Output>;
}
