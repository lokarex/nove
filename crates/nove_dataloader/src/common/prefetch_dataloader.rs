use std::sync::mpsc::{self, Receiver};
use std::thread::{self, JoinHandle};

use crate::{Dataloader, DataloaderError};

/// A wrapper dataloader that prefetches data in a background thread.
///
/// This dataloader wraps any existing dataloader and prefetches batches
/// in a separate thread to hide data loading latency during training.
///
/// # Note
/// * This struct cannot be constructed directly. Use [`PrefetchDataloaderBuilder`]
///   to create instances.
///
/// # Generic Type Parameters
/// * `D` - The type of the inner dataloader.
///
/// # Fields
/// * `receiver` - The receiver for prefetched data from the background thread.
/// * `handle` - The handle for the background thread.
pub struct PrefetchDataloader<D>
where
    D: Dataloader + Send + 'static,
    D::Output: Send + 'static,
{
    receiver: Receiver<Result<Option<D::Output>, DataloaderError>>,
    handle: Option<JoinHandle<D>>,
}

impl<D> Dataloader for PrefetchDataloader<D>
where
    D: Dataloader + Send + 'static,
    D::Output: Send + 'static,
{
    type Output = D::Output;

    fn next(&mut self) -> Result<Option<Self::Output>, DataloaderError> {
        self.receiver
            .recv()
            .map_err(|_| DataloaderError::OtherError("Prefetch thread disconnected".to_string()))?
    }

    fn reset(&mut self) -> Result<(), DataloaderError> {
        if let Some(handle) = self.handle.take() {
            let mut dataloader = handle.join().map_err(|_| {
                DataloaderError::OtherError("Failed to join prefetch thread".to_string())
            })?;
            dataloader.reset()?;

            let buffer_size = 2;
            let (sender, receiver) = mpsc::sync_channel(buffer_size);
            self.receiver = receiver;

            self.handle = Some(thread::spawn(move || {
                loop {
                    match dataloader.next() {
                        Ok(Some(batch)) => {
                            if sender.send(Ok(Some(batch))).is_err() {
                                break;
                            }
                        }
                        Ok(None) => {
                            let _ = sender.send(Ok(None));
                            break;
                        }
                        Err(e) => {
                            let _ = sender.send(Err(e));
                            break;
                        }
                    }
                }
                dataloader
            }));
        }
        Ok(())
    }
}

/// Builder for constructing [`PrefetchDataloader`] instances.
///
/// # Note
/// * The `PrefetchDataloaderBuilder` implements the `Default` trait. So you can use
///   `PrefetchDataloaderBuilder::default()` to create a new builder and then configure it.
///
/// # Generic Type Parameters
/// * `D` - The type of the inner dataloader.
pub struct PrefetchDataloaderBuilder<D>
where
    D: Dataloader + Send + 'static,
    D::Output: Send + 'static,
{
    dataloader: Option<D>,
    buffer_size: usize,
}

impl<D> Default for PrefetchDataloaderBuilder<D>
where
    D: Dataloader + Send + 'static,
    D::Output: Send + 'static,
{
    fn default() -> Self {
        Self {
            dataloader: None,
            buffer_size: 2,
        }
    }
}

impl<D> PrefetchDataloaderBuilder<D>
where
    D: Dataloader + Send + 'static,
    D::Output: Send + 'static,
{
    /// Configures the inner dataloader.
    ///
    /// # Arguments
    /// * `dataloader` - The inner dataloader to wrap.
    ///
    /// # Returns
    /// * `&mut Self` - The builder itself.
    pub fn dataloader(&mut self, dataloader: D) -> &mut Self {
        self.dataloader = Some(dataloader);
        self
    }

    /// Configures the prefetch buffer size.
    ///
    /// # Arguments
    /// * `buffer_size` - The number of items to prefetch. Default is 2.
    ///
    /// # Returns
    /// * `&mut Self` - The builder itself.
    pub fn buffer_size(&mut self, buffer_size: usize) -> &mut Self {
        self.buffer_size = buffer_size;
        self
    }

    /// Builds the prefetch dataloader.
    ///
    /// This method spawns a background thread to prefetch data from the inner dataloader.
    ///
    /// # Returns
    /// * `Ok(PrefetchDataloader<D>)` - The built prefetch dataloader.
    /// * `Err(DataloaderError)` - Some errors occur while building.
    pub fn build(&mut self) -> Result<PrefetchDataloader<D>, DataloaderError> {
        let mut dataloader = self
            .dataloader
            .take()
            .ok_or(DataloaderError::MissingArgument(
                "dataloader in PrefetchDataloaderBuilder".to_string(),
            ))?;
        if self.buffer_size == 0 {
            return Err(DataloaderError::OtherError(
                "buffer_size in PrefetchDataloaderBuilder must be greater than 0".to_string(),
            ));
        }

        let (sender, receiver) = mpsc::sync_channel(self.buffer_size);

        let handle = thread::spawn(move || {
            loop {
                match dataloader.next() {
                    Ok(Some(batch)) => {
                        if sender.send(Ok(Some(batch))).is_err() {
                            break;
                        }
                    }
                    Ok(None) => {
                        let _ = sender.send(Ok(None));
                        break;
                    }
                    Err(e) => {
                        let _ = sender.send(Err(e));
                        break;
                    }
                }
            }
            dataloader
        });

        Ok(PrefetchDataloader {
            receiver,
            handle: Some(handle),
        })
    }
}
