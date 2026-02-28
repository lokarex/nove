use std::marker::PhantomData;

use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};

use crate::{Dataloader, DataloaderError};
use nove_dataset::Dataset;

/// The `BatchDataloader` struct provides a basic data loader to
/// shuffle the dataset, load the data, process the data, and collate the
/// datas into batches.
///
/// # Note
/// * When create a `BatchDataloader` from `BatchDataloaderBuilder`,
///   the specific `BatchDataloader` type must be determined. It means
///   3 generic type parameters must be set manually, so please clearly
///   learn about the generic type parameters of `BatchDataloader`.
///
/// # Generic Type Parameters
/// * `D`(Need to be manually set) - The type of the inner dataset.
/// * `O`(Need to be manually set) - The type of the processed data which is the output of the process function.
/// * `B`(Need to be manually set) - The type of the batched data which is the output of the collate function
///   and the final output of the dataloader.
/// * `P` - The type of the process function. It needs a function that takes
///   an item(The `D::Item` type) from the dataset as input and returns the processed
///   data(The `Result<O, DataloaderError>` type) as output.
/// * `C` - The type of the collate function. It needs a function that takes
///   a vector of processed data(The `Vec<O>` type) from the process function as input and returns
///   the batched data(The `Result<B, DataloaderError>` type) as output.
///
/// # Fields
/// * `dataset` - The inner dataset.
/// * `batch_size` - The batch size.
/// * `process_fn` - The process function.
/// * `collate_fn` - The collate function.
/// * `shuffle_seed` - The shuffle seed.
/// * `index` - The current index.
/// * `dataset_len` - The cached length of the dataset.
/// * `datas` - The processed data.
/// * `indices` - The indices of the dataset which would be shuffled before the first batch if the shuffle_seed is not `None`.
/// * `phantom` - The phantom data only used to marks that the `B` generic type parameter which would be used later.
///
/// # Examples
/// ```rust
/// use nove::dataset::common::VecDataset;
/// use nove::dataloader::common::{BatchDataloaderBuilder, BatchDataloader};
/// use nove::dataloader::DataloaderError;
///
/// let dataset = VecDataset::from_vec(vec![1usize, 2usize, 3usize]);
/// let mut dataloader: BatchDataloader<VecDataset<usize>, usize, Vec<usize>, _, _> = BatchDataloaderBuilder::default()
///     .dataset(dataset)       // Required configuration
///     .batch_size(2)           // Required configuration
///     .process_fn(             // Required configuration
///         |x: usize| Ok::<usize, DataloaderError>(x)
///     )
///     .collate_fn(             // Required configuration
///         |x: Vec<usize>| Ok::<Vec<usize>, DataloaderError>(x)
///     )
///     .shuffle_seed(Some(42))  // Optional configuration
///     .build()
///     .unwrap();
/// ```
pub struct BatchDataloader<D: Dataset, O, B, P, C> {
    dataset: D,
    batch_size: usize,
    process_fn: P,
    collate_fn: C,
    shuffle_seed: Option<usize>,
    index: usize,
    dataset_len: usize,
    datas: Vec<O>,
    indices: Vec<usize>,
    phantom: std::marker::PhantomData<B>,
}

impl<D: Dataset, O, B, P, C> BatchDataloader<D, O, B, P, C>
where
    P: Fn(D::Item) -> Result<O, DataloaderError>,
    C: Fn(Vec<O>) -> Result<B, DataloaderError>,
{
    fn shuffle_indices(&mut self, seed: usize) {
        self.indices
            .shuffle(&mut StdRng::seed_from_u64(seed as u64));
    }
}

impl<D: Dataset, O, B, P, C> Dataloader for BatchDataloader<D, O, B, P, C>
where
    P: Fn(D::Item) -> Result<O, DataloaderError>,
    C: Fn(Vec<O>) -> Result<B, DataloaderError>,
{
    type Output = B;

    fn next(&mut self) -> Result<Option<Self::Output>, DataloaderError> {
        if self.index == 0 {
            if let Some(seed) = self.shuffle_seed {
                self.shuffle_indices(seed);
            }
        }

        if self.index >= self.dataset_len {
            return Ok(None);
        }

        let end = (self.index + self.batch_size).min(self.dataset_len);

        for &idx in &self.indices[self.index..end] {
            let item = self.dataset.get(idx)?;
            self.datas.push((self.process_fn)(item)?);
        }
        self.index = end;

        let batch = std::mem::take(&mut self.datas);
        self.datas.reserve(self.batch_size);

        Ok(Some((self.collate_fn)(batch)?))
    }

    fn reset(&mut self) -> Result<(), DataloaderError> {
        self.index = 0;
        Ok(())
    }
}

/// The builder for the `BatchDataloader`.
///
/// # Notes
/// * The `BatchDataloaderBuilder` implements the `Default` trait. So you can use
///   `BatchDataloaderBuilder::default()` to create a new builder and then configure it.
///
/// # Required Arguments
/// * `dataset` - The dataset to be used by the dataloader.
/// * `batch_size` - The batch size.
/// * `process_fn` - The process function.
/// * `collate_fn` - The collate function.
///
/// # Optional Arguments
/// * `shuffle_seed` - The shuffle seed. Default is `None` (no shuffling before the first batch).
///
/// # Generic Type Parameters
/// * `D` - The type of the dataset.
/// * `O` - The type of the processed data.
/// * `B` - The type of the batched data.
/// * `P` - The type of the process function.
/// * `C` - The type of the collate function.
///
/// # Fields
/// * `dataset` - The dataset to be used by the dataloader.
/// * `batch_size` - The batch size.
/// * `process_fn` - The process function.
/// * `collate_fn` - The collate function.
/// * `shuffle_seed` - The shuffle seed. Default is `None` (no shuffling before the first batch).
/// * `phantom` - The phantom data only used to marks that the `O` and `B` generic type parameters which would be used later.
///
/// # Examples
/// ```rust
/// use nove::dataset::common::VecDataset;
/// use nove::dataloader::common::{BatchDataloaderBuilder, BatchDataloader};
/// use nove::dataloader::DataloaderError;
///
/// let dataset = VecDataset::from_vec(vec![1usize, 2usize, 3usize]);
/// let mut dataloader: BatchDataloader<VecDataset<usize>, usize, Vec<usize>, _, _> = BatchDataloaderBuilder::default()
///     .dataset(dataset)       // Required configuration
///     .batch_size(2)           // Required configuration
///     .process_fn(             // Required configuration
///         |x: usize| Ok::<usize, DataloaderError>(x)
///     )
///     .collate_fn(             // Required configuration
///         |x: Vec<usize>| Ok::<Vec<usize>, DataloaderError>(x)
///     )
///     .shuffle_seed(Some(42))  // Optional configuration
///     .build()
///     .unwrap();
/// ```
pub struct BatchDataloaderBuilder<D: Dataset, O, B, P, C> {
    dataset: Option<D>,
    batch_size: Option<usize>,
    process_fn: Option<P>,
    collate_fn: Option<C>,
    shuffle_seed: Option<usize>,
    phantom: PhantomData<(O, B)>,
}

impl<D: Dataset, O, B, P, C> Default for BatchDataloaderBuilder<D, O, B, P, C> {
    fn default() -> Self {
        Self {
            dataset: None,
            batch_size: None,
            process_fn: None,
            collate_fn: None,
            shuffle_seed: None,
            phantom: PhantomData,
        }
    }
}

impl<D: Dataset, O, B, P, C> BatchDataloaderBuilder<D, O, B, P, C>
where
    P: Fn(D::Item) -> Result<O, DataloaderError>,
    C: Fn(Vec<O>) -> Result<B, DataloaderError>,
{
    /// Configures the dataset for the dataloader.
    ///
    /// # Arguments
    /// * `dataset` - The dataset to be used by the dataloader.
    ///
    /// # Returns
    /// * `&mut Self` - The builder itself.
    pub fn dataset(&mut self, dataset: D) -> &mut Self {
        self.dataset = Some(dataset);
        self
    }

    /// Configures the batch size for the dataloader.
    ///
    /// # Arguments
    /// * `batch_size` - The batch size to be used by the dataloader.
    ///
    /// # Returns
    /// * `&mut Self` - The builder itself.
    pub fn batch_size(&mut self, batch_size: usize) -> &mut Self {
        self.batch_size = Some(batch_size);
        self
    }

    /// Configures the process function for the dataloader.
    ///
    /// # Arguments
    /// * `process_fn` - The process function to be used by the dataloader.
    ///
    /// # Returns
    /// * `&mut Self` - The builder itself.
    pub fn process_fn(&mut self, process_fn: P) -> &mut Self {
        self.process_fn = Some(process_fn);
        self
    }

    /// Configures the collate function for the dataloader.
    ///
    /// # Arguments
    /// * `collate_fn` - The collate function to be used by the dataloader.
    ///
    /// # Returns
    /// * `&mut Self` - The builder itself.
    pub fn collate_fn(&mut self, collate_fn: C) -> &mut Self {
        self.collate_fn = Some(collate_fn);
        self
    }

    /// Configures the shuffle seed for the dataloader.
    ///
    /// # Arguments
    /// * `shuffle_seed` - The shuffle seed to be used by the dataloader.
    ///
    /// # Returns
    /// * `&mut Self` - The builder itself.
    pub fn shuffle_seed(&mut self, shuffle_seed: Option<usize>) -> &mut Self {
        self.shuffle_seed = shuffle_seed;
        self
    }

    /// Builds the dataloader.
    ///
    /// # Returns
    /// * `Ok(BatchDataloader<D, O, B, P, C>)` - The built dataloader.
    /// * `Err(DataloaderError)` - Some errors occur while building the dataloader.
    pub fn build(&mut self) -> Result<BatchDataloader<D, O, B, P, C>, DataloaderError> {
        let dataset = self.dataset.take().ok_or(DataloaderError::MissingArgument(
            "dataset in BatchDataloaderBuilder".to_string(),
        ))?;
        let batch_size = self.batch_size.ok_or(DataloaderError::MissingArgument(
            "batch_size in BatchDataloaderBuilder".to_string(),
        ))?;
        if batch_size == 0 {
            return Err(DataloaderError::OtherError(
                "batch_size in BatchDataloaderBuilder must be greater than 0".to_string(),
            ));
        }
        let process_fn = self
            .process_fn
            .take()
            .ok_or(DataloaderError::MissingArgument(
                "process_fn in BatchDataloaderBuilder".to_string(),
            ))?;
        let collate_fn = self
            .collate_fn
            .take()
            .ok_or(DataloaderError::MissingArgument(
                "collate_fn in BatchDataloaderBuilder".to_string(),
            ))?;
        let dataset_len = dataset.len()?;
        if dataset_len == 0 {
            return Err(DataloaderError::OtherError(
                "dataset in BatchDataloaderBuilder must have at least one item".to_string(),
            ));
        }

        Ok(BatchDataloader {
            dataset,
            batch_size,
            process_fn,
            collate_fn,
            shuffle_seed: self.shuffle_seed,
            index: 0,
            dataset_len,
            datas: Vec::with_capacity(batch_size),
            indices: (0..dataset_len).collect(),
            phantom: std::marker::PhantomData,
        })
    }
}
