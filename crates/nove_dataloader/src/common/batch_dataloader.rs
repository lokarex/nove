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
/// # Lifetime Parameters
/// * `'a` - The lifetime of the inner dataset.
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
/// * `index` - The current index.
/// * `shuffle_seed` - The shuffle seed.
/// * `indices` - The indices of the dataset which would be shuffled before the first batch if the shuffle_seed is not `None`.
/// * `datas` - The processed data.
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
///     .dataset(&dataset)       // Required configuration
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
pub struct BatchDataloader<'a, D: Dataset, O, B, P, C> {
    dataset: &'a dyn Dataset<Item = D::Item>,
    batch_size: usize,
    process_fn: P,
    collate_fn: C,
    shuffle_seed: Option<usize>,
    index: usize,
    datas: Vec<O>,
    indices: Vec<usize>,
    phantom: std::marker::PhantomData<B>,
}

impl<'a, D: Dataset, O, B, P, C> BatchDataloader<'a, D, O, B, P, C>
where
    P: Fn(D::Item) -> Result<O, DataloaderError>,
    C: Fn(Vec<O>) -> Result<B, DataloaderError>,
{
    /// Shuffle the indices of the dataloader.
    ///
    /// # Arguments
    /// * `seed` - The seed to shuffle the indices.
    fn shuffle_indices(&mut self, seed: usize) {
        self.indices
            .shuffle(&mut StdRng::seed_from_u64(seed as u64));
    }
}

impl<'a, D: Dataset, O, B, P, C> Dataloader for BatchDataloader<'a, D, O, B, P, C>
where
    P: Fn(D::Item) -> Result<O, DataloaderError>,
    C: Fn(Vec<O>) -> Result<B, DataloaderError>,
{
    type Output = B;

    fn next(&mut self) -> Result<Option<Self::Output>, DataloaderError> {
        let dataset_len = self.dataset.len()?;

        // Shuffle the indices before the first batch.
        if self.index == 0
            && let Some(seed) = self.shuffle_seed
        {
            self.shuffle_indices(seed);
        }

        if self.index >= dataset_len {
            return Ok(None);
        }

        for _ in 0..self.batch_size {
            if self.index >= dataset_len {
                break;
            }
            let item = self.dataset.get(self.indices[self.index])?;
            self.datas.push((self.process_fn)(item)?);
            self.index += 1;
        }

        Ok(Some((self.collate_fn)(std::mem::take(&mut self.datas))?))
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
/// # Lifetime Parameters
/// * `'a` - The lifetime of the inner dataset.
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
///     .dataset(&dataset)       // Required configuration
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
pub struct BatchDataloaderBuilder<'a, D: Dataset, O, B, P, C> {
    dataset: Option<&'a dyn Dataset<Item = D::Item>>,
    batch_size: Option<usize>,
    process_fn: Option<P>,
    collate_fn: Option<C>,
    shuffle_seed: Option<usize>,
    phantom: PhantomData<(O, B)>,
}

impl<'a, D: Dataset, O, B, P, C> Default for BatchDataloaderBuilder<'a, D, O, B, P, C> {
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

impl<'a, D: Dataset, O, B, P, C> BatchDataloaderBuilder<'a, D, O, B, P, C>
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
    pub fn dataset(&mut self, dataset: &'a D) -> &mut Self {
        self.dataset = Some(dataset as &'a dyn Dataset<Item = D::Item>);
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
    /// * `Ok(BatchDataloader<'a, D, O, B, P, C>)` - The built dataloader.
    /// * `Err(DataloaderError)` - Some errors occur while building the dataloader.
    pub fn build(&mut self) -> Result<BatchDataloader<'a, D, O, B, P, C>, DataloaderError> {
        let dataset = self.dataset.ok_or_else(|| {
            DataloaderError::MissingArgument("dataset in BatchDataloaderBuilder".to_string())
        })?;
        let batch_size = self.batch_size.ok_or_else(|| {
            DataloaderError::MissingArgument("batch_size in BatchDataloaderBuilder".to_string())
        })?;
        if batch_size == 0 {
            return Err(DataloaderError::OtherError(
                "batch_size in BatchDataloaderBuilder must be greater than 0".to_string(),
            ));
        }
        let process_fn = self.process_fn.take().ok_or_else(|| {
            DataloaderError::MissingArgument("process_fn in BatchDataloaderBuilder".to_string())
        })?;
        let collate_fn = self.collate_fn.take().ok_or_else(|| {
            DataloaderError::MissingArgument("collate_fn in BatchDataloaderBuilder".to_string())
        })?;
        let len = dataset.len()?;
        if len == 0 {
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
            datas: Vec::with_capacity(len),
            indices: (0..len).collect(),
            phantom: std::marker::PhantomData,
        })
    }
}
