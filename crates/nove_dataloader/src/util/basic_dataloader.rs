use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};

use crate::Dataloader;
use nove_dataset::Dataset;
use nove_tensor::Device;

/// The `BasicDataloader` struct provides a basic data loader to
/// shuffle the dataset, load the data, process the data, and collate the
/// datas into batches.
///
/// # Note
/// * When create a `BasicDataloader` from `from_dataset` function,
///   the specific `BasicDataloader` type must be determined. It means
///   3 generic type parameters must be set manually, so please clearly
///   learn about the generic type parameters of `BasicDataloader`.
///
/// # Lifetime Parameters
/// * `'a` - The lifetime of the inner dataset.
///
/// # Generic Type Parameters
/// * `D`(Need to be manually set) - The type of the inner dataset.
/// * `O`(Need to be manually set) - The type of the processed data which is the output of the process function.
/// * `B`(Need to be manually set) - The type of the batched data which is the output of the collate function
///    and the final output of the dataloader.
/// * `P` - The type of the process function. It needs a function that takes
///    an item(The `D::Item` type) from the dataset and the device as input and returns the processed
///    data(The `O` type) on the specified device as output.
/// * `C` - The type of the collate function. It needs a function that takes
///    a vector of processed data(The `Vec<O>` type) from the process function as input and returns
///    the batched data(The `B` type) as output.
///
/// # Fields
/// * `dataset` - The inner dataset.
/// * `device` - The device.
/// * `batch_size` - The batch size.
/// * `process_fn` - The process function.
/// * `collate_fn` - The collate function.
/// * `index` - The current index.
/// * `shuffle_seed` - The shuffle seed.
/// * `indices` - The indices of the dataset which would be shuffled before the first batch if the shuffle_seed is not `None`.
/// * `datas` - The processed data.
/// * `phantom` - The phantom data only used to marks that the `B` generic type parameter which would be used later.
pub struct BasicDataloader<'a, D: Dataset, O, B, P, C> {
    dataset: &'a dyn Dataset<Item = D::Item>,
    device: Option<&'a Device>,
    batch_size: usize,
    process_fn: Option<P>,
    collate_fn: Option<C>,
    index: usize,
    datas: Vec<O>,
    shuffle_seed: Option<usize>,
    indices: Vec<usize>,
    phantom: std::marker::PhantomData<B>,
}

impl<'a, D: Dataset, O, B, P, C> BasicDataloader<'a, D, O, B, P, C>
where
    P: Fn(D::Item, &Device) -> O,
    C: Fn(Vec<O>) -> B,
{
    /// Create a new `BasicDataloader` from the given dataset.
    ///
    /// # Arguments
    /// * `dataset` - The dataset to wrap.
    ///
    /// # Returns
    /// A new `BasicDataloader` instance.
    pub fn from_dataset(dataset: &'a dyn Dataset<Item = D::Item>) -> Self {
        Self {
            dataset,
            device: None,
            batch_size: 1,
            process_fn: None,
            collate_fn: None,
            index: 0,
            datas: Vec::with_capacity(1),
            shuffle_seed: None,
            indices: (0..dataset.len()).collect(),
            phantom: std::marker::PhantomData,
        }
    }

    /// Set the shuffle seed(The default value is `None`) of the dataloader when create the dataloader.
    /// If the shuffle seed is None, the dataloader will not shuffle the data.
    ///
    /// # Arguments
    /// * `shuffle_seed` - The shuffle seed to set.
    ///
    /// # Returns
    /// A new `BasicDataloader` instance with the shuffle seed set.
    pub fn with_shuffle_seed(mut self, shuffle_seed: Option<usize>) -> Self {
        self.shuffle_seed = shuffle_seed;
        self
    }

    /// Set the shuffle seed(The default value is `None`) of the dataloader.
    /// If the shuffle seed is None, the dataloader will not shuffle the data.
    ///
    /// # Arguments
    /// * `shuffle_seed` - The shuffle seed to set.
    pub fn set_shuffle_seed(&mut self, shuffle_seed: Option<usize>) {
        self.shuffle_seed = shuffle_seed;
    }

    /// Shuffle the indices of the dataloader.
    ///
    /// # Arguments
    /// * `seed` - The seed to shuffle the indices.
    fn shuffle_indices(&mut self, seed: usize) {
        self.indices
            .shuffle(&mut StdRng::seed_from_u64(seed as u64));
    }

    /// Set the batch size of the dataloader when create the dataloader.
    ///
    /// # Arguments
    /// * `batch_size` - The batch size to set.
    ///
    /// # Panics
    /// * If the batch size is 0.
    ///
    /// # Returns
    /// A new `BasicDataloader` instance with the batch size set.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        if batch_size == 0 {
            panic!("The batch size must be greater than 0.");
        }
        self.batch_size = batch_size;
        self.datas.reserve(batch_size);
        self
    }

    /// Set the batch size of the dataloader.
    ///
    /// # Arguments
    /// * `batch_size` - The batch size to set.
    ///
    /// # Panics
    /// * If the batch size is 0.
    pub fn set_batch_size(&mut self, batch_size: usize) {
        if batch_size == 0 {
            panic!("The batch size must be greater than 0.");
        }
        self.batch_size = batch_size;
        self.datas.reserve(batch_size);
    }

    /// Set the process function of the dataloader when create the dataloader.
    ///
    /// # Arguments
    /// * `process_fn` - The process function to set.
    ///
    /// # Returns
    /// A new `BasicDataloader` instance with the process function set.
    pub fn with_process_fn(mut self, process_fn: P) -> Self {
        self.process_fn = Some(process_fn);
        self
    }

    /// Set the process function of the dataloader.
    ///
    /// # Arguments
    /// * `process_fn` - The process function to set.
    pub fn set_process_fn(&mut self, process_fn: P) {
        self.process_fn = Some(process_fn);
    }

    /// Set the collate function of the dataloader when create the dataloader.
    ///
    /// # Arguments
    /// * `collate_fn` - The collate function to set.
    ///
    /// # Returns
    /// A new `BasicDataloader` instance with the collate function set.
    pub fn with_collate_fn(mut self, collate_fn: C) -> Self {
        self.collate_fn = Some(collate_fn);
        self
    }

    /// Set the collate function of the dataloader.
    ///
    /// # Arguments
    /// * `collate_fn` - The collate function to set.
    pub fn set_collate_fn(&mut self, collate_fn: C) {
        self.collate_fn = Some(collate_fn);
    }

    /// Set the device of the dataloader when create the dataloader.
    ///
    /// # Arguments
    /// * `device` - The device to set.
    ///
    /// # Returns
    /// A new `BasicDataloader` instance with the device set.
    pub fn with_device(mut self, device: &'a Device) -> Self {
        self.device = Some(device);
        self
    }
}

impl<'a, D: Dataset, O, B, P, C> Dataloader<'a> for BasicDataloader<'a, D, O, B, P, C>
where
    P: Fn(D::Item, &Device) -> O,
    C: Fn(Vec<O>) -> B,
{
    type Output = B;

    fn next(&mut self) -> Option<Self::Output> {
        if self.device.is_none() {
            panic!("The device in BasicDataloader is not set.");
        }
        if self.process_fn.is_none() {
            panic!("The process_fn in BasicDataloader is not set.");
        }
        if self.collate_fn.is_none() {
            panic!("The collate_fn in BasicDataloader is not set.");
        }

        // Shuffle the indices before the first batch.
        if self.index == 0 && self.shuffle_seed.is_some() {
            self.shuffle_indices(self.shuffle_seed.unwrap());
        }

        if self.index >= self.dataset.len() {
            return None;
        }

        for _ in 0..self.batch_size {
            if self.index >= self.dataset.len() {
                break;
            }
            let item = self.dataset.get(self.indices[self.index]);
            self.datas.push((self.process_fn.as_mut().unwrap())(
                item,
                self.device.unwrap(),
            ));
            self.index += 1;
        }

        Some((self.collate_fn.as_mut().unwrap())(std::mem::take(
            &mut self.datas,
        )))
    }

    fn reset(&mut self) {
        self.index = 0;
    }

    fn to_device(&mut self, device: &'a Device) {
        self.device = Some(device);
    }
}
