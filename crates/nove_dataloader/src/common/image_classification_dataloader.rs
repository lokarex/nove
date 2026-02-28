use std::marker::PhantomData;

use nove_dataset::Dataset;
use nove_tensor::{Device, Shape, Tensor};

use crate::{Dataloader, DataloaderError};

type ProcessFn = dyn Fn((String, usize)) -> Result<(Tensor, Tensor), DataloaderError>;
type CollateFn = dyn Fn(Vec<(Tensor, Tensor)>) -> Result<(Tensor, Tensor), DataloaderError>;

/// The `ImageClassificationDataloader` struct provides a specialized data loader
/// for image classification datasets, which loads images from file paths,
/// resizes them, converts them to tensors, and batches them together.
///
/// # Note
/// * This dataloader is designed for datasets where `Dataset::Item` is `(String, usize)`,
///   representing `(image_path, label)`.
/// * Images are resized to the specified dimensions and converted to RGB format.
/// * Pixel values are normalized to the range [0, 1].
///
/// # Generic Type Parameters
/// * `D` - The type of the dataset. Must implement `Dataset<Item = (String, usize)>`.
///
/// # Output
/// * `Output` - A tuple `(Tensor, Tensor)` where:
///   - The first tensor is the batched images with shape `[batch_size, height, width, 3]`.
///   - The second tensor is the batched labels with shape `[batch_size]`.
///
/// # Examples
/// ```rust,no_run
/// use nove::dataset::resource::Mnist;
/// use nove::dataloader::common::ImageClassificationDataloaderBuilder;
/// use nove::dataloader::Dataloader;
/// use nove::tensor::Device;
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let mnist = Mnist::new("data/mnist")?;
///     let dataset = mnist.train()?;
///     
///     let mut dataloader = ImageClassificationDataloaderBuilder::default()
///         .dataset(dataset)
///         .batch_size(32)
///         .resize(28, 28)
///         .device(Device::cpu())
///         .shuffle_seed(Some(42))
///         .build()?;
///
///     if let Some((images, labels)) = dataloader.next()? {
///         println!("Images shape: {:?}", images.shape()?);
///         println!("Labels shape: {:?}", labels.shape()?);
///     }
///     Ok(())
/// }
/// ```
pub struct ImageClassificationDataloader<D>
where
    D: Dataset<Item = (String, usize)>,
{
    inner: crate::common::BatchDataloader<
        D,
        (Tensor, Tensor),
        (Tensor, Tensor),
        Box<ProcessFn>,
        Box<CollateFn>,
    >,
}

impl<D> Dataloader for ImageClassificationDataloader<D>
where
    D: Dataset<Item = (String, usize)>,
{
    type Output = (Tensor, Tensor);

    fn next(&mut self) -> Result<Option<Self::Output>, DataloaderError> {
        self.inner.next()
    }

    fn reset(&mut self) -> Result<(), DataloaderError> {
        self.inner.reset()
    }
}

/// The builder for the `ImageClassificationDataloader`.
///
/// # Notes
/// * The `ImageClassificationDataloaderBuilder` implements the `Default` trait. So you can use
///   `ImageClassificationDataloaderBuilder::default()` to create a new builder and then configure it.
///
/// # Required Arguments
/// * `dataset` - The dataset to be used by the dataloader. Must be a dataset with `Item = (String, usize)`.
/// * `batch_size` - The batch size. Must be greater than 0.
/// * `resize` - The target height and width for resizing images.
/// * `device` - The device to place the tensors on.
///
/// # Optional Arguments
/// * `grad_enabled` - Whether to enable gradient tracking for image tensors. Default is `false`.
/// * `shuffle_seed` - The shuffle seed. Default is `None` (no shuffling).
///
/// # Generic Type Parameters
/// * `D` - The type of the dataset. Must implement `Dataset<Item = (String, usize)>`.
///
/// # Fields
/// * `dataset` - The dataset to be used by the dataloader.
/// * `batch_size` - The batch size.
/// * `resize_height` - The target height for resizing images.
/// * `resize_width` - The target width for resizing images.
/// * `device` - The device to place the tensors on.
/// * `grad_enabled` - Whether to enable gradient tracking for image tensors.
/// * `shuffle_seed` - The shuffle seed.
/// * `phantom` - The phantom data for marker purposes.
///
/// # Examples
/// ```rust,no_run
/// use nove::dataset::resource::Mnist;
/// use nove::dataloader::common::ImageClassificationDataloaderBuilder;
/// use nove::tensor::Device;
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let mnist = Mnist::new("data/mnist")?;
///     let dataset = mnist.train()?;
///     
///     let mut dataloader = ImageClassificationDataloaderBuilder::default()
///         .dataset(dataset)        // Required configuration
///         .batch_size(32)          // Required configuration
///         .resize(28, 28)          // Required configuration
///         .device(Device::cpu())   // Required configuration
///         .grad_enabled(false)     // Optional configuration
///         .shuffle_seed(Some(42))  // Optional configuration
///         .build()?;
///
///     Ok(())
/// }
/// ```
pub struct ImageClassificationDataloaderBuilder<D>
where
    D: Dataset<Item = (String, usize)>,
{
    dataset: Option<D>,
    batch_size: Option<usize>,
    resize_height: Option<usize>,
    resize_width: Option<usize>,
    device: Option<Device>,
    grad_enabled: bool,
    shuffle_seed: Option<usize>,
    phantom: PhantomData<D>,
}

impl<D> Default for ImageClassificationDataloaderBuilder<D>
where
    D: Dataset<Item = (String, usize)>,
{
    fn default() -> Self {
        Self {
            dataset: None,
            batch_size: None,
            resize_height: None,
            resize_width: None,
            device: None,
            grad_enabled: false,
            shuffle_seed: None,
            phantom: PhantomData,
        }
    }
}

impl<D> ImageClassificationDataloaderBuilder<D>
where
    D: Dataset<Item = (String, usize)>,
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

    /// Configures the resize dimensions for images.
    ///
    /// # Arguments
    /// * `height` - The target height for resizing images.
    /// * `width` - The target width for resizing images.
    ///
    /// # Returns
    /// * `&mut Self` - The builder itself.
    pub fn resize(&mut self, height: usize, width: usize) -> &mut Self {
        self.resize_height = Some(height);
        self.resize_width = Some(width);
        self
    }

    /// Configures the device for the dataloader.
    ///
    /// # Arguments
    /// * `device` - The device to place the tensors on.
    ///
    /// # Returns
    /// * `&mut Self` - The builder itself.
    pub fn device(&mut self, device: Device) -> &mut Self {
        self.device = Some(device);
        self
    }

    /// Configures whether to enable gradient tracking for image tensors.
    ///
    /// # Arguments
    /// * `grad_enabled` - Whether to enable gradient tracking.
    ///
    /// # Returns
    /// * `&mut Self` - The builder itself.
    pub fn grad_enabled(&mut self, grad_enabled: bool) -> &mut Self {
        self.grad_enabled = grad_enabled;
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
    /// * `Ok(ImageClassificationDataloader<D>)` - The built dataloader.
    /// * `Err(DataloaderError)` - Some errors occur while building the dataloader.
    pub fn build(&mut self) -> Result<ImageClassificationDataloader<D>, DataloaderError> {
        let dataset = self.dataset.take().ok_or(DataloaderError::MissingArgument(
            "dataset in ImageClassificationDataloaderBuilder".to_string(),
        ))?;
        let batch_size = self.batch_size.ok_or(DataloaderError::MissingArgument(
            "batch_size in ImageClassificationDataloaderBuilder".to_string(),
        ))?;
        if batch_size == 0 {
            return Err(DataloaderError::OtherError(
                "batch_size in ImageClassificationDataloaderBuilder must be greater than 0"
                    .to_string(),
            ));
        }
        let resize_height = self.resize_height.ok_or(DataloaderError::MissingArgument(
            "resize_height in ImageClassificationDataloaderBuilder".to_string(),
        ))?;
        let resize_width = self.resize_width.ok_or(DataloaderError::MissingArgument(
            "resize_width in ImageClassificationDataloaderBuilder".to_string(),
        ))?;
        let device = self.device.take().ok_or(DataloaderError::MissingArgument(
            "device in ImageClassificationDataloaderBuilder".to_string(),
        ))?;

        let grad_enabled = self.grad_enabled;
        let shuffle_seed = self.shuffle_seed;

        let process_fn: Box<ProcessFn> = Box::new(move |(image_path, label): (String, usize)| {
            let img = image::open(&image_path).map_err(|e| {
                DataloaderError::ImageError(format!("Failed to open {}: {}", image_path, e))
            })?;

            let img = img.resize_exact(
                resize_width as u32,
                resize_height as u32,
                image::imageops::FilterType::Triangle,
            );

            let img = img.to_rgb8();
            let (width, height) = img.dimensions();
            let data: Vec<f32> = img
                .pixels()
                .flat_map(|p| p.0.iter().map(|&v| v as f32 / 255.0))
                .collect();

            let shape = Shape::from(&[height as usize, width as usize, 3]);
            let image_tensor = Tensor::from_vec(data, &shape, &device, grad_enabled)?;

            let label_tensor =
                Tensor::from_slice(&[label as i64], &Shape::from(&[1]), &device, false)?;

            Ok((image_tensor, label_tensor))
        });

        let collate_fn: Box<CollateFn> = Box::new(|batch: Vec<(Tensor, Tensor)>| {
            if batch.is_empty() {
                return Err(DataloaderError::OtherError("Batch is empty".to_string()));
            }

            let images: Vec<Tensor> = batch.iter().map(|(img, _)| img.clone()).collect();
            let labels: Vec<Tensor> = batch.iter().map(|(_, lbl)| lbl.clone()).collect();

            let batch_images = Tensor::stack(&images, 0)?;
            let batch_labels = Tensor::stack(&labels, 0)?;

            Ok((batch_images, batch_labels))
        });

        let inner = crate::common::BatchDataloaderBuilder::default()
            .dataset(dataset)
            .batch_size(batch_size)
            .process_fn(process_fn)
            .collate_fn(collate_fn)
            .shuffle_seed(shuffle_seed)
            .build()?;

        Ok(ImageClassificationDataloader { inner })
    }
}
