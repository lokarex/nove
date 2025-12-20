use std::{array, vec};

use nove::{
    dataloader::{Dataloader, util::BasicDataloader},
    dataset::Dataset,
    device::Device,
    tensor::{Shape, Tensor, kind::Int},
};

use crate::dataset::util::simple_dataset::{self, SimpleDataset};

#[test]
fn test_batched_dataloader() {
    // Create a simple dataset.
    let dataset = simple_dataset::SimpleDataset {};

    // Prepare the batch size, device, process function and collate function.
    let batch_size = 12;
    let device = Device::DefaultDevice;
    let process_fn =
        |x: usize| -> Tensor<1, Int> { Tensor::<1, Int>::from_data([x as i32], &device) };
    let collate_fn = |x: Vec<Tensor<1, Int>>| -> Tensor<2, Int> { Tensor::<1, Int>::stack(x, 0) };

    // Create a batched dataloader.
    let mut dataloader: BasicDataloader<SimpleDataset, Tensor<1, Int>, Tensor<2, Int>, _, _> =
        BasicDataloader::from_dataset(&dataset)
            .with_batch_size(batch_size)
            .with_process_fn(process_fn)
            .with_collate_fn(collate_fn);

    // Calculate the expected shapes of the batches.
    let mut shapes = vec![Shape::new([batch_size, 1]); dataset.len() / batch_size];
    if dataset.len() % batch_size != 0 {
        shapes.push(Shape::new([dataset.len() % batch_size, 1]));
    }

    // Record the number of cycles.
    let mut counter = 0;

    // Record the number of datas.
    let mut index: i32 = 0;

    loop {
        // Get the next batch.
        let batch = dataloader.next();
        if batch.is_none() {
            break;
        }
        let batch = batch.unwrap();

        // Check the shape of the batch.
        assert_eq!(batch.shape(), shapes[counter]);

        // Check the data of the batch.
        batch
            .to_data()
            .bytes
            .chunks_exact(std::mem::size_of::<i32>())
            .for_each(|chunk| {
                let bytes = array::from_fn(|i| chunk[i]);
                let data = i32::from_le_bytes(bytes);

                assert_eq!(data, index);
                index += 1;
            });

        counter += 1;
    }

    // Check the number of cycles.
    assert_eq!(counter, shapes.len());

    // Check the number of datas.
    assert_eq!(index as usize, dataset.len());
}
