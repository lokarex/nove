use std::vec;

use nove::{
    dataloader::{Dataloader, util::BasicDataloader},
    dataset::Dataset,
    tensor::{Device, Shape, Tensor},
};

mod util;
use util::SimpleDataset;

#[test]
fn test_batched_dataloader() {
    // Create a simple dataset.
    let dataset = SimpleDataset {};

    // Prepare the batch size, device, process function and collate function.
    let batch_size = 12;
    let process_fn = |x: usize, device: &Device| -> Tensor {
        Tensor::from_data(&[x as i64], device, false).unwrap()
    };
    let collate_fn = |x: Vec<Tensor>| -> Tensor { Tensor::stack(&x.as_slice(), 0).unwrap() };
    let device = Device::cpu();

    // Create a basic dataloader.
    let mut dataloader: BasicDataloader<SimpleDataset, Tensor, Tensor, _, _> =
        BasicDataloader::from_dataset(&dataset)
            .with_batch_size(batch_size)
            .with_process_fn(process_fn)
            .with_collate_fn(collate_fn)
            .with_device(&device);

    // Calculate the expected shapes of the batches.
    let mut shapes = vec![Shape::from_dims(&[batch_size, 1]); dataset.len() / batch_size];
    if dataset.len() % batch_size != 0 {
        shapes.push(Shape::from_dims(&[dataset.len() % batch_size, 1]));
    }

    // Record the number of cycles.
    let mut counter = 0;

    // Record the number of datas.
    let mut index: i64 = 0;

    loop {
        // Get the next batch.
        let batch = dataloader.next();
        if batch.is_none() {
            break;
        }
        let batch = batch.unwrap();

        // Check the shape of the batch.
        assert_eq!(batch.shape().unwrap(), shapes[counter]);

        // Check the data of the batch.
        batch
            .to_vec::<i64>()
            .unwrap()
            .chunks_exact(1)
            .for_each(|chunk| {
                let data = chunk[0];

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
