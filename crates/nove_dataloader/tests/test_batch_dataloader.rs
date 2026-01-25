use nove::{
    dataloader::{Dataloader, common::BatchDataloader},
    dataset::Dataset,
    tensor::{Device, Tensor},
};

mod util;
use nove_dataloader::{DataloaderError, common::BatchDataloaderBuilder};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use util::SimpleDataset;

#[test]
fn test_batched_dataloader() {
    // Create a simple dataset.
    let dataset = SimpleDataset {};

    // Prepare the batch size, device, shuffle seed, process function and collate function.
    let device = Device::cpu();
    let batch_size = 12;
    let shuffle_seed = 42;
    let process_fn = |x: usize| -> Result<Tensor, DataloaderError> {
        Ok(Tensor::from_data(&[x as i64], &device, false).unwrap())
    };
    let collate_fn = |x: Vec<Tensor>| -> Result<Tensor, DataloaderError> {
        Ok(Tensor::stack(&x.as_slice(), 0).unwrap())
    };

    // Create a batch_dataloader.
    let mut dataloader: BatchDataloader<SimpleDataset, Tensor, Tensor, _, _> =
        BatchDataloaderBuilder::default()
            .dataset(&dataset)
            .batch_size(batch_size)
            .process_fn(process_fn)
            .collate_fn(collate_fn)
            .shuffle_seed(Some(shuffle_seed))
            .build()
            .unwrap();

    let mut counter = 0;
    // Generate the shuffled indices that are the same as the ones in the DataLoader.
    let mut indices = (0..dataset.len()).collect::<Vec<usize>>();
    indices.shuffle(&mut StdRng::seed_from_u64(shuffle_seed as u64));

    loop {
        // Get the next batch.
        let batch = dataloader.next().unwrap();
        if batch.is_none() {
            break;
        }
        let batch = batch.unwrap();

        batch
            .to_vec::<i64>()
            .unwrap()
            .chunks_exact(1)
            .for_each(|chunk| {
                let data = chunk[0];

                // Check the correspondence data.
                assert_eq!(data, dataset.get(indices[counter]) as i64);
                counter += 1;
            });
    }

    // Check the number of datas.
    assert_eq!(counter, dataset.len());
}
