use std::thread;
use std::time::{Duration, Instant};

use nove_dataloader::{Dataloader, DataloaderError, common::PrefetchDataloaderBuilder};

/// A mock dataloader that simulates data loading delay.
/// This helps demonstrate the performance benefit of prefetching.
struct SlowDataloader {
    data: Vec<i32>,
    index: usize,
    load_delay_ms: u64,
    process_delay_ms: u64,
}

impl SlowDataloader {
    fn new(data: Vec<i32>, load_delay_ms: u64, process_delay_ms: u64) -> Self {
        Self {
            data,
            index: 0,
            load_delay_ms,
            process_delay_ms,
        }
    }
}

impl Dataloader for SlowDataloader {
    type Output = i32;

    fn next(&mut self) -> Result<Option<Self::Output>, DataloaderError> {
        if self.index < self.data.len() {
            // Simulate data loading delay (e.g., disk I/O, network)
            thread::sleep(Duration::from_millis(self.load_delay_ms));

            let value = self.data[self.index];
            self.index += 1;

            // Simulate processing delay (e.g., decoding, augmentation)
            thread::sleep(Duration::from_millis(self.process_delay_ms));

            Ok(Some(value))
        } else {
            Ok(None)
        }
    }

    fn reset(&mut self) -> Result<(), DataloaderError> {
        self.index = 0;
        Ok(())
    }
}

/// Simulates a training step that takes some time.
fn simulate_training_step(batch: i32, train_delay_ms: u64) {
    thread::sleep(Duration::from_millis(train_delay_ms));
    let _ = batch;
}

/// Test that PrefetchDataloader provides performance improvement
/// when data loading is slower than training.
#[test]
fn test_prefetch_performance_improvement() {
    let data: Vec<i32> = (0..10).collect();
    let load_delay_ms = 10;
    let process_delay_ms = 5;
    let train_delay_ms = 5;

    // Test without prefetch
    let mut slow_dl = SlowDataloader::new(data.clone(), load_delay_ms, process_delay_ms);
    let start_no_prefetch = Instant::now();

    while let Some(batch) = slow_dl.next().unwrap() {
        simulate_training_step(batch, train_delay_ms);
    }

    let duration_no_prefetch = start_no_prefetch.elapsed();

    // Test with prefetch
    let slow_dl = SlowDataloader::new(data, load_delay_ms, process_delay_ms);
    let mut prefetch_dl = PrefetchDataloaderBuilder::default()
        .dataloader(slow_dl)
        .buffer_size(2)
        .build()
        .unwrap();

    let start_with_prefetch = Instant::now();

    while let Some(batch) = prefetch_dl.next().unwrap() {
        simulate_training_step(batch, train_delay_ms);
    }

    let duration_with_prefetch = start_with_prefetch.elapsed();

    // Calculate speedup
    let speedup =
        duration_no_prefetch.as_millis() as f64 / duration_with_prefetch.as_millis() as f64;

    println!("Without prefetch: {:?}", duration_no_prefetch);
    println!("With prefetch: {:?}", duration_with_prefetch);
    println!("Speedup: {:.2}x", speedup);

    // Prefetch should provide some improvement
    assert!(
        duration_with_prefetch < duration_no_prefetch,
        "Prefetch should be faster: {:?} should be < {:?}",
        duration_with_prefetch,
        duration_no_prefetch
    );
}

/// Test that PrefetchDataloader provides improvement
/// when data loading is much slower than training.
#[test]
fn test_prefetch_high_io_bound() {
    let data: Vec<i32> = (0..5).collect();
    let load_delay_ms = 20; // High I/O delay
    let process_delay_ms = 2;
    let train_delay_ms = 2; // Fast training

    // Test without prefetch
    let mut slow_dl = SlowDataloader::new(data.clone(), load_delay_ms, process_delay_ms);
    let start_no_prefetch = Instant::now();

    while let Some(batch) = slow_dl.next().unwrap() {
        simulate_training_step(batch, train_delay_ms);
    }

    let duration_no_prefetch = start_no_prefetch.elapsed();

    // Test with prefetch
    let slow_dl = SlowDataloader::new(data, load_delay_ms, process_delay_ms);
    let mut prefetch_dl = PrefetchDataloaderBuilder::default()
        .dataloader(slow_dl)
        .buffer_size(3)
        .build()
        .unwrap();

    let start_with_prefetch = Instant::now();

    while let Some(batch) = prefetch_dl.next().unwrap() {
        simulate_training_step(batch, train_delay_ms);
    }

    let duration_with_prefetch = start_with_prefetch.elapsed();

    let speedup =
        duration_no_prefetch.as_millis() as f64 / duration_with_prefetch.as_millis() as f64;

    println!(
        "High I/O bound - Without prefetch: {:?}",
        duration_no_prefetch
    );
    println!(
        "High I/O bound - With prefetch: {:?}",
        duration_with_prefetch
    );
    println!("Speedup: {:.2}x", speedup);

    // Prefetch should provide some improvement
    assert!(
        duration_with_prefetch < duration_no_prefetch,
        "Prefetch should be faster: {:?} should be < {:?}",
        duration_with_prefetch,
        duration_no_prefetch
    );
}

/// Test that PrefetchDataloader still works correctly
/// when training is slower than data loading.
#[test]
fn test_prefetch_compute_bound() {
    let data: Vec<i32> = (0..5).collect();
    let load_delay_ms = 2; // Fast loading
    let process_delay_ms = 2;
    let train_delay_ms = 20; // Slow training (compute bound)

    // Test without prefetch
    let mut slow_dl = SlowDataloader::new(data.clone(), load_delay_ms, process_delay_ms);
    let start_no_prefetch = Instant::now();

    while let Some(batch) = slow_dl.next().unwrap() {
        simulate_training_step(batch, train_delay_ms);
    }

    let duration_no_prefetch = start_no_prefetch.elapsed();

    // Test with prefetch
    let slow_dl = SlowDataloader::new(data, load_delay_ms, process_delay_ms);
    let mut prefetch_dl = PrefetchDataloaderBuilder::default()
        .dataloader(slow_dl)
        .buffer_size(2)
        .build()
        .unwrap();

    let start_with_prefetch = Instant::now();

    while let Some(batch) = prefetch_dl.next().unwrap() {
        simulate_training_step(batch, train_delay_ms);
    }

    let duration_with_prefetch = start_with_prefetch.elapsed();

    let speedup =
        duration_no_prefetch.as_millis() as f64 / duration_with_prefetch.as_millis() as f64;

    println!(
        "Compute bound - Without prefetch: {:?}",
        duration_no_prefetch
    );
    println!(
        "Compute bound - With prefetch: {:?}",
        duration_with_prefetch
    );
    println!("Speedup: {:.2}x", speedup);

    // For compute bound workload, prefetch should not add significant overhead
    assert!(
        duration_with_prefetch <= duration_no_prefetch,
        "Prefetch should not slow down compute bound workload: {:?} should be <= {:?}",
        duration_with_prefetch,
        duration_no_prefetch
    );
}

/// Test different buffer sizes and their impact on performance.
#[test]
fn test_prefetch_buffer_size_impact() {
    let data: Vec<i32> = (0..8).collect();
    let load_delay_ms = 10;
    let process_delay_ms = 5;
    let train_delay_ms = 5;

    // Baseline without prefetch
    let mut slow_dl = SlowDataloader::new(data.clone(), load_delay_ms, process_delay_ms);
    let start = Instant::now();
    while let Some(batch) = slow_dl.next().unwrap() {
        simulate_training_step(batch, train_delay_ms);
    }
    let baseline = start.elapsed();

    println!("\nBuffer size comparison:");
    println!("Baseline (no prefetch): {:?}", baseline);

    // Test different buffer sizes
    for buffer_size in [1, 2, 4] {
        let slow_dl = SlowDataloader::new(data.clone(), load_delay_ms, process_delay_ms);
        let mut prefetch_dl = PrefetchDataloaderBuilder::default()
            .dataloader(slow_dl)
            .buffer_size(buffer_size)
            .build()
            .unwrap();

        let start = Instant::now();
        while let Some(batch) = prefetch_dl.next().unwrap() {
            simulate_training_step(batch, train_delay_ms);
        }
        let duration = start.elapsed();

        let speedup = baseline.as_millis() as f64 / duration.as_millis() as f64;
        println!(
            "Buffer size {}: {:?} ({:.2}x speedup)",
            buffer_size, duration, speedup
        );

        // All buffer sizes should provide some improvement
        assert!(
            duration < baseline,
            "Buffer size {} should be faster than baseline",
            buffer_size
        );
    }
}
