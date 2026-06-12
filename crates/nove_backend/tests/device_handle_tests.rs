#[cfg(feature = "candle-cuda")]
#[test]
fn candle_cuda_device_conversions_reuse_the_same_handle() {
    let device = match nove_backend::device::candle::cuda(0) {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping CUDA handle reuse test: {error}");
            return;
        }
    };

    let first = device.to_candle_device().unwrap();
    let second = device.to_candle_device().unwrap();

    assert!(
        first.same_device(&second),
        "converting the same Nove CUDA device twice should reuse one Candle CUDA handle"
    );

    let from_clone = device.clone().to_candle_device().unwrap();
    assert!(
        first.same_device(&from_clone),
        "cloning a Nove CUDA device should keep sharing the same Candle CUDA handle"
    );
}
