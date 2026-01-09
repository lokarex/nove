use nove::tensor::Device;

#[test]
fn test_eq() {
    let device1 = Device::get_cpu();
    let device2 = Device::get_cpu();
    assert_eq!(device1, device2);
}

#[cfg(feature = "cuda")]
#[test]
fn test_get_cuda_with_invalid_index() {
    // Test get_cuda with an invalid index
    let device = Device::get_cuda(usize::MAX);
    assert_eq!(device.is_err(), true);
}

#[cfg(feature = "cuda")]
#[test]
fn test_get_cuda_if_available() {
    // Try to get a CUDA device
    let cuda_if_available = Device::get_cuda_if_available(0);

    // Get the CPU device used to compare with
    let cpu = Device::get_cpu();

    // Judge whether the CUDA device is available
    match Device::get_cuda(0) {
        // If CUDA device is available, cuda_if_available should be equal to it
        Ok(cuda) => assert_eq!(cuda, cuda_if_available),
        // If CUDA device is not available, cuda_if_available should be equal to cpu
        Err(_) => assert_eq!(cuda_if_available, cpu),
    }
}

#[cfg(feature = "metal")]
#[test]
fn test_get_metal_with_invalid_index() {
    // Test get_metal with an invalid index
    let device = Device::get_metal(usize::MAX);
    assert_eq!(device.is_err(), true);
}

#[cfg(feature = "metal")]
#[test]
fn test_get_metal_if_available() {
    // Try to get a Metal device
    let metal_if_available = Device::get_metal_if_available(0);

    // Get the CPU device used to compare with
    let cpu = Device::get_cpu();

    // Judge whether the Metal device is available
    match Device::get_metal(0) {
        // If Metal device is available, metal_if_available should be equal to it
        Ok(metal) => assert_eq!(metal, metal_if_available),
        // If Metal device is not available, metal_if_available should be equal to cpu
        Err(_) => assert_eq!(metal_if_available, cpu),
    }
}
