use nove::tensor::Device;

fn main() {
    let device = Device::cpu();
    println!("{}", device);

    #[cfg(feature = "cuda")]
    {
        let device = Device::cuda_if_available(0);
        println!("{}", device);
    }

    #[cfg(feature = "metal")]
    {
        let device = Device::metal_if_available(0);
        println!("{}", device);
    }
}
