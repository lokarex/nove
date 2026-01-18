use nove::tensor::{Device, Tensor};

fn main() {
    let tensor = Tensor::from_data(vec![2.0f32, 3.0f32], &Device::cpu(), false).unwrap();
    println!("{}", tensor);
}
