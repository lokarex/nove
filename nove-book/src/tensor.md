# Tensor

The Tensor in nove is just a product of fixing the backend of the Tensor in burn (fixed to Wgpu), and nove does not extend any additional functionality. Therefore, this chapter only briefly introduces the relevant knowledge.

## 1. Dimension and Kind

For a specific Tensor type, its dimension and kind are needed to be fixed by the two generic parameters `D`(dimension) and `K`(kind). 

The requirement for `D` is that it must be a fixed constant. And `K` only has three possible values, including `Float`(Default value), `Int` and `Bool`. 

For example, 

`Tensor<2>` and `Tensor<2, Float>` is a 2D tensor with float kind. 

`Tensor<2, Int>` is a 2D tensor with int kind. 

`Tensor<2, Bool>` is a 2D tensor with bool kind.

## 2. Shape

## 3. Initialization


### 3.1 Intialize a tensor with zeros
```rust
use nove::{
    device::Device,
    tensor::{Shape, Tensor},
};

fn main() {
    let device = Device::DefaultDevice;
    let shape = Shape::new([2, 3]);
    let t1 = Tensor::<2>::zeros(shape, &device);
    println!("{}", t1);
}

```

```
Tensor {
  data:
[[0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0]],
  shape:  [2, 3],
  device:  DefaultDevice,
  backend:  "autodiff<cubecl<wgpu<wgsl>>>",
  kind:  "Float",
  dtype:  "f32",
}
```

## 4. Operation

## 5. Backpropagation