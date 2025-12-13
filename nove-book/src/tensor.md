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

The `Shape` is a struct that represents the shape of a tensor.

It need a fixed-size array with a length of `D`, where each element is a `usize` that represents the size of the tensor along that dimension.

For example, `Shape::new([2, 3])` is a shape of a 2D tensor with size 2 along the first dimension and size 3 along the second dimension.


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

### 3.2 Intialize a tensor with ones

```rust
use nove::{
    device::Device,
    tensor::{Shape, Tensor},
};

fn main() {
    let device = Device::DefaultDevice;
    let shape = Shape::new([2, 3]);
    let t1: Tensor<2> = Tensor::<2>::ones(shape, &device);
    println!("{}", t1);
}

```

```
Tensor {
  data:
[[1.0, 1.0, 1.0],
 [1.0, 1.0, 1.0]],
  shape:  [2, 3],
  device:  DefaultDevice,
  backend:  "autodiff<cubecl<wgpu<wgsl>>>",
  kind:  "Float",
  dtype:  "f32",
}
```

### 3.3 Intialize a tensor from data

```rust
use nove::{device::Device, tensor::Tensor};

fn main() {
    let device = Device::DefaultDevice;

    let t1 = Tensor::<1>::from_data([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &device);
    println!("{}", t1);

    let t2 = Tensor::<2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
    println!("{}", t2);

    let data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let t3 = Tensor::<2>::from_data(data, &device);
    println!("{}", t3);
}

```

```
Tensor {
  data:
[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
  shape:  [6],
  device:  DefaultDevice,
  backend:  "autodiff<cubecl<wgpu<wgsl>>>",
  kind:  "Float",
  dtype:  "f32",
}
Tensor {
  data:
[[1.0, 2.0, 3.0],
 [4.0, 5.0, 6.0]],
  shape:  [2, 3],
  device:  DefaultDevice,
  backend:  "autodiff<cubecl<wgpu<wgsl>>>",
  kind:  "Float",
  dtype:  "f32",
}
Tensor {
  data:
[[1.0, 2.0],
 [3.0, 4.0],
 [5.0, 6.0]],
  shape:  [3, 2],
  device:  DefaultDevice,
  backend:  "autodiff<cubecl<wgpu<wgsl>>>",
  kind:  "Float",
  dtype:  "f32",
}
```

## 4. Operation

## 5. Backpropagation