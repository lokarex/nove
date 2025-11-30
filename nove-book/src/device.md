# Device

In nove, the backend is fixed to Wgpu. Therefore, the device is identified and obtained from Wgpu.

## 1. Supported Devices
| Device | Description |
| --- | --- |
| DefaultDevice | The default device, usually the integrated GPU |
| DiscreteGpu(index) | The discrete GPU with the given index |
| IntegratedGpu(index) | The integrated GPU with the given index |
| VirtualGpu(usize) | The virtual GPU with the given index |
| Cpu | The CPU |

## 2. Example

```rust
use nove::device::Device;

fn main() {
    let device1 = Device::DefaultDevice;
    let device2 = Device::DiscreteGpu(0); // The first discrete GPU
    let device3 = Device::Cpu;

    println!("device1: {:?}", device1);
    println!("device2: {:?}", device2);
    println!("device3: {:?}", device3);
}

```

```
device1: DefaultDevice
device2: DiscreteGpu(0)
device3: Cpu
```