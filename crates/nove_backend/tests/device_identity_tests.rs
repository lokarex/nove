#[cfg(any(feature = "candle-cuda", feature = "candle-metal"))]
use nove_backend::BackendKind;
use nove_backend::{Device, DeviceKind, DeviceLocation, backend::BackendStorage};

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn enabled_cpu_devices() -> Vec<Device> {
    vec![
        #[cfg(feature = "candle-cpu")]
        nove_backend::device::candle::cpu().unwrap(),
        #[cfg(feature = "native-cpu")]
        nove_backend::device::native::cpu().unwrap(),
    ]
}

#[test]
fn cpu_identity_and_location_agree_with_storage() -> TestResult {
    for first in enabled_cpu_devices() {
        let clone = first.clone();
        assert_eq!(first, clone);
        assert_eq!(first.location(), clone.location());
        assert_eq!(first.location().backend(), first.backend());
        assert_eq!(first.location().kind(), DeviceKind::Cpu);
        assert_eq!(first.location().index(), 0);
        let location: nove_backend::device::DeviceLocation = first.location();
        let _: DeviceLocation = location;
        let mut locations = std::collections::HashSet::new();
        locations.insert(location);
        assert!(locations.contains(&clone.location()));

        for second in enabled_cpu_devices() {
            let same_backend = first.backend() == second.backend();
            assert_eq!(first == second, same_backend);
            assert_eq!(first.location() == second.location(), same_backend);
            let lhs = BackendStorage::from_data(vec![1.0f32, 2.0], &first)?;
            let rhs = BackendStorage::from_data(vec![3.0f32, 4.0], &second)?;
            assert_eq!(lhs.same_device(&rhs), first == second);
        }
    }
    Ok(())
}

#[cfg(any(feature = "candle-cuda", feature = "candle-metal"))]
fn check_gpu_identity(first: Device, second: Device, kind: DeviceKind) -> TestResult {
    let clone = first.clone();
    let clone_again = clone.clone();
    let first_ref = &first;
    assert_eq!(&first, first_ref);
    assert_eq!(first, clone);
    assert_eq!(clone, first);
    assert_eq!(clone, clone_again);
    assert_eq!(first, clone_again);
    assert_ne!(first, second);
    assert_ne!(second, first);
    assert_eq!(first.location(), second.location());
    assert_eq!(first.location().backend(), BackendKind::Candle);
    assert_eq!(first.location().kind(), kind);
    assert_eq!(first.location().index(), 0);
    let lhs = BackendStorage::from_data(vec![1.0f32, 2.0], &first)?;
    let alias = BackendStorage::from_data(vec![3.0f32, 4.0], &clone)?;
    let rhs = BackendStorage::from_data(vec![5.0f32, 6.0], &second)?;
    assert!(lhs.same_device(&alias));
    assert!(!lhs.same_device(&rhs));
    assert!(!rhs.same_device(&lhs));
    for cpu in enabled_cpu_devices() {
        assert_ne!(first, cpu);
        assert_ne!(first.location(), cpu.location());
        let cpu_storage = BackendStorage::from_data(vec![1.0f32, 2.0], &cpu)?;
        assert!(!lhs.same_device(&cpu_storage));
    }
    Ok(())
}

#[cfg(feature = "candle-cuda")]
#[test]
#[ignore = "requires a CUDA device"]
fn cuda_identity_distinguishes_instances_at_the_same_location() -> TestResult {
    check_gpu_identity(
        nove_backend::device::candle::cuda(0)?,
        nove_backend::device::candle::cuda(0)?,
        DeviceKind::Cuda,
    )
}

#[cfg(feature = "candle-metal")]
#[test]
#[ignore = "requires a Metal device"]
fn metal_identity_distinguishes_instances_at_the_same_location() -> TestResult {
    check_gpu_identity(
        nove_backend::device::candle::metal(0)?,
        nove_backend::device::candle::metal(0)?,
        DeviceKind::Metal,
    )
}
