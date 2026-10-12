use nove_tensor::{Device, Tensor};

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn enabled_cpu_devices() -> Vec<Device> {
    vec![
        #[cfg(feature = "candle-cpu")]
        nove_tensor::device::candle::cpu().unwrap(),
        #[cfg(feature = "native-cpu")]
        nove_tensor::device::native::cpu().unwrap(),
    ]
}

#[test]
fn zeroing_one_leaf_keeps_other_leaf_and_snapshot() -> TestResult {
    for device in enabled_cpu_devices() {
        let mut x = Tensor::from_data(vec![1.0f32, 2.0], &device, true)?;
        let y = Tensor::from_data(vec![3.0f32, 4.0], &device, true)?;
        x.add(&y)?.sum(None)?.backward()?;
        let snapshot = x.grad()?.unwrap().try_clone()?;
        let gradient_alias = x.grad()?.unwrap();
        let tensor_alias = x.copy();
        x.zero_grad()?;
        assert_eq!(x.grad()?.unwrap().to_vec::<f32>()?, vec![0.0, 0.0]);
        assert_eq!(gradient_alias.to_vec::<f32>()?, vec![0.0, 0.0]);
        assert_eq!(
            tensor_alias.grad()?.unwrap().to_vec::<f32>()?,
            vec![0.0, 0.0]
        );
        assert_eq!(y.grad()?.unwrap().to_vec::<f32>()?, vec![1.0, 1.0]);
        assert_eq!(snapshot.to_vec::<f32>()?, vec![1.0, 1.0]);
    }
    Ok(())
}

#[test]
fn reused_leaf_accumulates_without_duplicate_storage() -> TestResult {
    for device in enabled_cpu_devices() {
        let mut x = Tensor::from_data(vec![1.0f32, 2.0], &device, true)?;
        let loss = x.add(&x)?.sum(None)?;
        loss.backward()?;
        let gradient_alias = x.grad()?.unwrap();
        assert_eq!(gradient_alias.to_vec::<f32>()?, vec![2.0, 2.0]);
        loss.backward()?;
        assert_eq!(gradient_alias.to_vec::<f32>()?, vec![4.0, 4.0]);
        x.zero_grad()?;
        x.add(&x)?.sum(None)?.backward()?;
        assert_eq!(gradient_alias.to_vec::<f32>()?, vec![2.0, 2.0]);
        x.clear_grad()?;
        assert!(x.grad()?.is_none());
    }
    Ok(())
}

#[test]
fn copy_keeps_shared_tensor_state() -> TestResult {
    for device in enabled_cpu_devices() {
        let mut x = Tensor::from_data(vec![1.0f32, 2.0], &device, true)?;
        let alias = x.copy();
        assert_eq!(alias.graph_id()?, x.graph_id()?);
        let source = Tensor::from_data(vec![3.0f32, 4.0], &device, false)?;
        x.update_from_tensor(&source)?;
        assert_eq!(alias.to_vec::<f32>()?, vec![3.0, 4.0]);
        x.sum(None)?.backward()?;
        x.zero_grad()?;
        assert_eq!(alias.grad()?.unwrap().to_vec::<f32>()?, vec![0.0, 0.0]);
    }
    Ok(())
}

#[test]
fn detach_preserves_values_and_stops_parent_gradients() -> TestResult {
    for device in enabled_cpu_devices() {
        let x = Tensor::from_data(vec![1.0f32, 2.0], &device, true)?;
        let mut detached = x.detach()?;
        assert_eq!(detached.to_vec::<f32>()?, vec![1.0, 2.0]);
        assert_eq!(detached.shape()?, x.shape()?);
        assert_eq!(detached.dtype()?, x.dtype()?);
        assert_eq!(detached.device()?, device);
        assert!(!detached.grad_enabled()?);
        assert_ne!(detached.graph_id()?, x.graph_id()?);
        let new_leaf = detached.require_grad(true)?;
        new_leaf.sum(None)?.backward()?;
        assert_eq!(new_leaf.grad()?.unwrap().to_vec::<f32>()?, vec![1.0, 1.0]);
        assert!(x.grad()?.is_none());
    }
    Ok(())
}

#[test]
fn try_clone_preserves_large_integers() -> TestResult {
    for device in enabled_cpu_devices() {
        let values = vec![9_007_199_254_740_993i64, -9_007_199_254_740_993];
        let original = Tensor::from_data(values.clone(), &device, false)?;
        let copy = original.try_clone()?;
        assert_eq!(copy.to_vec::<i64>()?, values);
        assert_eq!(copy.shape()?, original.shape()?);
        assert_eq!(copy.dtype()?, original.dtype()?);
        assert_eq!(copy.device()?, device);
    }
    Ok(())
}
