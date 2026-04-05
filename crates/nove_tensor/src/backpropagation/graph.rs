use crate::{
    Tensor, TensorError,
    tensor::{TensorData, TensorInner},
};
use crossbeam::queue::SegQueue;
use dashmap::DashMap;
use std::sync::{Arc, RwLock};

impl Tensor {
    /// Detach the tensor from the computational graph.
    ///
    /// This creates a new tensor that shares the same data but is disconnected
    /// from the computational graph. The detached tensor will:
    /// - Have no parent tensors
    /// - Disconnect from the computational graph
    ///
    /// # Returns
    /// * `Ok(Tensor)` - A new tensor detached from the computational graph.
    /// * `Err(TensorError)` - The error when detaching the tensor.
    pub fn detach(&self) -> Result<Tensor, TensorError> {
        let inner = self.data.read()?;
        let inner_tensor = match &inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = TensorInner::Tensor(inner_tensor.detach());

        let grad = match &inner.grad {
            Some(grad) => Some(grad.detach()?),
            None => None,
        };

        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                device: self.data.read()?.device.clone(),
                parents: vec![],
                grad,
                name: inner.name.clone(),
            })),
        })
    }

    /// Clear the computational graph.
    ///
    /// # Returns
    /// * `Ok(())` - The tensor's computational graph is successfully cleared.
    /// * `Err(TensorError)` - The error when clearing the tensor's computational graph.
    pub fn clear_graph(&self) -> Result<(), TensorError> {
        let queue = SegQueue::new();
        let visited = DashMap::new();

        queue.push(self.copy());
        visited.insert(Arc::as_ptr(&self.data) as usize, true);

        while let Some(current) = queue.pop() {
            let current_parents = {
                let mut data = current.data.write()?;
                if let Some(grad) = data.grad.as_ref() {
                    queue.push(grad.copy());
                }
                std::mem::take(&mut data.parents)
            };

            for parent in current_parents {
                let parent_id = Arc::as_ptr(&parent.data) as usize;
                if visited.insert(parent_id, true).is_none() {
                    queue.push(parent);
                }
            }
        }

        Ok(())
    }
}
