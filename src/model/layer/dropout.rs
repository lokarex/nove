use crate::{
    model::{Model, ModelError, paramstore::ParamStore},
    tensor::Tensor,
};
use std::sync::atomic::{AtomicUsize, Ordering};

static ID: AtomicUsize = AtomicUsize::new(1);

pub struct Dropout<P: ParamStore> {
    pub dropout_prob: f32,
    pub param_store: P,
}

impl<P: ParamStore> Dropout<P> {
    pub fn new(root_param_store: &P, dropout_prob: f32) -> Result<Self, ModelError> {
        // Check the dropout_prob is valid.
        if dropout_prob < 0.0 || dropout_prob > 1.0 {
            return Err(ModelError::OtherError(
                "dropout_prob must be between 0.0 and 1.0".to_string(),
            ));
        }

        // Create the param store for the dropout layer.
        let param_store = P::new(&format!(
            "(dropout{}): Dropout(p={})",
            ID.fetch_add(1, Ordering::SeqCst),
            dropout_prob
        ))?;
        // Add the param store of the dropout layer to the root param store.
        root_param_store.set_module(param_store.clone())?;

        Ok(Self {
            dropout_prob,
            param_store,
        })
    }
}

impl<P: ParamStore> Model for Dropout<P> {
    type ParamStore = P;
    type Input = (Tensor, bool);
    type Output = Tensor;

    fn param_stores(&self) -> Result<Vec<Self::ParamStore>, ModelError> {
        Ok(vec![self.param_store.clone()])
    }

    fn forward(&mut self, _input: Self::Input) -> Result<Self::Output, ModelError> {
        todo!()
    }
}
