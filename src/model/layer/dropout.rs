use crate::{
    model::{Model, ModelError, paramstore::ParamStore},
    tensor::Tensor,
};
use rand::{Rng, rngs::ThreadRng};
use std::sync::atomic::{AtomicUsize, Ordering};

static ID: AtomicUsize = AtomicUsize::new(1);

pub struct Dropout<P: ParamStore> {
    dropout_prob: f32,
    param_store: P,
    rng: ThreadRng,
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
            rng: rand::rng(),
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

    fn forward(&mut self, input: Self::Input) -> Result<Self::Output, ModelError> {
        let (x, training) = input;
        match training {
            true => {
                let scale = 1.0 / (1.0 - self.dropout_prob);
                let shape = x.shape()?;
                let total_size = shape.dims().iter().product::<usize>();

                let mask_vec = (0..total_size)
                    .map(|_| {
                        if self.rng.random::<f32>() >= self.dropout_prob {
                            scale
                        } else {
                            0.0
                        }
                    })
                    .collect::<Vec<_>>();

                let mask = Tensor::from_data(mask_vec, &x.device()?, false)?;
                mask.to_dtype(&x.dtype()?)?;
                mask.to_shape_inplace(&x.shape()?)?;

                Ok(x.mul(&mask)?)
            }
            false => Ok(x),
        }
    }
}
