use nove_tensor::{DType, Device, Tensor};
use std::{
    fmt::Display,
    sync::atomic::{AtomicUsize, Ordering},
};

use crate::{Model, ModelError};

static ID: AtomicUsize = AtomicUsize::new(1);

pub struct Dropout {
    probability: f32,
    id: usize,
}

impl Dropout {
    /// Create a new Dropout layer.
    ///
    /// # Notes
    /// * The probability of dropping out a unit must be in range [0, 1).
    ///
    /// # Arguments
    /// * `probability` - The probability of dropping out a unit.
    /// * `seed` - The seed for the random number generator.
    ///
    /// # Returns
    /// * `Ok(Self)` - The Dropout layer if successful.
    /// * `Err(ModelError)` - The error when creating the Dropout layer.
    ///
    /// # Examples
    /// ```
    /// use nove_model::layer::Dropout;
    ///
    /// let dropout = Dropout::new(0.5).unwrap();
    /// println!("{}", dropout);
    /// ```
    pub fn new(probability: f32) -> Result<Self, ModelError> {
        if !(0.0..1.0).contains(&probability) {
            return Err(ModelError::InvalidArgument(
                "Dropout probability must be in range [0, 1)".to_string(),
            ));
        }

        Ok(Self {
            probability,
            id: ID.fetch_add(1, Ordering::Relaxed),
        })
    }
}

impl Model for Dropout {
    type Input = (Tensor, bool);

    type Output = Tensor;

    /// Apply dropout layer to the input tensor.
    ///
    /// # Arguments
    /// * `input` - `(xs, training)` where `xs` is the input tensor and `training` is a boolean
    ///   indicating whether the model is in training mode.
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The output tensor if successful.
    /// * `Err(ModelError)` - The error when applying dropout layer to the input tensor.
    fn forward(&mut self, input: Self::Input) -> Result<Self::Output, crate::ModelError> {
        let (xs, training) = input;
        if !training {
            return Ok(xs);
        }

        let xs_shape = xs.shape()?;
        let xs_device = xs.device()?;
        let xs_dtype = xs.dtype()?;

        let scale = 1.0 / (1.0 - self.probability) as f64;

        let mask = Tensor::rand(0.0f32, 1.0f32, &xs_shape, &xs_device, false)?
            .ge(&Tensor::from_scalar(self.probability, &xs_device, false)?
                .broadcast(&xs_shape)?
                .to_dtype(&xs_dtype)?)?
            .to_dtype(&xs_dtype)?
            .affine(scale, 0.0)?;
        Ok(xs.mul(&mask)?)
    }

    fn require_grad(&mut self, _: bool) -> Result<(), crate::ModelError> {
        Ok(())
    }

    fn to_device(&mut self, _: &Device) -> Result<(), crate::ModelError> {
        Ok(())
    }

    fn to_dtype(&mut self, _: &DType) -> Result<(), crate::ModelError> {
        Ok(())
    }

    fn parameters(&self) -> Result<Vec<nove_tensor::Tensor>, crate::ModelError> {
        Ok(vec![])
    }
}

impl Display for Dropout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "dropout.{} (probability={})", self.id, self.probability)
    }
}
