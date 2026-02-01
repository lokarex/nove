use std::{
    collections::HashMap,
    fmt::Display,
    sync::atomic::{AtomicUsize, Ordering},
};

use nove_tensor::{DType, Device, Shape, Tensor};

use crate::{Model, ModelError};

static ID: AtomicUsize = AtomicUsize::new(0);

pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
    in_features: usize,
    out_features: usize,
    id: usize,
}

impl Linear {
    fn weight_name(&self) -> String {
        format!("linear.{}.weight", self.id)
    }

    fn bias_name(&self) -> String {
        format!("linear.{}.bias", self.id)
    }
}

impl Model for Linear {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&mut self, input: Self::Input) -> Result<Self::Output, ModelError> {
        let y = self.weight.matmul(&input)?;

        let y = if let Some(bias) = &self.bias {
            y.add(bias)?
        } else {
            y
        };

        Ok(y)
    }

    fn set_grad_enabled(&mut self, grad_enabled: bool) -> Result<(), ModelError> {
        self.weight.set_grad_enabled(grad_enabled)?;
        if let Some(bias) = &mut self.bias {
            bias.set_grad_enabled(grad_enabled)?;
        }
        Ok(())
    }

    fn to_device(&mut self, device: &Device) -> Result<(), ModelError> {
        self.weight.to_device(device)?;
        if let Some(bias) = &mut self.bias {
            bias.to_device(device)?;
        }
        Ok(())
    }

    fn to_dtype(&mut self, dtype: &DType) -> Result<(), ModelError> {
        self.weight.to_dtype(dtype)?;
        if let Some(bias) = &mut self.bias {
            bias.to_dtype(dtype)?;
        }
        Ok(())
    }

    fn to_safetensors(&self) -> Result<HashMap<String, Tensor>, ModelError> {
        let mut tensors = HashMap::new();

        tensors.insert(self.weight_name(), self.weight.clone());

        if let Some(bias) = &self.bias {
            tensors.insert(self.bias_name(), bias.clone());
        }
        Ok(tensors)
    }

    fn load_from_safetensors(
        &mut self,
        tensors: HashMap<String, Tensor>,
    ) -> Result<(), ModelError> {
        let weight_name = self.weight_name();
        let bias_name = self.bias_name();

        // Update weight
        let new_weight = tensors
            .get(&weight_name)
            .ok_or(ModelError::MissingParameter(weight_name))?;
        self.weight.update_from_tensor(new_weight)?;

        // Update bias
        match (&self.bias, tensors.get(&bias_name)) {
            // If the old and new biases exist, update the old bias.
            (Some(bias), Some(new_bias)) => bias.update_from_tensor(new_bias)?,
            // If the old bias does not exist, but new bias is provided, return an error.
            (None, Some(_)) => {
                return Err(ModelError::UnexpectedParameter(bias_name));
            }
            // If the old bias exists, but new bias is not provided, return an error.
            (Some(_), None) => {
                return Err(ModelError::MissingParameter(bias_name));
            }
            // If the old and new biases do not exist, do nothing.
            (None, None) => {}
        }

        Ok(())
    }
}

impl Display for Linear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "linear.{} (in_features={}, out_features={}, bias_enabled={})",
            self.id,
            self.in_features,
            self.out_features,
            self.bias.is_some()
        )
    }
}

pub struct LinearBuilder {
    in_features: Option<usize>,
    out_features: Option<usize>,
    bias_enabled: bool,
    device: Device,
    dtype: DType,
}

impl Default for LinearBuilder {
    fn default() -> Self {
        Self {
            in_features: None,
            out_features: None,
            bias_enabled: true,
            device: Device::cpu(),
            dtype: DType::F32,
        }
    }
}

impl LinearBuilder {
    pub fn in_features(mut self, in_features: usize) -> Self {
        self.in_features = Some(in_features);
        self
    }

    pub fn out_features(mut self, out_features: usize) -> Self {
        self.out_features = Some(out_features);
        self
    }

    pub fn bias_enabled(mut self, bias_enabled: bool) -> Self {
        self.bias_enabled = bias_enabled;
        self
    }

    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    pub fn dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }

    pub fn build(&self) -> Result<Linear, ModelError> {
        let in_features = self.in_features.ok_or(ModelError::MissingField(
            "in_features in LinearBuilder".to_string(),
        ))?;
        let out_features = self.out_features.ok_or(ModelError::MissingField(
            "out_features in LinearBuilder".to_string(),
        ))?;

        // Generate a unique ID for the linear layer.
        let id = ID.fetch_add(1, Ordering::Relaxed);

        // Determine the bounds for the weight and bias initialization.
        let bound: f32 = 1.0 / (self.in_features.unwrap() as f32).sqrt();
        let low: f32 = -bound;
        let high: f32 = bound;

        // Initialize the weight tensor.
        let weight = Tensor::rand(
            low,
            high,
            &Shape::from_dims(&[self.in_features.unwrap(), self.out_features.unwrap()]),
            &self.device,
            true,
        )?
        .to_dtype(&self.dtype)?;
        let mut weight = weight;
        weight.set_name(format!("linear.{}.weight", id))?;

        // Initialize the bias tensor if enabled.
        let bias = if self.bias_enabled {
            let mut bias = Tensor::rand(
                low,
                high,
                &Shape::from_dims(&[self.out_features.unwrap()]),
                &self.device,
                true,
            )?
            .to_dtype(&self.dtype)?;
            bias.set_name(format!("linear.{}.bias", id))?;
            Some(bias)
        } else {
            None
        };

        Ok(Linear {
            weight,
            bias,
            in_features,
            out_features,
            id,
        })
    }
}
