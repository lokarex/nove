use std::collections::HashMap;

use crate::{
    model::paramstore::{ParamStore, ParamStoreError},
    tensor::{Device, Tensor},
};

pub struct SafeTensorsParamStore {
    params: HashMap<String, Tensor>,
}

impl SafeTensorsParamStore {
    pub fn new() -> Self {
        Self {
            params: HashMap::new(),
        }
    }
}

impl ParamStore for SafeTensorsParamStore {
    fn save(&self, path: &str) -> Result<(), super::ParamStoreError> {
        let params = self
            .params
            .iter()
            .map(|(k, v)| Ok((k.clone(), v.to_candle_tensor()?)))
            .collect::<Result<HashMap<String, candle_core::Tensor>, ParamStoreError>>()?;

        candle_core::safetensors::save(&params, path)
            .map_err(|e| ParamStoreError::OtherError(e.to_string()))?;

        Ok(())
    }

    fn load(
        &mut self,
        path: &str,
        devices: &[Device],
        mut processor: impl FnMut((&str, &Tensor), &[Device]) -> Result<(), ParamStoreError>,
    ) -> Result<(), ParamStoreError> {
        let loaded_params = candle_core::safetensors::load(path, &Device::get_cpu())
            .map_err(|e| ParamStoreError::OtherError(e.to_string()))?;

        self.params
            .iter()
            .zip(loaded_params.iter())
            .map(|((k1, v1), (k2, v2))| {
                if k1 != k2 {
                    return Err(ParamStoreError::OtherError(format!(
                        "key {} in SafeTensorsParamStore does not match key {} in safetensors file",
                        k1, k2
                    )));
                }
                let new_tensor = Tensor::from_candle_tensor(v2.clone(), &Device::get_cpu(), false)?;
                v1.deep_clone_from(&new_tensor)?;
                processor((k1, v1), devices)
            })
            .collect::<Result<(), ParamStoreError>>()?;

        Ok(())
    }

    fn add_param(&mut self, name: &str, param: Tensor) -> Result<(), ParamStoreError> {
        if self.params.contains_key(name) {
            return Err(ParamStoreError::OtherError(format!(
                "param {} already exist in SafeTensorsParamStore",
                name
            )));
        }
        self.params.insert(name.to_string(), param);
        Ok(())
    }

    fn update_param(&mut self, name: &str, param: Tensor) -> Result<(), ParamStoreError> {
        if !self.params.contains_key(name) {
            return Err(ParamStoreError::OtherError(format!(
                "param {} does not exist in SafeTensorsParamStore",
                name
            )));
        }
        self.params.insert(name.to_string(), param);
        Ok(())
    }

    fn to_device(
        &mut self,
        devices: &[Device],
        mut processor: impl FnMut((&str, &Tensor), &[Device]) -> Result<(), ParamStoreError>,
    ) -> Result<(), ParamStoreError> {
        self.params
            .iter()
            .map(|(k, v)| processor((k.as_str(), v), devices))
            .collect::<Result<(), ParamStoreError>>()?;
        Ok(())
    }

    fn parameters(&self) -> Result<Vec<&Tensor>, ParamStoreError> {
        Ok(self.params.values().collect())
    }

    fn named_parameters(&self) -> Result<HashMap<&str, &Tensor>, ParamStoreError> {
        Ok(self.params.iter().map(|(k, v)| (k.as_str(), v)).collect())
    }
}
