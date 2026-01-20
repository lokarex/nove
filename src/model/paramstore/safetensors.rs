use std::{
    collections::HashMap,
    fmt::Display,
    sync::{Arc, RwLock},
};

use indexmap::IndexMap;

use crate::{
    model::{
        Parameter,
        paramstore::{ParamStore, ParamStoreError},
    },
    tensor::{Device, Tensor},
};

#[derive(Debug)]
struct SafeTensorsParamStoreInner {
    name: RwLock<String>,
    params: RwLock<IndexMap<String, Parameter>>,
    modules: RwLock<IndexMap<String, SafeTensorsParamStore>>,
}

#[derive(Debug, Clone)]
pub struct SafeTensorsParamStore {
    inner: Arc<SafeTensorsParamStoreInner>,
}

impl ParamStore for SafeTensorsParamStore {
    fn new(name: &str) -> Result<Self, ParamStoreError> {
        Ok(Self {
            inner: Arc::new(SafeTensorsParamStoreInner {
                name: RwLock::new(name.to_string()),
                params: RwLock::new(IndexMap::new()),
                modules: RwLock::new(IndexMap::new()),
            }),
        })
    }

    fn set_name(&self, name: &str) -> Result<(), ParamStoreError> {
        *self.inner.name.write()? = name.to_string();
        Ok(())
    }

    fn name(&self) -> Result<String, ParamStoreError> {
        Ok(self.inner.name.read()?.clone())
    }

    fn save(&self, folder_path: &str) -> Result<(), ParamStoreError> {
        // Create the folder if it does not exist.
        std::fs::create_dir_all(folder_path)?;

        let file_path = folder_path.to_string() + "/" + self.name()?.as_str() + ".safetensors";
        let all_params = Self::all_params_with_full_name(self, &self.name()?)?
            .iter()
            .map(|(k, v)| {
                Ok::<(String, candle_core::Tensor), ParamStoreError>((
                    k.clone(),
                    v.1.to_candle_tensor()?,
                ))
            })
            .collect::<Result<HashMap<_, _>, _>>()?;

        candle_core::safetensors::save(&all_params, file_path)
            .map_err(|e| ParamStoreError::OtherError(e.to_string()))?;

        Ok(())
    }

    fn load(
        &self,
        folder_path: &str,
        device: &Device,
        mut process_fn: impl FnMut(&str, &Self) -> Result<(), ParamStoreError>,
    ) -> Result<(), ParamStoreError> {
        let file_path = folder_path.to_string() + "/" + self.name()?.as_str() + ".safetensors";
        // Load the parameters from the file.
        let loaded_params = candle_core::safetensors::load(file_path.as_str(), device)
            .map_err(|e| ParamStoreError::OtherError(e.to_string()))?;

        // Get all parameters in the store.
        let all_params = Self::all_params_with_full_name(self, &self.name()?)?;

        // Check if the number of parameters in the file matches the number of parameters in the store.
        if loaded_params.len() != all_params.len() {
            return Err(ParamStoreError::OtherError(format!(
                "The number({}) of parameters in the file({}) does not match the number({}) of parameters in the store",
                loaded_params.len(),
                file_path,
                all_params.len()
            )));
        }

        for (full_name, param) in all_params {
            // Check if the parameter exists in the file.
            let loaded_param = loaded_params.get(full_name.as_str()).ok_or_else(|| {
                ParamStoreError::OtherError(format!(
                    "Parameter {} not found in the file({}).",
                    full_name, file_path
                ))
            })?;

            // Convert the loaded parameter to a `Tensor`.
            let loaded_tensor = Tensor::from_candle_tensor(loaded_param.clone(), device, false)?;
            // Update the parameter in the store with the loaded tensor.
            param.1.update_from_tensor(&loaded_tensor)?;
        }

        // Process each module in the store.
        self.process_all_modules_with_full_name(self.name()?.as_str(), &mut process_fn)?;

        Ok(())
    }

    fn set_module(&self, module: Self) -> Result<(), ParamStoreError> {
        self.inner.modules.write()?.insert(module.name()?, module);
        Ok(())
    }

    fn modules(&self) -> Result<Vec<Self>, ParamStoreError> {
        Ok(self
            .inner
            .modules
            .read()?
            .iter()
            .map(|(_, v)| v.clone())
            .collect())
    }

    fn set_paramter(&self, param: Parameter) -> Result<(), ParamStoreError> {
        self.inner.params.write()?.insert(param.0.clone(), param);
        Ok(())
    }

    fn parameters(&self) -> Result<Vec<Parameter>, ParamStoreError> {
        Ok(self
            .inner
            .params
            .read()?
            .iter()
            .map(|(_, v)| v.clone())
            .collect())
    }
}

impl SafeTensorsParamStore {
    fn all_params_with_full_name(
        &self,
        prefix: &str,
    ) -> Result<HashMap<String, Parameter>, ParamStoreError> {
        let mut all_params = HashMap::new();

        // Get the direct parameters in the current module.
        for (name, param) in &*self.inner.params.read()? {
            let full_name = format!("{}.{}", prefix, name);
            all_params.insert(full_name, param.clone());
        }

        // Recursively get the parameters in the submodules.
        for (module_name, module) in &*self.inner.modules.read()? {
            let new_prefix = format!("{}.{}", prefix, module_name);
            let module_params = Self::all_params_with_full_name(module, &new_prefix)?;
            all_params.extend(module_params);
        }

        Ok(all_params)
    }

    fn process_all_modules_with_full_name(
        &self,
        prefix: &str,
        process_fn: &mut impl FnMut(&str, &Self) -> Result<(), ParamStoreError>,
    ) -> Result<(), ParamStoreError> {
        // Process the current module
        process_fn(prefix, self)?;

        // Recursively process all submodules
        let modules = self.inner.modules.read()?.clone();
        for (module_name, module) in modules {
            let new_prefix = format!("{}.{}", prefix, module_name);
            module.process_all_modules_with_full_name(new_prefix.as_str(), process_fn)?;
        }

        Ok(())
    }
}

impl SafeTensorsParamStore {
    fn fmt_with_indent(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        indent: &mut String,
    ) -> std::fmt::Result {
        write!(f, "{}{}", indent, self.name().map_err(|_| std::fmt::Error)?)?;

        // Display submodules
        if !self
            .inner
            .modules
            .read()
            .map_err(|_| std::fmt::Error)?
            .is_empty()
        {
            writeln!(f, "(")?;
            indent.push_str("  ");
            for (_, module) in &*self.inner.modules.read().map_err(|_| std::fmt::Error)? {
                module.fmt_with_indent(f, indent)?;
            }
            indent.truncate(indent.len().saturating_sub(2));
            writeln!(f, "{})", indent)?;
        } else {
            writeln!(f, "")?;
        }

        Ok(())
    }
}

impl Display for SafeTensorsParamStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut indent = String::new();
        self.fmt_with_indent(f, &mut indent)?;
        Ok(())
    }
}
