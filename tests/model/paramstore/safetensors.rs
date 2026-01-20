use nove::model::paramstore::{ParamStore, safetensors::SafeTensorsParamStore};
use tempfile::TempDir;

#[test]
fn test_safetensors_paramstore() {
    let temp_dir = TempDir::new().unwrap();
    let param_store = SafeTensorsParamStore::new("test").unwrap();
    param_store
        .save(temp_dir.path().join("test").to_str().unwrap())
        .unwrap();
}
