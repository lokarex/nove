use nove::model::paramstore::ParamStore;
use nove::model::paramstore::safetensors::SafeTensorsParamStore;

fn main() {
    // Create a root param store
    let root = SafeTensorsParamStore::new("CompositeModel").unwrap();

    // Create a submodule1
    let submodule1 = SafeTensorsParamStore::new("(feature_extractor): FeatureExtractor").unwrap();
    submodule1
        .set_module(
            SafeTensorsParamStore::new(
                "(conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))",
            )
            .unwrap(),
        )
        .unwrap();
    submodule1
        .set_module(SafeTensorsParamStore::new(
            "(bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)",
        ).unwrap())
        .unwrap();
    submodule1
        .set_module(
            SafeTensorsParamStore::new(
                "(conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))",
            )
            .unwrap(),
        )
        .unwrap();
    submodule1
        .set_module(SafeTensorsParamStore::new(
            "(bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)",
        ).unwrap())
        .unwrap();
    submodule1
        .set_module(SafeTensorsParamStore::new(
            "(pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)",
        ).unwrap())
        .unwrap();

    // Add submodule1 to root
    root.set_module(submodule1).unwrap();

    // Create a submodule2
    let submodule2 = SafeTensorsParamStore::new("(classifier): Classifier").unwrap();
    submodule2
        .set_module(
            SafeTensorsParamStore::new(
                "(fc1): Linear(in_features=4096, out_features=256, bias=True)",
            )
            .unwrap(),
        )
        .unwrap();
    submodule2
        .set_module(SafeTensorsParamStore::new("(dropout): Dropout(p=0.5, inplace=False)").unwrap())
        .unwrap();
    submodule2
        .set_module(
            SafeTensorsParamStore::new(
                "(fc2): Linear(in_features=256, out_features=10, bias=True)",
            )
            .unwrap(),
        )
        .unwrap();

    // Add submodule2 to root
    root.set_module(submodule2).unwrap();

    root.set_module(
        SafeTensorsParamStore::new(
            "(attention_layer): Linear(in_features=4096, out_features=4096, bias=True)",
        )
        .unwrap(),
    )
    .unwrap();
    root.set_module(
        SafeTensorsParamStore::new(
            "(fusion_layer): Linear(in_features=4106, out_features=64, bias=True)",
        )
        .unwrap(),
    )
    .unwrap();
    root.set_module(
        SafeTensorsParamStore::new(
            "(output_layer): Linear(in_features=64, out_features=10, bias=True)",
        )
        .unwrap(),
    )
    .unwrap();

    // Display the param store
    println!("{}", root);
}
