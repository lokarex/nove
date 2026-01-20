use nove::{
    model::{
        Model,
        layer::Dropout,
        paramstore::{ParamStore, safetensors::SafeTensorsParamStore},
    },
    tensor::Tensor,
};

struct TestModel<P: ParamStore> {
    store: P,
    dropout: Dropout<P>,
}

impl<P: ParamStore> TestModel<P> {
    fn new() -> Self {
        let store = P::new("HHHHHHHH").unwrap();
        let dropout = Dropout::new(&store, 0.5).unwrap();
        Self { store, dropout }
    }
}

impl<P: ParamStore> Model for TestModel<P> {
    type ParamStore = P;
    type Input = (Tensor, bool);
    type Output = Tensor;

    fn param_stores(&self) -> Result<Vec<Self::ParamStore>, nove::model::ModelError> {
        Ok(vec![self.store.clone()])
    }

    fn forward(&mut self, input: Self::Input) -> Result<Self::Output, nove::model::ModelError> {
        let x = self.dropout.forward(input)?;
        Ok(x)
    }
}

fn main() {
    let model = TestModel::<SafeTensorsParamStore>::new();
    let store = model.param_stores().unwrap();
    println!("{}", store[0]);
    println!("{}", model.summary().unwrap());
}
