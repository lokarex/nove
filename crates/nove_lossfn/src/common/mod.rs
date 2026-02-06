use nove_tensor::{Tensor, TensorError};

pub fn log_softmax(xs: &Tensor, dim: usize) -> Result<Tensor, TensorError> {
    let max = xs.max(Some((dim, true)))?;

    let difference = xs.sub(&max)?;

    let log_sum = Tensor::log(&Tensor::sum(&Tensor::exp(&difference)?, Some((dim, true)))?)?;

    Ok(difference.sub(&log_sum)?)
}
