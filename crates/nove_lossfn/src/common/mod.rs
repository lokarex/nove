use nove_tensor::{Tensor, TensorError};

/// Compute the log-softmax of the tensor along a specified dimension.
///
/// <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" id="MathJax-script" async></script>
///
/// The softmax function is defined as:
///
/// $$ \text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}} $$
///
/// The log-softmax is the logarithm of the softmax:
///
/// $$ \text{log\_softmax}(x_i) = \log(\text{softmax}(x_i)) = \log\left(\frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}\right) = x_i - \log\left(\sum_{j=1}^{n} e^{x_j}\right) $$
///
/// For numerical stability, we subtract the maximum value before exponentiation:
///
/// $$ \text{log\_softmax}(x_i) = \log\left(\frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}\right) = \log\left(\frac{e^{x_i} \cdot e^{-\max(x)}}{\sum_{j=1}^{n} e^{x_j} \cdot e^{-\max(x)}}\right) = \log\left(\frac{e^{x_i-\max(x)}}{\sum_{j=1}^{n} e^{x_j-\max(x)}}\right) = (x_i - \max(x)) - \log\left(\sum_{j=1}^{n} e^{x_j - \max(x)}\right) $$
///
/// This prevents overflow when computing $$e^{x_j}$$ for large values of $$x_j$$.
///
/// Where:
/// - n is the number of classes
/// - x_i is the input value for the i-th class
/// - \max(x) is the maximum value in the input tensor along the specified dimension
///
/// # Arguments
/// * `xs` - The input tensor.
/// * `dim` - The dimension along which to compute log-softmax.
///
/// # Returns
/// * `Ok(Tensor)` - The result tensor containing log-softmax values.
/// * `Err(TensorError)` - The error when computing log-softmax.
///
/// # Examples
/// ```
/// use nove::tensor::{Device, Tensor};
/// use nove::lossfn::common::log_softmax;
/// let device = Device::cpu();
/// let t = Tensor::from_data(vec![1.0, 2.0, 3.0], &device, false).unwrap();
///
/// let result = log_softmax(&t, 0).unwrap();
/// println!("{:?}", result);
/// ```
pub fn log_softmax(xs: &Tensor, dim: usize) -> Result<Tensor, TensorError> {
    let max = xs.max(Some((dim, true)))?;

    let difference = xs.sub(&max)?;

    let log_sum = Tensor::log(&Tensor::sum(&Tensor::exp(&difference)?, Some((dim, true)))?)?;

    difference.sub(&log_sum)
}
