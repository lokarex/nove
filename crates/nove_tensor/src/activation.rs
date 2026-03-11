use std::sync::{Arc, RwLock};

use crate::{
    Tensor, TensorError,
    tensor::{TensorData, TensorInner},
};

impl Tensor {
    /// Apply the Rectified Linear Unit (ReLU) activation function element-wise.
    ///
    /// <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" id="MathJax-script" async></script>
    ///
    /// The ReLU function is a non-linear activation function that outputs the input
    /// directly if it is positive, otherwise it outputs zero. It is widely used in
    /// deep neural networks due to its simplicity and effectiveness.
    ///
    /// The ReLU function is defined as:
    ///
    /// $$ f(x) = \max(0, x) $$
    ///
    /// Where:
    /// - x is the input value
    /// - f(x) is the output value (0 if x < 0, otherwise x)
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The tensor after applying the ReLU activation function.
    /// * `Err(TensorError)` - The error when applying the ReLU activation function.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::cpu();
    /// let t = Tensor::from_data(vec![-1.0, 2.0, -3.0, 4.0], &device, false).unwrap();
    ///
    /// let result = t.relu().unwrap();
    /// println!("{:?}", result);
    /// ```
    pub fn relu(&self) -> Result<Self, TensorError> {
        let inner = self.data.read()?;
        let inner_tensor = match &inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = TensorInner::Tensor(inner_tensor.relu()?);

        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                device: self.data.read()?.device.clone(),
                parents: vec![self.copy()],
                grad: None,
                name: None,
            })),
        })
    }

    /// Apply the Sigmoid Linear Unit (SiLU) activation function element-wise.
    ///
    /// <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" id="MathJax-script" async></script>
    ///
    /// The SiLU function, also known as the Swish activation function, is a smooth,
    /// non-monotonic activation function that has shown improved performance over
    /// traditional activation functions like ReLU in some deep learning models.
    ///
    /// The SiLU function is defined as:
    ///
    /// $$ f(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}} $$
    ///
    /// Where:
    /// - x is the input value
    /// - σ(x) is the sigmoid function: σ(x) = 1 / (1 + e^(-x))
    /// - f(x) is the output value
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The tensor after applying the SiLU activation function.
    /// * `Err(TensorError)` - The error when applying the SiLU activation function.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::cpu();
    /// let t = Tensor::from_data(vec![-1.0, 0.0, 1.0], &device, false).unwrap();
    ///
    /// let result = t.silu().unwrap();
    /// println!("{:?}", result);
    /// assert_eq!(result.to_vec::<f64>().unwrap(), vec![-0.2689414213699951, 0.0, 0.7310585786300049]);
    /// ```
    pub fn silu(&self) -> Result<Self, TensorError> {
        let inner = self.data.read()?;
        let inner_tensor = match &inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = TensorInner::Tensor(inner_tensor.silu()?);

        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                device: self.data.read()?.device.clone(),
                parents: vec![self.copy()],
                grad: None,
                name: None,
            })),
        })
    }

    /// Apply the Gaussian Error Linear Unit (GELU) activation function element-wise.
    ///
    /// <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" id="MathJax-script" async></script>
    ///
    /// The GELU function is a smooth, non-linear activation function that is used in
    /// state-of-the-art transformer models like BERT and GPT. It combines the properties
    /// of dropout and zoneout, providing better performance than ReLU in many tasks.
    ///
    /// The GELU function is defined as:
    ///
    /// $$ f(x) = x \cdot \Phi(x) $$
    ///
    /// A common approximation used in practice is:
    ///
    /// $$ f(x) \approx 0.5 \cdot x \cdot \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \cdot \left(x + 0.044715 \cdot x^3\right)\right)\right) $$
    ///
    /// Where:
    /// - x is the input value
    /// - Φ(x) is the cumulative distribution function of the standard normal distribution:
    ///
    /// $$ \Phi(x) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{x} e^{-\frac{t^2}{2}} dt = \frac{1}{2} \left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right] $$
    ///
    /// - f(x) is the output value
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor after GELU operation.
    /// * `Err(TensorError)` - The error when applying the GELU activation function.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::cpu();
    /// let t = Tensor::from_data(vec![-1.0, 0.0, 1.0], &device, false).unwrap();
    ///
    /// let result = t.gelu().unwrap();
    /// println!("{}", result);
    /// assert_eq!(result.to_vec::<f64>().unwrap(), vec![-0.15880800939172324, 0.0, 0.8411919906082768]);
    /// ```
    pub fn gelu(&self) -> Result<Self, TensorError> {
        let inner = self.data.read()?;
        let inner_tensor = match &inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = TensorInner::Tensor(inner_tensor.gelu()?);

        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                device: self.data.read()?.device.clone(),
                parents: vec![self.copy()],
                grad: None,
                name: None,
            })),
        })
    }

    /// Apply the hyperbolic tangent (tanh) activation function element-wise.
    ///
    /// <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" id="MathJax-script" async></script>
    ///
    /// The tanh function is a smooth, non-linear activation function that maps
    /// input values to the range (-1, 1). It is commonly used in recurrent neural
    /// networks (RNNs) and as an alternative to the sigmoid function when outputs
    /// need to be centered around zero.
    ///
    /// The tanh function is defined as:
    ///
    /// $$ f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = \frac{1 - e^{-2x}}{1 + e^{-2x}} $$
    ///
    /// Where:
    /// - x is the input value
    /// - e is the base of the natural logarithm
    /// - f(x) is the output value in the range (-1, 1)
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The result tensor after hyperbolic tangent operation.
    /// * `Err(TensorError)` - The error when applying the hyperbolic tangent.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::cpu();
    /// let t = Tensor::from_data(vec![-1.0, 0.0, 1.0], &device, false).unwrap();
    ///
    /// let result = t.tanh().unwrap();
    /// println!("{}", result);
    /// assert_eq!(result.to_vec::<f64>().unwrap(), vec![-0.7615941559557649, 0.0, 0.7615941559557649]);
    /// ```
    pub fn tanh(&self) -> Result<Self, TensorError> {
        let inner = self.data.read()?;
        let inner_tensor = match &inner.inner {
            TensorInner::Tensor(tensor) => tensor,
            TensorInner::Var(var) => var,
        };

        let new_inner = TensorInner::Tensor(inner_tensor.tanh()?);

        Ok(Self {
            data: Arc::new(RwLock::new(TensorData {
                inner: new_inner,
                device: self.data.read()?.device.clone(),
                parents: vec![self.copy()],
                grad: None,
                name: None,
            })),
        })
    }

    /// Apply the sigmoid activation function element-wise.
    ///
    /// <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" id="MathJax-script" async></script>
    ///
    /// The sigmoid function is a smooth, non-linear activation function that maps
    /// input values to the range (0, 1). It is commonly used in binary classification
    /// problems and as the output activation function in neural networks where
    /// probability-like outputs are needed.
    ///
    /// The sigmoid function is defined as:
    ///
    /// $$ f(x) = \frac{1}{1 + e^{-x}} = \frac{e^x}{e^x + 1} $$
    ///
    /// Where:
    /// - x is the input value
    /// - e is the base of the natural logarithm
    /// - f(x) is the output value in the range (0, 1)
    ///
    /// # Returns
    /// * `Ok(Tensor)` - The tensor after applying the sigmoid activation function.
    /// * `Err(TensorError)` - The error when applying the sigmoid activation function.
    ///
    /// # Examples
    /// ```
    /// use nove::tensor::{Device, Tensor};
    /// let device = Device::cpu();
    /// let t = Tensor::from_data(vec![-1.0, 0.0, 1.0], &device, false).unwrap();
    ///
    /// let result = t.sigmoid().unwrap();
    /// println!("{}", result);
    /// assert_eq!(result.to_vec::<f64>().unwrap(), vec![0.2689414213699951, 0.5, 0.7310585786300049]);
    /// ```
    pub fn sigmoid(&self) -> Result<Self, TensorError> {
        let denom = self.affine(-1f64, 0f64)?.exp()?.affine(1f64, 1f64)?;
        let numer = Tensor::from_scalar(1u8, &self.device()?, false)?.to_dtype(&self.dtype()?)?;

        numer.div(&denom)
    }
}
