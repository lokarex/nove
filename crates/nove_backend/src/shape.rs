/// Tensor shape represented as an ordered list of dimensions.
///
/// # Notes
/// `Shape` is a Nove-owned public type. Backend adapters convert it into their
/// native shape representation only at the backend boundary.
///
/// # Examples
/// ```
/// use nove_backend::Shape;
///
/// let shape = Shape::from_dims(&[2, 3, 4]);
///
/// assert_eq!(shape.dims(), &[2, 3, 4]);
/// assert_eq!(shape.rank(), 3);
/// assert_eq!(shape.elem_count(), 24);
/// ```
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Shape(Vec<usize>);

impl Shape {
    /// Creates a shape from a dimension slice.
    ///
    /// # Arguments
    /// * `dims` - The dimensions to store in the shape.
    ///
    /// # Returns
    /// * [`Shape`] - A shape containing the provided dimensions.
    ///
    /// # Examples
    /// ```
    /// use nove_backend::Shape;
    ///
    /// let shape = Shape::from_dims(&[2, 3]);
    ///
    /// assert_eq!(shape.dims(), &[2, 3]);
    /// ```
    pub fn from_dims(dims: &[usize]) -> Self {
        Self(dims.to_vec())
    }

    /// Returns the shape dimensions.
    ///
    /// # Returns
    /// * `&[usize]` - The dimensions in row-major order.
    ///
    /// # Examples
    /// ```
    /// use nove_backend::Shape;
    ///
    /// let shape = Shape::from((2, 3, 4));
    ///
    /// assert_eq!(shape.dims(), &[2, 3, 4]);
    /// ```
    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    /// Returns the number of dimensions.
    ///
    /// # Returns
    /// * `usize` - The tensor rank.
    ///
    /// # Examples
    /// ```
    /// use nove_backend::Shape;
    ///
    /// assert_eq!(Shape::from(()).rank(), 0);
    /// assert_eq!(Shape::from((2, 3)).rank(), 2);
    /// ```
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// Returns the number of elements described by the shape.
    ///
    /// # Returns
    /// * `usize` - The product of all dimensions. Scalar shapes contain one element.
    ///
    /// # Examples
    /// ```
    /// use nove_backend::Shape;
    ///
    /// assert_eq!(Shape::from(()).elem_count(), 1);
    /// assert_eq!(Shape::from((2, 3, 4)).elem_count(), 24);
    /// ```
    pub fn elem_count(&self) -> usize {
        self.0.iter().product()
    }
}

impl std::fmt::Debug for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.dims())
    }
}

impl<const C: usize> From<&[usize; C]> for Shape {
    fn from(dims: &[usize; C]) -> Self {
        Self(dims.to_vec())
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Self(dims.to_vec())
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Self(dims)
    }
}

impl From<&Shape> for Shape {
    fn from(shape: &Shape) -> Self {
        shape.clone()
    }
}

impl From<()> for Shape {
    fn from(_: ()) -> Self {
        Self(vec![])
    }
}

impl From<usize> for Shape {
    fn from(d1: usize) -> Self {
        Self(vec![d1])
    }
}

macro_rules! impl_from_tuple {
    ($tuple:ty, $($index:tt),+) => {
        impl From<$tuple> for Shape {
            fn from(dims: $tuple) -> Self {
                Self(vec![$(dims.$index,)+])
            }
        }
    };
}

impl_from_tuple!((usize,), 0);
impl_from_tuple!((usize, usize), 0, 1);
impl_from_tuple!((usize, usize, usize), 0, 1, 2);
impl_from_tuple!((usize, usize, usize, usize), 0, 1, 2, 3);
impl_from_tuple!((usize, usize, usize, usize, usize), 0, 1, 2, 3, 4);
impl_from_tuple!((usize, usize, usize, usize, usize, usize), 0, 1, 2, 3, 4, 5);
