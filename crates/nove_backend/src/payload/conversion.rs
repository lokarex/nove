use super::{IntoTensorPayload, TensorElement, TensorPayload};
use crate::{BackendError, Shape};

impl<T> IntoTensorPayload for Vec<T>
where
    T: TensorElement,
{
    fn into_tensor_payload(self) -> Result<TensorPayload, BackendError> {
        let shape = Shape::from([self.len()].as_slice());
        TensorPayload::new(T::into_buffer(self), shape)
    }
}

impl<T> IntoTensorPayload for &[T]
where
    T: TensorElement,
{
    fn into_tensor_payload(self) -> Result<TensorPayload, BackendError> {
        let shape = Shape::from([self.len()].as_slice());
        TensorPayload::new(T::into_buffer(self.to_vec()), shape)
    }
}

impl<T, const N: usize> IntoTensorPayload for &[T; N]
where
    T: TensorElement,
{
    fn into_tensor_payload(self) -> Result<TensorPayload, BackendError> {
        self.as_slice().into_tensor_payload()
    }
}

impl<T> IntoTensorPayload for Vec<&[T]>
where
    T: TensorElement,
{
    fn into_tensor_payload(self) -> Result<TensorPayload, BackendError> {
        let rows = self.len();
        let cols = self.first().map(|row| row.len()).unwrap_or(0);
        if self.iter().any(|row| row.len() != cols) {
            return Err(BackendError::InvalidOperation(
                "ragged Vec<&[T]> cannot create a tensor payload".to_string(),
            ));
        }
        let data = self
            .into_iter()
            .flat_map(|row| row.iter().copied())
            .collect::<Vec<_>>();
        TensorPayload::new(T::into_buffer(data), Shape::from_dims(&[rows, cols]))
    }
}

impl<T> IntoTensorPayload for Vec<Vec<T>>
where
    T: TensorElement,
{
    fn into_tensor_payload(self) -> Result<TensorPayload, BackendError> {
        let rows = self.len();
        let cols = self.first().map(|row| row.len()).unwrap_or(0);
        if self.iter().any(|row| row.len() != cols) {
            return Err(BackendError::InvalidOperation(
                "ragged Vec<Vec<T>> cannot create a tensor payload".to_string(),
            ));
        }
        let data = self.into_iter().flatten().collect::<Vec<_>>();
        TensorPayload::new(T::into_buffer(data), Shape::from_dims(&[rows, cols]))
    }
}

impl<T> IntoTensorPayload for Vec<Vec<Vec<T>>>
where
    T: TensorElement,
{
    fn into_tensor_payload(self) -> Result<TensorPayload, BackendError> {
        let d0 = self.len();
        let d1 = self.first().map(|value| value.len()).unwrap_or(0);
        let d2 = self
            .first()
            .and_then(|value| value.first())
            .map(|value| value.len())
            .unwrap_or(0);
        if self
            .iter()
            .any(|value| value.len() != d1 || value.iter().any(|inner| inner.len() != d2))
        {
            return Err(BackendError::InvalidOperation(
                "ragged Vec<Vec<Vec<T>>> cannot create a tensor payload".to_string(),
            ));
        }
        let data = self.into_iter().flatten().flatten().collect::<Vec<_>>();
        TensorPayload::new(T::into_buffer(data), Shape::from_dims(&[d0, d1, d2]))
    }
}

impl<T> IntoTensorPayload for Vec<Vec<Vec<Vec<T>>>>
where
    T: TensorElement,
{
    fn into_tensor_payload(self) -> Result<TensorPayload, BackendError> {
        let d0 = self.len();
        let d1 = self.first().map(|value| value.len()).unwrap_or(0);
        let d2 = self
            .first()
            .and_then(|value| value.first())
            .map(|value| value.len())
            .unwrap_or(0);
        let d3 = self
            .first()
            .and_then(|value| value.first())
            .and_then(|value| value.first())
            .map(|value| value.len())
            .unwrap_or(0);
        if self.iter().any(|value| {
            value.len() != d1
                || value
                    .iter()
                    .any(|inner| inner.len() != d2 || inner.iter().any(|leaf| leaf.len() != d3))
        }) {
            return Err(BackendError::InvalidOperation(
                "ragged Vec<Vec<Vec<Vec<T>>>> cannot create a tensor payload".to_string(),
            ));
        }
        let data = self
            .into_iter()
            .flatten()
            .flatten()
            .flatten()
            .collect::<Vec<_>>();
        TensorPayload::new(T::into_buffer(data), Shape::from_dims(&[d0, d1, d2, d3]))
    }
}

impl<T, const M: usize, const N: usize> IntoTensorPayload for &[[T; N]; M]
where
    T: TensorElement,
{
    fn into_tensor_payload(self) -> Result<TensorPayload, BackendError> {
        let data = self.iter().flatten().copied().collect::<Vec<_>>();
        TensorPayload::new(T::into_buffer(data), Shape::from_dims(&[M, N]))
    }
}

impl<T, const A: usize, const B: usize, const C: usize> IntoTensorPayload for &[[[T; C]; B]; A]
where
    T: TensorElement,
{
    fn into_tensor_payload(self) -> Result<TensorPayload, BackendError> {
        let data = self.iter().flatten().flatten().copied().collect::<Vec<_>>();
        TensorPayload::new(T::into_buffer(data), Shape::from_dims(&[A, B, C]))
    }
}

impl<T, const A: usize, const B: usize, const C: usize, const D: usize> IntoTensorPayload
    for &[[[[T; D]; C]; B]; A]
where
    T: TensorElement,
{
    fn into_tensor_payload(self) -> Result<TensorPayload, BackendError> {
        let data = self
            .iter()
            .flatten()
            .flatten()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        TensorPayload::new(T::into_buffer(data), Shape::from_dims(&[A, B, C, D]))
    }
}
