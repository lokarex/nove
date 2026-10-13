//! Metadata and parameter validation before backend dispatch.

use crate::{BackendError, DType, Shape};

fn elem_count(shape: &Shape, operation: &'static str) -> Result<usize, BackendError> {
    shape
        .checked_elem_count()
        .map_err(|_| BackendError::ShapeOverflow {
            operation,
            shape: shape.clone(),
        })
}

fn shape_mismatch(
    lhs: &Shape,
    rhs: &Shape,
    axis: Option<usize>,
    operation: &'static str,
) -> BackendError {
    BackendError::ShapeMismatch {
        operation,
        lhs: lhs.clone(),
        rhs: rhs.clone(),
        axis,
    }
}

fn validate_rank(
    shape: &Shape,
    expected: usize,
    input_index: usize,
    operation: &'static str,
) -> Result<(), BackendError> {
    if shape.rank() != expected {
        return Err(BackendError::RankMismatch {
            operation,
            input_index,
            expected,
            found: shape.rank(),
        });
    }
    Ok(())
}

fn validate_dtype(
    found: DType,
    expected: DType,
    input_index: usize,
    operation: &'static str,
) -> Result<(), BackendError> {
    if found != expected {
        return Err(BackendError::DTypeMismatch {
            operation,
            input_index,
            expected,
            found,
        });
    }
    Ok(())
}

pub(crate) fn validate_axis(
    shape: &Shape,
    axis: usize,
    operation: &'static str,
) -> Result<(), BackendError> {
    if axis >= shape.rank() {
        return Err(BackendError::AxisOutOfBounds {
            operation,
            axis,
            rank: shape.rank(),
        });
    }
    Ok(())
}

pub(crate) fn validate_reshape(input: &Shape, target: &Shape) -> Result<(), BackendError> {
    if elem_count(input, "reshape")? != elem_count(target, "reshape")? {
        return Err(shape_mismatch(input, target, None, "reshape"));
    }
    Ok(())
}

pub(crate) fn validate_broadcast(input: &Shape, target: &Shape) -> Result<(), BackendError> {
    if input.rank() > target.rank()
        || input
            .dims()
            .iter()
            .rev()
            .zip(target.dims().iter().rev())
            .any(|(&a, &b)| a != 1 && a != b)
    {
        return Err(shape_mismatch(input, target, None, "broadcast_as"));
    }
    elem_count(target, "broadcast_as")?;
    Ok(())
}

pub(crate) fn validate_permutation(shape: &Shape, dims: &[usize]) -> Result<(), BackendError> {
    if dims.len() != shape.rank() {
        return Err(BackendError::InvalidPermutation {
            operation: "permute",
            permutation: dims.to_vec(),
            rank: shape.rank(),
        });
    }
    let mut seen = vec![false; shape.rank()];
    for &axis in dims {
        validate_axis(shape, axis, "permute")?;
        if seen[axis] {
            return Err(BackendError::InvalidPermutation {
                operation: "permute",
                permutation: dims.to_vec(),
                rank: shape.rank(),
            });
        }
        seen[axis] = true;
    }
    Ok(())
}

pub(crate) fn validate_narrow(
    shape: &Shape,
    dim: usize,
    start: usize,
    length: usize,
) -> Result<(), BackendError> {
    validate_axis(shape, dim, "narrow")?;
    let end = start
        .checked_add(length)
        .ok_or(BackendError::SizeOverflow {
            operation: "narrow",
            parameter: "slice_end",
            lhs: start,
            rhs: length,
        })?;
    if end > shape.dims()[dim] {
        return Err(BackendError::SliceOutOfBounds {
            operation: "narrow",
            axis: dim,
            start,
            length,
            axis_size: shape.dims()[dim],
        });
    }
    Ok(())
}

pub(crate) fn validate_flatten(
    shape: &Shape,
    start: Option<usize>,
    end: Option<usize>,
    operation: &'static str,
) -> Result<(), BackendError> {
    let rank = shape.rank().max(1);
    let start = start.unwrap_or(0);
    let end = end.unwrap_or(rank - 1);
    if start > end || end >= rank {
        return Err(BackendError::InvalidAxisRange {
            operation,
            start,
            end,
            rank,
        });
    }
    if shape.rank() != 0 {
        elem_count(&Shape::from_dims(&shape.dims()[start..=end]), operation)?;
    }
    Ok(())
}

pub(crate) fn validate_unsqueeze(shape: &Shape, dim: usize) -> Result<(), BackendError> {
    if dim > shape.rank() {
        return Err(BackendError::InsertionAxisOutOfBounds {
            operation: "unsqueeze",
            axis: dim,
            rank: shape.rank(),
        });
    }
    Ok(())
}

pub(crate) fn validate_reduction(
    shape: &Shape,
    axis: usize,
    keepdim: bool,
    operation: &'static str,
) -> Result<(), BackendError> {
    validate_axis(shape, axis, operation)?;
    let mut dims = shape.dims().to_vec();
    if keepdim {
        dims[axis] = 1;
    } else {
        dims.remove(axis);
    }
    elem_count(&Shape::from_dims(&dims), operation)?;
    Ok(())
}

pub(crate) fn validate_index_select(
    input: &Shape,
    indexes: &Shape,
    dtype: DType,
    dim: usize,
) -> Result<(), BackendError> {
    validate_axis(input, dim, "index_select")?;
    validate_dtype(dtype, DType::I64, 1, "index_select")?;
    validate_rank(indexes, 1, 1, "index_select")?;
    let mut dims = input.dims().to_vec();
    dims[dim] = indexes.dims()[0];
    elem_count(&Shape::from_dims(&dims), "index_select")?;
    Ok(())
}

pub(crate) fn validate_embedding(
    table: &Shape,
    indexes: &Shape,
    dtype: DType,
) -> Result<(), BackendError> {
    validate_dtype(dtype, DType::I64, 1, "embedding")?;
    validate_rank(table, 2, 0, "embedding")?;
    validate_rank(indexes, 1, 1, "embedding")?;
    elem_count(
        &Shape::from_dims(&[indexes.dims()[0], table.dims()[1]]),
        "embedding",
    )?;
    Ok(())
}

pub(crate) fn validate_gather(
    input: &Shape,
    indexes: &Shape,
    dtype: DType,
    dim: usize,
) -> Result<(), BackendError> {
    validate_axis(input, dim, "gather")?;
    validate_dtype(dtype, DType::I64, 1, "gather")?;
    validate_rank(indexes, input.rank(), 1, "gather")?;
    if input
        .dims()
        .iter()
        .zip(indexes.dims())
        .enumerate()
        .any(|(axis, (&a, &b))| axis != dim && b > a)
    {
        return Err(shape_mismatch(input, indexes, Some(dim), "gather"));
    }
    elem_count(indexes, "gather")?;
    Ok(())
}

pub(crate) fn validate_scatter_add(
    input: &Shape,
    dtype: DType,
    indexes: &Shape,
    index_dtype: DType,
    source: &Shape,
    source_dtype: DType,
    dim: usize,
) -> Result<(), BackendError> {
    validate_axis(input, dim, "scatter_add")?;
    validate_dtype(index_dtype, DType::I64, 1, "scatter_add")?;
    validate_dtype(source_dtype, dtype, 2, "scatter_add")?;
    validate_rank(source, input.rank(), 2, "scatter_add")?;
    if indexes != source {
        return Err(shape_mismatch(indexes, source, Some(dim), "scatter_add"));
    }
    if input
        .dims()
        .iter()
        .zip(source.dims())
        .enumerate()
        .any(|(axis, (&a, &b))| axis != dim && b > a)
    {
        return Err(shape_mismatch(input, source, Some(dim), "scatter_add"));
    }
    Ok(())
}

pub(crate) fn validate_index_add(
    input: &Shape,
    dtype: DType,
    indexes: &Shape,
    index_dtype: DType,
    source: &Shape,
    source_dtype: DType,
    dim: usize,
) -> Result<(), BackendError> {
    validate_axis(input, dim, "index_add")?;
    validate_dtype(index_dtype, DType::I64, 1, "index_add")?;
    validate_dtype(source_dtype, dtype, 2, "index_add")?;
    validate_rank(indexes, 1, 1, "index_add")?;
    validate_rank(source, input.rank(), 2, "index_add")?;
    if source.dims()[dim] != indexes.dims()[0] {
        return Err(shape_mismatch(indexes, source, Some(dim), "index_add"));
    }
    if input
        .dims()
        .iter()
        .zip(source.dims())
        .enumerate()
        .any(|(axis, (&a, &b))| axis != dim && a != b)
    {
        return Err(shape_mismatch(input, source, Some(dim), "index_add"));
    }
    Ok(())
}

pub(crate) fn validate_stack(inputs: &[(&Shape, DType)], dim: usize) -> Result<(), BackendError> {
    let operation = "stack";
    let Some(&(first, dtype)) = inputs.first() else {
        return Err(BackendError::EmptyInputs { operation });
    };
    if dim > first.rank() {
        return Err(BackendError::InsertionAxisOutOfBounds {
            operation,
            axis: dim,
            rank: first.rank(),
        });
    }
    for (index, &(shape, other_dtype)) in inputs.iter().enumerate() {
        if shape != first {
            return Err(shape_mismatch(first, shape, None, operation));
        }
        validate_dtype(other_dtype, dtype, index, operation)?;
    }
    let mut dims = first.dims().to_vec();
    dims.insert(dim, inputs.len());
    elem_count(&Shape::from_dims(&dims), operation)?;
    Ok(())
}

pub(crate) fn validate_cat(inputs: &[(&Shape, DType)], dim: usize) -> Result<(), BackendError> {
    let operation = "cat";
    let Some(&(first, dtype)) = inputs.first() else {
        return Err(BackendError::EmptyInputs { operation });
    };
    validate_axis(first, dim, operation)?;
    let mut length = 0usize;
    for (index, &(shape, other_dtype)) in inputs.iter().enumerate() {
        validate_rank(shape, first.rank(), index, operation)?;
        validate_dtype(other_dtype, dtype, index, operation)?;
        if shape
            .dims()
            .iter()
            .zip(first.dims())
            .enumerate()
            .any(|(axis, (&a, &b))| axis != dim && a != b)
        {
            return Err(shape_mismatch(first, shape, Some(dim), operation));
        }
        length = length
            .checked_add(shape.dims()[dim])
            .ok_or(BackendError::SizeOverflow {
                operation,
                parameter: "axis_length",
                lhs: length,
                rhs: shape.dims()[dim],
            })?;
    }
    let mut dims = first.dims().to_vec();
    dims[dim] = length;
    elem_count(&Shape::from_dims(&dims), operation)?;
    Ok(())
}

pub(crate) fn broadcast_shape(
    lhs: &Shape,
    rhs: &Shape,
    operation: &'static str,
) -> Result<Shape, BackendError> {
    let rank = lhs.rank().max(rhs.rank());
    let mut dims = vec![1; rank];
    for (offset, output) in dims.iter_mut().rev().enumerate() {
        let a = lhs.dims().iter().rev().nth(offset).copied().unwrap_or(1);
        let b = rhs.dims().iter().rev().nth(offset).copied().unwrap_or(1);
        *output = if a == b || b == 1 {
            a
        } else if a == 1 {
            b
        } else {
            return Err(shape_mismatch(lhs, rhs, None, operation));
        };
    }
    Ok(Shape::from_dims(&dims))
}

pub(crate) fn validate_binary_broadcast(
    lhs: &Shape,
    rhs: &Shape,
    operation: &'static str,
) -> Result<(), BackendError> {
    elem_count(&broadcast_shape(lhs, rhs, operation)?, operation)?;
    Ok(())
}

pub(crate) fn validate_matmul(lhs: &Shape, rhs: &Shape) -> Result<(), BackendError> {
    let operation = "broadcast_matmul";
    for (input_index, shape) in [lhs, rhs].into_iter().enumerate() {
        if shape.rank() < 2 {
            return Err(BackendError::RankTooSmall {
                operation,
                input_index,
                minimum: 2,
                found: shape.rank(),
            });
        }
    }
    if lhs.dims()[lhs.rank() - 1] != rhs.dims()[rhs.rank() - 2] {
        return Err(shape_mismatch(lhs, rhs, None, operation));
    }
    let lhs_batch = Shape::from_dims(&lhs.dims()[..lhs.rank() - 2]);
    let rhs_batch = Shape::from_dims(&rhs.dims()[..rhs.rank() - 2]);
    let mut dims = broadcast_shape(&lhs_batch, &rhs_batch, operation)?
        .dims()
        .to_vec();
    dims.extend_from_slice(&[lhs.dims()[lhs.rank() - 2], rhs.dims()[rhs.rank() - 1]]);
    elem_count(&Shape::from_dims(&dims), operation)?;
    Ok(())
}

pub(crate) fn validate_where(
    condition: DType,
    true_dtype: DType,
    false_dtype: DType,
) -> Result<(), BackendError> {
    validate_dtype(condition, DType::U8, 0, "where_cond")?;
    validate_dtype(false_dtype, true_dtype, 2, "where_cond")
}

pub(crate) fn validate_where_shapes(
    condition: &Shape,
    true_value: &Shape,
    false_value: &Shape,
) -> Result<(), BackendError> {
    let values = broadcast_shape(true_value, false_value, "where_cond")?;
    elem_count(
        &broadcast_shape(condition, &values, "where_cond")?,
        "where_cond",
    )?;
    Ok(())
}

pub(crate) fn validate_clamp(min: f64, max: f64) -> Result<(), BackendError> {
    if min.is_nan() || max.is_nan() || min > max {
        return Err(BackendError::InvalidBounds {
            operation: "clamp",
            lower: min,
            upper: max,
        });
    }
    Ok(())
}

pub(crate) fn validate_uniform(low: f64, high: f64) -> Result<(), BackendError> {
    if !low.is_finite() || !high.is_finite() || low >= high || !(high - low).is_finite() {
        return Err(BackendError::InvalidBounds {
            operation: "rand",
            lower: low,
            upper: high,
        });
    }
    Ok(())
}

pub(crate) fn validate_normal(mean: f64, std: f64) -> Result<(), BackendError> {
    if !mean.is_finite() || !std.is_finite() || std < 0.0 {
        return Err(BackendError::InvalidNormalParameters {
            operation: "randn",
            mean,
            std,
        });
    }
    Ok(())
}

pub(crate) fn validate_convolution_basics(
    input: &Shape,
    kernel: &Shape,
    spatial_rank: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
    transposed: bool,
) -> Result<(), BackendError> {
    let operation = match (spatial_rank, transposed) {
        (1, false) => "conv1d",
        (2, false) => "conv2d",
        (1, true) => "conv_transpose1d",
        (2, true) => "conv_transpose2d",
        _ => {
            return Err(BackendError::InvalidOperation(format!(
                "convolution: unsupported spatial rank {spatial_rank}"
            )));
        }
    };
    validate_rank(input, spatial_rank + 2, 0, operation)?;
    validate_rank(kernel, spatial_rank + 2, 1, operation)?;
    for (axis, &size) in kernel.dims().iter().enumerate().skip(2) {
        if size == 0 {
            return Err(BackendError::EmptySpatialDimension {
                operation,
                input_index: 1,
                axis,
            });
        }
    }
    for (parameter, value) in [
        ("stride", stride),
        ("dilation", dilation),
        ("groups", groups),
    ] {
        if value == 0 {
            return Err(BackendError::ZeroParameter {
                operation,
                parameter,
            });
        }
    }
    if transposed {
        for (axis, &size) in input.dims().iter().enumerate().skip(2) {
            if size == 0 {
                return Err(BackendError::EmptySpatialDimension {
                    operation,
                    input_index: 0,
                    axis,
                });
            }
        }
    }
    Ok(())
}

pub(crate) fn validate_pool2d(
    input: &Shape,
    kernel: (usize, usize),
    stride: (usize, usize),
    operation: &'static str,
) -> Result<(), BackendError> {
    validate_rank(input, 4, 0, operation)?;
    for (parameter, value) in [
        ("kernel.0", kernel.0),
        ("kernel.1", kernel.1),
        ("stride.0", stride.0),
        ("stride.1", stride.1),
    ] {
        if value == 0 {
            return Err(BackendError::ZeroParameter {
                operation,
                parameter,
            });
        }
    }
    if kernel.0 > input.dims()[2] || kernel.1 > input.dims()[3] {
        return Err(BackendError::KernelTooLarge {
            operation,
            kernel,
            spatial: (input.dims()[2], input.dims()[3]),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BackendError, Shape};

    fn overflow(result: Result<(), BackendError>, operation: &str, expected: &[usize]) {
        match result.unwrap_err() {
            BackendError::ShapeOverflow {
                operation: name,
                shape,
            } => {
                assert_eq!(name, operation);
                assert_eq!(shape.dims(), expected);
            }
            error => panic!("expected ShapeOverflow, got {error:?}"),
        }
    }

    #[test]
    fn reduction_outputs_reject_overflow() {
        let shape = Shape::from_dims(&[0, usize::MAX, 2]);
        overflow(
            validate_reduction(&shape, 0, false, "sum"),
            "sum",
            &[usize::MAX, 2],
        );
        overflow(
            validate_reduction(&shape, 0, true, "sum_keepdim"),
            "sum_keepdim",
            &[1, usize::MAX, 2],
        );
    }

    #[test]
    fn flatten_dimensions_reject_overflow() {
        let shape = Shape::from_dims(&[usize::MAX, 2, 0]);
        overflow(
            validate_flatten(&shape, Some(0), Some(1), "flatten"),
            "flatten",
            &[usize::MAX, 2],
        );
        validate_flatten(&shape, Some(0), Some(2), "flatten").unwrap();
        validate_flatten(&Shape::from(()), Some(0), None, "flatten_from").unwrap();
        assert!(matches!(
            validate_flatten(&Shape::from(()), Some(usize::MAX), None, "flatten_from"),
            Err(BackendError::InvalidAxisRange {
                operation: "flatten_from",
                start: usize::MAX,
                end: 0,
                rank: 1
            })
        ));
    }

    #[test]
    fn combination_checks_output_overflow() {
        let large = Shape::from_dims(&[usize::MAX / 2 + 1]);
        overflow(
            validate_stack(&[(&large, DType::F32), (&large, DType::F32)], 0),
            "stack",
            &[2, usize::MAX / 2 + 1],
        );
        assert!(
            matches!(validate_cat(&[(&large, DType::F32), (&large, DType::F32)], 0),
            Err(BackendError::SizeOverflow { operation: "cat", parameter: "axis_length", lhs, rhs })
            if lhs == usize::MAX / 2 + 1 && rhs == usize::MAX / 2 + 1)
        );
        let large = Shape::from_dims(&[usize::MAX / 2 + 1, 1]);
        overflow(
            validate_cat(&[(&large, DType::F32), (&large, DType::F32)], 1),
            "cat",
            &[usize::MAX / 2 + 1, 2],
        );
        let a = Shape::from_dims(&[0, usize::MAX]);
        let b = Shape::from_dims(&[0, 1]);
        assert!(matches!(
            validate_cat(&[(&a, DType::F32), (&b, DType::F32)], 1),
            Err(BackendError::SizeOverflow {
                operation: "cat",
                parameter: "axis_length",
                lhs: usize::MAX,
                rhs: 1
            })
        ));
    }

    #[test]
    fn index_outputs_reject_overflow() {
        overflow(
            validate_index_select(
                &Shape::from_dims(&[usize::MAX, 0]),
                &Shape::from_dims(&[2]),
                DType::I64,
                1,
            ),
            "index_select",
            &[usize::MAX, 2],
        );
        overflow(
            validate_embedding(
                &Shape::from_dims(&[0, usize::MAX]),
                &Shape::from_dims(&[2]),
                DType::I64,
            ),
            "embedding",
            &[2, usize::MAX],
        );
    }

    #[test]
    fn binary_outputs_reject_overflow() {
        overflow(
            validate_binary_broadcast(
                &Shape::from_dims(&[usize::MAX, 1]),
                &Shape::from_dims(&[1, 2]),
                "broadcast_add",
            ),
            "broadcast_add",
            &[usize::MAX, 2],
        );
        overflow(
            validate_matmul(
                &Shape::from_dims(&[usize::MAX, 0]),
                &Shape::from_dims(&[0, 2]),
            ),
            "broadcast_matmul",
            &[usize::MAX, 2],
        );
        validate_matmul(
            &Shape::from_dims(&[usize::MAX, 2, 0, 0]),
            &Shape::from_dims(&[usize::MAX, 2, 0, 0]),
        )
        .unwrap();
    }

    #[test]
    fn where_outputs_reject_overflow() {
        overflow(
            validate_where_shapes(
                &Shape::from_dims(&[1]),
                &Shape::from_dims(&[usize::MAX, 1]),
                &Shape::from_dims(&[1, 2]),
            ),
            "where_cond",
            &[usize::MAX, 2],
        );
    }

    #[test]
    fn broadcast_preserves_zero_dimensions() {
        assert_eq!(
            broadcast_shape(
                &Shape::from_dims(&[2, 0, 3]),
                &Shape::from_dims(&[1, 1, 3]),
                "test"
            )
            .unwrap()
            .dims(),
            &[2, 0, 3]
        );
        assert_eq!(
            broadcast_shape(&Shape::from(()), &Shape::from(()), "test")
                .unwrap()
                .dims(),
            &[] as &[usize]
        );
    }

    #[test]
    fn random_interval_rejects_overflow() {
        assert!(matches!(validate_uniform(-f64::MAX, f64::MAX),
            Err(BackendError::InvalidBounds { operation: "rand", lower, upper })
            if lower == -f64::MAX && upper == f64::MAX));
    }

    #[test]
    fn convolution_basics_preserve_empty_forward_input() {
        let input = Shape::from_dims(&[1, 1, 0]);
        let kernel = Shape::from_dims(&[1, 1, 1]);
        validate_convolution_basics(&input, &kernel, 1, 1, 1, 1, false).unwrap();
        assert!(matches!(
            validate_convolution_basics(&input, &kernel, 1, 1, 1, 1, true),
            Err(BackendError::EmptySpatialDimension {
                operation: "conv_transpose1d",
                input_index: 0,
                axis: 2
            })
        ));
    }
}
