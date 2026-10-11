use nove_backend::{BackendError, Shape};

#[test]
fn checked_count_handles_scalar_zero_and_overflow() {
    let cases: &[(&[usize], usize)] = &[
        (&[], 1),
        (&[2, 3], 6),
        (&[usize::MAX], usize::MAX),
        (&[2, 0, 3], 0),
        (&[0, usize::MAX, 2], 0),
        (&[usize::MAX, 0, 2], 0),
        (&[usize::MAX, 2, 0], 0),
    ];
    for &(dims, expected) in cases {
        assert_eq!(
            Shape::from_dims(dims).checked_elem_count().unwrap(),
            expected
        );
    }
    for dims in [[usize::MAX, 2], [usize::MAX / 2 + 1, 2]] {
        let error = Shape::from_dims(&dims).checked_elem_count().unwrap_err();
        match error {
            BackendError::InvalidOperation(message) => {
                assert!(message.contains("overflow"));
                assert!(message.contains(&format!("{dims:?}")));
            }
            other => panic!("expected InvalidOperation, got {other:?}"),
        }
    }
}
