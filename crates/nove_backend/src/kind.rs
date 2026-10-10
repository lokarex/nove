/// Identifies a tensor backend implementation.
///
/// # Examples
/// ```no_run
/// use nove_backend::BackendKind;
///
/// assert_eq!(BackendKind::Candle, BackendKind::Candle);
/// assert_ne!(BackendKind::Candle, BackendKind::Native);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKind {
    /// Candle-backed storage and numerical operations.
    Candle,
    /// Native reference storage and numerical operations.
    Native,
}
