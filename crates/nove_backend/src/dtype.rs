use std::{fmt::Display, str::FromStr};
use thiserror::Error;

/// Element data types supported by Nove tensors.
///
/// # Notes
/// `DType` is a Nove-owned public type. Backend adapters convert it into their
/// native dtype representation at the backend boundary.
///
/// # Examples
/// ```
/// use nove_backend::DType;
///
/// assert_eq!(DType::F32.as_str(), "f32");
/// assert_eq!(DType::F32.size_in_bytes(), 4);
/// assert!(DType::F32.is_float());
/// assert!(DType::I64.is_int());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// Unsigned 8-bit integer.
    U8,
    /// Unsigned 32-bit integer.
    U32,
    /// Signed 64-bit integer.
    I64,
    /// Brain floating point 16-bit value.
    BF16,
    /// IEEE 754 half precision floating point value.
    F16,
    /// IEEE 754 single precision floating point value.
    F32,
    /// IEEE 754 double precision floating point value.
    F64,
}

/// Error returned when parsing a dtype from text fails.
///
/// # Examples
/// ```
/// use nove_backend::DType;
///
/// assert!("float32".parse::<DType>().is_err());
/// ```
#[derive(Error, Debug, PartialEq, Eq)]
#[error("cannot parse '{0}' as a dtype")]
pub struct DTypeParseError(String);

impl DType {
    /// Returns the canonical lowercase dtype name.
    ///
    /// # Returns
    /// * `&'static str` - The lowercase dtype name used in display and parsing.
    ///
    /// # Examples
    /// ```
    /// use nove_backend::DType;
    ///
    /// assert_eq!(DType::BF16.as_str(), "bf16");
    /// assert_eq!(DType::U8.as_str(), "u8");
    /// ```
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::U32 => "u32",
            Self::I64 => "i64",
            Self::BF16 => "bf16",
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }

    /// Returns the element size in bytes.
    ///
    /// # Returns
    /// * `usize` - The number of bytes used by one element of this dtype.
    ///
    /// # Examples
    /// ```
    /// use nove_backend::DType;
    ///
    /// assert_eq!(DType::F16.size_in_bytes(), 2);
    /// assert_eq!(DType::F64.size_in_bytes(), 8);
    /// ```
    pub fn size_in_bytes(&self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U32 => 4,
            Self::I64 => 8,
            Self::BF16 | Self::F16 => 2,
            Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    /// Returns true for integer dtypes.
    ///
    /// # Returns
    /// * `bool` - `true` when the dtype stores integer values.
    ///
    /// # Examples
    /// ```
    /// use nove_backend::DType;
    ///
    /// assert!(DType::U32.is_int());
    /// assert!(!DType::F32.is_int());
    /// ```
    pub fn is_int(&self) -> bool {
        matches!(self, Self::U8 | Self::U32 | Self::I64)
    }

    /// Returns true for floating point dtypes.
    ///
    /// # Returns
    /// * `bool` - `true` when the dtype stores floating point values.
    ///
    /// # Examples
    /// ```
    /// use nove_backend::DType;
    ///
    /// assert!(DType::F32.is_float());
    /// assert!(!DType::I64.is_float());
    /// ```
    pub fn is_float(&self) -> bool {
        !self.is_int()
    }
}

impl Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl FromStr for DType {
    type Err = DTypeParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "u8" => Ok(Self::U8),
            "u32" => Ok(Self::U32),
            "i64" => Ok(Self::I64),
            "bf16" => Ok(Self::BF16),
            "f16" => Ok(Self::F16),
            "f32" => Ok(Self::F32),
            "f64" => Ok(Self::F64),
            _ => Err(DTypeParseError(s.to_string())),
        }
    }
}
