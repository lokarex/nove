macro_rules! dispatch {
    ($self:ident, $method:ident $(,)?) => {
        match $self {
            #[cfg(feature = "candle")]
            Self::Candle(storage) => Ok(Self::Candle(storage.$method()?)),
            #[cfg(feature = "native")]
            Self::Native(storage) => Ok(Self::Native(storage.$method()?)),
        }
    };
    ($self:ident, $method:ident, $($arg:expr),+ $(,)?) => {
        match $self {
            #[cfg(feature = "candle")]
            Self::Candle(storage) => Ok(Self::Candle(storage.$method($($arg),+)?)),
            #[cfg(feature = "native")]
            Self::Native(storage) => Ok(Self::Native(storage.$method($($arg),+)?)),
        }
    };
}

macro_rules! backend_unary_methods {
    ($($method:ident),+ $(,)?) => {
        impl BackendStorage {
            $(
                pub fn $method(&self) -> Result<Self, BackendError> {
                    match self {
                        #[cfg(feature = "candle")]
                        Self::Candle(storage) => Ok(Self::Candle(storage.$method()?)),
                        #[cfg(feature = "native")]
                        Self::Native(storage) => Ok(Self::Native(storage.$method()?)),

                    }
                }
            )+
        }
    };
}

macro_rules! backend_binary_methods {
    ($($method:ident),+ $(,)?) => {
        impl BackendStorage {
            $(
                pub fn $method(&self, rhs: &Self) -> Result<Self, BackendError> {
                    match (self, rhs) {
                        #[cfg(feature = "candle")]
                        (Self::Candle(lhs), Self::Candle(rhs)) => Ok(Self::Candle(lhs.$method(rhs)?)),
                        #[cfg(feature = "native")]
                        (Self::Native(lhs), Self::Native(rhs)) => Ok(Self::Native(lhs.$method(rhs)?)),
                        #[cfg(all(feature = "candle", feature = "native"))]
                        (Self::Candle(_), other) => Err(BackendError::BackendMismatch {
                            expected: BackendKind::Candle,
                            found: other.backend_kind().unwrap_or(BackendKind::Candle),
                        }),
                        #[cfg(all(feature = "candle", feature = "native"))]
                        (Self::Native(_), other) => Err(BackendError::BackendMismatch {
                            expected: BackendKind::Native,
                            found: other.backend_kind().unwrap_or(BackendKind::Native),
                        }),
                    }
                }
            )+
        }
    };
}

pub(super) use {backend_binary_methods, backend_unary_methods, dispatch};
