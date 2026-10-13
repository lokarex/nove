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
                    ensure_compatible_devices(stringify!($method), &[self, rhs])?;
                    validation::validate_binary_broadcast(&self.shape()?, &rhs.shape()?, stringify!($method))?;
                    dispatch_binary!(self, rhs, $method)
                }
            )+
        }
    };
}

pub(super) use {backend_binary_methods, backend_unary_methods, dispatch};

macro_rules! dispatch_binary {
    ($lhs:ident, $rhs:ident, $method:ident $(, $arg:expr)* $(,)?) => {
        match ($lhs, $rhs) {
            #[cfg(feature = "candle")]
            (Self::Candle(a), Self::Candle(b)) => Ok(Self::Candle(a.$method(b $(, $arg)*)?)),
            #[cfg(feature = "native")]
            (Self::Native(a), Self::Native(b)) => Ok(Self::Native(a.$method(b $(, $arg)*)?)),
            #[cfg(all(feature = "candle", feature = "native"))]
            (a, b) => Err(BackendError::BackendMismatch { expected: a.backend_kind()?, found: b.backend_kind()? }),
        }
    };
}

macro_rules! dispatch_ternary {
    ($lhs:ident, $rhs:ident, $third:ident, $method:ident $(, $arg:expr)* $(,)?) => {
        match ($lhs, $rhs, $third) {
            #[cfg(feature = "candle")]
            (Self::Candle(a), Self::Candle(b), Self::Candle(c)) => Ok(Self::Candle(Backend::$method(a, b, c $(, $arg)*)?)),
            #[cfg(feature = "native")]
            (Self::Native(a), Self::Native(b), Self::Native(c)) => Ok(Self::Native(Backend::$method(a, b, c $(, $arg)*)?)),
            #[cfg(all(feature = "candle", feature = "native"))]
            (a, b, c) => Err(BackendError::BackendMismatch { expected: a.backend_kind()?, found: if a.backend_kind()? != b.backend_kind()? { b.backend_kind()? } else { c.backend_kind()? } }),
        }
    };
}

pub(super) use {dispatch_binary, dispatch_ternary};
