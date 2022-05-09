use std::{fmt::Display, string::FromUtf8Error};

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    UTF8Error,
    Decode,
    IndexedFile(indexed_file::error::Error),
    InvalidIndex,
}

impl From<FromUtf8Error> for Error {
    #[inline]
    fn from(_: FromUtf8Error) -> Self {
        Self::UTF8Error
    }
}

impl From<std::io::Error> for Error {
    #[inline]
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<indexed_file::error::Error> for Error {
    #[inline]
    fn from(err: indexed_file::error::Error) -> Self {
        Self::IndexedFile(err)
    }
}

impl std::error::Error for Error {}

impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}
