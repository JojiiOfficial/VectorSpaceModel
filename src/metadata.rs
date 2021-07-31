use std::{
    convert::TryFrom,
    io::{Cursor, Read, Write},
};

use byteorder::{ByteOrder, LittleEndian, ReadBytesExt, WriteBytesExt};

use crate::{
    error::Error,
    index::IndexBuilder,
    traits::{Decodable, Encodable},
};

/// File name in the index tar
pub(crate) const FILE_NAME: &str = "metadata";

/// Version of the index file
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum IndexVersion {
    V1 = 0u8,
}

impl Default for IndexVersion {
    fn default() -> Self {
        Self::V1
    }
}

impl TryFrom<u8> for IndexVersion {
    type Error = crate::error::Error;

    #[inline]
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        Ok(match value {
            0 => Self::V1,
            _ => return Err(Error::Decode),
        })
    }
}

/// Various metadata for the given Index
#[derive(Debug, Clone, Copy, Default)]
pub struct Metadata {
    pub version: IndexVersion,
    pub document_count: usize,
}

impl Metadata {
    /// Creates a new `Metadata` with the given values
    pub fn new(version: IndexVersion, document_count: usize) -> Self {
        Self {
            version,
            document_count,
        }
    }

    /// Loads an existing Metadata file
    pub fn load<R: Read>(mut reader: R) -> Result<Metadata, Error> {
        let mut buf = vec![];
        reader.read_to_end(&mut buf)?;
        Self::decode::<LittleEndian, _>(Cursor::new(buf))
    }

    /// Builds a new Metadata file
    pub(crate) fn build(&self, index_builder: &mut IndexBuilder) -> Result<(), Error> {
        let mut out = Vec::new();
        out.write_all(&self.encode::<LittleEndian>()?)?;
        index_builder.write_metadata(&out)?;
        Ok(())
    }
}

impl Encodable for Metadata {
    fn encode<T: ByteOrder>(&self) -> Result<Vec<u8>, Error> {
        let mut out = vec![];

        out.write_u8(self.version as u8)?;
        out.write_u64::<T>(self.document_count as u64)?;

        Ok(out)
    }
}

impl Decodable for Metadata {
    fn decode<T: ByteOrder, R: Read>(mut data: R) -> Result<Self, Error> {
        let version = IndexVersion::try_from(data.read_u8()?)?;
        let document_count = data.read_u64::<T>()? as usize;

        Ok(Self {
            version,
            document_count,
        })
    }
}
