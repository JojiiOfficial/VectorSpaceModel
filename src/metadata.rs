use std::{
    intrinsics::transmute,
    io::{Cursor, Read, Write},
};

use byteorder::{ByteOrder, LittleEndian, ReadBytesExt, WriteBytesExt};

use crate::{error::Error, index::IndexBuilder};

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
        Self::decode::<LittleEndian>(&buf)
    }

    /// Builds a new Metadata file
    pub(crate) fn build(&self, index_builder: &mut IndexBuilder) -> Result<(), Error> {
        let mut out = Vec::new();
        out.write_all(&self.encode::<LittleEndian>()?)?;
        index_builder.write_metadata(&out)?;
        Ok(())
    }

    /// Encode Metadata into binary form
    fn encode<T>(&self) -> Result<Vec<u8>, Error>
    where
        T: ByteOrder,
    {
        let mut out = vec![];

        out.write_u8(self.version as u8)?;
        out.write_u64::<T>(self.document_count as u64)?;

        Ok(out)
    }

    /// Decode metadata information
    fn decode<T>(data: &[u8]) -> Result<Self, Error>
    where
        T: ByteOrder,
    {
        let mut data = Cursor::new(data);

        let version: IndexVersion = unsafe { transmute(data.read_u8()?) };
        let document_count = data.read_u64::<T>()? as usize;

        Ok(Self {
            version,
            document_count,
        })
    }
}
