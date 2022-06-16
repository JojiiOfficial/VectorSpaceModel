use crate::{
    build::output::OutputBuilder,
    error::Error,
    traits::{Decodable, Encodable},
};
use byteorder::{ByteOrder, LittleEndian, ReadBytesExt, WriteBytesExt};
use std::{
    convert::TryFrom,
    io::{Cursor, Read, Write},
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

/// Defines required functions of a metadata type. All types implementing this trait can be used as
/// metadata and will be included in the the index
pub trait Metadata: Encodable + Decodable {
    fn get_version(&self) -> IndexVersion;
    fn get_document_count(&self) -> usize;

    fn set_document_count(&mut self, count: usize);

    /// Loads an existing Metadata file
    fn load<R: Read>(mut reader: R) -> Result<Self, Error> {
        let mut buf = vec![];
        reader.read_to_end(&mut buf)?;
        Self::decode::<LittleEndian, _>(Cursor::new(buf))
    }
}

/// Internal behavior for creating new indexes. This should not be overwritten and is only public
/// within this crate.
pub(crate) trait MetadataBuild: Metadata {
    /// Builds a new Metadata file
    fn build<W: Write>(
        &mut self,
        index_builder: &mut OutputBuilder<W>,
        doc_count: usize,
    ) -> Result<(), Error> {
        self.set_document_count(doc_count);
        let mut out = Vec::new();
        out.write_all(&self.encode::<LittleEndian>()?)?;
        index_builder.write_metadata(&out)?;
        Ok(())
    }
}

impl<T: Metadata> MetadataBuild for T {}

/// Various metadata for the given Index
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultMetadata {
    pub version: IndexVersion,
    pub document_count: usize,
}

impl DefaultMetadata {
    /// Creates a new `Metadata` with the given values
    pub fn new(version: IndexVersion) -> Self {
        Self {
            version,
            document_count: 0,
        }
    }
}

impl Metadata for DefaultMetadata {
    fn get_version(&self) -> IndexVersion {
        self.version
    }

    fn get_document_count(&self) -> usize {
        self.document_count
    }

    fn set_document_count(&mut self, count: usize) {
        self.document_count = count;
    }
}

impl Encodable for DefaultMetadata {
    fn encode<T: ByteOrder>(&self) -> Result<Vec<u8>, Error> {
        let mut out = vec![];

        out.write_u8(self.version as u8)?;
        out.write_u64::<T>(self.document_count as u64)?;

        Ok(out)
    }
}

impl Decodable for DefaultMetadata {
    fn decode<T: ByteOrder, R: Read>(mut data: R) -> Result<Self, Error> {
        let version = IndexVersion::try_from(data.read_u8()?)?;
        let document_count = data.read_u64::<T>()? as usize;

        Ok(Self {
            version,
            document_count,
        })
    }
}
