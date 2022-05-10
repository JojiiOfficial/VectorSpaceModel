#![allow(unused)]

use crate::{dim_map, error::Error, metadata, term_store, vector_store};
use flate2::{write::GzEncoder, Compression};
use std::{fs::File, io::Write};

type Result<T> = std::result::Result<T, Error>;

/// An crate internal helper for building new index files.
pub(crate) struct OutputBuilder<W: Write> {
    builder: tar::Builder<GzEncoder<W>>,
}

impl<W: Write> OutputBuilder<W> {
    /// Create a new IndexBuilder
    pub fn new(out: W) -> Result<OutputBuilder<W>> {
        let enc = GzEncoder::new(out, Compression::best());
        let tar = tar::Builder::new(enc);
        Ok(Self { builder: tar })
    }

    pub fn write_vectors(&mut self, data: &[u8]) -> Result<()> {
        self.append_file(vector_store::FILE_NAME, data)?;
        Ok(())
    }

    pub fn write_term_indexer(&mut self, data: &[u8]) -> Result<()> {
        self.append_file(term_store::FILE_NAME, data)?;
        Ok(())
    }

    pub fn write_metadata(&mut self, data: &[u8]) -> Result<()> {
        self.append_file(metadata::FILE_NAME, data)?;
        Ok(())
    }

    pub fn write_dim_vec_map(&mut self, data: &[u8]) -> Result<()> {
        self.append_file(dim_map::FILE_NAME, data)?;
        Ok(())
    }

    pub fn finish(&mut self) -> Result<()> {
        self.builder.finish()?;
        Ok(())
    }

    /// Append a file to the index
    fn append_file(&mut self, name: &str, data: &[u8]) -> Result<()> {
        let mut header = tar::Header::new_gnu();
        header.set_path(name)?;
        header.set_size(data.len() as u64);
        header.set_entry_type(tar::EntryType::file());
        header.set_cksum();
        self.builder.append(&header, data)?;
        Ok(())
    }
}
