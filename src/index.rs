use std::{
    fs::File,
    io::{BufReader, Cursor, Read},
    path::Path,
};

use flate2::{read::GzDecoder, write::GzEncoder, Compression};
use tar::Entry;

use crate::{
    dim_map::{self, DimVecMap},
    document::DocumentVector,
    error::Error,
    metadata::{self, Metadata, MetadataBuild},
    term_store::{self, item::IndexTerm, TermIndexer},
    traits::{Decodable, Encodable},
    vector_store::{self, VectorStore},
};

type Result<T> = std::result::Result<T, Error>;

#[derive(Clone)]
pub struct Index<D: Decodable + Clone, M: Metadata> {
    metadata: M,
    indexer: TermIndexer,
    vector_store: VectorStore<D>,
}

impl<D: Decodable + Clone, M: Metadata> Index<D, M> {
    /// Opens an Index from a tar.gz file and returns a new `Index`
    pub fn open<P: AsRef<Path>>(file: P) -> Result<Index<D, M>> {
        Self::from_archive(file)
    }

    /// Returns the vector store of the index
    #[inline]
    pub fn get_vector_store(&self) -> &VectorStore<D> {
        &self.vector_store
    }

    /// Returns the indexer of the index
    #[inline]
    pub fn get_indexer(&self) -> &TermIndexer {
        &self.indexer
    }

    /// Returns the indexes metadata
    #[inline]
    pub fn get_metadata(&self) -> &M {
        &self.metadata
    }

    /// Read an index-archive and build an `Index` out of it
    fn from_archive<P: AsRef<Path>>(file: P) -> Result<Index<D, M>> {
        let file = BufReader::new(File::open(file)?);
        let mut archive = tar::Archive::new(GzDecoder::new(file));

        // We have to read the archives files in the same order as archives.entries() yields
        // the elements
        let mut metadata: Option<M> = None;
        let mut term_indexer: Option<TermIndexer> = None;
        let mut dim_map: Option<DimVecMap> = None;
        let mut vector_store: Option<VectorStore<D>> = None;

        for entry in archive.entries()? {
            let entry = entry?;
            let name = entry
                .path()
                .ok()
                .and_then(|i| i.file_name().and_then(|i| i.to_str().map(|i| i.to_owned())))
                .ok_or(Error::InvalidIndex)?;

            let size = entry.size();

            match name.as_str() {
                metadata::FILE_NAME => metadata = Some(Self::parse_metadata(entry)?),
                term_store::FILE_NAME => term_indexer = Some(Self::parse_indexer(entry, size)?),
                dim_map::FILE_NAME => dim_map = Some(Self::parse_dim_map(entry, size)?),
                vector_store::FILE_NAME => {
                    vector_store = Some(Self::parse_vector_store(entry, size)?)
                }
                _ => (),
            }
        }

        let metadata = metadata.ok_or(Error::InvalidIndex)?;
        let dim_map = dim_map.ok_or(Error::InvalidIndex)?;
        let mut term_indexer = term_indexer.ok_or(Error::InvalidIndex)?;
        let mut vector_store = vector_store.ok_or(Error::InvalidIndex)?;

        vector_store.set_dim_map(dim_map);
        term_indexer.set_total_documents(metadata.get_document_count());

        Ok(Self {
            metadata,
            indexer: term_indexer,
            vector_store,
        })
    }

    fn parse_vector_store<R: Read>(mut entry: Entry<R>, size: u64) -> Result<VectorStore<D>> {
        let mut data = Vec::with_capacity(size as usize);
        entry.read_to_end(&mut data)?;
        VectorStore::new(Cursor::new(data))
    }

    fn parse_indexer<R: Read>(mut entry: Entry<R>, size: u64) -> Result<TermIndexer> {
        let mut data = Vec::with_capacity(size as usize);
        entry.read_to_end(&mut data)?;
        TermIndexer::new(Cursor::new(data)).map_err(|_| Error::InvalidIndex)
    }

    fn parse_dim_map<R: Read>(mut entry: Entry<R>, size: u64) -> Result<DimVecMap> {
        let mut data = Vec::with_capacity(size as usize);
        entry.read_to_end(&mut data)?;
        DimVecMap::load(Cursor::new(data)).map_err(|_| Error::InvalidIndex)
    }

    fn parse_metadata<T: Read>(entry: Entry<T>) -> Result<M> {
        M::load(entry).map_err(|_| Error::InvalidIndex)
    }
}

/// An crate internal helper for building new index files.
pub(crate) struct IndexBuilder {
    builder: tar::Builder<GzEncoder<File>>,
}

impl IndexBuilder {
    /// Create a new IndexBuilder
    pub fn new<S: AsRef<str>>(file_name: S) -> Result<IndexBuilder> {
        std::fs::remove_file(file_name.as_ref()).ok();
        let enc = GzEncoder::new(File::create(file_name.as_ref())?, Compression::best());
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

/// Public API for generating new index files
pub struct NewIndex {
    output: String,
    doc_count: usize,
}

impl NewIndex {
    /// Constructs a new `NewIndex`
    pub fn new<S: AsRef<str>>(output: S, doc_count: usize) -> NewIndex {
        Self {
            output: output.as_ref().to_string(),
            doc_count,
        }
    }

    /// Compile a new index for a given space vector model
    pub fn build<E, D, T, FV, M>(
        &self,
        unique_terms: T,
        metadata: M,
        mut calc_vecs: FV,
    ) -> Result<()>
    where
        E: Encodable + Clone,
        D: Decodable + Clone,
        T: Iterator<Item = IndexTerm>,
        FV: FnMut(&TermIndexer) -> Vec<DocumentVector<E>>,
        M: Metadata,
    {
        let mut index_builder = IndexBuilder::new(&self.output)?;

        let term_indexer = TermIndexer::build(&mut index_builder, self.doc_count, unique_terms)?;

        vector_store::build(&mut index_builder, calc_vecs(&term_indexer).into_iter())?;

        metadata.build(&mut index_builder)?;

        index_builder.finish()?;

        Ok(())
    }
}
