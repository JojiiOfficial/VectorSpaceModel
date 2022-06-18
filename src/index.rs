use crate::{
    build::weights::TermWeight,
    dim_map::{self, DimVecMap},
    error::Error,
    metadata::{self, Metadata},
    term_store::{self, TermIndexer},
    traits::Decodable,
    vector_store::{self, VectorStore},
    Vector,
};
use flate2::read::GzDecoder;
use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    io::{BufReader, Cursor, Read},
    path::Path,
};
use tar::Entry;

type Result<T> = std::result::Result<T, Error>;

#[derive(Serialize, Deserialize)]
pub struct Index<D: Decodable, M: Metadata> {
    metadata: M,
    indexer: TermIndexer,
    vector_store: VectorStore<D>,
}

impl<D: Decodable, M: Metadata> Index<D, M> {
    /// Opens an Index from a tar.gz file and returns a new `Index`
    pub fn open<P: AsRef<Path>>(file: P) -> Result<Index<D, M>> {
        Self::from_reader(BufReader::new(File::open(file)?))
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

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.indexer.is_empty() || self.vector_store.is_empty()
    }

    // TODO: add function to create new vectors (incl. weights)
    pub fn build_vector_weights<S: AsRef<str>>(&self, terms: &[(S, f32)]) -> Option<Vector> {
        let terms: Vec<_> = terms
            .iter()
            .filter_map(|(term, weight)| {
                let item_pos = self.indexer.get_term(term.as_ref())?;
                Some((item_pos as u32, *weight))
            })
            .collect();

        if terms.is_empty() {
            return None;
        }

        Some(Vector::create_new_raw(terms))
    }

    // TODO: add function to create new vectors (incl. weights)
    pub fn build_vector<S: AsRef<str>>(
        &self,
        terms: &[S],
        weight: Option<&dyn TermWeight>,
    ) -> Option<Vector> {
        let terms: Vec<_> = terms
            .iter()
            .filter_map(|i| {
                let item_pos = self.indexer.get_term(i.as_ref())?;
                let item = self.indexer.load_term(item_pos)?;
                Some((item_pos, item))
            })
            .map(|(pos, i)| {
                let mut res_weight = 1.0;
                if let Some(w) = weight.as_ref() {
                    res_weight =
                        w.weight(1.0, 1, i.doc_frequency() as usize, self.vector_store.len());
                }
                (pos as u32, res_weight)
            })
            .collect();

        if terms.is_empty() {
            return None;
        }

        Some(Vector::create_new_raw(terms))
    }

    pub fn is_stopword_cust(&self, term: &str, threshold: f32) -> Option<bool> {
        let tot_docs = self.get_indexer().len() as f32;
        let term = self.get_indexer().find_term(term)?;
        let ratio = term.doc_frequency() as f32 / tot_docs * 100.0;
        Some(ratio >= threshold)
    }

    #[inline]
    pub fn is_stopword(&self, term: &str) -> Option<bool> {
        self.is_stopword_cust(term, 35.0)
    }

    /// Read an index-archive and build an `Index` out of it
    pub fn from_reader<R: Read>(reader: R) -> Result<Index<D, M>> {
        let mut archive = tar::Archive::new(GzDecoder::new(reader));

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

    /// Sets the vector storte
    pub fn set_vec_store(&mut self, new: VectorStore<D>) {
        self.vector_store = new;
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

impl<D: Decodable, M: Metadata + Default> Default for Index<D, M> {
    fn default() -> Self {
        Self {
            metadata: Default::default(),
            indexer: Default::default(),
            vector_store: Default::default(),
        }
    }
}
