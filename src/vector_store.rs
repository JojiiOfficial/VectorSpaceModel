use std::{
    cmp::min,
    collections::HashMap,
    future::Future,
    io::{BufReader, Cursor, Read, Seek, SeekFrom, Write},
    marker::PhantomData,
    mem,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use byteorder::LittleEndian;
use indexed_file::{
    any::IndexedReader, index::Header as IndexHeader, index::Index as FileIndex, Indexable,
    IndexableFile, ReadByLine,
};

use crate::{
    build::output::OutputBuilder,
    dim_map::{DimToVecs, DimVecMap, NewDimVecMap},
    document::DocumentVector,
    error::Error,
    traits::{Decodable, Encodable},
};

/// File name in the index tar
pub(crate) const FILE_NAME: &str = "vectors";

/// A struct containing raw data of vectors and a map from a dimension to a set of those vectors.
#[derive(Debug, Clone)]
pub struct VectorStore<D: Decodable> {
    store: IndexedReader<Vec<u8>>,
    map: Option<Arc<DimVecMap>>,
    vec_type: PhantomData<D>,
}

impl<D: Decodable> VectorStore<D> {
    /// Builds data for a new DocumentStore. This contains all generated vectors.
    ///
    /// The `map` gets initialized with `None` and hast to be set afterwards using
    /// `vec_store.set_dim_map(..)`. Otherwise VectorStore will panic
    pub fn new<R: Read + Seek + Unpin>(reader: R) -> Result<Self, Error> {
        let mut buf_read = BufReader::new(reader);

        let header = IndexHeader::decode(&mut buf_read)?;
        let index = FileIndex::decode(&mut buf_read, &header)?;

        // Seek to beginning of actual data
        buf_read.seek(SeekFrom::Start(index.len_bytes() as u64))?;

        // Read all vector-data into Vec<u8> and create new IndexedReader
        let mut s: Vec<u8> = Vec::new();
        buf_read.read_to_end(&mut s)?;
        let data = IndexedReader::new_custom(s, Arc::new(index.zero_len()));

        Ok(Self {
            store: data,
            map: None,
            vec_type: PhantomData,
        })
    }
    /// Get the amount of vectors in the `VectorStore`
    #[inline]
    pub fn len(&self) -> usize {
        self.store.total_lines()
    }

    /// Returns `true` if vector store does not contain any vectors
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns an iterator over all Vectors in the vecstore
    pub fn iter(&self) -> impl Iterator<Item = DocumentVector<D>> + '_ {
        let mut pos = 0;
        let mut reader = self.store.clone();
        std::iter::from_fn(move || {
            if pos >= self.len() {
                return None;
            }
            let vec = load_vector_reader(&mut reader, pos).ok()?;
            pos += 1;
            Some(vec)
        })
    }

    #[inline(always)]
    pub fn get_map(&self) -> &Arc<DimVecMap> {
        self.map.as_ref().expect("set_dim_map was not called")
    }

    /// Return the size of the given dimension. The size represents the amount of vectors which are
    /// laying in the dimension.
    #[inline]
    pub fn dimension_size(&self, dimension: u32) -> usize {
        self.get_map().get(dimension).map(|i| i.len()).unwrap_or(0)
    }

    /// Returns all vectors in `dimension`
    #[inline]
    pub fn get_in_dim(&mut self, dimension: u32) -> Option<Vec<DocumentVector<D>>> {
        let vec_refs = self.get_map().get(dimension)?;
        Some(self.load_documents(&vec_refs))
    }

    /// Returns all unique vector references laying in `dimensions`
    pub fn get_in_dims(&mut self, dimensions: &[u32]) -> Vec<u32> {
        let mut vec_refs: Vec<_> = dimensions
            .iter()
            .filter_map(|i| self.get_map().get(*i))
            .flatten()
            .collect();

        vec_refs.sort_unstable();
        vec_refs.dedup();

        vec_refs
    }

    /// Returns all vectors in given dimensions efficiently via an iterator
    #[inline]
    pub fn get_all_iter<'a>(
        &'a mut self,
        dimensions: &[u32],
    ) -> impl Iterator<Item = DocumentVector<D>> + 'a {
        let vec_refs = self.get_in_dims(dimensions);
        self.load_documents_iter(vec_refs.into_iter())
    }

    /// Load all documents by their ids
    #[inline]
    fn load_documents_iter<'a>(
        &'a mut self,
        vec_ids: impl Iterator<Item = u32> + 'a,
    ) -> impl Iterator<Item = DocumentVector<D>> + 'a {
        vec_ids.map(move |i| self.load_vector(i as usize).expect("invalid index format"))
    }

    /// Load all documents by their ids
    #[inline(always)]
    fn load_documents(&mut self, vec_ids: &[u32]) -> Vec<DocumentVector<D>> {
        vec_ids
            .iter()
            .map(|i| self.load_vector(*i as usize).expect("invalid index format"))
            .collect::<Vec<_>>()
    }

    /// Read and decode a vector from `self.store` and returns it
    #[inline(always)]
    fn load_vector(&mut self, line: usize) -> Result<DocumentVector<D>, Error> {
        load_vector_reader(&mut self.store, line)
    }

    /// Set the dim_vec_map to `map`
    #[inline(always)]
    pub(crate) fn set_dim_map(&mut self, map: DimVecMap) {
        self.map = Some(Arc::new(map));
    }
}

/// Read and decode a vector from `self.store` and returns it
#[inline(always)]
fn load_vector_reader<D: Decodable>(
    reader: &mut IndexedReader<Vec<u8>>,
    line: usize,
) -> Result<DocumentVector<D>, Error> {
    let mut buf = Vec::new();
    reader.read_line_raw(line, &mut buf)?;
    DocumentVector::decode::<LittleEndian, _>(Cursor::new(buf))
}

impl<D: Decodable + Unpin> VectorStore<D> {
    /// Returns all vectors in `dimension`
    pub async fn get_async(&mut self, dimension: u32) -> Result<Vec<DocumentVector<D>>, Error> {
        let vec_refs = self.get_map().get(dimension).unwrap_or_default();
        self.load_vecs_async(vec_refs).await
    }

    /// Returns all vectors in given dimensions efficiently
    pub async fn get_all_async(
        &mut self,
        dimensions: &[u32],
    ) -> Result<Vec<DocumentVector<D>>, Error> {
        let vec_refs = self.get_in_dims(dimensions);
        self.load_vecs_async(vec_refs).await
    }

    /// Loads all vectors by their references
    async fn load_vecs_async(&self, vec_refs: Vec<u32>) -> Result<Vec<DocumentVector<D>>, Error> {
        if vec_refs.is_empty() {
            return Ok(vec![]);
        }

        AsyncDocRetrieval::new(vec_refs, self.store.clone()).await
    }
}

/// Load document vectors asynchronously by chunking the load process into small pieces
struct AsyncDocRetrieval<D: Decodable + Unpin> {
    vec_refs: Vec<u32>,
    store: IndexedReader<Vec<u8>>,
    vec_type: PhantomData<D>,
    out: Vec<DocumentVector<D>>,
    last_pos: usize,
}

impl<D: Decodable + Unpin> AsyncDocRetrieval<D> {
    #[inline(always)]
    fn new(vec_refs: Vec<u32>, store: IndexedReader<Vec<u8>>) -> Self {
        Self {
            // output
            out: Vec::with_capacity(vec_refs.len()),
            last_pos: 0,

            // input
            store,
            vec_refs,
            vec_type: PhantomData,
        }
    }
}

impl<D: Decodable + Unpin> Future for AsyncDocRetrieval<D> {
    type Output = Result<Vec<DocumentVector<D>>, Error>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let start = self.last_pos;
        let mut buf = Vec::with_capacity(40);

        if start >= self.vec_refs.len() {
            return Poll::Ready(Ok(mem::take(&mut self.out)));
        }

        // ensure we're not going further than `self.vec_refs.len()`
        let end = min(self.vec_refs.len(), self.last_pos + 200);

        for vec_pos in start..end {
            let vec_id = self.vec_refs[vec_pos];
            if let Err(err) = self.store.read_line_raw(vec_id as usize, &mut buf) {
                return Poll::Ready(Err(err.into()));
            }

            let dv = match DocumentVector::<D>::decode::<LittleEndian, _>(Cursor::new(&buf)) {
                Ok(v) => v,
                Err(err) => return Poll::Ready(Err(err)),
            };

            self.out.push(dv);
            buf.clear();
        }

        self.last_pos = end;

        cx.waker().wake_by_ref();

        Poll::Pending
    }
}

/// Creates a new DocumentStore using a with `build` generated DocumentStore.
pub(crate) fn build<D: Encodable, W: Write>(
    index_builder: &mut OutputBuilder<W>,
    vectors: Vec<DocumentVector<D>>,
) -> Result<(), Error> {
    let mut encoded_vectors: Vec<u8> = Vec::new();

    // Map from dimensions to vectors in dimension
    let mut dim_vec_map: DimToVecs = HashMap::new();

    // Index position for each vector
    let mut file_index: Vec<u32> = Vec::new();

    for (pos, vector) in vectors.into_iter().enumerate() {
        // Bulid map from dimension to all vectors in this dimension
        for dim in vector.vector().vec_indices() {
            dim_vec_map.entry(dim).or_default().push(pos as u32);
        }

        file_index.push(encoded_vectors.len() as u32);

        // Encode document vector and push as new line to encoded_documents
        encoded_vectors.extend(&vector.encode::<LittleEndian>()?);
    }

    NewDimVecMap::new(dim_vec_map).build(index_builder)?;

    let index = Arc::new(FileIndex::new(file_index).zero_len());
    let mut indexed_vectors = IndexedReader::new_custom(encoded_vectors, index);

    let mut out = Vec::new();
    indexed_vectors.write_to(&mut out)?;
    index_builder.write_vectors(&out)?;
    Ok(())
}
