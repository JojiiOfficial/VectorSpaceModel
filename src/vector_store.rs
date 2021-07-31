use std::{
    cmp::min,
    collections::HashMap,
    future::Future,
    io::{BufReader, Cursor, Read, Seek, SeekFrom},
    marker::PhantomData,
    mem,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use byteorder::LittleEndian;
use indexed_file::{
    any::IndexedReader, index::Header as IndexHeader, index::Index as FileIndex, IndexableFile,
    ReadByLine,
};

use crate::{
    dim_map::{DimToVecs, DimVecMap},
    document_vector::DocumentVector,
    error::Error,
    index::IndexBuilder,
    traits::{Decodable, Encodable},
};

/// File name in the index tar
pub(crate) const FILE_NAME: &str = "vectors";

/// A struct containing raw data of vectors and a map from a dimension to a set of those vectors.
#[derive(Debug, Clone)]
pub struct VectorStore<D: Decodable + Clone> {
    store: IndexedReader<Vec<u8>>,
    map: Arc<DimVecMap>,
    vec_type: PhantomData<D>,
}

impl<D: Decodable + Clone> VectorStore<D> {
    /// Builds data for a new DocumentStore. This contains all generated vectors. The `map` gets
    /// initialized with `DimVecMap::default()` and hast to be set afterwards using
    /// `vec_store.set_dim_map(..)`
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
            map: Arc::new(DimVecMap::default()),
            vec_type: PhantomData,
        })
    }

    /// Set the dim_vec_map to `map`
    #[inline(always)]
    pub fn set_dim_map(&mut self, map: DimVecMap) {
        self.map = Arc::new(map);
    }

    /// Return the size of the given dimension. The size represents the amount of vectors which are
    /// laying in the dimension.
    #[inline]
    pub fn dimension_size(&self, dimension: u32) -> usize {
        self.map.get(dimension).map(|i| i.len()).unwrap_or(0)
    }

    /// Returns all vectors in `dimension`
    pub fn get(&mut self, dimension: u32) -> Option<Vec<DocumentVector<D>>> {
        let vec_refs = self.map.get(dimension)?.clone();
        Some(self.load_documents(&vec_refs))
    }

    /// Returns all vectors in given dimensions efficiently
    pub fn get_all(&mut self, dimensions: &[u32]) -> Option<Vec<DocumentVector<D>>> {
        let vec_refs = self.vectors_in_dimensions(dimensions);
        Some(self.load_documents(&vec_refs))
    }

    /// Returns all unique vector references laying in `dimensions`
    fn vectors_in_dimensions(&mut self, dimensions: &[u32]) -> Vec<usize> {
        let mut vec_refs: Vec<_> = dimensions
            .iter()
            .filter_map(|i| self.map.get(*i))
            .flatten()
            .copied()
            .collect();

        vec_refs.sort_unstable();
        vec_refs.dedup();

        vec_refs
    }

    /// Load all documents by their ids
    #[inline(always)]
    fn load_documents(&mut self, vec_ids: &[usize]) -> Vec<DocumentVector<D>> {
        vec_ids
            .iter()
            .map(|i| self.load_vector(*i).expect("invalid index format"))
            .collect::<Vec<_>>()
    }

    /// Read and decode a vector from `self.store` and returns it
    #[inline(always)]
    fn load_vector(&mut self, line: usize) -> Result<DocumentVector<D>, Error> {
        let mut buf = Vec::new();
        self.store.read_line_raw(line, &mut buf)?;
        DocumentVector::decode::<LittleEndian, _>(Cursor::new(buf))
    }
}

impl<D: Decodable + Clone + Unpin> VectorStore<D> {
    /// Returns all vectors in `dimension`
    pub async fn get_async(&mut self, dimension: u32) -> Result<Vec<DocumentVector<D>>, Error> {
        let vec_refs = self.map.get(dimension).cloned().unwrap_or_default();
        self.load_vecs_async(vec_refs).await
    }

    /// Returns all vectors in given dimensions efficiently
    pub async fn get_all_async(
        &mut self,
        dimensions: &[u32],
    ) -> Result<Vec<DocumentVector<D>>, Error> {
        let vec_refs = self.vectors_in_dimensions(dimensions);
        self.load_vecs_async(vec_refs).await
    }

    /// Loads all vectors by their references
    async fn load_vecs_async(&self, vec_refs: Vec<usize>) -> Result<Vec<DocumentVector<D>>, Error> {
        if vec_refs.is_empty() {
            return Ok(vec![]);
        }

        AsyncDocRetrieval::new(vec_refs, self.store.clone()).await
    }
}

/// Load document vectors asynchronously by chunking the load process into small pieces
struct AsyncDocRetrieval<D: Decodable + Clone + Unpin> {
    vec_refs: Vec<usize>,
    store: IndexedReader<Vec<u8>>,
    vec_type: PhantomData<D>,
    out: Vec<DocumentVector<D>>,
    last_pos: usize,
}

impl<D: Decodable + Clone + Unpin> AsyncDocRetrieval<D> {
    #[inline(always)]
    fn new(vec_refs: Vec<usize>, store: IndexedReader<Vec<u8>>) -> Self {
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

impl<D: Decodable + Clone + Unpin> Future for AsyncDocRetrieval<D> {
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
            if let Err(err) = self.store.read_line_raw(vec_id, &mut buf) {
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
pub(crate) fn build<V, D: Encodable + Clone>(
    index_builder: &mut IndexBuilder,
    vectors: V,
) -> Result<(), Error>
where
    V: Iterator<Item = DocumentVector<D>>,
{
    let mut encoded_vectors: Vec<u8> = Vec::new();

    // Map from dimensions to vectors in dimension
    let mut dim_vec_map: DimToVecs = HashMap::new();

    // Index position for each vector
    let mut file_index: Vec<u64> = Vec::new();

    for (pos, vector) in vectors.enumerate() {
        // Bulid map from dimension to all vectors in this dimension
        for dim in vector.vector().vec_indices() {
            dim_vec_map.entry(dim).or_default().push(pos);
        }

        file_index.push(encoded_vectors.len() as u64);

        // Encode document vector and push as new line to encoded_documents
        let encoded = vector.encode::<LittleEndian>()?;
        encoded_vectors.extend(&encoded);
        encoded_vectors.push(b'\n');
    }

    DimVecMap::new(dim_vec_map).build(index_builder)?;

    let index = Arc::new(FileIndex::new(file_index).zero_len());
    let mut indexed_vectors = IndexedReader::new_custom(encoded_vectors, index);

    let mut out = Vec::new();
    indexed_vectors.write_to(&mut out)?;
    index_builder.write_vectors(&out)?;
    drop(out);

    Ok(())
}
