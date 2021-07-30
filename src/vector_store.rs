use std::{
    collections::HashMap,
    io::{BufReader, Cursor, Read, Seek, SeekFrom},
    marker::PhantomData,
    sync::Arc,
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
    map: DimVecMap,
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
            map: DimVecMap::default(),
            vec_type: PhantomData,
        })
    }

    /// Set the dim_vec_map to `map`
    pub fn set_dim_map(&mut self, map: DimVecMap) {
        self.map = map;
    }

    /// Returns all vectors in `dimension`
    pub fn get(&mut self, dimension: u32) -> Option<Vec<DocumentVector<D>>> {
        let vec_refs = self.map.get(dimension)?.clone();

        let documents = vec_refs
            .into_iter()
            .map(|i| self.get_vec_by_position(i).expect("invalid index format"))
            .collect::<Vec<_>>();

        Some(documents)
    }

    /// Read and decode a vector from `self.store` and returns it
    #[inline(always)]
    fn get_vec_by_position(&mut self, line: usize) -> Result<DocumentVector<D>, Error> {
        let mut buf = Vec::new();
        self.store.read_line_raw(line, &mut buf)?;
        DocumentVector::decode::<LittleEndian, _>(Cursor::new(buf))
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

    let index = Arc::new(FileIndex::new(file_index));
    let mut indexed_vectors = IndexedReader::new_custom(encoded_vectors, index);

    let mut out = Vec::new();
    indexed_vectors.write_to(&mut out)?;
    index_builder.write_vectors(&out)?;
    drop(out);

    Ok(())
}
