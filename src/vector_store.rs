use crate::{
    build::output::OutputBuilder,
    dim_map::{DimToVecs, DimVecMap, NewDimVecMap},
    document::DocumentVector,
    error::Error,
    traits::{Decodable, Encodable},
};
use byteorder::LittleEndian;
use indexed_file::mem_file::MemFile;
use std::{
    collections::HashMap,
    io::{Read, Seek, Write},
    marker::PhantomData,
    sync::Arc,
};

/// File name in the index tar
pub(crate) const FILE_NAME: &str = "vectors";

/// A struct containing raw data of vectors and a map from a dimension to a set of those vectors.
#[derive(Debug, Clone)]
pub struct VectorStore<D: Decodable> {
    store: Arc<MemFile>,
    map: Option<Arc<DimVecMap>>,
    vec_type: PhantomData<D>,
}

impl<D: Decodable> VectorStore<D> {
    /// Builds data for a new DocumentStore. This contains all generated vectors.
    ///
    /// The `map` gets initialized with `None` and hast to be set afterwards using
    /// `vec_store.set_dim_map(..)`. Otherwise VectorStore will panic
    pub fn new<R: Read + Seek + Unpin>(reader: R) -> Result<Self, Error> {
        let store: MemFile = bincode::deserialize_from(reader)?;
        Ok(Self {
            store: Arc::new(store),
            map: None,
            vec_type: PhantomData,
        })
    }

    /// Get the amount of vectors in the `VectorStore`
    #[inline]
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Returns `true` if vector store does not contain any vectors
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns an iterator over all Vectors in the vecstore
    pub fn iter(&self) -> impl Iterator<Item = DocumentVector<D>> + '_ {
        self.store
            .iter()
            .map(|i| Self::decode_vec(i).expect("Invalid index format"))
    }

    #[inline(always)]
    pub fn get_map(&self) -> &DimVecMap {
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
    pub fn get_in_dim(&self, dimension: u32) -> Option<Vec<DocumentVector<D>>> {
        let vec_refs = self.get_map().get(dimension)?;
        Some(self.load_documents(&vec_refs))
    }

    /// Returns all unique vector references laying in `dimensions`
    pub fn get_in_dims(&self, dimensions: &[u32]) -> Vec<u32> {
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
        &'a self,
        dimensions: &[u32],
    ) -> impl Iterator<Item = DocumentVector<D>> + 'a {
        let vec_refs = self.get_in_dims(dimensions);
        self.load_documents_iter(vec_refs.into_iter())
    }

    /// Load all documents by their ids
    #[inline]
    fn load_documents_iter<'a>(
        &'a self,
        vec_ids: impl Iterator<Item = u32> + 'a,
    ) -> impl Iterator<Item = DocumentVector<D>> + 'a {
        vec_ids.map(move |i| self.load_vector(i as usize).expect("invalid index format"))
    }

    /// Load all documents by their ids
    #[inline]
    fn load_documents(&self, vec_ids: &[u32]) -> Vec<DocumentVector<D>> {
        vec_ids
            .iter()
            .map(|i| self.load_vector(*i as usize).expect("invalid index format"))
            .collect()
    }

    /// Read and decode a vector from `self.store` and returns it
    #[inline]
    fn load_vector(&self, id: usize) -> Option<DocumentVector<D>> {
        Self::decode_vec(self.store.get(id)?)
    }

    #[inline]
    fn decode_vec(data: &[u8]) -> Option<DocumentVector<D>> {
        DocumentVector::<D>::decode::<LittleEndian, _>(data).ok()
    }

    /// Set the dim_vec_map to `map`
    #[inline(always)]
    pub(crate) fn set_dim_map(&mut self, map: DimVecMap) {
        self.map = Some(Arc::new(map));
    }
}

/// Creates a new DocumentStore using a with `build` generated DocumentStore.
pub(crate) fn build<D: Encodable, W: Write>(
    index_builder: &mut OutputBuilder<W>,
    vectors: Vec<DocumentVector<D>>,
) -> Result<(), Error> {
    //let mut encoded_vectors: Vec<u8> = Vec::new();
    let mut index = MemFile::with_capacity(vectors.len());

    // Map from dimensions to vectors in dimension
    let mut dim_vec_map: DimToVecs = HashMap::new();

    for vector in vectors {
        let vec_enc = vector.encode::<LittleEndian>()?;
        let vec_id = index.insert(&vec_enc);

        // Bulid map from dimension to all vectors in this dimension
        for dim in vector.vector().vec_indices() {
            dim_vec_map.entry(dim).or_default().push(vec_id as u32);
        }
    }

    NewDimVecMap::new(dim_vec_map).build(index_builder)?;

    let enc = bincode::serialize(&index)?;
    index_builder.write_vectors(&enc)?;
    Ok(())
}

/*
pub fn build_from_vecs<D: Encodable + Decodable>(
    vectors: Vec<DocumentVector<D>>,
) -> Result<VectorStore<D>, Error> {
    //let mut encoded_vectors: Vec<u8> = Vec::new();
    let mut index = MemFile::with_capacity(vectors.len());

    // Map from dimensions to vectors in dimension
    let mut dim_vec_map: DimToVecs = HashMap::new();

    for (pos, vector) in vectors.into_iter().enumerate() {
        // Bulid map from dimension to all vectors in this dimension
        for dim in vector.vector().vec_indices() {
            dim_vec_map.entry(dim).or_default().push(pos as u32);
        }

        let vec_enc = vector.encode::<LittleEndian>()?;
        index.insert(&vec_enc);
    }

    let encoded_dv = Cursor::new(NewDimVecMap::new(dim_vec_map).encode::<LittleEndian>()?);
    let dv_map = DimVecMap::load(encoded_dv)?;

    let vec_store = VectorStore {
        store: Arc::new(index),
        map: Some(Arc::new(dv_map)),
        vec_type: PhantomData::<D>,
    };

    Ok(vec_store)
}
*/
