use crate::{
    document::DocumentVector,
    error::Error,
    inv_index::{DimToVecs, InvertedIndex, NewDimVecMap},
    traits::{Decodable, Encodable},
    Vector,
};
use byteorder::LittleEndian;
use indexed_file::mem_file::MemFile;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, marker::PhantomData};

/// A struct containing raw data of vectors and a map from a dimension to a set of those vectors.
#[derive(Debug, Serialize, Deserialize)]
pub struct VectorStore<D: Decodable> {
    store: MemFile,
    map: InvertedIndex,
    vec_type: PhantomData<D>,
}

impl<D: Decodable> VectorStore<D> {
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
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = DocumentVector<D>> + '_ {
        self.store
            .iter()
            .map(|i| Self::decode_vec(i).expect("Invalid index format"))
    }

    #[inline(always)]
    pub fn get_map(&self) -> &InvertedIndex {
        &self.map
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
    #[inline]
    pub fn get_in_dims_iter2<'a, I>(&'a self, dimensions: I) -> impl Iterator<Item = u32> + 'a
    where
        I: Iterator<Item = u32> + 'a,
    {
        dimensions
            .filter_map(move |i| self.get_map().get(i))
            .flatten()
            .unique()
    }

    /// Returns all unique vector references laying in `dimensions`
    pub fn get_in_dims_iter<I>(&self, dimensions: I) -> Vec<u32>
    where
        I: Iterator<Item = u32>,
    {
        let mut vec_refs: Vec<_> = dimensions
            .filter_map(|i| self.get_map().get(i))
            .flatten()
            .collect();

        vec_refs.sort_unstable();
        vec_refs.dedup();

        vec_refs
    }

    /// Returns all unique vector references laying in `dimensions`
    #[inline]
    pub fn get_in_dims(&self, dimensions: &[u32]) -> Vec<u32> {
        self.get_in_dims_iter(dimensions.iter().copied())
    }

    /// Returns all vectors in given dimensions efficiently via an iterator. May contain duplicates
    /// If vectors share multiple dimensions with the passed ones
    #[inline]
    pub fn get_all_iter2<'a, I>(
        &'a self,
        dimensions: I,
    ) -> impl Iterator<Item = DocumentVector<D>> + 'a
    where
        I: Iterator<Item = u32> + 'a,
    {
        let map = self.get_map();
        dimensions
            .filter_map(move |i| map.get(i))
            .flatten()
            .filter_map(move |i| self.load_vector(i as usize))
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

    /// Returns all for query_vec
    #[inline]
    pub fn get_for_vec<'a>(
        &'a self,
        q_vec: &'a Vector,
    ) -> impl Iterator<Item = DocumentVector<D>> + 'a {
        self.load_documents_iter(self.get_in_dims_iter2(q_vec.vec_indices()))
    }

    /// Load all documents by their ids
    #[inline]
    pub fn load_documents_iter<'a>(
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
    pub fn load_vector(&self, id: usize) -> Option<DocumentVector<D>> {
        Self::decode_vec(self.store.get(id)?)
    }

    #[inline]
    fn decode_vec(data: &[u8]) -> Option<DocumentVector<D>> {
        DocumentVector::<D>::decode::<LittleEndian, _>(data).ok()
    }

    #[inline]
    pub(crate) fn clone_full(&self) -> Self {
        Self {
            store: self.store.clone(),
            map: self.map.clone(),
            vec_type: self.vec_type,
        }
    }
}

/// Creates a new DocumentStore using a with `build` generated DocumentStore.
pub(crate) fn build<D: Encodable + Decodable>(
    vectors: Vec<DocumentVector<D>>,
) -> Result<VectorStore<D>, Error> {
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

    for (_, v) in dim_vec_map.iter_mut() {
        v.sort_unstable();
    }

    let map = NewDimVecMap::new(dim_vec_map).build();

    Ok(VectorStore {
        store: index,
        map,
        vec_type: PhantomData,
    })
}

impl<D: Decodable> Default for VectorStore<D> {
    #[inline]
    fn default() -> Self {
        Self {
            store: Default::default(),
            map: Default::default(),
            vec_type: Default::default(),
        }
    }
}
