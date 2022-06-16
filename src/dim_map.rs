use crate::{build::output::OutputBuilder, error::Error, traits::Encodable};
use byteorder::LittleEndian;
use compressed_vec::{buffered::BufCVecRef, CVec};
use indexed_file::{
    any::CloneableIndexedReader, index::Header as IndexHeader, index::Index, IndexableFile,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    io::{BufReader, Read, Seek, SeekFrom, Write},
    sync::Arc,
};

/// FileName of dim_maps
pub(crate) const FILE_NAME: &str = "dim_map";

/// A Dimension Vector map maps a dimension to all references of vectors which lay in the
/// dimension. This allows much more efficient searching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimVecMap {
    /// Maps dimension indexes to positions in `data`
    index: Index,
    /// Contains the vector ids for each dimension
    data: CVec,
}

impl DimVecMap {
    /// Reads and decodes a dimension-vector-map from a reader
    pub fn load<R: Read + Unpin + Seek>(reader: R) -> Result<Self, Error> {
        let mut buf_read = BufReader::new(reader);

        let header = IndexHeader::decode(&mut buf_read)?;
        let index = Index::decode(&mut buf_read, &header)?;

        // Seek to beginning of actual data
        buf_read.seek(SeekFrom::Start(index.len_bytes() as u64))?;

        // Read all vector-data into Vec<u8> and create new IndexedReader
        let mut s: Vec<u8> = Vec::new();
        buf_read.read_to_end(&mut s)?;

        let data = CVec::from_bytes(&s)?;
        let index = index.zero_len();

        Ok(Self { index, data })
    }

    /// Returns a vec over all Vector IDs in dimension `dim`
    #[inline]
    pub fn get(&self, dim: u32) -> Option<Vec<u32>> {
        let arr_start = self.index.get2(dim as usize)? as usize;

        let mut buf_vec = BufCVecRef::new(&self.data);

        // Length of following vec containing the vector IDs
        let arr_len = *buf_vec.get_buffered(arr_start)? as usize;

        // Padded values have a length of 0
        if arr_len == 0 {
            return None;
        }

        let mut out = Vec::with_capacity(arr_len);

        // Take all elements of array. `arr_len` contains the count of all items in the array
        for pos in (arr_start + 1)..(arr_start + 1 + arr_len) {
            out.push(*buf_vec.get_buffered(pos as usize)?);
        }

        Some(out)
    }

    /// Returns true if there is at least one vector in dimension `dim`
    #[inline]
    pub fn has(&self, dim: u32) -> bool {
        self.get(dim).is_some()
    }

    pub fn decoded_map(&self) -> DimToVecs {
        let mut map = HashMap::<u32, Vec<u32>>::with_capacity(self.index.len());

        for dim in 0..self.index.len() {
            let get = self.get(dim as u32).unwrap();
            map.insert(dim as u32, get);
        }

        map
    }
}

pub type DimToVecs = HashMap<u32, Vec<u32>>;

#[derive(Debug, Clone)]
pub(crate) struct NewDimVecMap {
    pub(crate) map: DimToVecs,
}

impl NewDimVecMap {
    #[inline]
    pub(crate) fn new(map: DimToVecs) -> Self {
        Self { map }
    }

    /// Build a dimension vector map and write it to `index_builder`
    pub(crate) fn build<W: Write>(
        &self,
        index_builder: &mut OutputBuilder<W>,
    ) -> Result<(), Error> {
        index_builder.write_dim_vec_map(&self.encode::<LittleEndian>()?)?;
        Ok(())
    }
}

impl Encodable for NewDimVecMap {
    fn encode<T: byteorder::ByteOrder>(&self) -> Result<Vec<u8>, Error> {
        // Index position for each vector
        let mut file_index = Vec::new();

        let mut sorted_map = self
            .map
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect::<Vec<_>>();
        sorted_map.sort_by(|a, b| a.0.cmp(&b.0));

        let mut map_store = CVec::new();
        let mut last_dim: Option<u32> = None;

        for (dim, mut vecs) in sorted_map {
            if last_dim.is_none() {
                last_dim = Some(dim);
            }

            // Fill non mapped dimensions with 0s to make the CVS replace a HashMap
            let ld = last_dim.as_ref().unwrap();
            for _ in ld + 1..dim {
                file_index.push(map_store.len() as u32);
                map_store.push(0);
            }

            vecs.sort_unstable();
            file_index.push(map_store.len() as u32);

            map_store.push(vecs.len() as u32);
            map_store.extend(vecs);

            last_dim = Some(dim);
        }

        let index = Arc::new(Index::new(file_index).zero_len());

        let data = map_store.as_bytes();
        let mut out = Vec::with_capacity(data.len());

        let mut indexed_vectors = CloneableIndexedReader::new_custom(data, index);
        indexed_vectors.write_to(&mut out).unwrap();

        Ok(out)
    }
}
