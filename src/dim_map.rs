use std::{
    collections::HashMap,
    convert::TryInto,
    io::{BufReader, Read, Seek, SeekFrom, Write},
    sync::Arc,
};

use byteorder::LittleEndian;

use crate::{error::Error, index::IndexBuilder, traits::Encodable};
use indexed_file::{
    any::IndexedReader, index::Header as IndexHeader, index::Index as FileIndex, Indexable,
    IndexableFile, ReadByLine,
};

/// FileName of dim_maps
pub(crate) const FILE_NAME: &str = "dim_map";

pub type DimToVecs = HashMap<u32, Vec<u32>>;

/// A Dimension Vector map maps a dimension to all references of vectors which lay in the
/// dimension. This allows much faster searching.
#[derive(Debug, Clone)]
pub struct DimVecMap {
    file: IndexedReader<Vec<u8>>,
}

impl DimVecMap {
    /// Reads and decodes a dimension-vector-map from a reader
    pub fn load<R: Read + Unpin + Seek>(reader: R) -> Result<Self, Error> {
        let mut buf_read = BufReader::new(reader);

        let header = IndexHeader::decode(&mut buf_read)?;
        let index = FileIndex::decode(&mut buf_read, &header)?;

        // Seek to beginning of actual data
        buf_read.seek(SeekFrom::Start(index.len_bytes() as u64))?;

        // Read all vector-data into Vec<u8> and create new IndexedReader
        let mut s: Vec<u8> = Vec::new();
        buf_read.read_to_end(&mut s)?;
        let data = IndexedReader::new_custom(s, Arc::new(index.zero_len()));

        println!("loaded: {}", data.total_lines());

        Ok(Self { file: data })
    }

    /// Returns a vec over all Vector IDs in dimension `dim`
    #[inline]
    pub fn get(&self, dim: u32) -> Option<Vec<u32>> {
        if dim as usize >= self.file.total_lines() {
            return None;
        }

        let mut file = self.file.clone();

        let mut buf = Vec::new();
        file.read_line_raw(dim as usize, &mut buf)
            .expect("Invalid dim map");

        Some(decode_map_item(&buf).expect("Invalid dim map").1)
    }

    /// Returns true if there is at least one vector in dimension `dim`
    #[inline]
    pub fn has(&self, dim: u32) -> bool {
        self.get(dim).is_some()
    }
}

#[derive(Debug, Clone)]
pub struct NewDimVecMap {
    pub(crate) map: DimToVecs,
}

impl NewDimVecMap {
    #[inline]
    pub fn new(map: DimToVecs) -> Self {
        Self { map }
    }

    /// Build a dimension vector map and write it to `index_builder`
    pub(crate) fn build(&self, index_builder: &mut IndexBuilder) -> Result<(), Error> {
        index_builder.write_dim_vec_map(&self.encode::<LittleEndian>()?)?;
        Ok(())
    }

    pub fn build_test(&self, out: &mut Vec<u8>) -> Result<(), Error> {
        out.extend(self.encode::<LittleEndian>()?);
        Ok(())
    }
}

impl Encodable for NewDimVecMap {
    fn encode<T: byteorder::ByteOrder>(&self) -> Result<Vec<u8>, Error> {
        let mut out = Vec::new();

        // Index position for each vector
        let mut file_index: Vec<u32> = Vec::new();

        let mut sorted_map = self
            .map
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect::<Vec<_>>();

        sorted_map.sort_by(|a, b| a.0.cmp(&b.0));

        println!("{:?}", &sorted_map[0..10]);
        println!("{:?}", sorted_map.len());

        for (_, vecs) in sorted_map {
            file_index.push(out.len() as u32);
            out.write_all(&encode_map_item(vecs))?;
        }

        let index = Arc::new(FileIndex::new(file_index).zero_len());
        let mut indexed_vectors = IndexedReader::new_custom(out.clone(), index);
        out.clear();
        indexed_vectors.write_to(&mut out).unwrap();

        Ok(out)
    }
}

/*
impl Decodable for DimVecMap {
    #[inline(always)]
    fn decode<T: ByteOrder, R: Read>(data: R) -> Result<Self, Error> {
        let lines = BufReader::new(data).lines();
        let mut map: DimToVecs = HashMap::new();

        for line in lines {
            let (dim, vecs) = decode_map_item(line?)?;
            map.insert(dim, vecs);
        }

        Ok(Self { map })
    }
}
*/

/// Encode a dim,vec pair
#[inline]
fn encode_map_item(vecs: Vec<u32>) -> Vec<u8> {
    vecs.into_iter()
        .map(|i| i.to_le_bytes())
        .flatten()
        .collect()
}

/// Decode a dim,vec pair
fn decode_map_item(data: &[u8]) -> Result<(u32, Vec<u32>), Error> {
    let mut vec = Vec::new();

    for chunk in data.chunks(4) {
        let u32_bits: [u8; 4] = chunk.try_into().expect("Invalid dim map");
        vec.push(u32::from_le_bytes(u32_bits));
    }

    Ok((0, vec))
}
