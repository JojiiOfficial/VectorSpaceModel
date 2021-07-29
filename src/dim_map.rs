use std::{
    collections::HashMap,
    io::{BufRead, BufReader, Read, Write},
};

use byteorder::{ByteOrder, LittleEndian};

use crate::{
    error::Error,
    index::IndexBuilder,
    traits::{Decodable, Encodable},
};

/// FileName of dim_maps
pub(crate) const FILE_NAME: &str = "dim_map";

pub(crate) type DimToVecs = HashMap<u32, Vec<usize>>;

/// A Dimension Vector map maps a dimension to all references of vectors which lay in the
/// dimension. This allows much faster searching.
#[derive(Debug, Clone, Default)]
pub struct DimVecMap {
    map: DimToVecs,
}

impl DimVecMap {
    /// Create a new map by a DimToVecs hashmap
    #[inline]
    pub fn new(map: DimToVecs) -> Self {
        Self { map }
    }

    /// Build a dimension vector map and write it to `index_builder`
    pub(crate) fn build(&self, index_builder: &mut IndexBuilder) -> Result<(), Error> {
        index_builder.write_dim_vec_map(&self.encode::<LittleEndian>()?)?;

        Ok(())
    }

    /// Reads and decodes a dimension-vector-map from a reader
    pub fn load<R: Read>(reader: R) -> Result<Self, Error> {
        Self::decode::<LittleEndian, _>(reader)
    }

    /// Returns a vec over all Vector IDs in dimension `dim`
    #[inline]
    pub fn get(&self, dim: u32) -> Option<&Vec<usize>> {
        self.map.get(&dim)
    }

    /// Returns true if there is at least one vector in dimension `dim`
    #[inline]
    pub fn has(&self, dim: u32) -> bool {
        self.map.contains_key(&dim)
    }
}

impl Encodable for DimVecMap {
    fn encode<T: byteorder::ByteOrder>(&self) -> Result<Vec<u8>, Error> {
        let mut out = Vec::new();

        for (dim, vecs) in self.map.iter() {
            out.write_all(&encode_map_item(*dim, vecs.clone()))?;
        }

        Ok(out)
    }
}

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

/// Encode a dim,vec pair
fn encode_map_item(dim: u32, vecs: Vec<usize>) -> Vec<u8> {
    let vecs_str = vecs
        .into_iter()
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(",");

    format!("{},{}\n", dim, vecs_str).as_bytes().to_vec()
}

/// Decode a dim,vec pair
fn decode_map_item(line: String) -> Result<(u32, Vec<usize>), Error> {
    let mut splitted = line.split(',');

    let dim: u32 = splitted
        .next()
        .unwrap()
        .parse()
        .map_err(|_| Error::Decode)?;

    let vecs = splitted
        .map(|i| i.parse::<usize>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| Error::Decode)?;

    Ok((dim, vecs))
}
