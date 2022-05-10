use std::{
    cmp::Ordering,
    hash::Hash,
    io::{Read, Write},
};

use byteorder::{ByteOrder, ReadBytesExt, WriteBytesExt};

use crate::{
    error::Error,
    traits::{Decodable, Encodable},
    vector::Vector,
};

/// A structure representing a document with its calculated document-vector
#[derive(Clone, Debug, Eq)]
pub struct DocumentVector<D> {
    pub document: D,
    vec: Vector,
}

impl<D> DocumentVector<D> {
    /// Create a new DocumentVector from a document and its vector
    #[inline(always)]
    pub fn new(document: D, vec: Vector) -> Self {
        Self { document, vec }
    }

    #[inline(always)]
    pub fn similarity<O>(&self, other: &DocumentVector<O>) -> f32 {
        self.vec.similarity(&other.vec)
    }

    #[inline(always)]
    pub fn vector(&self) -> &Vector {
        &self.vec
    }

    #[inline(always)]
    pub fn overlaps_with(&self, other: &Self) -> bool {
        self.vec.overlaps_with(&other.vec)
    }

    #[inline]
    pub(crate) fn set_vec(&mut self, new_vec: Vector) {
        self.vec = new_vec
    }
}

impl<D: Hash> Hash for DocumentVector<D> {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.document.hash(state);
    }
}

impl<D: PartialOrd> PartialOrd for DocumentVector<D> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.document.partial_cmp(&other.document)
    }
}

impl<D: PartialEq> PartialEq for DocumentVector<D> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.document.eq(&other.document)
    }
}

impl<D: Encodable> Encodable for DocumentVector<D> {
    fn encode<T: ByteOrder>(&self) -> Result<Vec<u8>, Error> {
        let doc_enc = self.document.encode::<T>()?;
        let svec = self.vec.sparse_vec();

        let mut encoded = Vec::with_capacity(6 + (svec.len() * 6) + doc_enc.len());

        // 0..4 vector length
        encoded.write_f32::<T>(self.vec.get_length())?;

        // 4..6 vector-dimension count
        encoded.write_u16::<T>(svec.len() as u16)?;

        // n*u48..
        for (dimension, value) in svec {
            encoded.write_u24::<T>(*dimension)?;
            encoded.write_f32::<T>(*value)?;
        }

        encoded.write_all(&doc_enc)?;

        Ok(encoded)
    }
}

impl<D: Decodable> Decodable for DocumentVector<D> {
    #[inline]
    fn decode<T: ByteOrder, R: Read>(mut data: R) -> Result<Self, Error> {
        // 0..4 vector length
        let vec_length = data.read_f32::<T>()?;

        // 4..6 vector-dimension count
        let vector_dim_count = data.read_u16::<T>()?;

        let dimensions: Vec<_> = (0..vector_dim_count)
            .map(|_| -> Result<_, std::io::Error> {
                let dim = data.read_u24::<T>()?;
                let val = data.read_f32::<T>()?;
                Ok((dim, val))
            })
            .collect::<Result<_, _>>()?;

        let doc = D::decode::<T, _>(data)?;

        let vec = Vector::new_raw(dimensions, vec_length);

        Ok(DocumentVector::new(doc, vec))
    }
}
