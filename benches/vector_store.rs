use std::{convert::TryFrom, io::Read};

use byteorder::{ByteOrder, ReadBytesExt, WriteBytesExt};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use vector_space_model2::{
    error::Error, index::Index, metadata::IndexVersion, traits::Decodable, DefaultMetadata,
};

fn get_index() -> Index<u32, DefaultMetadata> {
    Index::<u32, DefaultMetadata>::open("./out_index_de").unwrap()
}

fn get_all(c: &mut Criterion) {
    let index = get_index();
    let vec_store = index.get_vector_store();

    c.bench_function("get all", |b| {
        let to_get = vec![12073, 26015, 54225, 56717, 123781, 125995, 126438, 126515];
        b.iter(|| {
            vec_store
                .get_all_iter(black_box(&to_get))
                .collect::<Vec<_>>()
        })
    });
}

fn get_mult(c: &mut Criterion) {
    let index = get_index();
    let vec_store = index.get_vector_store();

    c.bench_function("get multiple indiv", |b| {
        let to_get = vec![12073, 26015, 54225, 56717, 123781, 125995, 126438, 126515];

        b.iter(|| {
            for i in to_get.iter() {
                vec_store.get_in_dim(black_box(*i));
            }
        })
    });
}

fn get(c: &mut Criterion) {
    let index = Index::<Vec<u32>, DefaultMetadata>::open("./out_index_de").unwrap();
    let term_indexer = index.get_indexer();
    let vec_store = index.get_vector_store();

    c.bench_function("get single", |b| {
        let t = term_indexer.get_term("あ").unwrap();
        b.iter(|| vec_store.get_in_dim(t as u32))
    });
}

/// Various metadata for the given Index
#[derive(Debug, Clone)]
pub struct Metadata {
    pub version: IndexVersion,
    pub document_count: usize,
    pub language: u8,
}

impl Metadata {
    /// Creates a new `Metadata` with the given values
    #[inline]
    pub fn new(version: IndexVersion, document_count: usize, language: u8) -> Self {
        Self {
            version,
            document_count,
            language,
        }
    }
}

impl vector_space_model2::metadata::Metadata for Metadata {
    #[inline]
    fn get_version(&self) -> IndexVersion {
        self.version
    }

    #[inline]
    fn get_document_count(&self) -> usize {
        self.document_count
    }

    fn set_document_count(&mut self, count: usize) {
        self.document_count = count;
    }
}

impl vector_space_model2::traits::Encodable for Metadata {
    fn encode<T: ByteOrder>(&self) -> Result<Vec<u8>, Error> {
        let mut out = vec![];

        out.write_u8(self.version as u8)?;
        out.write_u64::<T>(self.document_count as u64)?;
        out.write_i32::<T>(self.language.into())?;

        Ok(out)
    }
}

impl Decodable for Metadata {
    fn decode<T: ByteOrder, R: Read>(mut data: R) -> Result<Self, Error> {
        let version = IndexVersion::try_from(data.read_u8()?)?;
        let document_count = data.read_u64::<T>()? as usize;
        let _lang = data.read_i32::<T>()?;

        let language = 0;

        Ok(Self {
            version,
            document_count,
            language,
        })
    }
}

criterion_group!(benches, get, get_mult, get_all);
criterion_main!(benches);
