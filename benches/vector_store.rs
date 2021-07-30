use std::io::Read;

use byteorder::{ByteOrder, ReadBytesExt};
use criterion::{criterion_group, criterion_main, Criterion};
use vector_space_model::{
    document_vector::Indexable, error::Error, index::Index, traits::Decodable,
};

fn get(c: &mut Criterion) {
    let index = Index::<Document>::open("./out_index_de").unwrap();
    let term_indexer = index.get_indexer();
    let mut vec_store = index.get_vector_store();

    c.bench_function("vector store get", |b| {
        let t = term_indexer.index("einer").unwrap();
        b.iter(|| vec_store.get(t as u32))
    });
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Document {
    pub seq_ids: Vec<usize>,
}

impl Decodable for Document {
    #[inline(always)]
    fn decode<T: ByteOrder, R: Read>(mut data: R) -> Result<Self, Error> {
        let seq_id_count = data.read_u16::<T>()?;

        let seq_ids = (0..seq_id_count)
            .map(|_| data.read_u64::<T>().map(|i| i as usize))
            .collect::<Result<_, _>>()?;

        Ok(Self { seq_ids })
    }
}

criterion_group!(benches, get);
criterion_main!(benches);
