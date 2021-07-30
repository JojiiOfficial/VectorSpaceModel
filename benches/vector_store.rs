use std::io::Read;

use byteorder::{ByteOrder, ReadBytesExt};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use vector_space_model::{
    document_vector::Indexable, error::Error, index::Index, traits::Decodable,
};

fn get_all(c: &mut Criterion) {
    let index = Index::<Document>::open("./out_index_de").unwrap();
    let mut vec_store = index.get_vector_store();

    c.bench_function("get all", |b| {
        let to_get = vec![12073, 26015, 54225, 56717, 123781, 125995, 126438, 126515];

        b.iter(|| {
            vec_store.get_all(black_box(&to_get));
        })
    });
}

fn get_mult(c: &mut Criterion) {
    let index = Index::<Document>::open("./out_index_de").unwrap();
    let mut vec_store = index.get_vector_store();

    c.bench_function("get multiple indiv", |b| {
        let to_get = vec![12073, 26015, 54225, 56717, 123781, 125995, 126438, 126515];

        b.iter(|| {
            for i in to_get.iter() {
                vec_store.get(black_box(*i));
            }
        })
    });
}

fn get(c: &mut Criterion) {
    let index = Index::<Document>::open("./out_index_de").unwrap();
    let term_indexer = index.get_indexer();
    let mut vec_store = index.get_vector_store();

    c.bench_function("get single", |b| {
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

criterion_group!(benches, get, get_mult, get_all);
criterion_main!(benches);
