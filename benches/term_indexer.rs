use criterion::{criterion_group, criterion_main, Criterion};
use vector_space_model::term_indexer::IndexItem;

fn index_item_decode(c: &mut Criterion) {
    let mut data = vec![];
    data.extend((20003 as u16).to_le_bytes());
    data.extend("deine Oma liegt ganz schön lange im Koma".as_bytes());

    c.bench_function("index item decode", |b| {
        b.iter(|| IndexItem::decode(&data));
    });
}

criterion_group!(benches, index_item_decode);
criterion_main!(benches);
