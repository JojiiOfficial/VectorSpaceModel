use criterion::{criterion_group, criterion_main, Criterion};
use vector_space_model::term_indexer::IndexItem;

fn index_item_from_string(c: &mut Criterion) {
    c.bench_function("index item from string", |b| {
        b.iter(|| IndexItem::from("deine Oma liegt ganz schön lange im Koma,200003"))
    });
}

criterion_group!(benches, index_item_from_string);
criterion_main!(benches);
