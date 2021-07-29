#![allow(unused_imports)]
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::{distributions::Standard, Rng};
use vector_space_model::document_vector::WordVec;

fn get_word_vector(size: usize) -> WordVec {
    let rand_values: Vec<(u32, f32)> = rand::thread_rng()
        .sample_iter(Standard)
        .take(size)
        .collect();
    let mut vec = WordVec::new_raw(rand_values, 100f32);
    vec.update();
    vec
}

fn overlapping(c: &mut Criterion) {
    c.bench_function("overlapping", |b| {
        let vec1 = get_word_vector(1000);
        let vec2 = get_word_vector(600);
        b.iter(|| vec1.overlaps_with(&vec2))
    });
}

criterion_group!(benches, overlapping);
criterion_main!(benches);
