use std::time::Instant;

use criterion::{criterion_group, criterion_main, Criterion};
use vector_space_model::document_vector::Vector;

fn get_word_vector(size: usize, seed: usize) -> Vector {
    let values = (seed / size % 10..)
        .step_by(seed / size + 1)
        .map(|i| (i as u32, i as f32))
        .take(size)
        .collect();

    let mut vec = Vector::new_raw(values, 100f32);
    vec.update();
    vec
}

fn similarity(c: &mut Criterion) {
    c.bench_function("overlapping", |b| {
        let vec1 = get_word_vector(1071, 185692);
        let vec2 = get_word_vector(603, 185692);
        b.iter(|| vec1.overlaps_with(&vec2))
    });
}

fn overlapping(c: &mut Criterion) {
    c.bench_function("similarity", |b| {
        b.iter_custom(|iter| {
            let vec1 = get_word_vector(1071, 185692);
            let vec2 = get_word_vector(603, 185692);

            let start = Instant::now();
            for _ in 0..iter {
                vec1.similarity(&vec2);
            }
            start.elapsed()
        })
    });
}

criterion_group!(benches, overlapping, similarity);
criterion_main!(benches);
