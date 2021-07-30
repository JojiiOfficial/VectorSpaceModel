use std::{
    cmp::Ordering,
    io::{Read, Write},
    iter::Peekable,
};

use byteorder::{ByteOrder, ReadBytesExt, WriteBytesExt};

use crate::{
    error::Error,
    traits::{Decodable, Encodable},
};

/// Representing a document which can be indexed
pub trait Document: Ord + Eq {
    fn get_terms(&self) -> Vec<String>;
}

/// Maps strings into a unique index assigned to the given string. This index will be used to build
/// the document vector
pub trait Indexable {
    fn index(&self, part: &str) -> Option<usize>;
    fn index_size(&self) -> usize;
    fn document_count(&self) -> usize;

    fn word_occurrence(&self, _term: &str) -> Option<usize> {
        None
    }
}

/// A structure representing a document with its calculated document-vector
#[derive(Clone, Debug, Eq)]
pub struct DocumentVector<D> {
    pub document: D,
    vec: Vector,
}

impl<D: Document> DocumentVector<D> {
    pub fn new<V: Indexable>(index: &V, document: D) -> Option<Self> {
        let vec = Vector::new(index, &document)?;
        Some(Self { document, vec })
    }

    /// Adds multiple terms to the word vector. This function should be preferred over `add_term`
    /// if you have more than one term you want to add to a vector
    pub fn add_terms<I: Indexable, T: AsRef<str>>(
        &mut self,
        index: &I,
        terms: &[T],
        skip_existing: bool,
        weight_mult: Option<f32>,
    ) {
        self.vec
            .add_terms::<D, _, T>(index, terms, skip_existing, weight_mult)
    }

    /// Adds a term to the word vector. Shouldn't be called from a loop. If you have to add
    /// multiple terms, use `add_terms`
    pub fn add_term<I: Indexable>(
        &mut self,
        index: &I,
        term: &str,
        skip_existing: bool,
        weight_mult: Option<f32>,
    ) {
        self.vec
            .add_term::<D, _>(index, term, skip_existing, weight_mult)
    }
}

impl<D> DocumentVector<D> {
    /// Create a new DocumentVector from a document and its vector
    #[inline]
    pub fn new_from_vector(document: D, vec: Vector) -> Self {
        Self { document, vec }
    }

    #[inline(always)]
    pub fn similarity<O: Document>(&self, other: &DocumentVector<O>) -> f32 {
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
}

impl<D: Eq> Ord for DocumentVector<D> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.vec.cmp(&other.vec)
    }
}

impl<D> PartialOrd for DocumentVector<D> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.vec.partial_cmp(&other.vec)
    }
}

impl<D> PartialEq for DocumentVector<D> {
    fn eq(&self, other: &Self) -> bool {
        self.vec.eq(&other.vec)
    }
}

impl<D: Encodable> Encodable for DocumentVector<D> {
    fn encode<T: ByteOrder>(&self) -> Result<Vec<u8>, Error> {
        let mut encoded = Vec::new();
        // 0..4 vector length
        encoded.write_f32::<T>(self.vec.get_length())?;

        // 4..6 vector-dimension count
        encoded.write_u16::<T>(self.vec.sparse_vec().len() as u16)?;

        // n*u64..
        for (dimension, value) in self.vec.sparse_vec() {
            encoded.write_u32::<T>(*dimension)?;
            encoded.write_f32::<T>(*value)?;
        }

        encoded.write_all(&self.document.encode::<T>()?)?;

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
                let dim = data.read_u32::<T>()?;
                let val = data.read_f32::<T>()?;
                Ok((dim, val))
            })
            .collect::<Result<_, _>>()?;

        let doc = D::decode::<T, _>(data)?;

        let vec = Vector::new_raw(dimensions, vec_length);

        Ok(DocumentVector::new_from_vector(doc, vec))
    }
}

/// A document vector
#[derive(Clone, Debug)]
pub struct Vector {
    /// Dimensions mapped to values
    inner: Vec<(u32, f32)>,
    /// Length of the vector
    length: f32,
}

impl Vector {
    /// Creates a new word vector from a Document using an index
    pub fn new<V: Indexable, D: Document>(index: &V, document: &D) -> Option<Self> {
        let inner: Vec<(u32, f32)> = document
            .get_terms()
            .into_iter()
            .filter_map(|term| {
                let weight = weight(&term, index, Some(document));
                let index = index.index(&term)? as u32;
                (weight != 0_f32).then(|| (index, weight))
            })
            .collect();

        let mut word_vec = Self {
            inner,
            length: 0f32,
        };

        if word_vec.is_empty() {
            return None;
        }

        word_vec.update();

        Some(word_vec)
    }

    /// Create a new WordVec from raw values
    #[inline]
    pub fn new_raw(sparse: Vec<(u32, f32)>, length: f32) -> Self {
        Self {
            inner: sparse,
            length,
        }
    }

    /// Calculates the similarity between two vectors
    #[inline]
    pub fn similarity(&self, other: &Vector) -> f32 {
        self.scalar(other) / (self.length * other.length)
    }

    /// Returns the reference to the inner vector
    #[inline]
    pub fn sparse_vec(&self) -> &Vec<(u32, f32)> {
        &self.inner
    }

    /// Returns true if the vector is zero
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns true if both vectors have at least one dimension in common
    pub fn overlaps_with(&self, other: &Vector) -> bool {
        // little speedup
        if self.first_indice() > other.last_indice() || self.last_indice() < other.first_indice() {
            return false;
        }

        LockStepIter::new(self.inner.iter().copied(), other.inner.iter().copied())
            .next()
            .is_some()
    }

    /// Returns the amount of dimensions the vector uses
    #[inline]
    pub fn dimen_count(&self) -> usize {
        self.inner.len()
    }

    /// Returns true if vector has a certain dimension
    #[inline]
    pub fn has_dim(&self, dim: u32) -> bool {
        self.vec_indices().any(|i| i == dim)
    }

    /// Returns an iterator over all dimensions of the vector
    #[inline]
    pub fn vec_indices(&self) -> impl Iterator<Item = u32> + '_ {
        self.inner.iter().map(|i| i.0)
    }

    /// Returns an iterator over all values of the vector
    #[inline]
    pub fn vec_values(&self) -> impl Iterator<Item = f32> + '_ {
        self.inner.iter().map(|i| i.1)
    }

    /// Adds a term to the word vector. Shouldn't be called from a loop. If you have to add
    /// multiple terms, use `add_terms`
    fn add_term<D: Document, I: Indexable>(
        &mut self,
        index: &I,
        term: &str,
        skip_existing: bool,
        weight_mult: Option<f32>,
    ) {
        self.add_single_term::<D, _>(index, term, skip_existing, weight_mult);
        self.update();
    }

    /// Adds multiple terms to the word vector. This function should be preferred over `add_term`
    /// if you have more than one term you want to add to a vector
    pub fn add_terms<D: Document, I: Indexable, T: AsRef<str>>(
        &mut self,
        index: &I,
        terms: &[T],
        skip_existing: bool,
        weight_mult: Option<f32>,
    ) {
        for term in terms.iter().map(|i| i.as_ref()) {
            self.add_single_term::<D, _>(index, term, skip_existing, weight_mult);
        }

        self.update();
    }

    /// Update the vector values
    pub fn update(&mut self) {
        // Calculate new vector length
        self.length = self.calc_len();

        // Sort the elements since order might be different
        self.sort();
    }

    /// Adds a term to the word vector. Afterwards `self.update()` should be called
    fn add_single_term<D: Document, I: Indexable>(
        &mut self,
        index: &I,
        term: &str,
        skip_existing: bool,
        weight_mult: Option<f32>,
    ) {
        let indexed = match index.index(&term) {
            Some(s) => s as u32,
            None => return,
        };

        let dim_used = self.has_dim(indexed);

        if dim_used && skip_existing {
            return;
        } else if dim_used {
            self.delete_dim(indexed);
        }

        let weight = weight::<D, _>(&term, index, None) * weight_mult.unwrap_or(1f32);
        self.inner.push((indexed, weight));
    }

    /// Get the length of the vector
    #[inline(always)]
    pub fn get_length(&self) -> f32 {
        self.length
    }

    /// Deletes a given dimension and its value from the vector
    fn delete_dim(&mut self, dim: u32) {
        self.inner.retain(|(curr_dim, _)| *curr_dim == dim);
    }

    #[inline]
    fn scalar(&self, other: &Vector) -> f32 {
        LockStepIter::new(self.inner.iter().copied(), other.inner.iter().copied())
            .map(|(_, a, b)| a * b)
            .sum()
    }

    /// Calculate the vector length
    fn calc_len(&self) -> f32 {
        self.inner
            .iter()
            .map(|(_, i)| i.powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Sort the Vec<> by the dimensions
    fn sort(&mut self) {
        self.inner.sort_by(|a, b| a.0.cmp(&b.0));
        self.inner.dedup_by(|a, b| a.0 == b.0);
    }

    #[inline(always)]
    fn last_indice(&self) -> u32 {
        self.inner.last().unwrap().0
    }

    #[inline(always)]
    fn first_indice(&self) -> u32 {
        self.inner.first().unwrap().0
    }
}

fn weight<D: Document, I: Indexable>(term: &str, index: &I, document: Option<&D>) -> f32 {
    let tf = match document {
        Some(d) => occurences(term, d) as f32,
        None => 1f32,
    };

    if let Some(idf) = idf(index, term) {
        // Adjust idf to not have that much of a difference between non stop-words
        let idf = if idf > 1f32 { 2.5 } else { 0.3f32 };

        (tf.log10() + 1f32) * idf
    } else {
        tf.log10() + 1f32
    }
}

fn idf<I: Indexable>(index: &I, term: &str) -> Option<f32> {
    Some((index.document_count() as f32 / index.word_occurrence(term)? as f32).log10())
}

fn occurences<D: Document>(term: &str, document: &D) -> usize {
    document.get_terms().iter().filter(|i| *i == term).count()
}

impl PartialEq for Vector {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl Eq for Vector {
    fn assert_receiver_is_total_eq(&self) {}
}

impl PartialOrd for Vector {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.inner.partial_cmp(&other.inner)
    }
}

impl Ord for Vector {
    fn cmp(&self, other: &Self) -> Ordering {
        self.inner.partial_cmp(&other.inner).unwrap()
    }
}

struct LockStepIter<A, B, K, V, W>
where
    A: Iterator<Item = (K, V)>,
    B: Iterator<Item = (K, W)>,
{
    a: Peekable<A>,
    b: Peekable<B>,
}

impl<A, B, K, V, W> LockStepIter<A, B, K, V, W>
where
    A: Iterator<Item = (K, V)>,
    B: Iterator<Item = (K, W)>,
{
    pub fn new(a: A, b: B) -> Self {
        Self {
            a: a.peekable(),
            b: b.peekable(),
        }
    }
}

impl<A, B, K, V, W> Iterator for LockStepIter<A, B, K, V, W>
where
    A: Iterator<Item = (K, V)>,
    B: Iterator<Item = (K, W)>,
    K: Ord,
{
    type Item = (K, V, W);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match (self.a.peek(), self.b.peek()) {
                (Some((dim_a, _)), Some((dim_b, _))) => match dim_a.cmp(&dim_b) {
                    Ordering::Less => {
                        self.a.next()?;
                    }
                    Ordering::Greater => {
                        self.b.next()?;
                    }
                    Ordering::Equal => {
                        let (dim, value_a) = self.a.next().unwrap();
                        let (_, value_b) = self.b.next().unwrap();
                        return Some((dim, value_a, value_b));
                    }
                },

                // At least one iterator finished
                _ => return None,
            }
        }
    }
}
