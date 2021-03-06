//pub mod output;
pub mod term_store;
pub mod weights;

use crate::{
    term_store::TermIndexer,
    traits::{Decodable, Encodable},
    vector_store, DocumentVector, Error, Index, Vector,
};
use serde::Serialize;
use std::{collections::HashSet, io::Write};
use term_store::TermStoreBuilder;

use self::weights::TermWeight;

/// Helper for building new indexes
pub struct IndexBuilder<D> {
    vectors: Vec<DocumentVector<D>>,
    terms: TermStoreBuilder,
    term_weight: Option<Box<dyn TermWeight>>,
    output_filter:
        Option<Box<dyn Fn(DocumentVector<D>, &TermIndexer) -> Option<DocumentVector<D>> + 'static>>,
}

impl<D> IndexBuilder<D> {
    /// Create a new Indexer
    #[inline]
    pub fn new() -> Self {
        Self {
            vectors: vec![],
            terms: TermStoreBuilder::new(),
            term_weight: None,
            output_filter: None,
        }
    }

    pub fn with_weight<U: TermWeight + 'static>(mut self, weight: U) -> Self {
        self.term_weight = Some(Box::new(weight));
        self
    }

    pub fn with_filter<F>(&mut self, filter: F)
    where
        F: Fn(DocumentVector<D>, &TermIndexer) -> Option<DocumentVector<D>> + 'static,
    {
        self.output_filter = Some(Box::new(filter));
    }

    /// Creates a new doc-vec and inserts it into the indexer. Returns the ID of the new vec
    pub fn insert_new_vec<S: AsRef<str>>(&mut self, doc: D, terms: &[S]) -> usize {
        let doc_id = self.vectors.len();

        let dimensions = terms
            .iter()
            .map(|term| {
                let term_id = self.terms.get_or_add_term(term.as_ref());
                self.terms.update_term_freq(term_id, doc_id as u32);
                term_id
            })
            .collect::<HashSet<_>>();

        self.terms.update_doc_freq(dimensions.iter().copied());

        // Initialize with same weights for all of them
        // We'll adjust the weigts later in `finish()`
        let vec = Vector::create_new_raw(dimensions.into_iter().map(|i| (i, 1.0)).collect());
        self.vectors.push(DocumentVector::new(doc, vec));
        doc_id
    }

    /// Creates a new doc-vec and inserts it into the indexer. Returns the ID of the new vec
    /// Requires `terms` to be free of duplicates
    pub fn insert_new_weighted_vec<S: AsRef<str>>(&mut self, doc: D, terms: &[(S, f32)]) -> usize {
        let doc_id = self.vectors.len();

        let dimensions = terms
            .iter()
            .map(|(term, weight)| {
                let term_id = self.terms.get_or_add_term(term.as_ref());
                self.terms.update_term_freq(term_id, doc_id as u32);
                (term_id, *weight)
            })
            .collect::<Vec<_>>();

        self.terms.update_doc_freq(dimensions.iter().map(|i| i.0));

        let vec = Vector::create_new_raw(dimensions);
        self.vectors.push(DocumentVector::new(doc, vec));
        doc_id
    }

    pub fn insert_custom_vec<F>(&mut self, func: F) -> usize
    where
        F: Fn(&mut TermStoreBuilder) -> DocumentVector<D>,
    {
        let doc_id = self.vectors.len();
        self.vectors.push(func(&mut self.terms));
        doc_id
    }

    /// Returns the current amount of vectors in the builder
    #[inline]
    pub fn vec_count(&self) -> usize {
        self.vectors.len()
    }

    #[inline]
    pub fn vecs_mut(&mut self) -> &mut Vec<DocumentVector<D>> {
        &mut self.vectors
    }

    /// Get a reference to the index builder's vectors.
    #[inline]
    pub fn vectors(&self) -> &[DocumentVector<D>] {
        self.vectors.as_ref()
    }
}

impl<D: Decodable + Encodable> IndexBuilder<D> {
    pub fn build_to_writer<W: Write, M: Serialize>(
        self,
        out: W,
        metadata: M,
    ) -> Result<Index<D, M>, Error> {
        let index = self.build(metadata)?;
        bincode::serialize_into(out, &index)?;
        Ok(index)
    }

    pub fn build<M>(mut self, metadata: M) -> Result<Index<D, M>, Error> {
        self.terms.adjust_vecs(&mut self.vectors, &self.term_weight);

        let indexer = TermIndexer::build(self.terms)?;

        if let Some(filter) = self.output_filter {
            self.vectors = self
                .vectors
                .into_iter()
                .filter_map(|vec| filter(vec, &indexer))
                .collect();
        }

        let vstore = vector_store::build(self.vectors)?;

        Ok(Index {
            metadata,
            indexer,
            vector_store: vstore,
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_indexer() {
        let insert_documents: &[&[&str]] = &[
            &["to", "drive", "a", "car"],
            &["to", "have", "a", "call"],
            &["to", "make", "a", "stand", "a"],
        ];

        let mut indexer = IndexBuilder::new();

        for (pos, terms) in insert_documents.iter().enumerate() {
            indexer.insert_new_vec(pos, *terms);
        }

        let term_store_builder = &indexer.terms;

        assert!(indexer.vectors.len() == insert_documents.len());
        let term_a = term_store_builder.get_term_id("a");
        assert!(term_a.is_some());
        assert_eq!(
            term_store_builder.doc_frequencies().get(&term_a.unwrap()),
            Some(&3)
        );
    }
}

impl<D> Default for IndexBuilder<D> {
    fn default() -> Self {
        Self::new()
    }
}
