pub mod output;
pub mod term_store;
pub mod weights;

use crate::{
    build::output::OutputBuilder,
    metadata::{Metadata, MetadataBuild},
    traits::{Decodable, Encodable},
    DocumentVector, Error, Vector,
};
use std::{collections::HashSet, io::Write};
use term_store::TermStoreBuilder;

use self::weights::TermWeight;

/// Helper for building new indexes
pub struct IndexBuilder<D, T> {
    vectors: Vec<DocumentVector<D>>,
    terms: TermStoreBuilder,
    term_weight: Option<T>,
}

impl<D, T> IndexBuilder<D, T> {
    /// Create a new Indexer
    #[inline]
    pub fn new() -> Self {
        Self {
            vectors: vec![],
            terms: TermStoreBuilder::new(),
            term_weight: None,
        }
    }

    pub fn with_weight(mut self, weight: T) -> Self {
        self.term_weight = Some(weight);
        self
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
}

impl<D: Decodable + Encodable, T: TermWeight> IndexBuilder<D, T> {
    pub fn build<M: Metadata, W: Write>(mut self, output: W, mut metadata: M) -> Result<(), Error> {
        self.terms.adjust_vecs(&mut self.vectors, &self.term_weight);

        let mut out_builder = OutputBuilder::new(output)?;

        crate::term_store::TermIndexer::build_from_termstore(self.terms, &mut out_builder)?;

        metadata.build(&mut out_builder, self.vectors.len())?;

        crate::vector_store::build(&mut out_builder, self.vectors)?;

        out_builder.finish()?;

        Ok(())
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

        let mut indexer = IndexBuilder::<_, ()>::new();

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

impl<D, T> Default for IndexBuilder<D, T> {
    fn default() -> Self {
        Self::new()
    }
}
