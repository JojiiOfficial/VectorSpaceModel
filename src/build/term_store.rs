use super::weights::TermWeight;
use crate::{DocumentVector, Vector};
use std::collections::HashMap;

/// Builds a new term storage
pub(crate) struct TermStoreBuilder {
    // Maps a term to its ID and frequency
    terms: HashMap<String, u32>,
    // Document frequencies for terms. Maps term-id to amount of docs containing this term
    doc_freq: HashMap<u32, u32>,
    // Term fequency. Frequencies for terms within a given document
    term_freq: HashMap<(u32, u32), u32>,

    // Map from ID to final position since terms have to be ordered
    order_map: HashMap<u32, u32>,
}

impl TermStoreBuilder {
    pub fn new() -> Self {
        Self {
            terms: HashMap::new(),
            doc_freq: HashMap::new(),
            term_freq: HashMap::new(),
            order_map: HashMap::new(),
        }
    }

    /// Returns a term from the Term store or `None` if term does not exist
    #[inline]
    pub fn get_term_id(&self, term: &str) -> Option<u32> {
        self.terms.get(term).copied()
    }

    /// Returns or creates a new term
    #[inline]
    pub fn get_or_add_term(&mut self, term: &str) -> u32 {
        if let Some(term) = self.get_term_id(term) {
            return term;
        }

        let next_id = self.terms.len() as u32;
        self.terms.insert(term.to_string(), next_id);
        next_id
    }

    #[inline]
    pub fn update_doc_freq<I: Iterator<Item = u32>>(&mut self, ids: I) {
        for id in ids {
            *self.doc_freq.entry(id).or_default() += 1;
        }
    }

    #[inline]
    pub fn update_term_freq(&mut self, term_id: u32, doc_id: u32) {
        *self.term_freq.entry((term_id, doc_id)).or_default() += 1;
    }

    #[inline]
    pub fn get_term_freq(&self, term_id: u32, doc_id: u32) -> Option<u32> {
        self.term_freq.get(&(term_id, doc_id)).copied()
    }

    /// Get a reference to the term store builder's frequencies.
    #[allow(unused)]
    pub fn term_frequencies(&self) -> &HashMap<(u32, u32), u32> {
        &self.term_freq
    }

    /// Get a reference to the term store builder's frequencies.
    #[allow(unused)]
    pub fn doc_frequencies(&self) -> &HashMap<u32, u32> {
        &self.doc_freq
    }

    /// Get a reference to the term store builder's terms.
    pub fn terms(&self) -> &HashMap<String, u32> {
        &self.terms
    }

    #[inline]
    pub fn get_sorted_term_pos(&self, term: u32) -> u32 {
        assert!(!self.order_map.is_empty());
        *self.order_map.get(&term).unwrap()
    }

    pub fn adjust_vecs<D, T: TermWeight>(
        &mut self,
        ves: &mut Vec<DocumentVector<D>>,
        weight: &Option<T>,
    ) {
        self.build_order_map();

        let doc_count = ves.len();

        for (doc_id, vec) in ves.iter_mut().enumerate() {
            let replaced = vec
                .vector()
                .sparse_vec()
                .iter()
                .copied()
                .map(|(old_dim, old_weight)| {
                    let new_dim = self.order_map.get(&old_dim).unwrap();

                    if let Some(w) = weight {
                        let tf = self.get_term_freq(old_dim, doc_id as u32).unwrap_or(0) as usize;
                        let df = self.doc_freq.get(&old_dim).copied().unwrap_or(0) as usize;
                        return (*new_dim, w.weight(tf, df, doc_count));
                    }

                    (*new_dim, old_weight)
                })
                .collect::<Vec<_>>();

            vec.set_vec(Vector::create_new_raw(replaced));
        }
    }

    /// Builds the a map of ID to ordered position of the term if the terms were sorted
    fn build_order_map(&mut self) {
        self.order_map.reserve(self.terms.len());

        let mut term_vec: Vec<_> = self.terms.iter().map(|(term, id)| (term, *id)).collect();
        term_vec.sort_by(|a, b| a.0.cmp(&b.0));

        for (pos, (_, id)) in term_vec.into_iter().enumerate() {
            self.order_map.insert(id, pos as u32);
        }
    }
}
