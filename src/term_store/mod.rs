pub mod item;

use self::item::IndexTerm;
use crate::{build::term_store::TermStoreBuilder, error::Error, traits::Encodable};
use byteorder::LittleEndian;
use indexed_file::mem_file::MemFile;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// An in memory TermIndexer that allows efficient indexing of terms which is requried for document
/// vectors being calculated.
#[derive(Debug, Deserialize, Serialize, Default)]
pub struct TermIndexer {
    index: MemFile,
    tot_documents: usize,
    sort_index: Vec<u32>,
}

impl TermIndexer {
    /// Returns the total amount of terms in the term index
    #[inline]
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Returns `true` if there is no term in the term store
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Finds a term in the termindex
    /// Term_str -> TermObj
    #[inline]
    pub fn find_term(&self, term: &str) -> Option<IndexTerm> {
        self.get_term_raw(term).map(|i| i.1)
    }

    /// Term_str -> Dimension
    #[inline]
    pub fn get_term(&self, term: &str) -> Option<usize> {
        self.get_term_raw(term).map(|i| i.0)
    }

    /// Gets a term by its dimension
    /// Dimension -> Term
    #[inline]
    pub fn load_term(&self, dimension: usize) -> Option<IndexTerm> {
        let res = self.index.get(dimension)?;
        Some(IndexTerm::decode(res))
    }

    /// Returns an iterator over all Indexed terms
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = IndexTerm> + '_ {
        self.index.iter().map(IndexTerm::decode)
    }

    /// Returns `true` if the items are sorted and no custom sort index exists
    #[inline]
    pub fn is_sorted(&self) -> bool {
        self.sort_index.is_empty()
    }

    #[cfg(feature = "genbktree")]
    pub fn gen_term_tree(&self) -> bktree::BkTree<String> {
        let mut terms = Vec::with_capacity(self.index.len());

        for i in self.index.iter() {
            terms.push(IndexTerm::decode(&i).text().to_string());
        }

        terms.into_iter().collect::<bktree::BkTree<_>>()
    }

    /// Builds a new TermIndexer from TermStoreBuilder.
    pub(crate) fn build(ts_builder: TermStoreBuilder) -> Result<Self, Error> {
        let sort_index = vec![];

        let mut terms = ts_builder
            .terms()
            .iter()
            .map(|(term, id)| {
                let doc_freq = ts_builder.doc_frequencies().get(id).unwrap();
                let term = IndexTerm::new(term.to_string(), *doc_freq);
                let pos = ts_builder.get_sorted_term_pos(*id);
                (pos, term)
            })
            .collect::<Vec<_>>();

        terms.sort_by(|a, b| a.0.cmp(&b.0));

        let mut index = MemFile::with_capacity(terms.len());
        for term in terms {
            index.insert(&term.1.encode::<LittleEndian>()?);
        }

        Ok(Self {
            index,
            tot_documents: 0,
            sort_index,
        })
    }

    /// Builds a new cust sort mapping index
    pub fn build_cust_sort(&mut self) {
        if !self.is_sorted() {
            return;
        }

        self.sort_index = (0..self.len()).map(|i| i as u32).collect();
    }

    /// Inserts a new term into the indexer. This requires `build_cust_sort` being called first (once)
    pub fn insert_new(&mut self, term: IndexTerm) -> Option<u32> {
        if self.is_sorted() {
            return None;
        }

        let enc = term.encode::<LittleEndian>().expect("Invalid item");
        let id = self.index.insert(&enc) as u32;

        self.sort_index.push(id);

        self.update_sort_index();

        Some(id)
    }

    fn update_sort_index(&mut self) {
        let index = &self.index;

        self.sort_index.sort_by_cached_key(|i| {
            IndexTerm::decode(index.get(*i as usize).unwrap())
                .text()
                .to_string()
        });
    }

    fn get_term_raw(&self, term: &str) -> Option<(usize, IndexTerm)> {
        if self.is_sorted() {
            // No special sort-mapping specified so we can assume terms are sorted
            return gen_bin_search_by(&self.index, self.index.len(), |idx, pos| {
                let i = idx.get_unchecked(pos);
                let item = IndexTerm::decode(i);
                (item.text().cmp(term), item)
            })
            .ok();
        }

        // Custom sort-mapping available so we need to map those indices
        let mpr = gen_bin_search_by(&self.index, self.index.len(), |idx, pos| {
            let mp_pos = self.sort_index[pos];
            let i = idx.get_unchecked(mp_pos as usize);
            let item = IndexTerm::decode(i);
            (item.text().cmp(term), mp_pos as usize)
        })
        .ok()?;
        Some((mpr.1, self.load_term(mpr.1)?))
    }

    #[inline]
    pub(crate) fn clone_heavy(&self) -> Self {
        Self {
            index: self.index.clone(),
            tot_documents: self.tot_documents,
            sort_index: self.sort_index.clone(),
        }
    }
}

/// Generic bin search over any value
fn gen_bin_search_by<I, F, T>(over: I, mut size: usize, f: F) -> Result<(usize, T), usize>
where
    F: Fn(&I, usize) -> (Ordering, T),
{
    let mut left = 0;
    let mut right = size;

    while left < right {
        let mid = left + size / 2;

        let (cmp, item) = f(&over, mid);

        if cmp == Ordering::Less {
            left = mid + 1;
        } else if cmp == Ordering::Greater {
            right = mid;
        } else {
            return Ok((mid, item));
        }

        size = right - left;
    }

    Err(left)
}
