pub mod item;

use self::item::IndexTerm;
use crate::{
    build::{output::OutputBuilder, term_store::TermStoreBuilder},
    error::Error,
    traits::Encodable,
};
use byteorder::LittleEndian;
use indexed_file::mem_file::MemFile;
use std::{
    cmp::Ordering,
    io::{Read, Seek, Write},
    sync::Arc,
};

/// File name in the index tar
pub(crate) const FILE_NAME: &str = "term_indexer";

/// An in memory TermIndexer that allows efficient indexing of terms which is requried for document
/// vectors being calculated.
#[derive(Clone, Debug)]
pub struct TermIndexer {
    index: Arc<MemFile>,
    tot_documents: usize,
}

impl TermIndexer {
    /// Creates a new DocumentStore using a with `build` generated DocumentStore.
    /// `term_indexer.set_total_documents(..)` has to be called afterwards
    /// with the correct number of documents.
    pub fn new<R: Read + Seek + Unpin>(reader: R) -> Result<Self, Error> {
        let index: MemFile = bincode::deserialize_from(reader)?;
        Ok(TermIndexer {
            index: Arc::new(index),
            tot_documents: 0,
        })
    }

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
    #[inline]
    pub fn find_term(&mut self, term: &str) -> Option<IndexTerm> {
        let dimension = binary_search(&mut self.index, term)?;
        self.load_term(dimension)
    }

    /// Gets a term by its dimension
    #[inline]
    pub fn load_term(&mut self, dimension: usize) -> Option<IndexTerm> {
        let res = self.index.get(dimension)?;
        Some(IndexTerm::decode(res))
    }

    /// Sets the total amount of documents in the index. This is required for better indexing
    #[inline]
    pub(crate) fn set_total_documents(&mut self, tot_documents: usize) {
        self.tot_documents = tot_documents;
    }

    /// Returns an iterator over all Indexed terms
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = IndexTerm> + '_ {
        self.index.iter().map(IndexTerm::decode)
    }

    /// Builds a new TermIndexer from TermStoreBuilder and writes it into an OutputBuilder.
    /// This requires the terms to be sorted
    pub(crate) fn build_from_termstore<W: Write>(
        term_store: TermStoreBuilder,
        index_builder: &mut OutputBuilder<W>,
    ) -> Result<Self, Error> {
        let mut terms = term_store
            .terms()
            .iter()
            .map(|(term, id)| {
                let doc_freq = term_store.doc_frequencies().get(id).unwrap();
                let term = IndexTerm::new(term.to_string(), *doc_freq);
                let pos = term_store.get_sorted_term_pos(*id);
                (pos, term)
            })
            .collect::<Vec<_>>();

        terms.sort_by(|a, b| a.0.cmp(&b.0));

        let mut index = MemFile::with_capacity(terms.len());
        for term in terms {
            index.insert(&term.1.encode::<LittleEndian>()?);
        }

        index_builder.write_term_indexer(&bincode::serialize(&index)?)?;

        Ok(Self {
            index: Arc::new(index),
            tot_documents: 0,
        })
    }

    #[inline]
    pub fn get_term(&self, term: &str) -> Option<usize> {
        binary_search(&self.index, term)
    }

    #[cfg(feature = "genbktree")]
    pub fn gen_term_tree(&self) -> bktree::BkTree<String> {
        let mut terms = Vec::with_capacity(self.index.len());

        for i in self.index.iter() {
            terms.push(IndexTerm::decode(&i).text().to_string());
        }

        terms.into_iter().collect::<bktree::BkTree<_>>()
    }
}

pub fn binary_search(index: &MemFile, query: &str) -> Option<usize> {
    binary_search_raw_by(index, |i| IndexTerm::decode(i).text().cmp(query)).ok()
}

fn binary_search_raw_by<F>(index: &MemFile, f: F) -> Result<usize, usize>
where
    F: Fn(&[u8]) -> std::cmp::Ordering,
{
    let mut size = index.len();
    let mut left = 0;
    let mut right = size;

    while left < right {
        let mid = left + size / 2;

        let cmp = f(&index.get_unchecked(mid));

        if cmp == Ordering::Less {
            left = mid + 1;
        } else if cmp == Ordering::Greater {
            right = mid;
        } else {
            return Ok(mid);
        }

        size = right - left;
    }

    Err(left)
}

/*
impl document::Indexable for TermIndexer {

    #[inline]
    fn document_count(&self) -> usize {
        self.tot_documents
    }

    #[inline]
    fn word_occurrence(&self, term: &str) -> Option<usize> {
        let mut index = self.index.clone();

        let item_pos = match binary_search(&mut index, term) {
            Some(s) => s,
            None => return Some(1),
        };

        let mut buf = Vec::with_capacity(6);
        let item = index
            .read_line_raw(item_pos, &mut buf)
            .ok()
            .and_then(|_| IndexItem::decode(&buf).ok())
            .expect("Invalid term index file");

        Some(item.frequency() as usize)
    }
}
*/
