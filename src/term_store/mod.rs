pub mod item;

use self::item::IndexTerm;
use crate::{
    build::{output::OutputBuilder, term_store::TermStoreBuilder},
    error::Error,
    traits::Encodable,
};
use byteorder::LittleEndian;
use indexed_file::{
    any::IndexedReader, index::Header as IndexHeader, index::Index as FileIndex, Indexable,
    IndexableFile, ReadByLine,
};
use std::{
    io::{BufReader, Read, Seek, SeekFrom, Write},
    sync::Arc,
};

/// File name in the index tar
pub(crate) const FILE_NAME: &str = "term_indexer";

/// An in memory TermIndexer that allows efficient indexing of terms which is requried for document
/// vectors being calculated.
#[derive(Clone)]
pub struct TermIndexer {
    pub index: IndexedReader<Vec<u8>>,
    tot_documents: usize,
}

impl TermIndexer {
    /// Creates a new DocumentStore using a with `build` generated DocumentStore.
    /// `term_indexer.set_total_documents(..)` has to be called afterwards
    /// with the correct number of documents.
    pub fn new<R: Read + Seek + Unpin>(reader: R) -> Result<Self, indexed_file::error::Error> {
        let mut reader = BufReader::new(reader);
        let header = IndexHeader::decode(&mut reader)?;
        let index = FileIndex::decode(&mut reader, &header)?;
        reader.seek(SeekFrom::Start(index.len_bytes() as u64))?;

        let mut s = Vec::new();
        reader.read_to_end(&mut s)?;
        let mem_index = IndexedReader::new_custom(s, Arc::new(index.zero_len()));

        Ok(TermIndexer {
            index: mem_index,
            tot_documents: 0,
        })
    }

    /// Returns the total amount of terms in the term index
    #[inline]
    pub fn len(&self) -> usize {
        self.index.total_lines()
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
        let mut buf = Vec::with_capacity(10);
        self.index.read_line_raw(dimension, &mut buf).ok()?;
        Some(IndexTerm::decode(&buf))
    }

    /// Sets the total amount of documents in the index. This is required for better indexing
    #[inline]
    pub(crate) fn set_total_documents(&mut self, tot_documents: usize) {
        self.tot_documents = tot_documents;
    }

    /// Returns an iterator over all Indexed terms
    pub fn iter(&self) -> impl Iterator<Item = IndexTerm> + '_ {
        let mut reader = self.index.clone();
        (0..reader.total_lines()).map(move |i| {
            let mut buf = Vec::with_capacity(10);
            reader.read_line_raw(i, &mut buf).unwrap();
            IndexTerm::decode(&buf)
        })
    }

    /// Builds a new TermIndexer from TermStoreBuilder and writes it into an OutputBuilder.
    /// This requires the terms to be sorted
    pub(crate) fn build_from_termstore<W: Write>(
        term_store: TermStoreBuilder,
        index_builder: &mut OutputBuilder<W>,
    ) -> Result<Self, Error> {
        let mut index = Vec::new();
        let mut text = Vec::new();

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

        for (_, term) in terms {
            index.push(text.len() as u32);
            text.extend(term.encode::<LittleEndian>()?);
        }

        let mut indexed_terms =
            IndexedReader::new_custom(text, Arc::new(FileIndex::new(index).zero_len()));

        // Write term indexer
        let mut data = Vec::new();
        indexed_terms.write_to(&mut data)?;
        index_builder.write_term_indexer(&data)?;

        Ok(Self {
            index: indexed_terms,
            tot_documents: 0,
        })
    }

    #[inline]
    pub fn get_term(&self, term: &str) -> Option<usize> {
        binary_search(&mut self.index.clone(), term)
    }

    #[cfg(feature = "genbktree")]
    pub fn gen_term_tree(&self) -> bktree::BkTree<String> {
        let mut index = self.index.clone();
        let mut terms = Vec::with_capacity(index.total_lines());

        for i in 0..index.total_lines() {
            let mut buf = Vec::with_capacity(6);
            index.read_line_raw(i, &mut buf).unwrap();
            let index_item = IndexTerm::decode(&buf);
            terms.push(index_item.text().to_string());
        }

        terms.into_iter().collect::<bktree::BkTree<_>>()
    }
}

#[inline]
pub fn binary_search(index: &mut IndexedReader<Vec<u8>>, query: &str) -> Option<usize> {
    index
        .binary_search_raw_by(|i| IndexTerm::decode(i).text().cmp(query))
        .ok()
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
