pub mod item;

use self::item::IndexItem;
use crate::{ error::Error, index::IndexBuilder, traits::Encodable};
use byteorder::LittleEndian;
use indexed_file::{
    any::IndexedReader, index::Header as IndexHeader, index::Index as FileIndex, Indexable,
    IndexableFile, ReadByLine,
};
use std::{
    io::{BufReader, Read, Seek, SeekFrom},
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
    pub fn find_term(&mut self, term: &str) -> Option<IndexItem> {
        let dimension = binary_search(&mut self.index, term)?;
        self.load_term(dimension)
    }

    /// Gets a term by its dimension
    #[inline]
    pub fn load_term(&mut self, dimension: usize) -> Option<IndexItem> {
        let mut buf = Vec::with_capacity(10);
        self.index.read_line_raw(dimension, &mut buf).ok()?;
        IndexItem::decode(&buf).ok()
    }

    /// Sets the total amount of documents in the index. This is required for better indexing
    #[inline]
    pub(crate) fn set_total_documents(&mut self, tot_documents: usize) {
        self.tot_documents = tot_documents;
    }

    /// Returns an iterator over all Indexed terms
    pub fn iter(&self) -> impl Iterator<Item = IndexItem> + '_ {
        let mut reader = self.index.clone();
        (0..self.index.total_lines()).map(move |i| {
            let mut buf = Vec::with_capacity(10);
            reader.read_line_raw(i, &mut buf).unwrap();
            IndexItem::decode(&buf).unwrap()
        })
    }

    /// Build a new term indexer for language `lang` using JMdict.
    pub(crate) fn build<T>(
        index_builder: &mut IndexBuilder,
        tot_documents: usize,
        terms: T,
    ) -> Result<Self, Error>
    where
        T: Iterator<Item = IndexItem>,
    {
        let mut index = Vec::new();
        let mut text = Vec::new();

        let mut terms = terms.collect::<Vec<_>>();
        terms.sort_by(|a, b| a.text().cmp(b.text()));

        for term in terms {
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
            tot_documents,
        })
    }
}

#[inline]
pub fn binary_search(index: &mut IndexedReader<Vec<u8>>, query: &str) -> Option<usize> {
    index
        .binary_search_raw_by(|i| {
            let item = IndexItem::decode(i).unwrap();
            item.text().cmp(query)
        })
        .ok()
}

/*
impl document::Indexable for TermIndexer {
    #[inline]
    fn index(&self, part: &str) -> Option<usize> {
        binary_search(&mut self.index.clone(), part)
    }

    #[inline]
    fn index_size(&self) -> usize {
        indexed_file::Indexable::total_lines(&self.index)
    }

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
