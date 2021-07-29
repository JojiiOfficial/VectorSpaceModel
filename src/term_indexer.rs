use std::{
    io::{BufReader, Read, Seek, SeekFrom},
    sync::Arc,
};

/// File name in the index tar
pub(crate) const FILE_NAME: &str = "term_indexer";

use indexed_file::{
    index::Header as IndexHeader, index::Index as FileIndex, IndexableFile, IndexedString,
    ReadByLine,
};

use crate::{document_vector, error::Error, index::IndexBuilder};

/// An in memory TermIndexer that allows efficient indexing of terms which is requried for document
/// vectors being calculated.
#[derive(Debug, Clone)]
pub struct TermIndexer {
    index: IndexedString,
    tot_documents: usize,
}

/// A single term info represented by a single line in `TermIndexer`
#[derive(Debug, Clone)]
struct IndexItem {
    text: String,
    frequency: u32,
}

impl<T: ToString> From<T> for IndexItem {
    fn from(s: T) -> Self {
        let s = s.to_string();
        let split_pos = s.char_indices().rev().find(|i| i.1 == ',').unwrap().0;
        let text = s[..split_pos].to_owned();
        let frequency: u32 = s[split_pos + 1..].parse().unwrap();
        IndexItem { text, frequency }
    }
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

        let mut s: String = String::new();
        reader.read_to_string(&mut s)?;
        let mem_index = IndexedString::new_custom(s, Arc::new(index.zero_len()));

        Ok(TermIndexer {
            index: mem_index,
            tot_documents: 0,
        })
    }

    /// Build a new term indexer for language `lang` using JMdict.
    pub(crate) fn build<T, U: AsRef<str>>(
        index_builder: &mut IndexBuilder,
        tot_documents: usize,
        terms: T,
    ) -> Result<Self, Error>
    where
        T: Iterator<Item = U>,
    {
        let terms: String = terms.fold(String::new(), |a, b| a + b.as_ref() + "\n");

        let mut indexed_terms = IndexedString::new_raw(terms)?;

        // Write term indexer
        let mut data = Vec::new();
        indexed_terms.write_to(&mut data)?;
        index_builder.write_term_indexer(&data)?;

        Ok(Self {
            index: indexed_terms,
            tot_documents,
        })
    }

    /// Sets the total amount of documents in the index. This is required for better indexing
    pub(crate) fn set_total_documents(&mut self, tot_documents: usize) {
        self.tot_documents = tot_documents;
    }

    fn binary_search(index: &mut IndexedString, query: &str) -> Option<usize> {
        index
            .binary_search_by(|i| {
                let item = IndexItem::from(i);
                item.text.as_str().cmp(&query)
            })
            .ok()
    }
}

impl document_vector::Indexable for TermIndexer {
    fn index(&self, part: &str) -> Option<usize> {
        // We can clone here since IndexedString's clone is very light
        let mut index = self.index.clone();

        let mut query = part;
        loop {
            if let Some(index) = Self::binary_search(&mut index, query) {
                break Some(index);
            }

            // Only go down to 3
            if query.len() <= 3 {
                break None;
            }

            // Remove last character
            query = &query[..query.char_indices().rev().nth(1).unwrap().0];
        }
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

        let item_pos = match Self::binary_search(&mut index, term) {
            Some(s) => s,
            None => return Some(1),
        };

        let frequency = index
            .read_line(item_pos)
            .map(IndexItem::from)
            .map(|i| i.frequency as usize)
            .unwrap_or(1);

        Some(frequency)
    }
}
