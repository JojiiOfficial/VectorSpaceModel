use std::{
    convert::TryInto,
    io::{BufReader, Read, Seek, SeekFrom},
    sync::Arc,
};

/// File name in the index tar
pub(crate) const FILE_NAME: &str = "term_indexer";

use byteorder::{LittleEndian, WriteBytesExt};
use indexed_file::{
    any::IndexedReader, index::Header as IndexHeader, index::Index as FileIndex, IndexableFile,
    ReadByLine,
};

use crate::{document_vector, error::Error, index::IndexBuilder, traits::Encodable};

/// An in memory TermIndexer that allows efficient indexing of terms which is requried for document
/// vectors being calculated.
#[derive(Debug, Clone)]
pub struct TermIndexer {
    index: IndexedReader<Vec<u8>>,
    tot_documents: usize,
}

/// A single term info represented by a single line in `TermIndexer`
#[derive(Debug, Clone)]
pub struct IndexItem {
    text: String,
    frequency: u16,
}

impl IndexItem {
    #[inline]
    pub fn new(text: String, frequency: u16) -> IndexItem {
        Self { text, frequency }
    }

    #[inline(always)]
    pub fn decode(data: &[u8]) -> Result<Self, Error> {
        let frequency = u16::from_le_bytes(data[0..2].try_into().unwrap());
        let text = String::from_utf8_lossy(&data[2..]).to_string();
        Ok(Self { text, frequency })
    }
}

impl Encodable for IndexItem {
    fn encode<T: byteorder::ByteOrder>(&self) -> Result<Vec<u8>, Error> {
        let mut out = Vec::new();
        out.write_u16::<T>(self.frequency)?;
        out.extend(self.text.as_bytes());
        Ok(out)
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

        let mut s = Vec::new();
        reader.read_to_end(&mut s)?;
        let mem_index = IndexedReader::new_custom(s, Arc::new(index.zero_len()));

        Ok(TermIndexer {
            index: mem_index,
            tot_documents: 0,
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

        for term in terms {
            index.push(text.len() as u64);
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

    /// Sets the total amount of documents in the index. This is required for better indexing
    pub(crate) fn set_total_documents(&mut self, tot_documents: usize) {
        self.tot_documents = tot_documents;
    }

    fn binary_search(index: &mut IndexedReader<Vec<u8>>, query: &str) -> Option<usize> {
        index
            .binary_search_raw_by(|i| {
                let item = IndexItem::decode(&i).unwrap();
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

        let mut buf = Vec::with_capacity(6);
        let item = index
            .read_line_raw(item_pos, &mut buf)
            .ok()
            .and_then(|_| IndexItem::decode(&buf).ok())
            .expect("Invalid term index file");

        Some(item.frequency as usize)
    }
}
