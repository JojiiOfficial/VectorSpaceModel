use crate::{traits::Encodable, Error};
use byteorder::WriteBytesExt;
use std::convert::TryInto;

/// A term in the index
#[derive(Debug, Clone)]
pub struct IndexTerm {
    /// The terms text/value
    text: String,
    /// number of documents with this term
    doc_frequency: u32,
}

impl IndexTerm {
    /// Creates a new term
    #[inline]
    pub fn new(text: String, doc_frequency: u32) -> IndexTerm {
        Self {
            text,
            doc_frequency,
        }
    }

    /// Decodes an index item from raw data. Panics if the data is malformed
    #[inline(always)]
    pub fn decode(data: &[u8]) -> Self {
        let frequency = u32::from_le_bytes(data[0..4].try_into().unwrap());
        let text = String::from_utf8_lossy(&data[4..]).to_string();
        Self {
            text,
            doc_frequency: frequency,
        }
    }

    /// Get a reference to the index item's text.
    #[inline]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Get the index item's frequency.
    #[inline]
    pub fn doc_frequency(&self) -> u32 {
        self.doc_frequency
    }
}

impl AsRef<str> for IndexTerm {
    #[inline]
    fn as_ref(&self) -> &str {
        &self.text
    }
}

impl Encodable for IndexTerm {
    #[inline]
    fn encode<T: byteorder::ByteOrder>(&self) -> Result<Vec<u8>, Error> {
        let text = self.text.as_bytes();
        let mut out = Vec::with_capacity(text.len() + 4);
        out.write_u32::<T>(self.doc_frequency)?;
        out.extend(text);
        Ok(out)
    }
}
