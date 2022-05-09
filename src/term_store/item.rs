use crate::{traits::Encodable, Error};
use byteorder::WriteBytesExt;
use std::convert::TryInto;

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

    /// Get a reference to the index item's text.
    #[inline]
    pub fn text(&self) -> &str {
        self.text.as_ref()
    }

    /// Get the index item's frequency.
    #[inline]
    pub fn frequency(&self) -> u16 {
        self.frequency
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
