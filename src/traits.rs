use std::io::Read;

use byteorder::ByteOrder;

use crate::error::Error;

pub trait Encodable {
    fn encode<T: ByteOrder>(&self) -> Result<Vec<u8>, Error>;
}

pub trait Decodable: Sized {
    fn decode<T: ByteOrder, R: Read>(data: R) -> Result<Self, Error>;
}
