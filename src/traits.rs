use crate::error::Error;
use byteorder::{ByteOrder, ReadBytesExt, WriteBytesExt};
use std::io::{Cursor, Read};

pub trait SizedSerialize {
    fn size() -> usize;
}

pub trait Encodable {
    fn encode<T: ByteOrder>(&self) -> Result<Vec<u8>, Error>;
}

pub trait Decodable: Sized {
    fn decode<T: ByteOrder, R: Read>(data: R) -> Result<Self, Error>;
}

impl Encodable for u32 {
    #[inline]
    fn encode<T: ByteOrder>(&self) -> Result<Vec<u8>, Error> {
        let mut out = Vec::with_capacity(Self::size());
        out.write_u32::<T>(*self)?;
        Ok(out)
    }
}

impl Decodable for u32 {
    #[inline]
    fn decode<T: ByteOrder, R: Read>(mut data: R) -> Result<Self, Error> {
        Ok(data.read_u32::<T>()?)
    }
}

impl SizedSerialize for u32 {
    #[inline(always)]
    fn size() -> usize {
        4
    }
}

impl Encodable for u64 {
    #[inline]
    fn encode<T: ByteOrder>(&self) -> Result<Vec<u8>, Error> {
        let mut out = Vec::with_capacity(Self::size());
        out.write_u64::<T>(*self)?;
        Ok(out)
    }
}

impl Decodable for u64 {
    #[inline]
    fn decode<T: ByteOrder, R: Read>(mut data: R) -> Result<Self, Error> {
        Ok(data.read_u64::<T>()?)
    }
}

impl SizedSerialize for u64 {
    #[inline(always)]
    fn size() -> usize {
        8
    }
}

impl<DE: Encodable + SizedSerialize> Encodable for Vec<DE> {
    fn encode<T: ByteOrder>(&self) -> Result<Vec<u8>, Error> {
        let mut out = Vec::with_capacity(4 + self.len() * 10);

        out.write_u32::<T>(self.len() as u32)?;

        for i in self.iter() {
            out.extend(i.encode::<T>()?);
        }

        Ok(out)
    }
}

impl<DE: Decodable + SizedSerialize> Decodable for Vec<DE> {
    #[inline]
    fn decode<T: ByteOrder, R: Read>(mut data: R) -> Result<Self, Error> {
        let len = data.read_u32::<T>()?;
        let mut out_list = Vec::with_capacity(len as usize);

        let mut buf = vec![0u8; DE::size()];
        for _ in 0..len {
            data.read_exact(&mut buf)?;
            out_list.push(DE::decode::<T, _>(Cursor::new(&buf))?);
        }

        Ok(out_list)
    }
}

#[cfg(test)]
mod test {
    use byteorder::LittleEndian;

    use super::*;

    #[test]
    fn test_vec_encode() {
        let input: Vec<u32> = vec![10, 7697, 16, 76];
        let encoded = input.encode::<LittleEndian>().unwrap();
        let decoded = Vec::<u32>::decode::<LittleEndian, _>(Cursor::new(encoded)).unwrap();
        assert_eq!(decoded, input);
    }
}
