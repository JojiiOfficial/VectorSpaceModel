use crate::{
    build::weights::TermWeight, error::Error, metadata::Metadata, term_store::TermIndexer,
    traits::Decodable, vector_store::VectorStore, Vector,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    fs::File,
    io::{BufReader, Read},
    path::Path,
};

type Result<T> = std::result::Result<T, Error>;

#[derive(Serialize, Deserialize)]
pub struct Index<D: Decodable, M> {
    pub(crate) metadata: M,
    pub(crate) indexer: TermIndexer,
    #[serde(serialize_with = "serialize_vs")]
    #[serde(deserialize_with = "deserialize_vs::<D,_>")]
    pub(crate) vector_store: VectorStore<D>,
}

impl<D: Decodable, M: DeserializeOwned + Serialize> Index<D, M> {
    /// Opens an Index from a tar.gz file and returns a new `Index`
    pub fn open<P: AsRef<Path>>(file: P) -> Result<Index<D, M>> {
        Self::from_reader(BufReader::new(File::open(file)?))
    }

    /// Read an index-archive and build an `Index` out of it
    pub fn from_reader<R: Read>(reader: R) -> Result<Index<D, M>> {
        bincode::deserialize_from(reader).map_err(|_| Error::InvalidIndex)
    }
}

impl<D: Decodable, M> Index<D, M> {
    /// Returns the vector store of the index
    #[inline]
    pub fn get_vector_store(&self) -> &VectorStore<D> {
        &self.vector_store
    }

    /// Returns the indexer of the index
    #[inline]
    pub fn get_indexer(&self) -> &TermIndexer {
        &self.indexer
    }

    /// Returns the indexes metadata
    #[inline]
    pub fn get_metadata(&self) -> &M {
        &self.metadata
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.indexer.is_empty() || self.vector_store.is_empty()
    }

    // TODO: add function to create new vectors (incl. weights)
    pub fn build_vector_weights<S: AsRef<str>>(&self, terms: &[(S, f32)]) -> Option<Vector> {
        let terms: Vec<_> = terms
            .iter()
            .filter_map(|(term, weight)| {
                let item_pos = self.indexer.get_term(term.as_ref())?;
                Some((item_pos as u32, *weight))
            })
            .collect();

        if terms.is_empty() {
            return None;
        }

        Some(Vector::create_new_raw(terms))
    }

    // TODO: add function to create new vectors (incl. weights)
    pub fn build_vector<S: AsRef<str>>(
        &self,
        terms: &[S],
        weight: Option<&dyn TermWeight>,
    ) -> Option<Vector> {
        let terms: Vec<_> = terms
            .iter()
            .filter_map(|i| {
                let item_pos = self.indexer.get_term(i.as_ref())?;
                let item = self.indexer.load_term(item_pos)?;
                Some((item_pos, item))
            })
            .map(|(pos, i)| {
                let mut res_weight = 1.0;
                if let Some(w) = weight.as_ref() {
                    res_weight =
                        w.weight(1.0, 1, i.doc_frequency() as usize, self.vector_store.len());
                }
                (pos as u32, res_weight)
            })
            .collect();

        if terms.is_empty() {
            return None;
        }

        Some(Vector::create_new_raw(terms))
    }

    pub fn is_stopword_cust(&self, term: &str, threshold: f32) -> Option<bool> {
        let tot_docs = self.get_indexer().len() as f32;
        let term = self.get_indexer().find_term(term)?;
        let ratio = term.doc_frequency() as f32 / tot_docs * 100.0;
        Some(ratio >= threshold)
    }

    #[inline]
    pub fn is_stopword(&self, term: &str) -> Option<bool> {
        self.is_stopword_cust(term, 35.0)
    }

    #[inline]
    pub fn indexer_mut(&mut self) -> &mut TermIndexer {
        &mut self.indexer
    }
}

impl<D: Decodable + DeserializeOwned, M: DeserializeOwned + Serialize + Clone> Index<D, M> {
    pub fn clone_heavy(&self) -> Self {
        let m_c = self.metadata.clone();
        let indexer = self.indexer.clone_heavy();
        let v_store = self.vector_store.clone_full();
        Self {
            metadata: m_c,
            indexer,
            vector_store: v_store,
        }
    }
}

impl<D: Decodable, M: Metadata + Default> Default for Index<D, M> {
    fn default() -> Self {
        Self {
            metadata: Default::default(),
            indexer: Default::default(),
            vector_store: Default::default(),
        }
    }
}

fn serialize_vs<D: Decodable, S>(v: &VectorStore<D>, ser: S) -> std::result::Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    let s = v.serialize(ser)?;
    Ok(s)
}

fn deserialize_vs<'de, D: Decodable, Des>(
    de: Des,
) -> std::result::Result<VectorStore<D>, Des::Error>
where
    Des: serde::Deserializer<'de>,
{
    let o: VectorStore<D> = Deserialize::deserialize(de)?;
    Ok(o)
}
