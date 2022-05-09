use std::{cmp::Ordering, iter::Peekable};

pub(crate) struct LockStepIter<A, B, K, V, W>
where
    A: Iterator<Item = (K, V)>,
    B: Iterator<Item = (K, W)>,
{
    a: Peekable<A>,
    b: Peekable<B>,
}

impl<A, B, K, V, W> LockStepIter<A, B, K, V, W>
where
    A: Iterator<Item = (K, V)>,
    B: Iterator<Item = (K, W)>,
{
    #[inline]
    pub fn new(a: A, b: B) -> Self {
        Self {
            a: a.peekable(),
            b: b.peekable(),
        }
    }
}

impl<A, B, K, V, W> Iterator for LockStepIter<A, B, K, V, W>
where
    A: Iterator<Item = (K, V)>,
    B: Iterator<Item = (K, W)>,
    K: Ord,
{
    type Item = (K, V, W);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match (self.a.peek(), self.b.peek()) {
                (Some((dim_a, _)), Some((dim_b, _))) => match dim_a.cmp(dim_b) {
                    Ordering::Less => {
                        self.a.next()?;
                    }
                    Ordering::Greater => {
                        self.b.next()?;
                    }
                    Ordering::Equal => {
                        let (dim, value_a) = unsafe { self.a.next().unwrap_unchecked() };
                        let (_, value_b) = unsafe { self.b.next().unwrap_unchecked() };
                        return Some((dim, value_a, value_b));
                    }
                },
                _ => return None,
            }
        }
    }
}
