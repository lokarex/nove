pub trait Dataset {
    type Item;

    fn get(&self, index: usize) -> Option<Self::Item>;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
