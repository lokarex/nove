use rand::rngs::ThreadRng;
use std::sync::atomic::AtomicUsize;

static ID: AtomicUsize = AtomicUsize::new(1);

pub struct Dropout {
    dropout_prob: f32,
    rng: ThreadRng,
}
