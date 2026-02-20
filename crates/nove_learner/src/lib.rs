use thiserror::Error;

pub mod common;

#[derive(Error, Debug)]
pub enum LearnerError {}

pub trait Learner {
    fn fit(&mut self) -> Result<(), LearnerError>;
}
