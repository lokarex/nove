//! Indexing, indexed accumulation and conditional selection.

use super::super::{Backend, validation};
use super::dispatch::{dispatch_binary, dispatch_ternary};
use super::{BackendStorage, ensure_compatible_devices};
use crate::BackendError;

impl BackendStorage {
    pub fn gather(&self, indexes: &Self, dim: usize) -> Result<Self, BackendError> {
        ensure_compatible_devices("gather", &[self, indexes])?;
        validation::validate_gather(&self.shape()?, &indexes.shape()?, indexes.dtype()?, dim)?;
        dispatch_binary!(self, indexes, gather, dim)
    }

    pub fn scatter_add(
        &self,
        indexes: &Self,
        source: &Self,
        dim: usize,
    ) -> Result<Self, BackendError> {
        ensure_compatible_devices("scatter_add", &[self, indexes, source])?;
        validation::validate_scatter_add(
            &self.shape()?,
            self.dtype()?,
            &indexes.shape()?,
            indexes.dtype()?,
            &source.shape()?,
            source.dtype()?,
            dim,
        )?;
        dispatch_ternary!(self, indexes, source, scatter_add, dim)
    }

    pub fn index_select(&self, indexes: &Self, dim: usize) -> Result<Self, BackendError> {
        ensure_compatible_devices("index_select", &[self, indexes])?;
        validation::validate_index_select(
            &self.shape()?,
            &indexes.shape()?,
            indexes.dtype()?,
            dim,
        )?;
        dispatch_binary!(self, indexes, index_select, dim)
    }

    pub fn index_add(
        &self,
        indexes: &Self,
        source: &Self,
        dim: usize,
    ) -> Result<Self, BackendError> {
        ensure_compatible_devices("index_add", &[self, indexes, source])?;
        validation::validate_index_add(
            &self.shape()?,
            self.dtype()?,
            &indexes.shape()?,
            indexes.dtype()?,
            &source.shape()?,
            source.dtype()?,
            dim,
        )?;
        dispatch_ternary!(self, indexes, source, index_add, dim)
    }

    pub fn where_cond(
        condition: &Self,
        true_value: &Self,
        false_value: &Self,
    ) -> Result<Self, BackendError> {
        ensure_compatible_devices("where_cond", &[condition, true_value, false_value])?;
        validation::validate_where(
            condition.dtype()?,
            true_value.dtype()?,
            false_value.dtype()?,
        )?;
        validation::validate_where_shapes(
            &condition.shape()?,
            &true_value.shape()?,
            &false_value.shape()?,
        )?;
        dispatch_ternary!(condition, true_value, false_value, where_cond)
    }
}
