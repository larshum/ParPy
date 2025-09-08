use crate::utils::ast::ElemSize;
use crate::utils::name::Name;

use pyo3::prelude::*;

#[pyclass(eq, frozen)]
#[derive(Clone, Debug, PartialEq)]
pub struct Symbol {
    id: Name
}

#[pyfunction]
pub fn symbol(id: Option<String>) -> Symbol {
    if let Some(s) = id {
        Symbol {id: Name::sym_str(&s)}
    } else {
        Symbol {id: Name::sym_str("")}
    }
}

#[pyclass(eq, frozen)]
#[derive(Clone, Debug, PartialEq)]
pub enum ExtType {
    Buffer(ElemSize, Vec<Symbol>),
}
