use crate::utils::ast::ElemSize;
use crate::utils::name::Name;

use pyo3::prelude::*;

#[pyclass(eq, frozen)]
#[derive(Clone, Debug, PartialEq)]
pub struct Symbol {
    pub id: Name
}

#[pymethods]
impl Symbol {
    #[new]
    fn new() -> Symbol {
        Symbol {id: Name::sym_str("")}
    }
}

#[pyclass(eq, frozen)]
#[derive(Clone, Debug, PartialEq)]
pub struct TypeVar {
    pub id: Name
}

#[pymethods]
impl TypeVar {
    #[new]
    fn new() -> TypeVar {
        TypeVar {id: Name::sym_str("")}
    }
}

#[pyclass(eq, frozen)]
#[derive(Clone, Debug, PartialEq)]
pub enum ExtType {
    Buffer(ElemSize, Vec<Symbol>),
    VarBuffer(TypeVar, Vec<Symbol>),
}
