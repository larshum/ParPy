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
    fn new(s: String) -> Symbol {
        Symbol {id: Name::sym_str(&s)}
    }
}

#[pyclass(eq, frozen)]
#[derive(Clone, Debug, PartialEq)]
pub enum ExtType {
    Buffer(ElemSize, Vec<Symbol>),
}
