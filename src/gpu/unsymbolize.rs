use super::ast::*;
use crate::utils::name::Name;
use crate::utils::smap::SMapAccum;

pub trait Unsymbolize {
    fn unsymbolize(self) -> Self where Self: Sized;
}

// TODO(larshum, 2025-11-05): This is a minimal implementation of un-symbolization as needed for
// the current tests. It should be expanded to support more kinds of nodes when used for more
// tests.
impl Unsymbolize for Name {
    fn unsymbolize(self) -> Self {
        Name::new(self.get_str().clone())
    }
}

impl Unsymbolize for Expr {
    fn unsymbolize(self) -> Self {
        match self {
            Expr::Var {id, ty, i} => {
                Expr::Var {id: id.unsymbolize(), ty, i}
            },
            _ => self.smap(|e| e.unsymbolize())
        }
    }
}

impl Unsymbolize for Stmt {
    fn unsymbolize(self) -> Self {
        match self {
            Stmt::Definition {ty, id, expr, i} => {
                Stmt::Definition {
                    ty,
                    id: id.unsymbolize(),
                    expr: expr.unsymbolize(),
                    i
                }
            },
            Stmt::AllocShared {elem_ty, id, sz, i} => {
                Stmt::AllocShared {elem_ty, id: id.unsymbolize(), sz, i}
            },
            _ => {
                self.smap(|s: Stmt| s.unsymbolize())
                    .smap(|e: Expr| e.unsymbolize())
            }
        }
    }
}
