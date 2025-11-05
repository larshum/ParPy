use super::ast::*;
use crate::py::ast as py_ast;
use crate::utils::name::Name;
use crate::utils::smap::SMapAccum;

pub trait Unsymbolize {
    fn unsymbolize(self) -> Self where Self: Sized;
}

impl Unsymbolize for Name {
    fn unsymbolize(self) -> Self {
        Name::new(self.get_str().clone())
    }
}

// TODO: Implement un-symbolization for Python AST expressions and for types. Left out for now
// because it is not needed for tests.
impl Unsymbolize for py_ast::Expr {
    fn unsymbolize(self) -> Self {
        self
    }
}

impl Unsymbolize for Expr {
    fn unsymbolize(self) -> Self {
        match self {
            Expr::Var {id, ty, i} => {
                Expr::Var {id: id.unsymbolize(), ty, i}
            },
            Expr::Call {id, args, ty, i} => {
                Expr::Call {
                    id: id.unsymbolize(),
                    args: args.smap(|e| e.unsymbolize()),
                    ty,
                    i
                }
            },
            Expr::PyCallback {id, args, ty, i} => {
                Expr::PyCallback {
                    id: id.unsymbolize(),
                    args: args.smap(|e| e.unsymbolize()),
                    ty,
                    i
                }
            },
            Expr::Struct {id, fields, ty, i} => {
                let fields = fields.into_iter()
                    .map(|(s, e)| (s, e.unsymbolize()))
                    .collect::<Vec<(String, Expr)>>();
                Expr::Struct {
                    id: id.unsymbolize(),
                    fields,
                    ty,
                    i
                }
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
            Stmt::For {var_ty, var, init, cond, incr, body, i} => {
                Stmt::For {
                    var_ty,
                    var: var.unsymbolize(),
                    init: init.unsymbolize(),
                    cond: cond.unsymbolize(),
                    incr: incr.unsymbolize(),
                    body: body.smap(|s| s.unsymbolize()),
                    i
                }
            },
            Stmt::Expr {e, i} => {
                Stmt::Expr {e: e.unsymbolize(), i}
            },
            Stmt::ParallelReduction {var_ty, var, init, cond, incr, body, nthreads, tpb, i} => {
                Stmt::ParallelReduction {
                    var_ty,
                    var: var.unsymbolize(),
                    init: init.unsymbolize(),
                    cond: cond.unsymbolize(),
                    incr: incr.unsymbolize(),
                    body: body.smap(|s| s.unsymbolize()),
                    nthreads,
                    tpb,
                    i
                }
            },
            Stmt::KernelLaunch {id, args, grid, i} => {
                Stmt::KernelLaunch {
                    id: id.unsymbolize(),
                    args: args.smap(|a| a.unsymbolize()),
                    grid,
                    i
                }
            },
            Stmt::AllocDevice {elem_ty, id, sz, i} => {
                Stmt::AllocDevice {elem_ty, id: id.unsymbolize(), sz, i}
            },
            Stmt::AllocShared {elem_ty, id, sz, i} => {
                Stmt::AllocShared {elem_ty, id: id.unsymbolize(), sz, i}
            },
            Stmt::FreeDevice {id, i} => Stmt::FreeDevice {id: id.unsymbolize(), i},
            _ => {
                self.smap(|s: Stmt| s.unsymbolize())
                    .smap(|e: Expr| e.unsymbolize())
            }
        }
    }
}
