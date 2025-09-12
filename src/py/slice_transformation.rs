use super::ast::*;
use crate::utils::info::Info;
use crate::utils::name::Name;
use crate::utils::smap::SMapAccum;

use pyo3::prelude::*;

fn replace_slices_assignment(
    reconstruct_stmt: impl Fn(Expr, Expr, Vec<String>, Info) -> Stmt,
    def_id: Option<Name>,
    lhs: Expr,
    rhs: Expr,
    mut labels: Vec<String>,
    i: Info
) -> PyResult<Stmt> {
    todo!()
}

fn replace_slices_stmt(s: Stmt) -> PyResult<Stmt> {
    match s {
        Stmt::Definition {ty, id, expr, labels, i} => {
            let reconstruct_def = |lhs, rhs, labels, i| {
                if let Expr::Var {id, ty, ..} = lhs {
                    Stmt::Definition {ty, id, expr: rhs, labels, i}
                } else {
                    unreachable!()
                }
            };
            let def_data = Some(id.clone());
            let lhs = Expr::Var {id, ty, i: i.clone()};
            replace_slices_assignment(reconstruct_def, def_data, lhs, expr, labels, i)
        },
        Stmt::Assign {dst, expr, labels, i} => {
            let reconstruct_assign = |lhs, rhs, labels, i| {
                Stmt::Assign {dst: lhs, expr: rhs, labels, i}
            };
            let def_data = None;
            replace_slices_assignment(reconstruct_assign, def_data, dst, expr, labels, i)
        },
        Stmt::Label {..} | Stmt::For {..} | Stmt::While {..} | Stmt::If {..} |
        Stmt::Return {..} | Stmt::WithGpuContext {..} | Stmt::Call {..} => {
            s.smap_result(replace_slices_stmt)
        }
    }
}

fn replace_slices_fun_def(def: FunDef) -> PyResult<FunDef> {
    // TODO: what slice properties to validate?
    let body = def.body.smap_result(replace_slices_stmt)?;
    Ok(FunDef {body, ..def})
}

fn replace_slices_top(t: Top) -> PyResult<Top> {
    match t {
        Top::ExtDecl {..} => Ok(t),
        Top::FunDef {v} => Ok(Top::FunDef {v: replace_slices_fun_def(v)?})
    }
}

pub fn apply(ast: Ast) -> PyResult<Ast> {
    let tops = ast.tops.smap_result(replace_slices_top)?;
    let main = replace_slices_fun_def(ast.main)?;
    Ok(Ast {tops, main})
}
