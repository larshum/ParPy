use super::ast::*;
use crate::option::CompileOptions;
use crate::utils::info::Info;
use crate::utils::name::Name;
use crate::utils::smap::{SFlatten, SMapAccum};
use crate::utils::substitute::SubVars;

use std::collections::BTreeMap;

#[derive(Clone)]
struct LoopInfo {
    var_ty: Type,
    var: Name,
    init: Expr,
    cond: Expr,
    incr: Expr,
    body: Vec<Stmt>,
    i: Info
}

fn extract_integer(e: &Expr) -> Option<i128> {
    match e {
        Expr::Int {v, ..} => Some(*v),
        _ => None
    }
}

fn lower_bound_of_first_thread(init: &Expr) -> Option<i128> {
    match init {
        Expr::BinOp {lhs, op: BinOp::Add, rhs, ..} => {
            match rhs.as_ref() {
                Expr::ThreadIdx {dim: Dim::X, ..} => extract_integer(lhs),
                _ => None
            }
        },
        Expr::ThreadIdx {dim: Dim::X, ..} => Some(0),
        Expr::Convert {e, ..} => lower_bound_of_first_thread(e),
        _ => None
    }
}

fn upper_bound_value(var: &Name, cond: &Expr) -> Option<i128> {
    match cond {
        Expr::BinOp {lhs, op: BinOp::Lt, rhs, ..} => {
            match lhs.as_ref() {
                Expr::Var {id, ..} if id.eq(var) => extract_integer(rhs),
                _ => None
            }
        },
        _ => None
    }
}

fn extract_step_size(var: &Name, incr: &Expr) -> Option<i128> {
    match incr {
        Expr::BinOp {lhs, op: BinOp::Add, rhs, ..} => {
            match lhs.as_ref() {
                Expr::Var {id, ..} if id.eq(var) => extract_integer(rhs),
                _ => None
            }
        },
        _ => None
    }
}

fn unroll_stmt(
    id: Name,
    lo: i128,
    hi: i128,
    num_threads: i128,
    stmt: Stmt,
    ty: &Type,
    i: &Info
) -> Vec<Stmt> {
    let thread_idx_plus_offset = |ofs| {
        let thread_expr = Expr::Convert {
            e: Box::new(Expr::ThreadIdx {
                dim: Dim::X, ty: Type::Scalar {sz: ElemSize::U32}, i: i.clone()
            }),
            ty: ty.clone()
        };
        if ofs == 0 {
            thread_expr
        } else {
            Expr::BinOp {
                lhs: Box::new(thread_expr),
                op: BinOp::Add,
                rhs: Box::new(Expr::Int {v: ofs, ty: ty.clone(), i: i.clone()}),
                ty: ty.clone(),
                i: i.clone()
            }
        }
    };
    let mk_sub_env = |ofs| {
        let sub_expr = thread_idx_plus_offset(ofs);
        let mut sub_env = BTreeMap::new();
        sub_env.insert(id.clone(), sub_expr);
        sub_env
    };
    let mut acc = vec![];
    let mut ofs = lo;
    while ofs + num_threads <= hi {
        let sub_env = mk_sub_env(ofs);
        acc.push(stmt.clone().sub_vars(&sub_env));
        ofs += num_threads;
    }
    // Insert a conditionally executed statement, that we only run for threads whose index is below
    // a certain upper-bound.
    if ofs < hi {
        let sub_env = mk_sub_env(ofs);
        let max_thread_idx = hi - ofs;
        let cond_stmt = Stmt::If {
            cond: Expr::BinOp {
                lhs: Box::new(Expr::ThreadIdx {
                    dim: Dim::X, ty: Type::Scalar {sz: ElemSize::U32}, i: i.clone()
                }),
                op: BinOp::Lt,
                rhs: Box::new(Expr::Int {
                    v: max_thread_idx, ty: ty.clone(), i: i.clone()
                }),
                ty: ty.clone(),
                i: i.clone()
            },
            thn: vec![stmt.sub_vars(&sub_env)],
            els: vec![],
            i: i.clone()
        };
        acc.push(cond_stmt);
    }
    acc
}

fn try_unroll_loop(
    mut li: LoopInfo,
    opts: &CompileOptions
) -> Option<Vec<Stmt>> {
    if li.body.len() == 1 {
        let lo = lower_bound_of_first_thread(&li.init)?;
        let hi = upper_bound_value(&li.var, &li.cond)?;
        let num_threads = extract_step_size(&li.var, &li.incr)?;
        let niters = ((hi - lo) + num_threads - 1) / num_threads;
        if niters <= opts.max_unroll_count as i128 {
            let stmt = li.body.pop().unwrap();
            Some(unroll_stmt(li.var, lo, hi, num_threads, stmt, &li.var_ty, &li.i))
        } else {
            None
        }
    } else {
        None
    }
}

fn unroll_loops_stmt(mut acc: Vec<Stmt>, s: Stmt, opts: &CompileOptions) -> Vec<Stmt> {
    match s {
        Stmt::For {var_ty, var, init, cond, incr, body, i} => {
            let li = LoopInfo {
                var_ty: var_ty.clone(),
                var: var.clone(),
                init: init.clone(),
                cond: cond.clone(),
                incr: incr.clone(),
                body: body.clone(),
                i: i.clone()
            };
            match try_unroll_loop(li, &opts) {
                Some(mut unrolled_body) => acc.append(&mut unrolled_body),
                None => {
                    let body = body.sflatten(vec![], |acc, s| {
                        unroll_loops_stmt(acc, s, &opts)
                    });
                    acc.push(Stmt::For {
                        var_ty, var, init, cond, incr, body, i
                    });
                }
            };
            acc
        },
        _ => s.sflatten(acc, |acc, s| unroll_loops_stmt(acc, s, &opts))
    }
}

fn unroll_loops_top(t: Top, opts: &CompileOptions) -> Top {
    match t {
        Top::ExtDecl {..} | Top::FunDef {..} | Top::StructDef {..} => {
            t
        },
        Top::KernelFunDef {attrs, id, params, body, i} => {
            let body = body.sflatten(vec![], |acc, s| unroll_loops_stmt(acc, s, &opts));
            Top::KernelFunDef {attrs, id, params, body, i}
        }
    }
}

pub fn apply(ast: Ast, opts: &CompileOptions) -> Ast {
    ast.smap(|t| unroll_loops_top(t, &opts))
}
