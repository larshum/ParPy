use super::ast::*;
use crate::utils::name::Name;
use crate::utils::smap::{SFold, SMapAccum};

use std::collections::BTreeSet;

fn extract_callback_decls_top(
    acc: (BTreeSet<Name>, Vec<Top>),
    t: Top
) -> (BTreeSet<Name>, Vec<Top>) {
    let (mut callbacks, mut tops) = acc;
    match t {
        Top::CallbackDecl {id, ..} => {
            callbacks.insert(id);
            (callbacks, tops)
        },
        _ => {
            tops.push(t);
            (callbacks, tops)
        }
    }
}

fn specialize_callback_calls_expr(callbacks: &BTreeSet<Name>, e: Expr) -> Expr {
    match e {
        Expr::Call {id, args, ty, i} if callbacks.contains(&id) => {
            Expr::Callback {id, args, ty, i}
        },
        _ => e.smap(|e| specialize_callback_calls_expr(&callbacks, e))
    }
}

fn specialize_callback_calls_stmt(callbacks: &BTreeSet<Name>, s: Stmt) -> Stmt {
    s.smap(|s| specialize_callback_calls_stmt(&callbacks, s))
        .smap(|e| specialize_callback_calls_expr(&callbacks, e))
}

fn specialize_callback_calls_def(callbacks: &BTreeSet<Name>, def: FunDef) -> FunDef {
    let body = def.body.smap(|s| specialize_callback_calls_stmt(&callbacks, s));
    FunDef {body, ..def}
}

pub fn apply(ast: Ast) -> Ast {
    let (callbacks, tops) = ast.tops.sfold_owned(
        (BTreeSet::new(), vec![]),
        extract_callback_decls_top
    );
    let main = specialize_callback_calls_def(&callbacks, ast.main);
    Ast {tops, main}
}
