use super::ast::*;
use crate::utils::name::Name;
use crate::utils::smap::*;

use std::collections::BTreeMap;

struct SpecEnv<'a> {
    shape_map: &'a BTreeMap<Name, i64>
}

impl<'a> SpecEnv<'a> {
    fn new(shape_map: &'a BTreeMap<Name, i64>) -> SpecEnv<'a> {
        SpecEnv { shape_map }
    }
}

fn specialize_expr<'a>(
    env: &SpecEnv<'a>,
    e: Expr
) -> Expr {
    match e {
        Expr::Var {id, ty, i} => {
            match env.shape_map.get(&id) {
                Some(v) => Expr::Int {v: *v as i128, ty, i},
                None => Expr::Var {id, ty, i}
            }
        },
        _ => e.smap(|e| specialize_expr(env, e))
    }
}

fn specialize_stmt<'a>(
    env: &SpecEnv<'a>,
    s: Stmt
) -> Stmt {
    s.smap(|s| specialize_stmt(env, s))
        .smap(|s| specialize_expr(env, s))
}

pub fn apply<'py>(
    shape_map: &BTreeMap<Name, i64>,
    body: Vec<Stmt>
) -> Vec<Stmt> {
    let env = SpecEnv::new(shape_map);
    body.smap(|s| specialize_stmt(&env, s))
}
