use super::ast::*;
use super::type_check::UnifyEnv;
use crate::utils::name::Name;
use crate::utils::smap::*;

use std::collections::BTreeMap;

struct SpecEnv<'a> {
    shape_map: &'a BTreeMap<Name, i64>
}

impl<'a> SpecEnv<'a> {
    fn new(unify_env: &'a UnifyEnv) -> SpecEnv<'a> {
        SpecEnv {
            shape_map: &unify_env.shape_vars
        }
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
    unify_env: &UnifyEnv,
    body: Vec<Stmt>
) -> Vec<Stmt> {
    let env = SpecEnv::new(unify_env);
    body.smap(|s| specialize_stmt(&env, s))
}
