/// The type-checker will always generate distinct versions of top-level definitions that are
/// called with distinct argument types. However, there may be situations where this specialization
/// is not necessary but which are not identified by the type-checker (to keep it simpler):
/// - External function declarations should never be duplicated, even though they may be
///   specialized differently based on shape parameters.
/// - A function definition may be duplicated based on a distinct shape parameter that is never
///   actually used in its body.
///
/// This module provides functionality for eliminating the former category. The latter category
/// does not apply in all cases, so the implementation must be precise in determining which
/// function definitions to eliminate. Therefore, we leave it out for now to avoid unnecessary
/// bugs, until we find indications that this has a noticeably negative impact on performance.

use super::ast::*;
use crate::utils::name::Name;
use crate::utils::smap::*;

use pyo3::prelude::*;
use std::collections::BTreeMap;

struct ElimDupEnv {
    // Maps the string representing an external function to a unique name used to refer to it.
    ext_decls: BTreeMap<String, Name>,

    // Maps the names of a duplicated top-level definition to its unique name.
    names: BTreeMap<Name, Name>
}

impl Default for ElimDupEnv {
    fn default() -> Self {
        ElimDupEnv {
            ext_decls: BTreeMap::new(),
            names: BTreeMap::new()
        }
    }
}

fn collect_duplicates_top(
    mut env: ElimDupEnv,
    t: Top
) -> (ElimDupEnv, Option<Top>) {
    match t {
        Top::ExtDecl {ref id, ref ext_id, ..} => {
            match env.ext_decls.get(ext_id) {
                Some(unique_id) => {
                    env.names.insert(id.clone(), unique_id.clone());
                    (env, None)
                },
                None => {
                    env.ext_decls.insert(ext_id.clone(), id.clone());
                    (env, Some(t))
                }
            }
        },
        Top::CallbackDecl {..} | Top::FunDef {..} => (env, Some(t))
    }
}

fn lookup_name(env: &ElimDupEnv, id: Name) -> Name {
    env.names.get(&id).cloned().unwrap_or(id)
}

fn replace_names_expr(env: &ElimDupEnv, e: Expr) -> Expr {
    match e {
        Expr::Call {id, args, ty, i} => {
            let id = lookup_name(env, id);
            Expr::Call {id, args, ty, i}
        },
        _ => e.smap(|e| replace_names_expr(env, e))
    }
}

fn replace_names_stmt(env: &ElimDupEnv, s: Stmt) -> Stmt {
    s.smap(|s| replace_names_stmt(env, s))
        .smap(|e| replace_names_expr(env, e))
}

fn replace_names_fun_def(env: &ElimDupEnv, def: FunDef) -> FunDef {
    let body = def.body.smap(|s| replace_names_stmt(env, s));
    FunDef {body, ..def}
}

fn replace_names_top(env: &ElimDupEnv, t: Top) -> Top {
    match t {
        Top::CallbackDecl {id, params, i} => {
            let id = lookup_name(env, id);
            Top::CallbackDecl {id, params, i}
        },
        Top::ExtDecl {id, ext_id, params, res_ty, target, header, par, i} => {
            let id = lookup_name(env, id);
            Top::ExtDecl {id, ext_id, params, res_ty, target, header, par, i}
        },
        Top::FunDef {v} => Top::FunDef {v: replace_names_fun_def(env, v)}
    }
}

pub fn apply(ast: Ast) -> PyResult<Ast> {
    let env = ElimDupEnv::default();
    let (env, tops) = ast.tops.into_iter()
        .fold((env, vec![]), |(env, mut tops), t| {
            match collect_duplicates_top(env, t) {
                (env, Some(t)) => {
                    tops.push(t);
                    (env, tops)
                },
                (env, None) => (env, tops)
            }
        });
    Ok(Ast {
        tops: tops.smap(|t| replace_names_top(&env, t)),
        main: replace_names_fun_def(&env, ast.main)
    })
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test::*;
    use crate::py::ast_builder::*;

    fn ext_decl(id: &Name) -> Top {
        Top::ExtDecl {
            id: id.clone(), ext_id: "".to_string(), params: vec![],
            res_ty: Type::Void, target: Target::Host, header: None,
            par: LoopPar::default(), i: i()
        }
    }

    fn get_identifier(t: &Top) -> Name {
        match t {
            Top::CallbackDecl {id, ..} | Top::ExtDecl {id, ..} |
            Top::FunDef {v: FunDef {id, ..}} => id.clone()
        }
    }

    #[test]
    fn eliminate_duplicate_externals() {
        let n1 = Name::sym_str("x");
        let ext1 = ext_decl(&n1);
        let n2 = Name::sym_str("x");
        let ext2 = ext_decl(&n2);
        let body = vec![
            assignment(
                var("x", tyuk()),
                binop(
                    call(n1.clone(), vec![], tyuk()),
                    BinOp::Add,
                    call(n2.clone(), vec![], tyuk()),
                    tyuk()
                )
            )
        ];
        let main = FunDef {
            id: id("f"), params: vec![], body, res_ty: Type::Void, i: i()
        };
        let ast = Ast {tops: vec![ext1, ext2], main};
        let Ast {tops, ..} = apply(ast).unwrap();
        assert_eq!(tops.len(), 1);
        assert_eq!(get_identifier(&tops[0]), n1);
    }
}
