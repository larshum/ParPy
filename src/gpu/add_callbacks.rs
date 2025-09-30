use super::ast::*;
use crate::parpy_compile_error;
use crate::utils::err::*;
use crate::utils::info::Info;
use crate::utils::name::Name;
use crate::utils::smap::*;

use std::collections::BTreeMap;

type Callbacks = BTreeMap<Name, (Vec<Param>, Info)>;

fn collect_callbacks_top(mut acc: Callbacks, t: &Top) -> Callbacks {
    match t {
        Top::CallbackDecl {id, params, i} => {
            acc.insert(id.clone(), (params.clone(), i.clone()));
            acc
        },
        _ => acc
    }
}

fn collect_used_callbacks_expr(
    mut used: Callbacks,
    e: &Expr,
    callbacks: &Callbacks
) -> Callbacks {
    match e {
        Expr::Call {id, args, ..} => {
            if let Some(entry) = callbacks.get(&id) {
                used.insert(id.clone(), entry.clone());
            }
            args.sfold(used, |acc, e| collect_used_callbacks_expr(acc, e, &callbacks))
        },
        _ => e.sfold(used, |acc, e| collect_used_callbacks_expr(acc, e, &callbacks))
    }
}

fn collect_used_callbacks_stmt(
    used: Callbacks,
    s: &Stmt,
    callbacks: &Callbacks
) -> Callbacks {
    let used = s.sfold(used, |acc, s| collect_used_callbacks_stmt(acc, s, &callbacks));
    s.sfold(used, |acc, e| collect_used_callbacks_expr(acc, e, &callbacks))
}

fn to_callback_parameter(id: Name, params: Vec<Param>, i: Info) -> Param {
    let param_types = params.into_iter()
        .map(|Param {ty, ..}| ty)
        .collect::<Vec<Type>>();
    let ty = Type::Pointer {
        ty: Box::new(Type::Function {
            result: Box::new(Type::Void),
            args: param_types
        }),
        mem: MemSpace::Host
    };
    Param {id, ty, i}
}

fn insert_callback_parameters(
    params: Vec<Param>,
    used_callbacks: Callbacks
) -> Vec<Param> {
    params.into_iter()
        .chain(used_callbacks.into_iter()
            .map(|(id, (params, i))| to_callback_parameter(id, params, i)))
        .collect::<Vec<Param>>()
}

fn add_callback_parameters(
    t: Top,
    callbacks: &Callbacks
) -> CompileResult<Top> {
    match t {
        Top::FunDef {ret_ty, id, params, body, target: Target::Host, i} => {
            let used_callbacks = body.sfold(BTreeMap::new(), |acc, s| {
                collect_used_callbacks_stmt(acc, s, &callbacks)
            });
            let params = insert_callback_parameters(params, used_callbacks);
            Ok(Top::FunDef {ret_ty, id, params, body, target: Target::Host, i})
        },
        Top::KernelFunDef {ref body, ref i, ..} |
        Top::FunDef {ref body, ref i, ..} => {
            let used_callbacks = body.sfold(BTreeMap::new(), |acc, s| {
                collect_used_callbacks_stmt(acc, s, &callbacks)
            });
            if used_callbacks.is_empty() {
                Ok(t)
            } else {
                parpy_compile_error!(i, "Callback functions can only be used \
                                         from the main function.")
            }
        },
        _ => Ok(t)
    }
}

fn remove_callback_decls_top(mut acc: Vec<Top>, t: Top) -> Vec<Top> {
    match t {
        Top::CallbackDecl {..} => acc,
        _ => {
            acc.push(t);
            acc
        }
    }
}

pub fn apply(ast: Ast) -> CompileResult<Ast> {
    let callbacks = ast.sfold(BTreeMap::new(), collect_callbacks_top);
    let ast = ast.smap_result(|t| add_callback_parameters(t, &callbacks))?;
    Ok(ast.sfold_owned(vec![], remove_callback_decls_top))
}
