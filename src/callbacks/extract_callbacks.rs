use crate::parpy_compile_error;
use crate::gpu::ast::*;
use crate::py::ast as py_ast;
use crate::utils::ast::ExprType;
use crate::utils::err::*;
use crate::utils::free_vars::*;
use crate::utils::info::*;
use crate::utils::name::Name;
use crate::utils::smap::*;

#[derive(Clone, Debug)]
pub struct Callback {
    pub id: Name,
    pub params: Vec<py_ast::Param>,
    pub ty: Type,
    pub body: Stmt,
}

fn extract_callback(body: &Vec<Stmt>) -> Option<(Name, Info)> {
    match &body[..] {
        [Stmt::Expr {e: Expr::PyCallback {id, i, ..}, ..}] => {
            Some((id.clone(), i.clone()))
        },
        [Stmt::For {ref body, ..}] => extract_callback(&body),
        _ => None
    }
}

fn py_to_gpu_type(ty: py_ast::Type) -> Type {
    // Converts a Python AST type to a GPU AST type. As arguments to a callback function have to be
    // tensors (scalar or non-scalar), this should work correctly for supported types. If a
    // user-defined callback function expects another kind of type, this should have been caught
    // elsewhere in the compiler.
    match ty {
        py_ast::Type::Tensor {sz: py_ast::TensorElemSize::Fixed {sz}, shape} => {
            if shape.is_empty() {
                Type::Scalar {sz}
            } else {
                let ty = Box::new(Type::Scalar {sz});
                Type::Pointer {ty, mem: MemSpace::Device}
            }
        },
        py_ast::Type::Void => Type::Void,
        _ => Type::Void
    }
}

fn generate_callback(
    s: Stmt,
    callback_id: Name,
    i: Info
) -> (Callback, Stmt) {
    let fvs = s.free_vars();
    let params = fvs.clone()
        .into_iter()
        .map(|(id, ty)| py_ast::Param {id, ty, i: Info::default()})
        .collect::<Vec<py_ast::Param>>();
    let fv_args = fvs.into_iter()
        .map(|(id, ty)| Expr::Var {
            id: id.clone(),
            ty: py_to_gpu_type(ty.clone()),
            i: Info::default()
        })
        .collect::<Vec<Expr>>();
    let arg_types = fv_args.iter()
        .map(|e| e.get_type().clone())
        .collect::<Vec<Type>>();
    let ty = Type::Pointer {
        ty: Box::new(Type::Function {
            result: Box::new(Type::Void),
            args: arg_types
        }),
        mem: MemSpace::Host
    };
    let callback_wrap_stmt = Stmt::Expr {
        e: Expr::Call {
            id: callback_id.clone(),
            args: fv_args,
            ty: Type::Void,
            i: i.clone()
        },
        i
    };
    ( Callback {id: callback_id, params, ty, body: s}
    , callback_wrap_stmt )
}

fn collect_used_callbacks_stmt(
    mut acc: Vec<Callback>,
    s: Stmt
) -> (Vec<Callback>, Stmt) {
    match s {
        Stmt::For {ref body, ..} => {
            if let Some((id, i)) = extract_callback(&body) {
                let (callback, callback_wrap_stmt) = generate_callback(s, id, i);
                acc.push(callback);
                (acc, callback_wrap_stmt)
            } else {
                s.smap_accum_l(acc, collect_used_callbacks_stmt)
            }
        },
        Stmt::Expr {ref e, ..} => {
            if let Expr::PyCallback {id, i, ..} = &e {
                let (id, i) = (id.clone(), i.clone());
                let (callback, callback_wrap_stmt) = generate_callback(s, id, i);
                acc.push(callback);
                (acc, callback_wrap_stmt)
            } else {
                s.smap_accum_l(acc, collect_used_callbacks_stmt)
            }
        },
        _ => s.smap_accum_l(acc, collect_used_callbacks_stmt)
    }
}

fn insert_callback_parameters(
    params: Vec<Param>,
    callbacks: Vec<Callback>,
) -> Vec<Param> {
    params.into_iter()
        .chain(callbacks.into_iter()
            .map(|Callback {id, ty, body, ..}| {
                Param {id, ty, i: body.get_info()}
            }))
        .collect::<Vec<Param>>()
}

fn add_callback_parameters(
    mut acc: Vec<Callback>,
    t: Top
) -> CompileResult<(Vec<Callback>, Top)> {
    match t {
        Top::FunDef {ret_ty, id, params, body, target: Target::Host, i} => {
            let (mut cbs, body) = body.smap_accum_l(vec![], collect_used_callbacks_stmt);
            let params = insert_callback_parameters(params, cbs.clone());
            acc.append(&mut cbs);
            Ok((acc, Top::FunDef {ret_ty, id, params, body, target: Target::Host, i}))
        },
        Top::KernelFunDef {ref body, ref i, ..} |
        Top::FunDef {ref body, ref i, ..} => {
            let (cbs, _) = body.clone().smap_accum_l(vec![], collect_used_callbacks_stmt);
            if cbs.is_empty() {
                Ok((acc, t))
            } else {
                parpy_compile_error!(i, "Callback functions can only be used \
                                         from the main function.")
            }
        },
        _ => Ok((acc, t))
    }
}

pub fn apply(ast: Ast) -> CompileResult<(Vec<Callback>, Ast)> {
    ast.smap_accum_l_result(Ok(vec![]), |acc, t| {
        add_callback_parameters(acc, t)
    })
}
