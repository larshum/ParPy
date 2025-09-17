/// Replaces any remaining use of builtin operations with other expression nodes, based on the
/// number of arguments they are provided. Specifically:
/// - Reduction builtins are replaced with a dedicated 'ReduceOp' node.
/// - Other builtins accepting one argument are replaced with a 'UnOp' node.
/// - Builtins accepting two arguments are replaced with a 'BinOp' node.
/// - Builtins accepting zero arguments are replaced with literal expressions.
/// - Other kinds of builtins are not supported; we report an error in this case.

use super::ast::*;
use crate::py_runtime_error;
use crate::utils::err::*;
use crate::utils::info::Info;
use crate::utils::smap::*;

use pyo3::prelude::*;

fn replace_literal_builtin(func: Builtin, i: Info) -> PyResult<Expr> {
    match func {
        Builtin::Inf => Ok(Expr::Float {v: f64::INFINITY, ty: Type::Unknown, i}),
        Builtin::Exp | Builtin::Log | Builtin::Max | Builtin::Min | Builtin::Abs |
        Builtin::Cos | Builtin::Sin | Builtin::Sqrt | Builtin::Tanh |
        Builtin::Atan2 | Builtin::Sum | Builtin::Prod | Builtin::Convert {..} |
        Builtin::Label | Builtin::GpuContext => {
            py_runtime_error!(i, "Builtin {func} does not accept zero arguments.")
        }
    }
}

fn to_unop(func: Builtin, i: &Info) -> PyResult<UnOp> {
    match func {
        Builtin::Exp => Ok(UnOp::Exp),
        Builtin::Log => Ok(UnOp::Log),
        Builtin::Cos => Ok(UnOp::Cos),
        Builtin::Sin => Ok(UnOp::Sin),
        Builtin::Sqrt => Ok(UnOp::Sqrt),
        Builtin::Tanh => Ok(UnOp::Tanh),
        Builtin::Abs => Ok(UnOp::Abs),
        Builtin::Inf | Builtin::Max | Builtin::Min | Builtin::Atan2 |
        Builtin::Sum | Builtin::Prod | Builtin::Convert {..} | Builtin::Label |
        Builtin::GpuContext => {
            py_runtime_error!(i, "Builtin {func} does not accept one argument.")
        }
    }
}

fn replace_unary_builtin(
    func: Builtin,
    arg: Expr,
    i: Info
) -> PyResult<Expr> {
    let arg = Box::new(arg);
    let ty = Type::Unknown;
    match func {
        Builtin::Max => Ok(Expr::ReduceOp {op: ReduceOp::Max, arg, ty, i}),
        Builtin::Min => Ok(Expr::ReduceOp {op: ReduceOp::Min, arg, ty, i}),
        Builtin::Sum => Ok(Expr::ReduceOp {op: ReduceOp::Sum, arg, ty, i}),
        Builtin::Prod => Ok(Expr::ReduceOp {op: ReduceOp::Prod, arg, ty, i}),
        Builtin::Convert {sz} => {
            let ty = Type::Tensor {sz, shape: vec![]};
            Ok(Expr::Convert {e: arg, ty})
        },
        _ => {
            let op = to_unop(func, &i)?;
            Ok(Expr::UnOp {op, arg, ty, i})
        }
    }
}

fn to_binop(func: Builtin, i: &Info) -> PyResult<BinOp> {
    match func {
        Builtin::Max => Ok(BinOp::Max),
        Builtin::Min => Ok(BinOp::Min),
        Builtin::Atan2 => Ok(BinOp::Atan2),
        Builtin::Exp | Builtin::Inf | Builtin::Log | Builtin::Abs | Builtin::Cos |
        Builtin::Sin | Builtin::Sqrt | Builtin::Tanh | Builtin::Sum | Builtin::Prod |
        Builtin::Convert {..} | Builtin::Label | Builtin::GpuContext => {
            py_runtime_error!(i, "Builtin {func} does not accept two arguments.")
        }
    }
}

fn replace_binary_builtin(
    func: Builtin,
    lhs: Expr,
    rhs: Expr,
    i: Info
) -> PyResult<Expr> {
    let op = to_binop(func, &i)?;
    let ty = Type::Unknown;
    Ok(Expr::BinOp {lhs: Box::new(lhs), op, rhs: Box::new(rhs), ty, i})
}

fn replace_builtins_expr(e: Expr) -> PyResult<Expr> {
    match e {
        Expr::Builtin {func, args, ty: _, i} => {
            let mut args = args.smap_result(replace_builtins_expr)?;
            match args.len() {
                0 => replace_literal_builtin(func, i),
                1 => {
                    let arg = args.pop().unwrap();
                    replace_unary_builtin(func, arg, i)
                },
                2 => {
                    let rhs = args.pop().unwrap();
                    let lhs = args.pop().unwrap();
                    replace_binary_builtin(func, lhs, rhs, i)
                },
                n => py_runtime_error!(i, "Builtin {func} does not accept {n} arguments.")
            }
        },
        _ => e.smap_result(replace_builtins_expr)
    }
}

fn replace_builtins_stmt(s: Stmt) -> PyResult<Stmt> {
    s.smap_result(replace_builtins_stmt)?
        .smap_result(replace_builtins_expr)
}

pub fn apply(def: FunDef) -> PyResult<FunDef> {
    let body = def.body.smap_result(replace_builtins_stmt)?;
    Ok(FunDef {body, ..def})
}
