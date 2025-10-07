use super::ast::*;
use crate::py::ast as py_ast;
use crate::utils::ast::ExprType;
use crate::utils::free_vars::*;
use crate::utils::name::Name;
use crate::utils::smap::SFold;

use std::collections::BTreeMap;

fn copy_as_py_env(env: &FVEnv<Type>) -> FVEnv<py_ast::Type> {
    let mut py_env = FVEnv::<py_ast::Type>::default();
    let insert_id = |env: &mut BTreeMap<Name, py_ast::Type>, id: &Name| {
        env.insert(id.clone(), py_ast::Type::Void);
    };
    env.bound.iter().for_each(|(id, _)| insert_id(&mut py_env.bound, id));
    env.free.iter().for_each(|(id, _)| insert_id(&mut py_env.free, id));
    py_env
}

fn convert_to_gpu_type(py_type: py_ast::Type) -> Type {
    // NOTE(larshum, 2025-10-06): We use 'Void' to represent any type we cannot immediately
    // translate from the Python AST to the GPU AST. As this is currently only used for
    // callback functions, where we only expect tensor arguments, this should be fine.
    match py_type {
        py_ast::Type::Tensor {sz, shape} => {
            match sz {
                py_ast::TensorElemSize::Fixed {sz} => {
                    if shape.is_empty() {
                        Type::Scalar {sz}
                    } else {
                        let ty = Box::new(Type::Scalar {sz});
                        Type::Pointer {ty, mem: MemSpace::Device}
                    }
                },
                py_ast::TensorElemSize::Variable {..} => Type::Void
            }
        },
        py_ast::Type::Void => Type::Void,
        _ => Type::Void,
    }
}

fn add_to_gpu_env(mut env: FVEnv<Type>, py_env: FVEnv<py_ast::Type>) -> FVEnv<Type> {
    let add_entry_if_new = |env: &mut BTreeMap<Name, Type>, id: Name, py_type: py_ast::Type| {
        env.entry(id).or_insert_with(|| convert_to_gpu_type(py_type));
    };
    py_env.bound.into_iter()
        .for_each(|(id, py_type)| add_entry_if_new(&mut env.bound, id, py_type));
    py_env.free.into_iter()
        .for_each(|(id, py_type)| add_entry_if_new(&mut env.free, id, py_type));
    env
}

impl FreeVars<py_ast::Type> for Expr {
    fn fv(&self, env: FVEnv<py_ast::Type>) -> FVEnv<py_ast::Type> {
        match self {
            Expr::Var {id, ..} => use_variable(env, &id, &py_ast::Type::Unknown),
            Expr::PyCallback {args, ..} => {
                args.sfold(env, |env, e: &py_ast::Expr| e.fv(env))
            },
            Expr::Bool {..} | Expr::Int {..} | Expr::Float {..} |
            Expr::UnOp {..} | Expr::BinOp {..} | Expr::IfExpr {..} |
            Expr::StructFieldAccess {..} | Expr::ArrayAccess {..} |
            Expr::Call {..} | Expr::Convert {..} | Expr::Struct {..} |
            Expr::ThreadIdx {..} | Expr::BlockIdx {..} => {
                self.sfold(env, |env, e| e.fv(env))
            }
        }
    }
}

impl FreeVars<py_ast::Type> for Stmt {
    fn fv(&self, env: FVEnv<py_ast::Type>) -> FVEnv<py_ast::Type> {
        match self {
            Stmt::Definition {id, expr, ..} => {
                let env = expr.fv(env);
                bind_variable(env, &id, &py_ast::Type::Unknown)
            },
            Stmt::For {var, init, cond, incr, body, ..} |
            Stmt::ParallelReduction {var, init, cond, incr, body, ..} => {
                let env = init.fv(env);
                let env = bind_variable(env, &var, &py_ast::Type::Unknown);
                let env = cond.fv(env);
                let env = incr.fv(env);
                body.sfold(env, |env, s| s.fv(env))
            },
            Stmt::AllocShared {id, ..} => {
                bind_variable(env, &id, &py_ast::Type::Unknown)
            },
            Stmt::Assign {..} | Stmt::If {..} | Stmt::While {..} |
            Stmt::Return {..} | Stmt::Scope {..} | Stmt::Expr {..} |
            Stmt::Synchronize {..} | Stmt::WarpReduce {..} |
            Stmt::ClusterReduce {..} | Stmt::KernelLaunch {..} |
            Stmt::AllocDevice {..} | Stmt::FreeDevice {..} |
            Stmt::CopyMemory {..} => {
                let env = self.sfold(env, |env, s: &Stmt| s.fv(env));
                self.sfold(env, |env, e: &Expr| e.fv(env))
            }
        }
    }
}

impl FreeVars<Type> for Expr {
    fn fv(&self, env: FVEnv<Type>) -> FVEnv<Type> {
        match self {
            Expr::Var {id, ty, ..} => use_variable(env, &id, &ty),
            Expr::PyCallback {args, ..} => {
                let py_env = copy_as_py_env(&env);
                let py_env = args.sfold(py_env, |env, e: &py_ast::Expr| e.fv(env));
                let env = add_to_gpu_env(env, py_env);
                env
            },
            Expr::Bool {..} | Expr::Int {..} | Expr::Float {..} |
            Expr::UnOp {..} | Expr::BinOp {..} | Expr::IfExpr {..} |
            Expr::StructFieldAccess {..} | Expr::ArrayAccess {..} |
            Expr::Call {..} | Expr::Convert {..} | Expr::Struct {..} |
            Expr::ThreadIdx {..} | Expr::BlockIdx {..} => {
                self.sfold(env, |env, e| e.fv(env))
            }
        }
    }
}

impl FreeVars<Type> for Stmt {
    fn fv(&self, env: FVEnv<Type>) -> FVEnv<Type> {
        match self {
            Stmt::Definition {id, expr, ..} => {
                let env = expr.fv(env);
                bind_variable(env, &id, expr.get_type())
            },
            Stmt::For {var, init, cond, incr, body, ..} |
            Stmt::ParallelReduction {var, init, cond, incr, body, ..} => {
                let env = init.fv(env);
                let env = bind_variable(env, &var, init.get_type());
                let env = cond.fv(env);
                let env = incr.fv(env);
                body.sfold(env, |env, s| s.fv(env))
            },
            Stmt::AllocShared {elem_ty, id, ..} => {
                bind_variable(env, &id, &elem_ty)
            },
            Stmt::Assign {..} | Stmt::If {..} | Stmt::While {..} |
            Stmt::Return {..} | Stmt::Scope {..} | Stmt::Expr {..} |
            Stmt::Synchronize {..} | Stmt::WarpReduce {..} |
            Stmt::ClusterReduce {..} | Stmt::KernelLaunch {..} |
            Stmt::AllocDevice {..} | Stmt::FreeDevice {..} |
            Stmt::CopyMemory {..} => {
                let env = self.sfold(env, |env, s: &Stmt| s.fv(env));
                self.sfold(env, |env, e: &Expr| e.fv(env))
            }
        }
    }
}

pub fn free_variables(s: &Vec<Stmt>) -> BTreeMap<Name, Type> {
    let env = s.sfold(FVEnv::default(), |env, s| s.fv(env));
    env.free
}
