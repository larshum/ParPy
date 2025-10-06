use super::ast::*;
use crate::utils::ast::ExprType;
use crate::utils::free_vars::*;
use crate::utils::name::Name;
use crate::utils::smap::SFold;

use std::collections::BTreeMap;

impl FreeVars<Type> for Expr {
    fn fv(&self, env: FVEnv<Type>) -> FVEnv<Type> {
        match self {
            Expr::Var {id, ty, ..} => use_variable(env, &id, &ty),
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
