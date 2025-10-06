use super::ast::*;
use crate::utils::free_vars::*;

impl FreeVars<Type> for Expr {
    fn fv(&self, env: FVEnv<Type>) -> FVEnv<Type> {
        match self {
            Expr::Var {id, ty, ..} => use_variable(env, &id, &ty),
            Expr::String {..} | Expr::Bool {..} | Expr::Int {..} |
            Expr::Float {..} | Expr::UnOp {..} | Expr::BinOp {..} |
            Expr::ReduceOp {..} | Expr::IfExpr {..} | Expr::Subscript {..} |
            Expr::Slice {..} | Expr::Tuple {..} | Expr::Call {..} |
            Expr::Convert {..} | Expr::GpuContext {..} | Expr::Inline {..} |
            Expr::Label {..} | Expr::StaticBackendEq {..} |
            Expr::StaticTypesEq {..} | Expr::StaticFail {..} => {
                self.sfold(env, |env, e| e.fv(env))
            }
        }
    }
}
