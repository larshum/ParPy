use super::ast::*;
use crate::utils::ast::ExprType;
use crate::utils::info::*;
use crate::utils::name::Name;

pub fn scalar(sz: ElemSize) -> Type {
    Type::Scalar {sz}
}

pub fn i64_ty() -> Type {
    scalar(ElemSize::I64)
}

pub fn id(x: &str) -> Name {
    Name::new(x.to_string())
}

pub fn var(v: &str, ty: Type) -> Expr {
    Expr::Var {id: id(v), ty, i: Info::default()}
}

pub fn int(v: i64, sz: ElemSize) -> Expr {
    Expr::Int {v: v as i128, ty: scalar(sz), i: Info::default()}
}

pub fn binop(lhs: Expr, op: BinOp, rhs: Expr, ty: Type) -> Expr {
    Expr::BinOp {lhs: Box::new(lhs), op, rhs: Box::new(rhs), ty, i: Info::default()}
}

pub fn add(lhs: Expr, rhs: Expr, ty: Type) -> Expr {
    binop(lhs, BinOp::Add, rhs, ty)
}

pub fn mul(lhs: Expr, rhs: Expr, ty: Type) -> Expr {
    binop(lhs, BinOp::Mul, rhs, ty)
}

pub fn rem(lhs: Expr, rhs: Expr, ty: Type) -> Expr {
    binop(lhs, BinOp::Rem, rhs, ty)
}

pub fn max(lhs: Expr, rhs: Expr, ty: Type) -> Expr {
    binop(lhs, BinOp::Max, rhs, ty)
}

pub fn defn(ty: Type, id: Name, expr: Option<Expr>) -> Stmt {
    Stmt::Definition {ty, id, expr}
}

pub fn assign(dst: Expr, expr: Expr) -> Stmt {
    let ty = expr.get_type().clone();
    Stmt::Expr {
        e: Expr::Assign {
            lhs: Box::new(dst),
            rhs: Box::new(expr),
            ty,
            i: Info::default()
        }
    }
}
