use crate::py::ast::*;
use crate::utils::ast::ExprType;
use crate::utils::info::*;
use crate::utils::name::Name;

use std::collections::BTreeMap;

pub fn tyuk() -> Type {
    Type::Unknown
}

pub fn fixed_elem_sz(sz: ElemSize) -> TensorElemSize {
    TensorElemSize::Fixed {sz}
}

pub fn shape(v: Vec<i64>) -> Type {
    let sz = fixed_elem_sz(ElemSize::I64);
    let shape = v.into_iter()
        .map(|n| TensorShape::Num {n})
        .collect::<Vec<TensorShape>>();
    Type::Tensor {sz, shape}
}

pub fn scalar(sz: ElemSize) -> Type {
    Type::fixed_scalar(sz)
}

pub fn dict_ty(fields: Vec<(&str, Type)>) -> Type {
    let fields = fields.into_iter()
        .map(|(s, ty)| (s.to_string(), ty))
        .collect::<BTreeMap<String, Type>>();
    Type::Dict {fields}
}

pub fn id(x: &str) -> Name {
    Name::new(x.to_string())
}

pub fn var(v: &str, ty: Type) -> Expr {
    Expr::Var {id: id(v), ty, i: Info::default()}
}

pub fn bool_expr(v: bool, o: Option<ElemSize>) -> Expr {
    let ty = o.map(scalar).unwrap_or(Type::Unknown);
    Expr::Bool {v, ty, i: Info::default()}
}

pub fn int(v: i128, o: Option<ElemSize>) -> Expr {
    let ty = o.map(scalar).unwrap_or(Type::Unknown);
    Expr::Int {v, ty, i: Info::default()}
}

pub fn float(v: f64, o: Option<ElemSize>) -> Expr {
    let ty = o.map(scalar).unwrap_or(Type::Unknown);
    Expr::Float {v, ty, i: Info::default()}
}

pub fn string(v: &str) -> Expr {
    Expr::String {v: v.to_string(), ty: Type::String, i: Info::default()}
}

pub fn unop(op: UnOp, arg: Expr) -> Expr {
    Expr::UnOp {op, arg: Box::new(arg), ty: Type::Unknown, i: Info::default()}
}

pub fn binop(lhs: Expr, op: BinOp, rhs: Expr, ty: Type) -> Expr {
    Expr::BinOp {
        lhs: Box::new(lhs), op, rhs: Box::new(rhs), ty, i: Info::default()
    }
}

pub fn subscript(target: Expr, idx: Expr, ty: Type) -> Expr {
    Expr::Subscript {
        target: Box::new(target), idx: Box::new(idx),
        ty, i: Info::default()
    }
}

pub fn slice(lo: Option<Expr>, hi: Option<Expr>) -> Expr {
    let wrap_box = |o: Option<Expr>| {
        if let Some(e) = o {
            Some(Box::new(e))
        } else {
            None
        }
    };
    let ty = Type::Unknown;
    Expr::Slice {lo: wrap_box(lo), hi: wrap_box(hi), ty, i: Info::default()}
}

pub fn tuple(elems: Vec<Expr>) -> Expr {
    let elem_tys = elems.iter()
        .map(|e| e.get_type().clone())
        .collect::<Vec<Type>>();
    let ty = Type::Tuple {elems: elem_tys};
    Expr::Tuple {elems, ty, i: Info::default()}
}

pub fn call(id: Name, args: Vec<Expr>, ty: Type) -> Expr {
    Expr::Call {id, args, ty, i: Info::default()}
}

pub fn convert(e: Expr, ty: Type) -> Expr {
    Expr::Convert {e: Box::new(e), ty}
}

pub fn assignment(lhs: Expr, rhs: Expr) -> Stmt {
    Stmt::Assign {dst: lhs, expr: rhs, labels: vec![], i: Info::default()}
}

pub fn if_stmt(cond: Expr, thn: Vec<Stmt>, els: Vec<Stmt>) -> Stmt {
    Stmt::If {cond, thn, els, i: Info::default()}
}

pub fn label(l: &str) -> Stmt {
    Stmt::Label {label: l.to_string(), i: Info::default()}
}

pub fn return_stmt(value: Expr) -> Stmt {
    Stmt::Return {value, i: Info::default()}
}
