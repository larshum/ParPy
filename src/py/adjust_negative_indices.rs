use super::ast::*;
use crate::py_internal_error;
use crate::utils::ast::ExprType;
use crate::utils::err::*;
use crate::utils::info::InfoNode;
use crate::utils::smap::SMapAccum;

use pyo3::prelude::*;

fn extract_target_shape(target: &Expr) -> PyResult<Vec<i64>> {
    let i = target.get_info();
    let ty = target.get_type();
    match ty {
        Type::Tensor {shape, ..} => {
            shape.iter()
                .map(|sh| match sh {
                    TensorShape::Num {n} => Ok(*n),
                    TensorShape::Symbol {..} => {
                        py_internal_error!(i, "Found shape symbol after type-checking.")
                    }
                })
                .collect::<PyResult<Vec<i64>>>()
        },
        _ => py_internal_error!(i, "Invalid type of slice target {ty}.")
    }
}

fn add_offset_if_negative_index(idx: Expr, dim: i128) -> Expr {
    match idx {
        Expr::Int {v, ty, i} if v < 0 => Expr::Int {v: v + dim, ty, i},
        _ => idx
    }
}

fn add_offset_to_negative_indices(idx: Expr, shape: Vec<i64>) -> Expr {
    match idx {
        Expr::Tuple {elems, ty, i} => {
            let elems = elems.into_iter()
                .zip(shape.into_iter())
                .map(|(e, dim)| add_offset_if_negative_index(e, dim as i128))
                .collect::<Vec<Expr>>();
            Expr::Tuple {elems, ty, i}
        },
        Expr::Int {..} => add_offset_if_negative_index(idx, shape[0] as i128),
        _ => idx
    }
}

fn apply_expr(e: Expr) -> PyResult<Expr> {
    match e {
        Expr::Subscript {target, idx, ty, i} => {
            if let Type::Tensor {..} = &target.get_type() {
                let shape = extract_target_shape(&target)?;
                let idx = add_offset_to_negative_indices(*idx, shape);
                Ok(Expr::Subscript {target, idx: Box::new(idx), ty, i})
            } else {
                Ok(Expr::Subscript {target, idx, ty, i})
            }
        },
        _ => e.smap_result(apply_expr)
    }
}

fn apply_stmt(s: Stmt) -> PyResult<Stmt> {
    s.smap_result(apply_stmt)?.smap_result(apply_expr)
}

fn apply_def(def: FunDef) -> PyResult<FunDef> {
    let body = def.body.smap_result(apply_stmt)?;
    Ok(FunDef {body, ..def})
}

fn apply_top(t: Top) -> PyResult<Top> {
    match t {
        Top::CallbackDecl {..} | Top::ExtDecl {..} => Ok(t),
        Top::FunDef {v} => Ok(Top::FunDef {v: apply_def(v)?})
    }
}

pub fn apply(ast: Ast) -> PyResult<Ast> {
    Ok(Ast {
        tops: ast.tops.smap_result(apply_top)?,
        main: apply_def(ast.main)?
    })
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test::*;
    use crate::py::ast_builder::*;

    #[test]
    fn single_negative_index() {
        let ty = shape(vec![10]);
        let e = subscript(var("x", ty.clone()), int(-1, None), scalar(ElemSize::I64));
        let expected = subscript(var("x", ty), int(9, None), scalar(ElemSize::I64));
        assert_eq!(apply_expr(e).unwrap(), expected);
    }

    #[test]
    fn negative_second_index() {
        let ty = shape(vec![10, 20]);
        let e = subscript(
            var("x", ty.clone()),
            tuple(vec![int(2, None), int(-2, None)]),
            scalar(ElemSize::I64)
        );
        let expected = subscript(
            var("x", ty),
            tuple(vec![int(2, None), int(18, None)]),
            scalar(ElemSize::I64)
        );
        assert_eq!(apply_expr(e).unwrap(), expected);
    }

    #[test]
    fn multiple_negative_indices() {
        let ty = shape(vec![10, 20, 30]);
        let e = subscript(
            var("x", ty.clone()),
            tuple(vec![int(-1, None), int(2, None), int(-2, None)]),
            scalar(ElemSize::I64)
        );
        let expected = subscript(
            var("x", ty),
            tuple(vec![int(9, None), int(2, None), int(28, None)]),
            scalar(ElemSize::I64)
        );
        assert_eq!(apply_expr(e).unwrap(), expected);
    }

    #[test]
    fn no_operation_on_dict_indexing() {
        let ty = dict_ty(vec![("x", scalar(ElemSize::F32)), ("y", scalar(ElemSize::F32))]);
        let e = subscript(
            var("a", ty.clone()),
            Expr::String {v: "x".to_string(), ty: Type::String, i: i()},
            scalar(ElemSize::F32)
        );
        assert_eq!(apply_expr(e.clone()).unwrap(), e);
    }
}
