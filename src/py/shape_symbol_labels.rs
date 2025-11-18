/// Implicitly adds labels to parallelizable statements that directly make use of shape symbols.
/// Assume we have an assignment statement of the form a[:N] += b[:N], where 'N' refers to a shape
/// symbol. In this case, this transformation will add a label 'N' to the assignment statement,
/// unless the 'implicit_shape_labels' compiler option is set to false.

use super::ast::*;
use crate::option::CompileOptions;
use crate::utils::name::*;
use crate::utils::smap::*;

use std::collections::BTreeSet;

fn collect_shape_variables_shape(
    mut acc: BTreeSet<Name>,
    sh: &TensorShape
) -> BTreeSet<Name> {
    match sh {
        TensorShape::Symbol {id} => {
            acc.insert(id.clone());
            acc
        },
        TensorShape::Num {..} => acc
    }
}

fn collect_shape_variables_type(acc: BTreeSet<Name>, ty: &Type) -> BTreeSet<Name> {
    match ty {
        Type::Tensor {shape, ..} => shape.sfold(acc, collect_shape_variables_shape),
        _ => ty.sfold(acc, collect_shape_variables_type)
    }
}

fn collect_shape_variables(acc: BTreeSet<Name>, p: &Param) -> BTreeSet<Name> {
    collect_shape_variables_type(acc, &p.ty)
}

fn collect_implicit_labels_helper(
    shape_vars: &BTreeSet<Name>,
    mut labels: BTreeSet<String>,
    e: &Expr
) -> BTreeSet<String> {
    match e {
        Expr::Var {id, ..} if shape_vars.contains(id) => {
            labels.insert(id.get_str().clone());
            labels
        },
        _ => e.sfold(labels, |labels, e| {
            collect_implicit_labels_helper(shape_vars, labels, e)
        })
    }
}

fn collect_implicit_labels(shape_vars: &BTreeSet<Name>, e: &Expr) -> Vec<String> {
    let l = collect_implicit_labels_helper(shape_vars, BTreeSet::new(), e);
    l.into_iter().collect::<Vec<String>>()
}

fn add_implicit_labels_stmt(
    shape_vars: &BTreeSet<Name>,
    s: Stmt
) -> Stmt {
    let add_implicit_labels = |mut labels: Vec<String>, e: &Expr| -> Vec<String> {
        labels.append(&mut collect_implicit_labels(&shape_vars, e));
        labels
    };
    match s {
        Stmt::Definition {ty, id, expr, labels, i} => {
            let labels = add_implicit_labels(labels, &expr);
            Stmt::Definition {ty, id, expr, labels, i}
        },
        Stmt::Assign {dst, expr, labels, i} => {
            let labels = add_implicit_labels(labels, &dst);
            let labels = add_implicit_labels(labels, &expr);
            Stmt::Assign {dst, expr, labels, i}
        },
        Stmt::For {var, lo, hi, step, body, labels, i} => {
            let labels = add_implicit_labels(labels, &lo);
            let labels = add_implicit_labels(labels, &hi);
            let body = body.smap(|s| add_implicit_labels_stmt(shape_vars, s));
            Stmt::For {var, lo, hi, step, body, labels, i}
        },
        _ => s.smap(|s| add_implicit_labels_stmt(shape_vars, s))
    }
}

pub fn add_implicit_labels(opts: &CompileOptions, def: FunDef) -> FunDef {
    if opts.implicit_shape_labels {
        let shape_vars = def.params.sfold(BTreeSet::new(), collect_shape_variables);
        let body = def.body.smap(|s| add_implicit_labels_stmt(&shape_vars, s));
        FunDef {body, ..def}
    } else {
        def
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::py::ast_builder::*;
    use crate::test::*;

    fn ts_sym(id: Name) -> TensorShape {
        TensorShape::Symbol {id}
    }

    #[test]
    fn collect_shape_variables_empty() {
        let p = Param {id: id("x"), ty: tyuk(), i: i()};
        let shape_vars = collect_shape_variables(BTreeSet::new(), &p);
        assert!(shape_vars.is_empty());
    }

    #[test]
    fn collect_shape_variables_single_shape() {
        let n = Name::sym_str("");
        let ty = Type::Tensor {
            sz: fixed_elem_sz(ElemSize::I32), shape: vec![ts_sym(n.clone())]
        };
        let p = Param {id: id("x"), ty, i: i()};
        let shape_vars = collect_shape_variables(BTreeSet::new(), &p);
        assert_eq!(shape_vars.len(), 1);
        assert!(shape_vars.contains(&n));
    }

    #[test]
    fn collect_shape_variables_repeated() {
        let n = Name::sym_str("");
        let ty = Type::Tensor {
            sz: fixed_elem_sz(ElemSize::I32),
            shape: vec![ts_sym(n.clone()), ts_sym(n.clone())]
        };
        let p = Param {id: id("x"), ty, i: i()};
        let shape_vars = collect_shape_variables(BTreeSet::new(), &p);
        assert_eq!(shape_vars.len(), 1);
        assert!(shape_vars.contains(&n));
    }

    #[test]
    fn collect_shape_variables_multiple() {
        let n1 = Name::sym_str("");
        let n2 = Name::sym_str("");
        let ty = Type::Tensor {
            sz: fixed_elem_sz(ElemSize::I32),
            shape: vec![ts_sym(n1.clone()), ts_sym(n2.clone())]
        };
        let p = Param {id: id("x"), ty, i: i()};
        let shape_vars = collect_shape_variables(BTreeSet::new(), &p);
        assert_eq!(shape_vars.len(), 2);
        assert!(shape_vars.contains(&n1));
        assert!(shape_vars.contains(&n2));
    }

    fn mk_shape_vars(v: Vec<Name>) -> BTreeSet<Name> {
        v.into_iter().collect::<_>()
    }

    #[test]
    fn add_implicit_label_definition() {
        let n = Name::sym_str("N");
        let shape_vars = mk_shape_vars(vec![n.clone()]);
        let s = Stmt::Definition {
            ty: tyuk(),
            id: id("x"),
            expr: Expr::Var {id: n.clone(), ty: tyuk(), i: i()},
            labels: vec![],
            i: i()
        };
        if let Stmt::Definition {labels, ..} = add_implicit_labels_stmt(&shape_vars, s) {
            assert_eq!(labels, vec!["N".to_string()]);
        } else {
            assert!(false);
        }
    }

    #[test]
    fn add_implicit_label_assign() {
        let n = Name::sym_str("N");
        let shape_vars = mk_shape_vars(vec![n.clone()]);
        let s = Stmt::Assign {
            dst: var("x", tyuk()),
            expr: Expr::Var {id: n.clone(), ty: tyuk(), i: i()},
            labels: vec![],
            i: i()
        };
        if let Stmt::Assign {labels, ..} = add_implicit_labels_stmt(&shape_vars, s) {
            assert_eq!(labels, vec!["N".to_string()]);
        } else {
            assert!(false);
        }
    }

    #[test]
    fn add_implicit_label_for_bounds() {
        let n = Name::sym_str("N");
        let m = Name::sym_str("M");
        let shape_vars = mk_shape_vars(vec![n.clone(), m.clone()]);
        let s = Stmt::For {
            var: id("i"),
            lo: Expr::Var {id: n.clone(), ty: tyuk(), i: i()},
            hi: Expr::Var {id: m.clone(), ty: tyuk(), i: i()},
            step: int(1, None),
            body: vec![],
            labels: vec![],
            i: i()
        };
        if let Stmt::For {labels, ..} = add_implicit_labels_stmt(&shape_vars, s) {
            assert_eq!(labels, vec!["N".to_string(), "M".to_string()]);
        } else {
            assert!(false);
        }
    }

    #[test]
    fn add_implicit_label_nested_for_loop_bounds() {
        let n = Name::sym_str("N");
        let shape_vars = mk_shape_vars(vec![n.clone()]);
        let inner_loop = Stmt::For {
            var: id("j"),
            lo: int(1, None),
            hi: binop(
                Expr::Var {id: n.clone(), ty: tyuk(), i: i()},
                BinOp::Sub,
                int(1, None),
                tyuk()
            ),
            step: int(1, None),
            body: vec![],
            labels: vec![],
            i: i()
        };
        let s = Stmt::For {
            var: id("i"),
            lo: int(1, None),
            hi: int(10, None),
            step: int(1, None),
            body: vec![inner_loop],
            labels: vec![],
            i: i()
        };
        if let Stmt::For {labels, mut body, ..} = add_implicit_labels_stmt(&shape_vars, s) {
            assert!(labels.is_empty());
            let s = body.pop().unwrap();
            if let Stmt::For {labels, ..} = s {
                assert_eq!(labels, vec!["N".to_string()]);
            } else {
                assert!(false);
            }
        } else {
            assert!(false);
        }
    }

    #[test]
    fn collect_no_duplicated_labels() {
        let n = Name::sym_str("N");
        let shape_vars = mk_shape_vars(vec![n.clone()]);
        let v = Expr::Var {id: n, ty: tyuk(), i: i()};
        let idx_expr = Expr::Tuple {
            elems: vec![
                Expr::Slice {
                    lo: None,
                    hi: Some(Box::new(v.clone())),
                    ty: tyuk(),
                    i: i()
                },
                Expr::Slice {
                    lo: None,
                    hi: Some(Box::new(v)),
                    ty: tyuk(),
                    i: i()
                },
            ],
            ty: tyuk(),
            i: i()
        };
        let labels = collect_implicit_labels(&shape_vars, &idx_expr);
        assert_eq!(labels, vec!["N".to_string()]);
    }
}
