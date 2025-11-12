use super::ast::*;
use crate::utils::info::Info;
use crate::utils::smap::SMapAccum;

fn find_thread_count_attr(acc: Option<i128>, attr: &KernelAttribute) -> Option<i128> {
    match attr {
        KernelAttribute::LaunchBounds {threads} => Some(*threads as i128),
        _ => acc
    }
}

fn find_thread_count(attrs: &Vec<KernelAttribute>) -> Option<i128> {
    attrs.iter().fold(None, find_thread_count_attr)
}

fn insert_thread_hints_top(t: Top) -> Top {
    match t {
        Top::FunDef {
            dev_attr: dev_attr @ Attribute::Global, ret_ty, attrs, id, params, mut body
        } => {
            // If the thread count is specified, we specify that the compiler may assume we have no
            // more than that number of threads in the kernel.
            if let Some (nthreads) = find_thread_count(&attrs) {
                let thread_idx = Expr::ThreadIdx {
                    dim: Dim::X,
                    ty: Type::Scalar {sz: ElemSize::U32},
                    i: Info::default()
                };
                let thread_count = Expr::Int {
                    v: nthreads,
                    ty: Type::Scalar {sz: ElemSize::U32},
                    i: Info::default()
                };
                let thread_constraint = Expr::BinOp {
                    lhs: Box::new(thread_idx),
                    op: BinOp::Lt,
                    rhs: Box::new(thread_count),
                    ty: Type::Scalar {sz: ElemSize::Bool},
                    i: Info::default()
                };
                body.insert(0, Stmt::BuiltinAssume {e: thread_constraint});
            }
            Top::FunDef {dev_attr, ret_ty, attrs, id, params, body}
        },
        _ => t
    }
}

pub fn insert_thread_hints(ast: Ast) -> Ast {
    ast.smap(insert_thread_hints_top)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test::*;
    use crate::cuda::ast_builder::*;

    #[test]
    fn insert_thread_hints_non_kernel_top() {
        let t = Top::FunDef {
            dev_attr: Attribute::Device,
            ret_ty: Type::Void,
            attrs: vec![],
            id: id("x"),
            params: vec![],
            body: vec![]
        };
        assert_eq!(insert_thread_hints_top(t.clone()), t);
    }

    #[test]
    fn insert_thread_hints_kernel_top() {
        let t = Top::FunDef {
            dev_attr: Attribute::Global,
            ret_ty: scalar(ElemSize::I64),
            attrs: vec![KernelAttribute::LaunchBounds {threads: 128}],
            id: id("f"),
            params: vec![],
            body: vec![Stmt::Return {value: int(2, ElemSize::I64)}]
        };
        let constraint = binop(
            Expr::ThreadIdx {dim: Dim::X, ty: scalar(ElemSize::U32), i: i()},
            BinOp::Lt,
            int(128, ElemSize::U32),
            scalar(ElemSize::Bool)
        );
        let expected = Top::FunDef {
            dev_attr: Attribute::Global,
            ret_ty: scalar(ElemSize::I64),
            attrs: vec![KernelAttribute::LaunchBounds {threads: 128}],
            id: id("f"),
            params: vec![],
            body: vec![
                Stmt::BuiltinAssume {e: constraint},
                Stmt::Return {value: int(2, ElemSize::I64)}
            ]
        };
        assert_eq!(insert_thread_hints_top(t), expected);
    }
}
