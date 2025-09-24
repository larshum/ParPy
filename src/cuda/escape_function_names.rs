use super::ast::*;
use crate::utils::name::Name;
use crate::utils::smap::SMapAccum;

fn escape_name(id: Name) -> Name {
    let Name {s, sym} = id;
    Name {s: format!("parpy_{s}"), sym}
}

fn escape_expr(e: Expr) -> Expr {
    match e {
        Expr::Call {id, args, ty, i} => {
            let id = escape_name(id);
            let args = args.smap(escape_expr);
            Expr::Call {id, args, ty, i}
        },
        _ => e.smap(escape_expr)
    }
}

fn escape_stmt(s: Stmt) -> Stmt {
    match s {
        Stmt::KernelLaunch {id, blocks, threads, stream, args} => {
            let id = escape_name(id);
            let args = args.smap(escape_expr);
            Stmt::KernelLaunch {id, blocks, threads, stream, args}
        },
        _ => s.smap(escape_stmt).smap(escape_expr)
    }
}

fn escape_top(t: Top) -> Top {
    match t {
        Top::ExtDecl {ret_ty, id, ext_id, params} => {
            let id = escape_name(id);
            Top::ExtDecl {ret_ty, id, ext_id, params}
        },
        Top::FunDef {dev_attr, ret_ty, attrs, id, params, body} => {
            let body = body.smap(escape_stmt);
            let id = match &dev_attr {
                Attribute::Global | Attribute::Device => escape_name(id),
                Attribute::Entry => id
            };
            Top::FunDef {dev_attr, ret_ty, attrs, id, params, body}
        },
        _ => t
    }
}

pub fn apply(ast: Ast) -> Ast {
    ast.smap(escape_top)
}
