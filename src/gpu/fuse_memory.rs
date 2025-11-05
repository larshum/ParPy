/// This file defines a simple form of fusion of memory operations for code running within a single
/// block. When the generated code results in subsequent kernels which operate on the same memory,
/// we may end up performing more operations on global memory than necessary. This transformation
/// detects a situation of the form:
///
///   x[mem] = ...;
///   ... = ... x[mem] ...;
///   x[mem] = ...;
///
/// where we write multiple times to the same memory location, and we read from it at some point
/// in-between these writes. In this case, we can perform the first write to a local variable, and
/// have any reads in-between refer to this local variable. This reduces memory bandwidth usage.

use super::ast::*;
use crate::parpy_internal_error;
use crate::utils::ast::ExprType;
use crate::utils::err::*;
use crate::utils::info::Info;
use crate::utils::name::Name;
use crate::utils::reduce::ExprLit;
use crate::utils::smap::{SFold, SMapAccum};

use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ArrayLoc {
    id: Name,
    idx: Expr
}

#[derive(Debug)]
enum MemOp {
    Write {loc: ArrayLoc, conds: Vec<Expr>},
    Read {loc: ArrayLoc, conds: Vec<Expr>},
}

struct FuseEnv {
    ops: Vec<MemOp>,
    conds: Vec<Expr>,
    shared_mem_vars: BTreeSet<Name>
}

fn memory_operation_location(
    env: &FuseEnv,
    target: &Expr,
    idx: &Expr
) -> Option<ArrayLoc> {
    match target {
        Expr::Var {id, ..} if !env.shared_mem_vars.contains(&id) => {
            Some(ArrayLoc {
                id: id.clone(),
                idx: idx.clone()
            })
        },
        _ => None
    }
}

fn collect_memory_operations_expr(
    mut env: FuseEnv,
    e: &Expr
) -> FuseEnv {
    match e {
        Expr::Assign {lhs, rhs, ..} => {
            let mut env = rhs.sfold(env, collect_memory_operations_expr);
            if let Expr::ArrayAccess {target, idx, ..} = lhs.as_ref() {
                if let Some(loc) = memory_operation_location(&env, &target, &idx) {
                    env.ops.push(MemOp::Write {loc, conds: env.conds.clone()});
                }
            };
            env
        },
        Expr::ArrayAccess {target, idx, ..} => {
            if let Some(loc) = memory_operation_location(&env, &target, &idx) {
                env.ops.push(MemOp::Read {loc, conds: env.conds.clone()});
            }
            env
        },
        Expr::IfExpr {cond, thn, els, ..} => {
            env.conds.push(*cond.clone());
            let mut env = collect_memory_operations_expr(env, thn);
            let c = env.conds.pop().unwrap();
            env.conds.push(Expr::UnOp {
                op: UnOp::Not,
                arg: Box::new(c),
                ty: Type::Scalar {sz: ElemSize::Bool},
                i: Info::default()
            });
            let mut env = collect_memory_operations_expr(env, els);
            env.conds.pop();
            env
        },
        _ => e.sfold(env, collect_memory_operations_expr)
    }
}

fn collect_memory_operations_stmt(
    mut env: FuseEnv,
    s: &Stmt
) -> FuseEnv {
    match s {
        Stmt::If {cond, thn, els, ..} => {
            env.conds.push(cond.clone());
            let mut env = thn.sfold(env, collect_memory_operations_stmt);
            let c = env.conds.pop().unwrap();
            env.conds.push(Expr::UnOp {
                op: UnOp::Not,
                arg: Box::new(c),
                ty: Type::Scalar {sz: ElemSize::Bool},
                i: Info::default()
            });
            let mut env = els.sfold(env, collect_memory_operations_stmt);
            env.conds.pop();
            env
        },
        Stmt::For {cond, body, ..} | Stmt::While {cond, body, ..} => {
            env.conds.push(cond.clone());
            let mut env = body.sfold(env, collect_memory_operations_stmt);
            env.conds.pop();
            env
        },
        Stmt::AllocShared {id, ..} => {
            env.shared_mem_vars.insert(id.clone());
            env
        },
        _ => {
            let env = s.sfold(env, collect_memory_operations_stmt);
            s.sfold(env, collect_memory_operations_expr)
        }
    }
}

fn count_writes_per_index(
    mut counts: BTreeMap<ArrayLoc, (isize, Vec<Expr>)>,
    op: &MemOp
) -> BTreeMap<ArrayLoc, (isize, Vec<Expr>)> {
    // Ensures that all memory operations targeting the same location are performed under exactly
    // the same set of constraints. Otherwise, we cannot ensure the transformation produces a
    // correct program, so we disable it entirely for that memory location.
    match op {
        MemOp::Write {loc, conds} => {
            counts.entry(loc.clone())
                .and_modify(|x| {
                    let (count, old_conds) = x;
                    if *count > 0 && old_conds.eq(&conds) {
                        *count += 1;
                    } else {
                        *count = 0;
                    }
                })
                .or_insert((1, conds.clone()));
            counts
        },
        MemOp::Read {loc, conds} => {
            counts.entry(loc.clone())
                .and_modify(|x| {
                    let (count, old_conds) = x;
                    if *count > 0 && !old_conds.eq(&conds) {
                        *count = 0;
                    }
                });
            counts
        },
    }
}

struct FuseAccessEnv {
    write_count: BTreeMap<ArrayLoc, isize>,
    sub_map: BTreeMap<ArrayLoc, Name>,
}

fn fuse_memory_accesses_expr(
    sub_map: &BTreeMap<ArrayLoc, Name>,
    e: Expr
) -> Expr {
    match e {
        Expr::ArrayAccess {target, idx, ty, i} => {
            if let Expr::Var {id, ..} = target.as_ref() {
                let loc = ArrayLoc {
                    id: id.clone(),
                    idx: *idx.clone()
                };
                match sub_map.get(&loc) {
                    Some(new_id) => Expr::Var {id: new_id.clone(), ty, i},
                    None => Expr::ArrayAccess {target, idx, ty, i}
                }
            } else {
                Expr::ArrayAccess {target, idx, ty, i}
            }
        },
        _ => e.smap(|e| fuse_memory_accesses_expr(&sub_map, e))
    }
}

fn fuse_memory_array_access(
    mut env: FuseAccessEnv,
    dst: Expr,
    expr: Expr,
    i: Info
) -> (FuseAccessEnv, Stmt) {
    let reconstruct_dst = |target, idx, ty, i| Expr::ArrayAccess {
        target, idx, ty, i
    };
    let reconstruct_assign = |dst, expr: Expr, i: Info| {
        let ty = expr.get_type().clone();
        Stmt::Expr {
            e: Expr::Assign {
                lhs: Box::new(dst),
                rhs: Box::new(expr),
                ty,
                i: i.clone()
            },
            i
        }
    };
    if let Expr::ArrayAccess {target, idx, ty, i: dst_i} = dst {
        if let Expr::Var {id, ..} = target.as_ref() {
            let loc = ArrayLoc {
                id: id.clone(),
                idx: *idx.clone()
            };
            let c = env.write_count.entry(loc.clone())
                .and_modify(|x| *x -= 1)
                .or_insert(0);
            // If the remaining count is greater than zero, it means this is not the last
            // time we write to this location. In this case, we want to replace any
            // subsequent reads from this memory location with the actual expression.
            if *c > 0 {
                let id = Name::sym_str("t");
                env.sub_map.insert(loc, id.clone());
                let s = Stmt::Definition {
                    ty: ty.clone(),
                    id,
                    expr,
                    i
                };
                (env, s)
            } else {
                let dst = reconstruct_dst(target, idx, ty, dst_i);
                (env, reconstruct_assign(dst, expr, i))
            }
        } else {
            let dst = reconstruct_dst(target, idx, ty, dst_i);
            (env, reconstruct_assign(dst, expr, i))
        }
    } else {
        (env, reconstruct_assign(dst, expr, i))
    }
}

fn fuse_memory_cond(
    cond: Expr,
    els: Expr,
    cond_ty: Type,
    s: Stmt
) -> CompileResult<Stmt> {
    match s {
        Stmt::Expr {e, i} => {
            Ok(Stmt::Expr {
                e: Expr::IfExpr {
                    cond: Box::new(cond),
                    thn: Box::new(Expr::Convert {
                        e: Box::new(e),
                        ty: Type::Void
                    }),
                    els: Box::new(els),
                    ty: cond_ty,
                    i: i.clone()
                },
                i
            })
        },
        Stmt::Definition {ty, id, expr, i} => {
            let zero = match &ty {
                Type::Scalar {sz} => {
                    Ok(Expr::generate_literal(0.0, &sz, i.clone()))
                },
                _ => {
                    parpy_internal_error!(i, "Cannot generate zero literal \
                                              for non-scalar types")
                }
            }?;
            Ok(Stmt::Definition {
                ty: ty.clone(),
                id,
                expr: Expr::IfExpr {
                    cond: Box::new(cond),
                    thn: Box::new(expr),
                    els: Box::new(zero),
                    ty,
                    i: i.clone()
                },
                i
            })
        },
        _ => Ok(s)
    }
}

fn fuse_memory_accesses_stmt(
    env: FuseAccessEnv,
    s: Stmt
) -> CompileResult<(FuseAccessEnv, Stmt)> {
    match s {
        Stmt::Expr {e: Expr::Assign {lhs, rhs, ..}, i} => {
            let rhs = fuse_memory_accesses_expr(&env.sub_map, *rhs);
            Ok(fuse_memory_array_access(env, *lhs, rhs, i))
        },
        Stmt::Expr {e: Expr::IfExpr {cond, thn, els, ty: cond_ty, ..}, i} => {
            if let Expr::Convert {e, ty} = *thn {
                if let Expr::Assign {lhs, rhs, ..} = *e {
                    let rhs = fuse_memory_accesses_expr(&env.sub_map, *rhs);
                    let (env, s) = fuse_memory_array_access(env, *lhs, rhs, i);
                    Ok((env, fuse_memory_cond(*cond, *els, cond_ty, s)?))
                } else {
                    let thn = fuse_memory_accesses_expr(&env.sub_map, Expr::Convert {e, ty});
                    let els = fuse_memory_accesses_expr(&env.sub_map, *els);
                    Ok((env
                    , Stmt::Expr {
                        e: Expr::IfExpr {
                            cond,
                            thn: Box::new(thn),
                            els: Box::new(els),
                            ty: cond_ty,
                            i: i.clone()
                        },
                        i
                    }))
                }
            } else {
                let thn = fuse_memory_accesses_expr(&env.sub_map, *thn);
                let els = fuse_memory_accesses_expr(&env.sub_map, *els);
                Ok(( env
                , Stmt::Expr {
                    e: Expr::IfExpr {
                        cond,
                        thn: Box::new(thn),
                        els: Box::new(els),
                        ty: cond_ty,
                        i: i.clone()
                    },
                    i
                }))
            }
        },
        _ => {
            let s = s.smap(|e| fuse_memory_accesses_expr(&env.sub_map, e));
            s.smap_accum_l_result(Ok(env), fuse_memory_accesses_stmt)
        }
    }
}

fn apply_kernel_body(body: Vec<Stmt>) -> CompileResult<Vec<Stmt>> {
    let env = FuseEnv {
        ops: vec![],
        conds: vec![],
        shared_mem_vars: BTreeSet::new()
    };
    let env = body.sfold(env, collect_memory_operations_stmt);
    let wc = env.ops.iter()
        .fold(BTreeMap::new(), count_writes_per_index)
        .into_iter()
        .filter(|(_, (count, _))| *count > 0)
        .map(|(k, (count, _))| (k, count))
        .collect::<BTreeMap<ArrayLoc, isize>>();
    let env = FuseAccessEnv {
        write_count: wc,
        sub_map: BTreeMap::new()
    };
    let (_, body) = body.smap_accum_l_result(Ok(env), fuse_memory_accesses_stmt)?;
    Ok(body)
}

fn apply_top(t: Top) -> CompileResult<Top> {
    match t {
        Top::KernelFunDef {attrs, id, params, body, i} => {
            let body = apply_kernel_body(body)?;
            Ok(Top::KernelFunDef {attrs, id, params, body, i})
        },
        _ => Ok(t),
    }
}

pub fn apply(ast: Ast) -> CompileResult<Ast> {
    ast.smap_result(apply_top)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test::*;
    use crate::gpu::ast_builder::*;
    use crate::gpu::unsymbolize::Unsymbolize;
    use crate::utils::pprint::*;

    fn strip_symbols(body: Vec<Stmt>) -> Vec<Stmt> {
        body.into_iter()
            .map(|s| s.unsymbolize())
            .collect::<_>()
    }

    fn assert_eq_bodies(l: Vec<Stmt>, r: Vec<Stmt>) {
        let l = strip_symbols(l);
        let r = strip_symbols(r);
        assert_eq!(
            l,
            r,
            "LHS:\n{}\nRHS:\n{}",
            pprint_iter(l.iter(), PrettyPrintEnv::default(), "\n").1,
            pprint_iter(r.iter(), PrettyPrintEnv::default(), "\n").1,
        );
    }

    #[test]
    fn fuse_write_write() {
        let loc = array_access(
            var("x", pointer(scalar(ElemSize::F32), MemSpace::Device)),
            int(1, Some(ElemSize::I32)),
            scalar(ElemSize::F32)
        );
        let body = vec![
            assign(loc.clone(), float(1.0, Some(ElemSize::F32))),
            assign(loc.clone(), float(2.0, Some(ElemSize::F32)))
        ];
        let expected = vec![
            Stmt::Definition {
                ty: scalar(ElemSize::F32),
                id: Name::new("t".to_string()),
                expr: float(1.0, Some(ElemSize::F32)),
                i: i()
            },
            assign(loc, float(2.0, Some(ElemSize::F32)))
        ];
        assert_eq_bodies(apply_kernel_body(body.clone()).unwrap(), expected);
    }

    #[test]
    fn fuse_write_read_write() {
        let loc = array_access(
            var("x", pointer(scalar(ElemSize::F32), MemSpace::Device)),
            int(1, Some(ElemSize::I32)),
            scalar(ElemSize::F32)
        );
        let body = vec![
            assign(loc.clone(), float(1.0, Some(ElemSize::F32))),
            assign(var("y", scalar(ElemSize::F32)), loc.clone()),
            assign(loc.clone(), float(2.0, Some(ElemSize::F32))),
        ];
        let id = Name::new("t".to_string());
        let temp_var = Expr::Var {
            id: id.clone(),
            ty: scalar(ElemSize::F32),
            i: i()
        };
        let expected = vec![
            Stmt::Definition {
                ty: scalar(ElemSize::F32),
                id,
                expr: float(1.0, Some(ElemSize::F32)),
                i: i()
            },
            assign(var("y", scalar(ElemSize::F32)), temp_var),
            assign(loc, float(2.0, Some(ElemSize::F32)))
        ];
        assert_eq_bodies(apply_kernel_body(body).unwrap(), expected);
    }
}
