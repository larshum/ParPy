use super::ast::*;
use crate::parpy_compile_error;
use crate::parpy_internal_error;
use crate::utils::err::*;
use crate::utils::info::Info;
use crate::utils::name::Name;
use crate::utils::smap::SFold;

use std::collections::BTreeMap;

struct ClassifyEnv {
    mapping: BTreeMap<Name, TargetClass>,
    id: Name,
    i: Info
}

impl Default for ClassifyEnv {
    fn default() -> Self {
        ClassifyEnv {
            mapping: BTreeMap::new(),
            id: Name::new("".to_string()),
            i: Info::default()
        }
    }
}

impl ClassifyEnv {
    fn enter_function(mut self, id: &Name, i: &Info) -> Self {
        self.id = id.clone();
        self.i = i.clone();
        self
    }
}

#[derive(Clone, Debug)]
pub enum TargetClass {
    HostOnly, DeviceOnly, Either
}

fn classify_target(target: &Target) -> TargetClass {
    match target {
        Target::Host => TargetClass::HostOnly,
        Target::Device => TargetClass::DeviceOnly
    }
}

fn unify_classifications(
    env: &ClassifyEnv,
    lcls: TargetClass,
    rcls: TargetClass
) -> CompileResult<TargetClass> {
    match (&lcls, &rcls) {
        (TargetClass::HostOnly, TargetClass::HostOnly) |
        (TargetClass::DeviceOnly, TargetClass::DeviceOnly) |
        (_, TargetClass::Either) => {
            Ok(lcls)
        },
        (TargetClass::Either, _) => Ok(rcls),
        (TargetClass::HostOnly, TargetClass::DeviceOnly) |
        (TargetClass::DeviceOnly, TargetClass::HostOnly) => {
            parpy_compile_error!(env.i, "Function {0} calls both host and device \
                                         functions, which is not allowed", env.id)
        }
    }
}

fn classify_expr(
    env: &ClassifyEnv,
    acc: TargetClass,
    e: &Expr
) -> CompileResult<TargetClass> {
    match e {
        Expr::Call {id, i, ..} => {
            match env.mapping.get(&id) {
                Some(cls) => unify_classifications(&env, acc, cls.clone()),
                None => {
                    parpy_internal_error!(i, "Found call to function {id} with \
                                              no classification")
                }
            }
        },
        Expr::PyCallback {..} => {
            unify_classifications(&env, acc, TargetClass::HostOnly)
        },
        _ => e.sfold_result(Ok(acc), |acc, e| classify_expr(&env, acc, e))
    }
}

fn classify_stmt(
    env: &ClassifyEnv,
    acc: TargetClass,
    s: &Stmt
) -> CompileResult<TargetClass> {
    let acc = s.sfold_result(Ok(acc), |acc, s| classify_stmt(&env, acc, s));
    s.sfold_result(acc, |acc, e| classify_expr(&env, acc, e))
}

fn classify_top(
    mut acc: ClassifyEnv,
    t: &Top
) -> CompileResult<ClassifyEnv> {
    match t {
        Top::StructDef {..} => Ok(acc),
        Top::ExtDecl {id, target, ..} => {
            let cl = classify_target(target);
            acc.mapping.insert(id.clone(), cl);
            Ok(acc)
        },
        Top::FunDef {v: FunDef {id, body, i, ..}} => {
            let mut acc = acc.enter_function(&id, &i);
            let cl = body.sfold_result(Ok(TargetClass::Either), |cl, s| {
                classify_stmt(&acc, cl, s)
            })?;
            acc.mapping.insert(id.clone(), cl);
            Ok(acc)
        },
    }
}

pub fn apply(ast: &Ast) -> CompileResult<BTreeMap<Name, TargetClass>> {
    let env = ast.tops.sfold_result(Ok(ClassifyEnv::default()), classify_top)?;
    Ok(env.mapping)
}
