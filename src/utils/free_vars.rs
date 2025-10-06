use crate::utils::name::Name;

use std::collections::BTreeMap;

pub struct FVEnv<T> {
    pub bound: BTreeMap<Name, T>,
    pub free: BTreeMap<Name, T>,
}

impl<T> Default for FVEnv<T> {
    fn default() -> FVEnv<T> {
        FVEnv {
            bound: BTreeMap::new(),
            free: BTreeMap::new()
        }
    }
}

pub trait FreeVars<T: Clone> {
    fn fv(&self, env: FVEnv<T>) -> FVEnv<T> where Self: Sized;
}

pub fn bind_variable<T: Clone>(mut env: FVEnv<T>, id: &Name, ty: &T) -> FVEnv<T> {
    env.bound.insert(id.clone(), ty.clone());
    env
}

pub fn use_variable<T: Clone>(mut env: FVEnv<T>, id: &Name, ty: &T) -> FVEnv<T> {
    if !env.bound.contains_key(&id) {
        env.free.insert(id.clone(), ty.clone());
    }
    env
}
