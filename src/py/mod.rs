pub mod ast;
mod constant_fold;
mod from_py;
mod inline_calls;
mod inline_const;
mod labels;
mod par;
mod pprint;
mod replace_builtins;
mod slice_transformation;
mod specialize;
mod symbolize;
mod type_check;

#[cfg(test)]
pub mod ast_builder;

use crate::py_runtime_error;
use crate::option::*;
use crate::utils::ast::ScalarSizes;
use crate::utils::debug;
use crate::utils::err::CompileError;
use crate::utils::info::Info;

use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyDict};
use std::collections::BTreeMap;

pub use from_py::convert_external;
pub use inline_calls::inline_function_calls;

pub fn parse_untyped_ast<'py>(
    ast: Bound<'py, PyAny>,
    info: (String, usize, usize),
    tops: &BTreeMap<String, Bound<'py, PyCapsule>>,
    vars: (Bound<'py, PyDict>, Bound<'py, PyDict>)
) -> PyResult<ast::FunDef> {
    let ast = from_py::to_untyped_ir(ast, info, tops, vars)?;
    let ast = symbolize::with_tops(tops, ast)?;
    let ast = replace_builtins::apply(ast)?;
    labels::associate_labels(ast)
}

pub fn specialize_ast_on_arguments<'py>(
    t: ast::Top,
    args: Vec<Bound<'py, PyAny>>,
    opts: &CompileOptions,
    tops: BTreeMap<String, Bound<'py, PyCapsule>>,
    debug_env: &debug::DebugEnv
) -> PyResult<ast::Ast> {
    // We expect the top to contain a function definition, but it could also be an external
    // declaration, in which case we report an error.
    let main = match t {
        ast::Top::FunDef {v} => Ok(v),
        ast::Top::ExtDecl {id, ..} => {
            py_runtime_error!(
                Info::default(),
                "Expected {id} to be a function definition, but it is an \
                 external function declaration"
            )
        }
    }?;

    // Ensure the AST contains any degree of parallelism - otherwise, there is no point in using
    // this framework at all.
    par::ensure_parallelism(&main, &opts.parallelize)?;

    // Perform the type-checking and inlining of literal values in an intertwined manner. First, we
    // type-check the parameters based on the corresponding arguments provided in the function
    // call. Second, once the parameters have been typed, we inline the values of scalar parameters
    // into the AST.
    //
    // This particular order is important, because it allows us to reason about the exact sizes of
    // all slices and by extension the correctness of dimensions of slice operations.
    let scalar_sizes = ScalarSizes::from_opts(&opts);
    let main = inline_const::inline_scalar_values(main, &args)?;
    debug_env.print("Python-like AST after inlining", &main);

    // Applies the type-checker, which resolves shape sizes and monomorphizes functions based on
    // the provided arguments. The result is an AST containing all monomorphized versions of
    // top-level definitions.
    let ast = type_check::apply(main, &args, tops, &scalar_sizes)?;
    debug_env.print("Python-like AST after type-checking", &ast);

    // Transform slice statements into for-loops.
    let ast = slice_transformation::apply(ast)?;
    debug_env.print("Python-like AST after slice transformation", &ast);

    Ok(ast)
}

#[macro_export]
macro_rules! py_runtime_error {
    ($i:expr,$($t:tt)*) => {
        Err(Into::<PyErr>::into(CompileError::compile_err($i.error_msg(format!($($t)*)))))
    }
}

#[macro_export]
macro_rules! py_name_error {
    ($i:expr,$($t:tt)*) => {
        Err(Into::<PyErr>::into(CompileError::name_err($i.error_msg(format!($($t)*)))))
    }
}

#[macro_export]
macro_rules! py_type_error {
    ($i:expr,$($t:tt)*) => {
        Err(Into::<PyErr>::into(CompileError::type_err($i.error_msg(format!($($t)*)))))
    }
}

#[macro_export]
macro_rules! py_internal_error {
    ($i:expr,$($t:tt)*) => {
        Err(Into::<PyErr>::into(CompileError::internal_err($i.error_msg(format!($($t)*)))))
    }
}
