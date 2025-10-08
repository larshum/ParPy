mod codegen;
mod ctypes;
mod extract_callbacks;

use crate::gpu::ast as gpu_ast;
use crate::option::CompileOptions;
use crate::utils::pprint::PrettyPrint;

use pyo3::prelude::*;

fn print_callback_asts(opts: &CompileOptions, callbacks: &Vec<String>) {
    // When the compiler option is enabled, and we have at least one callback function, we print
    // them to the terminal along with a unique message to keep it distinct from debug messages
    // produced by other parts of the compiler.
    if opts.debug_callbacks && !callbacks.is_empty() {
        let bounds = "=".repeat(5);
        println!("{0} START GENERATED CALLBACKS {0}\n", bounds);
        callbacks.iter().for_each(|s| println!("{s}\n"));
        println!("{0} END GENERATED CALLBACKS {0}\n", bounds);
    }
}

pub fn from_gpu_ast<'py>(
    opts: &CompileOptions,
    gpu_ast: gpu_ast::Ast,
    py: Python<'py>
) -> PyResult<(Vec<Bound<'py, PyAny>>, Vec<String>, gpu_ast::Ast)> {
    // Extracts all callback functions used in the entry point function of the GPU AST. The result
    // consists of:
    // - A list of the callback functions added as arguments, represented in terms of name, type,
    //   and a GPU AST statement node representing the code invoking the callback. This code may
    //   include looping constructs.
    // - The updated GPU AST, where callbacks are added as arguments in place of top-level
    //   declarations.
    let (callbacks, gpu_ast) = extract_callbacks::apply(gpu_ast)?;

    // Produces a vector containing the Ctypes types of all arguments passed to the entry point
    // function of the GPU AST, including the callback functions.
    let entry_point_argtypes = ctypes::produce_argument_list(&gpu_ast, py)?;

    // For each callback function, builds a Python AST representing a wrapping callback function
    // that handles the conversion of low-level types to ParPy buffers and scalar values, and
    // iteratively calls the user-provided callback.
    let callback_asts = codegen::generate_callbacks(&opts, callbacks)?;
    let callback_asts = callback_asts.into_iter()
        .map(|ast| ast.pprint_default())
        .collect::<Vec<String>>();
    print_callback_asts(&opts, &callback_asts);

    Ok((entry_point_argtypes, callback_asts, gpu_ast))
}
