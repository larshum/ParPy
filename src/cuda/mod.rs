pub mod ast;
mod clusters;
mod codegen;
mod error;
mod escape_function_names;
mod graphs;
mod memory;
mod pprint;
mod reduce;
mod thread_hint;

#[cfg(test)]
mod ast_builder;

use ast::*;
use crate::gpu::ast as gpu_ast;
use crate::option;
use crate::utils::err::*;

pub fn codegen(
    gpu_ast: gpu_ast::Ast,
    opts: &option::CompileOptions
) -> CompileResult<Ast> {
    // Expand the abstract representations of warp and cluster reductions. We do this separately
    // from this codegen to avoid making it unnecessarily complex.
    let gpu_ast = reduce::expand_parallel_reductions(gpu_ast);

    // Ensure we perform no accesses into GPU memory from the host code.
    memory::validate_gpu_memory_access(&gpu_ast)?;

    // Convert the GPU AST to a CUDA C++ AST.
    let cuda_ast = codegen::from_gpu_ir(gpu_ast, opts)?;

    // Adds a prefix to the names of all called functions and their definitions to avoid naming
    // collisions with existing functions.
    let cuda_ast = escape_function_names::apply(cuda_ast);

    // Update all kernel entry points to make use of CUDA graphs.
    let cuda_ast = graphs::use_if_enabled(cuda_ast, opts);

    // Add an attribute to the kernels for which we use a non-standard amount of thread blocks per
    // cluster (currently, only up to 8 blocks are supported by default).
    let cuda_ast = clusters::insert_attribute_for_nonstandard_blocks_per_cluster(cuda_ast, opts);

    // Insert hints about the number of threads each kernel is using, via the '__builtin_assume'
    // construct. This informs the CUDA compiler that it may assume we have no more than a certain
    // number of threads, which may help it generate more efficient code.
    let cuda_ast = thread_hint::insert_thread_hints(cuda_ast);

    Ok(error::add_error_handling(cuda_ast))
}
