use super::ast::*;
use super::constant_fold;
use super::specialize;
use crate::py_internal_error;
use crate::py_type_error;
use crate::ext::buffer::DataType;
use crate::utils::ast::*;
use crate::utils::err::*;
use crate::utils::info::*;
use crate::utils::name::Name;
use crate::utils::smap::*;

use itertools::Itertools;
use pyo3::PyTypeInfo;
use pyo3::prelude::*;
use pyo3::types::*;
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialOrd, Ord, PartialEq, Eq)]
struct UnifyEnv {
    id: Name,
    shape_vars: BTreeMap<Name, i64>,
}

impl UnifyEnv {
    fn new(id: Option<&Name>) -> UnifyEnv {
        let id = id.cloned().unwrap_or(Name::sym_str(""));
        UnifyEnv {id: id, shape_vars: BTreeMap::new()}
    }

    fn lookup_shape_symbol(&self, id: &Name) -> Option<i64> {
        self.shape_vars.get(id).copied()
    }

    fn insert_shape_symbol(mut self, id: Name, n: i64) -> Self {
        self.shape_vars.insert(id, n);
        self
    }
}

fn unify_elem_size(l: ElemSize, r: ElemSize, i: &Info) -> PyResult<ElemSize> {
    match (&l, &r) {
        (ElemSize::Bool, ElemSize::Bool) => Ok(r),
        (ElemSize::I8, _) if r.is_signed_integer() => Ok(r),
        (ElemSize::I16, ElemSize::I8) => Ok(ElemSize::I16),
        (ElemSize::I16, _) if r.is_signed_integer() => Ok(r),
        (ElemSize::I32, ElemSize::I8 | ElemSize::I16) => Ok(ElemSize::I32),
        (ElemSize::I32, _) if r.is_signed_integer() => Ok(r),
        (ElemSize::I64, _) if r.is_signed_integer() => Ok(l),
        (ElemSize::U8, _) if r.is_unsigned_integer() => Ok(r),
        (ElemSize::U16, ElemSize::U8) => Ok(ElemSize::U16),
        (ElemSize::U16, _) if r.is_unsigned_integer() => Ok(r),
        (ElemSize::U32, ElemSize::U8 | ElemSize::U16) => Ok(ElemSize::U32),
        (ElemSize::U32, _) if r.is_unsigned_integer() => Ok(r),
        (ElemSize::U64, _) if r.is_unsigned_integer() => Ok(l),
        (ElemSize::F16, _) if r.is_floating_point() => Ok(r),
        (ElemSize::F32, ElemSize::F16) => Ok(ElemSize::F32),
        (ElemSize::F32, _) if r.is_floating_point() => Ok(r),
        (ElemSize::F64, _) if r.is_floating_point() => Ok(l),
        _ => py_type_error!(i, "Incompatible element types {l} and {r}")
    }
}

fn unify_shape(
    env: UnifyEnv,
    lsh: TensorShape,
    rsh: TensorShape,
    i: &Info
) -> PyResult<(UnifyEnv, TensorShape)> {
    match (lsh, rsh) {
        (TensorShape::Num {n: ln}, TensorShape::Num {n: rn}) => {
            if ln == rn {
                Ok((env, TensorShape::Num {n: ln}))
            } else {
                py_type_error!(i, "Failed to unify distinct dimensions {ln} and {rn}.")
            }
        },
        (TensorShape::Num {n}, TensorShape::Symbol {id}) |
        (TensorShape::Symbol {id}, TensorShape::Num {n}) => {
            match env.lookup_shape_symbol(&id) {
                Some(m) if n == m => Ok((env, TensorShape::Num {n})),
                Some(m) => py_type_error!(i, "Failed to unify shape variable \
                                              {id} (={m}) with {n}."),
                None => {
                    let env = env.insert_shape_symbol(id, n);
                    Ok((env, TensorShape::Num {n}))
                }
            }
        },
        (TensorShape::Symbol {id: lid}, TensorShape::Symbol {id: rid}) => {
            match (env.lookup_shape_symbol(&lid), env.lookup_shape_symbol(&rid)) {
                (Some(l), Some(r)) if l == r => Ok((env, TensorShape::Num {n: l})),
                (Some(l), Some(r)) => {
                    py_type_error!(i, "Failed to unify shape variables \
                                       {lid} (={l}) and {rid} (={r}).")
                },
                (None, Some(n)) => {
                    let env = env.insert_shape_symbol(lid, n);
                    Ok((env, TensorShape::Num {n}))
                },
                (Some(n), None) => {
                    let env = env.insert_shape_symbol(rid, n);
                    Ok((env, TensorShape::Num {n}))
                },
                (None, None) => {
                    py_internal_error!(i, "Encountered two unresolved shape \
                                           symbols {lid} and {rid}.")
                }
            }
        }
    }
}

fn unify_shapes(
    env: UnifyEnv,
    lshapes: Vec<TensorShape>,
    rshapes: Vec<TensorShape>,
    i: &Info
) -> PyResult<(UnifyEnv, Vec<TensorShape>)> {
    if lshapes.len() == rshapes.len() {
        lshapes.into_iter()
            .zip(rshapes.into_iter())
            .fold(Ok((env, vec![])), |acc, (lsh, rsh)| {
                let (env, mut shape) = acc?;
                let (env, sh) = unify_shape(env, lsh, rsh, i)?;
                shape.push(sh);
                Ok((env, shape))
            })
    } else {
        let ls = lshapes.iter().join(", ");
        let rs = rshapes.iter().join(", ");
        py_type_error!(i, "Found incompatible tensor shapes [{ls}] and [{rs}]")
    }
}

fn unify_types(
    env: UnifyEnv,
    lty: Type,
    rty: Type,
    i: &Info
) -> PyResult<(UnifyEnv, Type)> {
    match (lty, rty) {
        (Type::Unknown, Type::Unknown) => {
            py_internal_error!(i, "Cannot unify two unknown types")
        },
        (Type::Unknown, ty) | (ty, Type::Unknown) => Ok((env, ty)),
        (Type::String, Type::String) => Ok((env, Type::String)),
        ( Type::Tensor {sz: lsz, shape: lshape}
        , Type::Tensor {sz: rsz, shape: rshape} ) => {
            if lshape.is_empty() && rshape.is_empty() {
                let sz = unify_elem_size(lsz, rsz, i)?;
                Ok((env, Type::Tensor {sz, shape: vec![]}))
            } else if lsz == rsz {
                let (env, shape) = unify_shapes(env, lshape, rshape, i)?;
                Ok((env, Type::Tensor {sz: lsz, shape}))
            } else {
                py_type_error!(i, "Failed to unify non-scalar tensors containing \
                                   distinct element sizes {lsz} and {rsz}.")
            }
        },
        (Type::Tuple {elems: lelems}, Type::Tuple {elems: relems}) => {
            if lelems.len() == relems.len() {
                let (env, elem_types) = lelems.into_iter()
                    .zip(relems.into_iter())
                    .fold(Ok((env, vec![])), |acc: PyResult<_>, (lty, rty)| {
                        let (env, mut types) = acc?;
                        let (env, ty) = unify_types(env, lty, rty, i)?;
                        types.push(ty);
                        Ok((env, types))
                    })?;
                Ok((env, Type::Tuple {elems: elem_types}))
            } else {
                py_type_error!(i, "Failed to unify tuple types of different lengths")
            }
        },
        (Type::Dict {fields: lfields}, Type::Dict {fields: rfields}) => {
            if lfields.len() == rfields.len() {
                let (env, field_types) = lfields.into_iter()
                    .zip(rfields.into_iter())
                    .fold(Ok((env, BTreeMap::new())), |acc, (lfield, rfield)| {
                        let (env, mut fields) = acc?;
                        let (lk, lv) = lfield;
                        let (rk, rv) = rfield;
                        if lk == rk {
                            let (env, ty) = unify_types(env, lv, rv, i)?;
                            fields.insert(lk, ty);
                            Ok((env, fields))
                        } else {
                            py_type_error!(i, "Failed to unify dictionary types with distinct keys")
                        }
                    })?;
                Ok((env, Type::Dict {fields: field_types}))
            } else {
                py_type_error!(i, "Failed to unify dictionary types of \
                                   different number of entries")
            }
        },
        (Type::Void, Type::Void) => Ok((env, Type::Void)),
        (l, r) => py_type_error!(i, "Failed to unify incompatible types {l} and {r}")
    }
}

fn unify_parameter_type(
    acc: PyResult<(UnifyEnv, Vec<Param>)>,
    param: Param,
    arg_type: Type
) -> PyResult<(UnifyEnv, Vec<Param>)> {
    let (env, mut params) = acc?;
    let Param {id, ty, i} = param;
    let (env, ty) = unify_types(env, ty, arg_type, &i)?;
    params.push(Param {id, ty, i});
    Ok((env, params))
}

fn unify_parameter_types(
    params: Vec<Param>,
    arg_types: Vec<Type>,
    id: &Name,
    i: &Info
) -> PyResult<(UnifyEnv, Vec<Param>)> {
    let pn = params.len();
    let an = arg_types.len();
    if pn == an {
        params.into_iter()
            .zip(arg_types.into_iter())
            .fold(Ok((UnifyEnv::new(Some(id)), vec![])), |acc, (param, arg_type)| {
                unify_parameter_type(acc, param, arg_type)
            })
    } else {
        py_type_error!(i, "Function {id} expects {pn} parameters, but was \
                           called with {an} arguments.")
    }
}

#[derive(Debug)]
struct TypeCheckState {
    // Maps the name of a variable to its type.
    vars: BTreeMap<Name, Type>,
}

impl TypeCheckState {
    fn new(params: Vec<Param>) -> Self {
        let vars = params.into_iter()
            .map(|Param {id, ty, ..}| (id, ty))
            .collect::<BTreeMap<Name, Type>>();
        TypeCheckState {vars}
    }
}

#[derive(Debug)]
struct TypeCheckEnv<'py> {
    tops: BTreeMap<String, Bound<'py, PyCapsule>>,

    // The default sizes to use for int and float literals.
    scalar_sizes: ScalarSizes,

    // Maps a unification environment to the specialized function.
    specs: BTreeMap<UnifyEnv, FunDef>,

    // Contains the names of the specialized functions in the order in which they should be
    // inserted, where the first entry should be placed at the top.
    spec_list: Vec<Top>,

    // A stack of states of type-checking. Each state in the stack corresponds to the intermediate
    // results of a (specialized) function. When a function call node is encountered, we invoke it
    // recursively and insert a new entry to this stack, so that the state of the current function
    // is kept.
    state_stack: Vec<TypeCheckState>
}

impl<'py> TypeCheckEnv<'py> {
    fn new(tops: BTreeMap<String, Bound<'py, PyCapsule>>, scalar_sizes: ScalarSizes) -> Self {
        TypeCheckEnv {
            tops,
            scalar_sizes,
            specs: BTreeMap::new(),
            spec_list: vec![],
            state_stack: vec![]
        }
    }

    fn enter_function(mut self, params: Vec<Param>) -> Self {
        let new_state = TypeCheckState::new(params);
        self.state_stack.push(new_state);
        self
    }

    fn exit_function(mut self, unify_env: UnifyEnv, def: &FunDef) -> PyResult<Self> {
        if self.state_stack.is_empty() {
            py_internal_error!(Info::default(), "Type environment attempted to \
                                                 pop with empty stack stack.")
        } else {
            self.state_stack.pop().unwrap();
            self.specs.insert(unify_env, def.clone());
            Ok(self)
        }
    }

    fn lookup_top(&self, id: &Name) -> Option<Top> {
        self.tops.get(id.get_str())
            .map(|cap| unsafe { cap.reference::<Top>() }.clone())
    }

    fn lookup_var(&self, id: &Name) -> PyResult<Option<Type>> {
        if self.state_stack.is_empty() {
            py_internal_error!(Info::default(), "Type environment variable lookup \
                                                 with empty state stack.")
        } else {
            let curr_state = self.state_stack.last().unwrap();
            Ok(curr_state.vars.get(&id).cloned())
        }
    }

    fn insert_var(mut self, id: &Name, ty: &Type) -> PyResult<Self> {
        if self.state_stack.is_empty() {
            py_internal_error!(Info::default(), "Type environment variable insertion \
                                                 with empty state stack.")
        } else {
            let curr_state = self.state_stack.last_mut().unwrap();
            curr_state.vars.insert(id.clone(), ty.clone());
            Ok(self)
        }
    }
}

type TypeCheckResult<'py, T> = PyResult<(TypeCheckEnv<'py>, T)>;

fn extract_type<'py>(
    arg: &Bound<'py, PyAny>,
    scalar_sizes: &ScalarSizes,
    i: &Info
) -> PyResult<Type> {
    let py = arg.py();
    let parpy = py.import("parpy")?;
    let buffer = parpy.getattr("buffer")?;
    let ty = arg.get_type();
    if arg.is_instance(&buffer.getattr("Buffer")?)? {
        let dtype = arg.getattr("dtype")?.extract::<DataType>()?;
        let sz = dtype.sz;
        let shape = arg.getattr("shape")?
            .extract::<Vec<i64>>()?
            .into_iter()
            .map(|n| TensorShape::Num {n})
            .collect::<Vec<TensorShape>>();
        Ok(Type::Tensor {sz, shape})
    } else if arg.is_instance(&PyBool::type_object(py))? {
        Ok(Type::Tensor {sz: ElemSize::Bool, shape: vec![]})
    } else if arg.is_instance(&PyInt::type_object(py))? {
        Ok(Type::Tensor {sz: scalar_sizes.int.clone(), shape: vec![]})
    } else if arg.is_instance(&PyFloat::type_object(py))? {
        Ok(Type::Tensor {sz: scalar_sizes.float.clone(), shape: vec![]})
    } else if arg.is_instance(&PyDict::type_object(py))? {
        let fields = arg.call_method0("items")?
            .try_iter()?
            .map(|f| {
                let f = f?;
                let id = f.get_item(0)?.extract::<String>()?;
                let ty = extract_type(&f.get_item(1)?, scalar_sizes, i)?;
                Ok((id, ty))
            })
            .collect::<PyResult<BTreeMap<String, Type>>>()?;
        Ok(Type::Dict {fields})
    } else {
        py_type_error!(i, "Argument has unsupported type {ty}")
    }
}

fn extract_argument_types<'py>(
    args: &Vec<Bound<'py, PyAny>>,
    scalar_sizes: &ScalarSizes,
    i: &Info
) -> PyResult<Vec<Type>> {
    args.iter()
        .map(|arg| extract_type(arg, scalar_sizes, i))
        .collect::<PyResult<Vec<Type>>>()
}

fn coerce_type(e: Expr, expected_ty: &Type) -> PyResult<Expr> {
    if e.get_type().eq(expected_ty) {
        Ok(e)
    } else {
        let i = e.get_info();
        let actual = e.get_type();
        match (actual, expected_ty) {
            (Type::Tensor {sz: lsz, shape: lsh}, Type::Tensor {sz: rsz, shape: rsh}) => {
                if lsh.is_empty() && rsh.is_empty() {
                    let sz = unify_elem_size(lsz.clone(), rsz.clone(), &i)?;
                    let ty = Type::Tensor {sz, shape: vec![]};
                    Ok(Expr::Convert {e: Box::new(e), ty})
                } else if lsz == rsz {
                    let unify_env = UnifyEnv::new(None);
                    let (_, shape) = unify_shapes(unify_env, lsh.clone(), rsh.clone(), &i)?;
                    let ty = Type::Tensor {sz: lsz.clone(), shape};
                    Ok(Expr::Convert {e: Box::new(e), ty})
                } else {
                    py_type_error!(i, "Cannot coerce non-empty tensor of element size {lsz} to {rsz}.")
                }
            },
            (Type::Tuple {..}, Type::Tuple {elems: r}) => {
                if let Expr::Tuple {elems, i, ..} = e {
                    let elems = elems.into_iter()
                        .zip(r.iter())
                        .map(|(e, ty)| coerce_type(e, ty))
                        .collect::<PyResult<Vec<Expr>>>()?;
                    let elem_tys = elems.iter()
                        .map(|e| e.get_type().clone())
                        .collect::<Vec<Type>>();
                    let ty = Type::Tuple {elems: elem_tys};
                    Ok(Expr::Tuple {elems, ty, i})
                } else {
                    py_type_error!(i, "Cannot coerce non-literal tuple value {e}")
                }
            },
            _ => py_type_error!(i, "Coercion of expression {e} to type {expected_ty} failed")
        }
    }
}

fn extract_return_type_stmt(
    ret_ty: Type,
    s: &Stmt
) -> PyResult<Type> {
    match s {
        Stmt::Return {value, i} => {
            let ty = value.get_type();
            match ret_ty {
                Type::Void => Ok(ty.clone()),
                actual_ty => {
                    let env = UnifyEnv::new(None);
                    match unify_types(env, actual_ty.clone(), ty.clone(), &i) {
                        Ok((_, r)) => Ok(r),
                        Err(_) => py_type_error!(i, "Found incompatible return types \
                                                     {ty} and {actual_ty}.")
                    }
                }
            }
        },
        _ => s.sfold_result(Ok(ret_ty), |ty, s| extract_return_type_stmt(ty, s))
    }
}

fn extract_return_type(body: &Vec<Stmt>) -> PyResult<Type> {
    body.sfold_result(Ok(Type::Void), extract_return_type_stmt)
}

trait TypeCheck {
    fn type_check<'py>(
        self,
        env: TypeCheckEnv<'py>,
    ) -> TypeCheckResult<'py, Self> where Self: Sized;
}

impl<T: Clone + TypeCheck + SMapAccum<T>> TypeCheck for Vec<T> {
    fn type_check<'py>(
        self,
        env: TypeCheckEnv<'py>
    ) -> TypeCheckResult<'py, Self> {
        self.smap_accum_l_result(Ok(env), |env, t| t.type_check(env))
    }
}

fn type_check_unop(
    op: &UnOp,
    arg: &Expr,
    i: &Info
) -> PyResult<Type> {
    let ty = arg.get_type();
    match op {
        UnOp::Sub if ty.is_int_scalar() || ty.is_float_scalar() => Ok(ty.clone()),
        UnOp::Not if ty.is_bool_scalar() => Ok(ty.clone()),
        UnOp::BitNeg if ty.is_int_scalar() => Ok(ty.clone()),
        UnOp::Exp if ty.is_float_scalar() => Ok(ty.clone()),
        UnOp::Log if ty.is_float_scalar() => Ok(ty.clone()),
        UnOp::Cos if ty.is_float_scalar() => Ok(ty.clone()),
        UnOp::Sin if ty.is_float_scalar() => Ok(ty.clone()),
        UnOp::Sqrt if ty.is_float_scalar() => Ok(ty.clone()),
        UnOp::Tanh if ty.is_float_scalar() => Ok(ty.clone()),
        UnOp::Abs if ty.is_int_scalar() || ty.is_float_scalar() => Ok(ty.clone()),
        _ => py_type_error!(i, "Unsupported argument type {ty} of unary operator {op:?}")
    }
}

fn type_check_binop(
    lhs: Expr,
    op: &BinOp,
    rhs: Expr,
    i: &Info
) -> PyResult<(Box<Expr>, Type, Box<Expr>)> {
    let lty = lhs.get_type().clone();
    let rty = rhs.get_type().clone();
    let (_, ty) = unify_types(UnifyEnv::new(None), lty, rty, i)?;
    let lhs = coerce_type(lhs, &ty)?;
    let rhs = coerce_type(rhs, &ty)?;
    let ty = match op {
        BinOp::Add | BinOp::Sub | BinOp::Mul |
        BinOp::Div if ty.is_int_scalar() || ty.is_float_scalar() => {
            Ok(ty)
        },
        BinOp::FloorDiv | BinOp::Rem if ty.is_int_scalar() => {
            Ok(ty)
        },
        BinOp::Pow | BinOp::Atan2 if ty.is_float_scalar() => {
            Ok(ty)
        },
        BinOp::And | BinOp::Or if ty.is_bool_scalar() => {
            Ok(ty)
        },
        BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor |
        BinOp::BitShl | BinOp::BitShr if ty.is_int_scalar() => {
            Ok(ty)
        },
        BinOp::Eq | BinOp::Neq | BinOp::Leq | BinOp::Geq |
        BinOp::Lt | BinOp::Gt if ty.is_scalar() => {
            Ok(Type::Tensor {sz: ElemSize::Bool, shape: vec![]})
        },
        BinOp::Max | BinOp::Min if ty.is_int_scalar() || ty.is_float_scalar() => {
            Ok(ty)
        },
        _ => py_type_error!(i, "Unsupported type {ty} of binary operator {op:?}")
    }?;
    Ok((Box::new(lhs), ty, Box::new(rhs)))
}

fn type_check_dict_indexing(
    target: Expr,
    key: String,
    str_info: Info,
    target_info: Info
) -> PyResult<Expr> {
    if let Type::Dict {fields} = target.get_type() {
        if let Some(ty) = fields.get(&key) {
            let result_ty = ty.clone();
            Ok(Expr::Subscript {
                target: Box::new(target),
                idx: Box::new(Expr::String {v: key, ty: Type::String, i: str_info}),
                ty: result_ty,
                i: target_info
            })
        } else {
            py_type_error!(
                target_info, 
                "Field {key} not found in type {0}",
                target.get_type()
            )
        }
    } else {
        py_type_error!(target_info, "Strings can only be used as keys on \
                                     non-dictionary targets")
    }
}

fn count_dimensions(idx: &Expr) -> usize {
    match idx {
        Expr::Tuple {elems, ..} => elems.len(),
        _ => 1
    }
}

fn contains_slices(acc: bool, e: &Expr) -> bool {
    match e {
        Expr::Slice {..} => true,
        _ => e.sfold(acc, contains_slices)
    }
}

fn type_check_tensor_indexing<'py>(
    env: TypeCheckEnv<'py>,
    target: Expr,
    idx: Expr,
    i: Info
) -> TypeCheckResult<'py, Expr> {
    // Add coercion of each component index of the index expression.
    let int_ty = Type::Tensor {sz: env.scalar_sizes.int.clone(), shape: vec![]};
    let expected_index_type = match idx.get_type() {
        Type::Tensor {shape, ..} if shape.len() == 0 => Ok(int_ty),
        Type::Tuple {elems} => {
            let expected_elem_types = elems.into_iter()
                .map(|_| int_ty.clone())
                .collect::<Vec<Type>>();
            Ok(Type::Tuple {elems: expected_elem_types})
        },
        ty => py_type_error!(i, "Found index of unsupported type {ty}")
    }?;
    let idx = coerce_type(idx, &expected_index_type)?;

    // Determine the result type based on the shape of the index and the target.
    let target_ty = target.get_type();
    let ty = match target_ty {
        Type::Tensor {sz, shape} => {
            let ndims = count_dimensions(&idx);
            if contains_slices(false, &idx) {
                if ndims == shape.len() {
                    Ok(Type::Tensor {sz: sz.clone(), shape: vec![]})
                } else {
                    let sh = shape.iter().join(", ");
                    py_type_error!(i, "Slice expressions must address all \
                                       dimensions of the target.\n\
                                       Index refers to {ndims} dimensions, while \
                                       the target has shape [{sh}].")
                }
            } else if ndims <= shape.len() {
                let shape = shape.clone()
                    .into_iter()
                    .skip(ndims)
                    .collect::<Vec<TensorShape>>();
                Ok(Type::Tensor {sz: sz.clone(), shape})
            } else {
                py_type_error!(i, "Indexing with {ndims} dimensions on tensor \
                                   of type {target_ty}.")
            }
        },
        _ => py_type_error!(i, "Indexing into unsupported target of type {target_ty}")
    }?;
    Ok((env, Expr::Subscript {target: Box::new(target), idx: Box::new(idx), ty, i}))
}

impl TypeCheck for Expr {
    fn type_check<'py>(
        self,
        env: TypeCheckEnv<'py>
    ) -> TypeCheckResult<'py, Expr> {
        match self {
            Expr::Var {id, ty: _, i} => {
                let ty = match env.lookup_var(&id)? {
                    Some(ty) => Ok(ty.clone()),
                    None => {
                        println!("{env:#?}");
                        py_type_error!(i, "Variable {id} has unknown type")
                    }
                }?;
                Ok((env, Expr::Var {id, ty, i}))
            },
            Expr::String {v, ty: _, i} => {
                Ok((env, Expr::String {v, ty: Type::String, i}))
            },
            Expr::Bool {v, ty: _, i} => {
                let ty = Type::Tensor {sz: ElemSize::Bool, shape: vec![]};
                Ok((env, Expr::Bool {v, ty, i}))
            },
            Expr::Int {v, ty: _, i} => {
                let ty = Type::Tensor {sz: env.scalar_sizes.int.clone(), shape: vec![]};
                Ok((env, Expr::Int {v, ty, i}))
            },
            Expr::Float {v, ty: _, i} => {
                let ty = Type::Tensor {sz: env.scalar_sizes.float.clone(), shape: vec![]};
                Ok((env, Expr::Float {v, ty, i}))
            },
            Expr::UnOp {op, arg, ty: _, i} => {
                let (env, arg) = arg.type_check(env)?;
                let ty = type_check_unop(&op, &arg, &i)?;
                Ok((env, Expr::UnOp {op, arg: Box::new(arg), ty, i}))
            },
            Expr::BinOp {lhs, op, rhs, ty: _, i} => {
                let (env, lhs) = lhs.type_check(env)?;
                let (env, rhs) = rhs.type_check(env)?;
                let (lhs, ty, rhs) = type_check_binop(lhs, &op, rhs, &i)?;
                Ok((env, Expr::BinOp {lhs, op, rhs, ty, i}))
            },
            Expr::ReduceOp {op, arg, ty: _, i} => {
                let (env, arg) = arg.type_check(env)?;
                let arg_ty = arg.get_type();
                if arg_ty.is_int_scalar() || arg_ty.is_float_scalar() {
                    let ty = arg_ty.clone();
                    Ok((env, Expr::ReduceOp {op, arg: Box::new(arg), ty, i}))
                } else {
                    py_type_error!(i, "Reduction operations are only supported \
                                       on integer and float types.")
                }
            },
            Expr::IfExpr {cond, thn, els, ty: _, i} => {
                let (env, cond) = cond.type_check(env)?;
                let cond_ty = cond.get_type();
                if cond_ty.is_bool_scalar() {
                    let (env, thn) = thn.type_check(env)?;
                    let (env, els) = els.type_check(env)?;
                    let thn_type = thn.get_type().clone();
                    let els_type = els.get_type().clone();
                    let (_, ty) = unify_types(UnifyEnv::new(None), thn_type, els_type, &i)?;
                    let thn = Box::new(coerce_type(thn, &ty)?);
                    let els = Box::new(coerce_type(els, &ty)?);
                    Ok((env, Expr::IfExpr {cond: Box::new(cond), thn, els, ty, i}))
                } else {
                    py_type_error!(i, "Expected boolean condition in \
                                       if-expression, found {cond_ty}.")
                }
            },
            Expr::Subscript {target, idx, ty: _, i} => {
                let (env, target) = target.type_check(env)?;
                let (env, idx) = idx.type_check(env)?;
                match idx {
                    Expr::String {v, ty: _, i: str_info} => {
                        Ok((env, type_check_dict_indexing(target, v, str_info, i)?))
                    },
                    idx => type_check_tensor_indexing(env, target, idx, i)
                }
            },
            Expr::Slice {lo, hi, ty: _, i} => {
                let type_check_boxed = |env, o: Option<Box<Expr>>| match o {
                    Some(e) => {
                        let (env, e) = e.type_check(env)?;
                        Ok::<_, PyErr>((env, Some(Box::new(e))))
                    },
                    None => Ok((env, None))
                };
                let (env, lo) = type_check_boxed(env, lo)?;
                let (env, hi) = type_check_boxed(env, hi)?;
                // NOTE(larshum, 2025-09-12): Slices are considered as integers. Assuming they are
                // never used outside of indexing (we cannot parse such Python code), this is fine.
                let ty = Type::Tensor {sz: env.scalar_sizes.int.clone(), shape: vec![]};
                Ok((env, Expr::Slice {lo, hi, ty, i}))
            },
            Expr::Tuple {elems, ty: _, i} => {
                let (env, elems) = elems.type_check(env)?;
                let elem_types = elems.iter()
                    .map(|e| e.get_type().clone())
                    .collect::<Vec<Type>>();
                let ty = Type::Tuple {elems: elem_types};
                Ok((env, Expr::Tuple {elems, ty, i}))
            },
            Expr::Call {id, args, ty: _, i} => {
                let (env, args) = args.type_check(env)?;
                let (env, new_id, ret_ty) = type_check_call(env, &id, &args, &i)?;
                match ret_ty {
                    Type::Void => py_type_error!(i, "Function call expression cannot \
                                                     have void result."),
                    ty => Ok((env, Expr::Call {id: new_id, args, ty, i}))
                }
            },
            Expr::NeutralElement {op, ty, i} => {
                Ok((env, Expr::NeutralElement {op, ty, i}))
            },
            Expr::Builtin {i, ..} => {
                py_internal_error!(i, "Found builtin expression in type-checker.")
            },
            Expr::Convert {e, ty} => {
                let (env, e) = e.type_check(env)?;
                Ok((env, Expr::Convert {e: Box::new(e), ty}))
            },
        }
    }
}

impl TypeCheck for Stmt {
    fn type_check<'py>(
        self,
        env: TypeCheckEnv<'py>
    ) -> TypeCheckResult<'py, Stmt> {
        match self {
            Stmt::Definition {ty: _, id, expr, labels, i} => {
                let (env, expr) = expr.type_check(env)?;
                let ty = expr.get_type().clone();
                let env = env.insert_var(&id, &ty)?;
                Ok((env, Stmt::Definition {ty, id, expr, labels, i}))
            },
            Stmt::Assign {dst, expr, labels, i} => {
                let (env, dst) = dst.type_check(env)?;
                let (env, expr) = expr.type_check(env)?;
                let expr = coerce_type(expr, dst.get_type())?;
                Ok((env, Stmt::Assign {dst, expr, labels, i}))
            },
            Stmt::For {var, lo, hi, step, body, labels, i} => {
                let (env, lo) = lo.type_check(env)?;
                let (env, hi) = hi.type_check(env)?;
                let int_ty = Type::Tensor {sz: env.scalar_sizes.int.clone(), shape: vec![]};
                let lo = coerce_type(lo, &int_ty)?;
                let hi = coerce_type(hi, &int_ty)?;
                let env = env.insert_var(&var, &int_ty)?;
                let (env, body) = body.type_check(env)?;
                Ok((env, Stmt::For {var, lo, hi, step, body, labels, i}))
            },
            Stmt::While {cond, body, i} => {
                let (env, cond) = cond.type_check(env)?;
                if cond.get_type().is_bool_scalar() {
                    let (env, body) = body.type_check(env)?;
                    Ok((env, Stmt::While {cond, body, i}))
                } else {
                    py_type_error!(i, "While loop condition must be a boolean value.")
                }
            },
            Stmt::If {cond, thn, els, i} => {
                let (env, cond) = cond.type_check(env)?;
                if cond.get_type().is_bool_scalar() {
                    let (env, thn) = thn.type_check(env)?;
                    let (env, els) = els.type_check(env)?;
                    Ok((env, Stmt::If {cond, thn, els, i}))
                } else {
                    py_type_error!(i, "If condition must be a boolean value")
                }
            },
            Stmt::Return {value, i} => {
                let (env, value) = value.type_check(env)?;
                Ok((env, Stmt::Return {value, i}))
            },
            Stmt::WithGpuContext {body, i} => {
                let (env, body) = body.type_check(env)?;
                Ok((env, Stmt::WithGpuContext {body, i}))
            },
            Stmt::Call {func, args, i} => {
                let (env, args) = args.type_check(env)?;
                let (env, new_id, ret_ty) = type_check_call(env, &func, &args, &i)?;
                match ret_ty {
                    Type::Void => Ok((env, Stmt::Call {func: new_id, args, i})),
                    ty => py_type_error!(i, "Function call statement must have \
                                             void result, but found {ty}.")
                }
            },
            Stmt::Label {label, i} => Ok((env, Stmt::Label {label, i}))
        }
    }
}

fn type_check_call<'py>(
    env: TypeCheckEnv<'py>,
    id: &Name,
    args: &Vec<Expr>,
    i: &Info
) -> PyResult<(TypeCheckEnv<'py>, Name, Type)> {
    let arg_types = args.into_iter()
        .map(|arg| arg.get_type().clone())
        .collect::<Vec<Type>>();
    match env.lookup_top(id) {
        Some(t) => {
            let (env, t) = type_check_top(env, t, arg_types)?;
            let (new_id, ret_ty) = match t {
                Top::ExtDecl {id, res_ty, ..} |
                Top::FunDef {v: FunDef {id, res_ty, ..}} => {
                    (id.clone(), res_ty.clone())
                }
            };
            Ok((env, new_id, ret_ty))
        },
        None => py_type_error!(i, "Call to unknown function {id}")
    }
}

fn type_check_fun_def<'py>(
    env: TypeCheckEnv<'py>,
    def: FunDef,
    arg_types: Vec<Type>
) -> TypeCheckResult<'py, FunDef> {
    // 1. Produce environment for the function by unifying the parameter declarations with the
    //    types of the provided arguments.
    let (unify_env, params) = unify_parameter_types(def.params, arg_types, &def.id, &def.i)?;

    // If this particular function has already been specialized, we immediately return.
    match env.specs.get(&unify_env).cloned() {
        Some(mono_def) => Ok((env, mono_def)),
        None => {
            let env = env.enter_function(params.clone());

            // 2. Specialize the function body by replacing expressions according to the
            //    environment.
            let body = specialize::apply(&unify_env.shape_vars, def.body);

            // 3. Apply constant folding to the function body, to ensure that slice bounds can be
            //    determined given the shapes.
            let body = constant_fold::fold(body);

            // 4. Perform an extended type-check of the body. In addition to normal type-checking,
            //    this will:
            //    - Insert the bounds of slices based on derived type information.
            //    - Recursively run these steps on any called function, or retrieving it from the
            //      cache immediately.
            let (env, body) = body.type_check(env)?;

            let id = def.id.with_new_sym();
            let res_ty = extract_return_type(&body)?;
            let def = FunDef {id, params, body, res_ty, ..def};
            let env = env.exit_function(unify_env, &def)?;

            Ok((env, def))
        }
    }
}

fn type_check_top<'py>(
    mut env: TypeCheckEnv<'py>,
    t: Top,
    arg_types: Vec<Type>
) -> TypeCheckResult<'py, Top> {
    match t {
        Top::ExtDecl {id, ext_id, params, res_ty, target, header, par, i} => {
            let (_, params) = unify_parameter_types(params, arg_types, &id, &i)?;
            let t = Top::ExtDecl {id, ext_id, params, res_ty, target, header, par, i};
            env.spec_list.push(t.clone());
            Ok((env, t))
        },
        Top::FunDef {v} => {
            let (mut env, v) = type_check_fun_def(env, v, arg_types)?;
            let t = Top::FunDef {v};
            env.spec_list.push(t.clone());
            Ok((env, t))
        },
    }
}

fn type_check_main<'py>(
    env: TypeCheckEnv<'py>,
    main: FunDef,
    arg_types: Vec<Type>
) -> PyResult<Ast> {
    let (env, main) = type_check_fun_def(env, main, arg_types)?;
    let FunDef {ref res_ty, ref id, ref i, ..} = main;
    match res_ty {
        Type::Void => Ok(Ast {tops: env.spec_list, main}),
        _ => py_type_error!(i, "Main function {id} cannot return a value.")
    }
}

pub fn apply<'py>(
    main: FunDef,
    args: &Vec<Bound<'py, PyAny>>,
    tops: BTreeMap<String, Bound<'py, PyCapsule>>,
    scalar_sizes: &ScalarSizes
) -> PyResult<Ast> {
    let i = main.i.clone();
    let env = TypeCheckEnv::new(tops, scalar_sizes.clone());
    let arg_types = extract_argument_types(args, &scalar_sizes, &i)?;
    type_check_main(env, main, arg_types)
}
