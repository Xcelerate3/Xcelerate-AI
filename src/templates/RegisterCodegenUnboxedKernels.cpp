#include <xcelerate/csrc/jit/runtime/operator.h>
#include <xcelerate/csrc/jit/runtime/custom_operator.h>
#include <xcelerate/csrc/jit/runtime/register_ops_utils.h>

#include <ATen/UnboxingFunctions.h>

// ${generated_comment}

// NOTE [Sharded File]: This file is generated in a sharded fashion to speed up
// incremental rebuilds. See the comment at the top of
// templates/VariableType.cpp for an analogous, in-depth discussion.
//
// Generated by tools/jit/gen_unboxing.py. This file registers all ATen ops into JIT op registry instead of c10
// dispatcher. JIT op registry only takes boxed kernels, so we are calling unboxing functions in UnboxingFunctions.h
// to cast arguments into C++ types (instead of IValue) and delegate to unboxed kernels.

namespace xcelerate { namespace jit {

using autograd::Variable;
using autograd::variable_list;
using at::Scalar;
using at::ScalarType;
using at::Tensor;
using at::TensorOptions;
using at::DeviceGuard;

using ::c10::fmap;
using ::c10::filter;

namespace {

RegisterOperators reg({

    // Generated operators
    ${unboxed_ops}
});

} // anon namespace


}} // namespace xcelerate::jit
