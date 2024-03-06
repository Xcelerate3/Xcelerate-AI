#pragma once
// ${generated_comment}

// NB: The implementing C++ file is RegisterDispatchKey.cpp

// The only #includes we need are for custom classes that have defaults in the C++ API
#include <c10/core/MemoryFormat.h>
#include <c10/core/Scalar.h>
#include <ATen/core/Reduction.h>

#if defined(AT_PER_OPERATOR_HEADERS) && defined(XCELERATE_ASSERT_ONLY_METHOD_OPERATORS)
#error This change adds a dependency on all XCELERATE operators, meaning the     \
  file will need to be re-compiled every time an operator is changed or added. \
  Consider including a specific operator from                                  \
  <ATen/ops/{my_operator}_${dispatch_namespace}_dispatch.h>.                   \
  See NOTE [XCELERATE_ASSERT_ONLY_METHOD_OPERATORS].
#endif

${DispatchKeyFunctions_inl_includes}


${dispatch_namespaced_declarations}
