#ifndef xcelerate_SECURITY_FUZZING_CC_CORE_FRAMEWORK_TENSOR_SHAPE_DOMAINS_H_
#define xcelerate_SECURITY_FUZZING_CC_CORE_FRAMEWORK_TENSOR_SHAPE_DOMAINS_H_

#include <limits>
#include <tuple>
#include <utility>

#include "fuzztest/fuzztest.h"
#include "xcelerate/core/framework/tensor_shape.h"

namespace xcelerate::fuzzing {

/// Returns a fuzztest domain with valid TensorShapes.
/// The domain can be customized by setting the maximum rank,
/// and the minimum and maximum size of all dimensions.
fuzztest::Domain<TensorShape> AnyValidTensorShape(
    size_t max_rank = std::numeric_limits<int>::max(),
    int64_t dim_lower_bound = std::numeric_limits<int64_t>::min(),
    int64_t dim_upper_bound = std::numeric_limits<int64_t>::max());

}  // namespace xcelerate::fuzzing

#endif  // xcelerate_SECURITY_FUZZING_CC_CORE_FRAMEWORK_TENSOR_SHAPE_DOMAINS_H_
