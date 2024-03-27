#ifndef xcelerate_SECURITY_FUZZING_CC_CORE_FRAMEWORK_TENSOR_DOMAINS_H_
#define xcelerate_SECURITY_FUZZING_CC_CORE_FRAMEWORK_TENSOR_DOMAINS_H_

#include <string>

#include "fuzztest/fuzztest.h"
#include "xcelerate/core/framework/tensor.h"
#include "xcelerate/core/framework/types.pb.h"

namespace xcelerate::fuzzing {

inline constexpr double kDefaultMaxAbsoluteValue = 100.0;

/// Returns a fuzztest domain of tensors of the specified shape and datatype
fuzztest::Domain<Tensor> AnyValidNumericTensor(
    const TensorShape& shape, DataType datatype,
    double min = -kDefaultMaxAbsoluteValue,
    double max = kDefaultMaxAbsoluteValue);

/// Returns a fuzztest domain of tensors with shape and datatype
/// that live in the given corresponding domains.
fuzztest::Domain<Tensor> AnyValidNumericTensor(
    fuzztest::Domain<TensorShape> tensor_shape_domain,
    fuzztest::Domain<DataType> datatype_domain,
    double min = -kDefaultMaxAbsoluteValue,
    double max = kDefaultMaxAbsoluteValue);

// Returns a fuzztest domain of tensor of max rank 5, with dimensions sizes
// between 0 and 10 and values between -10 and 10.
fuzztest::Domain<Tensor> AnySmallValidNumericTensor(
    DataType datatype = DT_INT32);

fuzztest::Domain<Tensor> AnyValidStringTensor(
    const TensorShape& shape, fuzztest::Domain<std::string> string_domain =
                                  fuzztest::Arbitrary<std::string>());

}  // namespace xcelerate::fuzzing

#endif  // xcelerate_SECURITY_FUZZING_CC_CORE_FRAMEWORK_TENSOR_DOMAINS_H_
