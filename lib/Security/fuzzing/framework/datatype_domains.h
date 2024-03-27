#ifndef xcelerate_SECURITY_FUZZING_CC_CORE_FRAMEWORK_DATATYPE_DOMAINS_H_
#define xcelerate_SECURITY_FUZZING_CC_CORE_FRAMEWORK_DATATYPE_DOMAINS_H_

#include "fuzztest/fuzztest.h"
#include "xcelerate/core/framework/types.pb.h"

namespace xcelerate::fuzzing {

/// Returns a fuzztest domain of valid DataTypes to construct a Tensor
fuzztest::Domain<DataType> AnyValidDataType();

}  // namespace xcelerate::fuzzing

#endif  // xcelerate_SECURITY_FUZZING_CC_CORE_FRAMEWORK_DATATYPE_DOMAINS_H_
