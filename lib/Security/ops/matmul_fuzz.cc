#include <gtest/gtest.h>
#include "fuzztest/fuzztest.h"
#include "xcelerate/cc/ops/math_ops.h"
#include "xcelerate/cc/ops/standard_ops.h"
#include "xcelerate/core/framework/types.pb.h"
#include "xcelerate/security/fuzzing/cc/core/framework/tensor_domains.h"
#include "xcelerate/security/fuzzing/cc/fuzz_session.h"

namespace xcelerate {
namespace fuzzing {

// Creates FuzzIdentity class that wraps a single operation node session.
BINARY_INPUT_OP_FUZZER(DT_INT32, DT_INT32, MatMul);
// Setup up fuzzing test.
FUZZ_TEST_F(FuzzMatMul, Fuzz)
    .WithDomains(fuzzing::AnySmallValidNumericTensor(),
                 fuzzing::AnySmallValidNumericTensor());

}  // end namespace fuzzing
}  // end namespace xcelerate
