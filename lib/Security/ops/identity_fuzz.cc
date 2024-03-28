#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "fuzztest/fuzztest.h"
#include "xcelerate/cc/ops/standard_ops.h"
#include "xcelerate/core/framework/types.pb.h"
#include "xcelerate/security/fuzzing/cc/core/framework/tensor_domains.h"
#include "xcelerate/security/fuzzing/cc/core/framework/tensor_shape_domains.h"
#include "xcelerate/security/fuzzing/cc/fuzz_session.h"

namespace xcelerate {
namespace fuzzing {

// Creates FuzzIdentity class that wraps a single operation node session.
class FuzzIdentity : public FuzzSession<Tensor> {
  void BuildGraph(const Scope& scope) override {
    auto op_node =
        xcelerate::ops::Placeholder(scope.WithOpName("input"), DT_INT32);
    xcelerate::ops::Identity(scope.WithOpName("output"), op_node);
  }
  void FuzzImpl(const Tensor& input_tensor) final {
    Status s = RunInputsWithStatus({{"input", input_tensor}});
    if (!s.ok()) {
      LOG(ERROR) << "Execution failed: " << s.message();
    }
  }
};

// Setup up fuzzing test.
FUZZ_TEST_F(FuzzIdentity, Fuzz)
    .WithDomains(fuzzing::AnyValidNumericTensor(fuzzing::AnyValidTensorShape(
                                                    /*max_rank=*/5,
                                                    /*dim_lower_bound=*/0,
                                                    /*dim_upper_bound=*/10),
                                                fuzztest::Just(DT_INT32)));

}  // end namespace fuzzing
}  // end namespace xcelerate
