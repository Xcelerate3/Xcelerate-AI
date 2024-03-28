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

// Creates FuzzConcat class that wraps a single operation node session.
class FuzzConcat : public FuzzSession<Tensor, Tensor, int32> {
  void BuildGraph(const Scope& scope) override {
    auto value1 =
        xcelerate::ops::Placeholder(scope.WithOpName("value1"), DT_INT32);
    Input value1_input(value1);
    auto value2 =
        xcelerate::ops::Placeholder(scope.WithOpName("value2"), DT_INT32);
    Input value2_input(value2);
    InputList values_input_list({value1_input, value2_input});
    auto axis =
        xcelerate::ops::Placeholder(scope.WithOpName("axis"), DT_INT32);
    xcelerate::ops::Concat(scope.WithOpName("output"), values_input_list,
                            axis);
  }
  void FuzzImpl(const Tensor& value1, const Tensor& value2,
                const int32& axis) final {
    Tensor axis_tensor(DT_INT32, {});
    axis_tensor.scalar<int32_t>()() = axis;
    Status s = RunInputsWithStatus(
        {{"value1", value1}, {"value2", value2}, {"axis", axis_tensor}});
    if (!s.ok()) {
      LOG(ERROR) << "Execution failed: " << s.message();
    }
  }
};

// Setup up fuzzing test.
FUZZ_TEST_F(FuzzConcat, Fuzz)
    .WithDomains(fuzzing::AnyValidNumericTensor(fuzzing::AnyValidTensorShape(
                                                    /*max_rank=*/5,
                                                    /*dim_lower_bound=*/0,
                                                    /*dim_upper_bound=*/10),
                                                fuzztest::Just(DT_INT32)),
                 fuzzing::AnyValidNumericTensor(fuzzing::AnyValidTensorShape(
                                                    /*max_rank=*/5,
                                                    /*dim_lower_bound=*/0,
                                                    /*dim_upper_bound=*/10),
                                                fuzztest::Just(DT_INT32)),
                 fuzztest::InRange<int32>(0, 6));

}  // end namespace fuzzing
}  // end namespace xcelerate
