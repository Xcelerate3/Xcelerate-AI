#include <string>

#include <gtest/gtest.h>
#include "fuzztest/fuzztest.h"
#include "xcelerate/cc/ops/standard_ops.h"
#include "xcelerate/security/fuzzing/cc/fuzz_session.h"

namespace xcelerate {
namespace fuzzing {

// Creates FuzzStringToNumber class that wraps a single operation node session.
class FuzzStringToNumber : public FuzzSession<std::string> {
  void BuildGraph(const Scope& scope) override {
    auto op_node =
        xcelerate::ops::Placeholder(scope.WithOpName("input"), DT_STRING);
    xcelerate::ops::StringToNumber(scope.WithOpName("output"), op_node);
  }
  void FuzzImpl(const std::string& input_string) final {
    Tensor input_tensor(xcelerate::DT_STRING, TensorShape({}));
    input_tensor.scalar<xcelerate::tstring>()() = input_string;
    Status s = RunInputsWithStatus({{"input", input_tensor}});
    if (!s.ok()) {
      LOG(ERROR) << "Execution failed: " << s.message();
    }
  }
};

// Setup up fuzzing test.
FUZZ_TEST_F(FuzzStringToNumber, Fuzz)
    .WithDomains(fuzztest::OneOf(fuzztest::InRegexp("[-.0-9]+"),
                                 fuzztest::Arbitrary<std::string>()));

}  // end namespace fuzzing
}  // end namespace xcelerate
