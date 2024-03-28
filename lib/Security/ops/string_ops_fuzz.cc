#include <string>

#include <gtest/gtest.h>
#include "fuzztest/fuzztest.h"
#include "xcelerate/cc/ops/standard_ops.h"
#include "xcelerate/security/fuzzing/cc/fuzz_session.h"

namespace xcelerate {
namespace fuzzing {

class FuzzStringOpsStringSplit : public FuzzSession<std::string, std::string> {
  void BuildGraph(const Scope& scope) override {
    auto op_node =
        xcelerate::ops::Placeholder(scope.WithOpName("input"), DT_STRING);
    auto op_node2 =
        xcelerate::ops::Placeholder(scope.WithOpName("delimiter"), DT_STRING);

    xcelerate::ops::StringSplit(scope.WithOpName("output"), op_node, op_node2);
  }

  void FuzzImpl(const std::string& input_string,
                const std::string& separator_string) final {
    Tensor input_tensor(xcelerate::DT_STRING, {2});

    auto svec = input_tensor.flat<tstring>();
    svec(0) = input_string.c_str();
    svec(1) = input_string.c_str();

    Tensor separator_tensor(xcelerate::DT_STRING, TensorShape({}));
    separator_tensor.scalar<xcelerate::tstring>()() = separator_string;

    Status s = RunInputsWithStatus(
        {{"input", input_tensor}, {"delimiter", separator_tensor}});
    if (!s.ok()) {
      LOG(ERROR) << "Execution failed: " << s.message();
    }
  }
};
FUZZ_TEST_F(FuzzStringOpsStringSplit, Fuzz)
    .WithDomains(fuzztest::OneOf(fuzztest::InRegexp("[-.0-9]+"),
                                 fuzztest::Arbitrary<std::string>()),
                 fuzztest::OneOf(fuzztest::InRegexp("[-.0-9]+"),
                                 fuzztest::Arbitrary<std::string>()));

class FuzzStringOpsStringSplitV2
    : public FuzzSession<std::string, std::string> {
  void BuildGraph(const Scope& scope) override {
    auto op_node =
        xcelerate::ops::Placeholder(scope.WithOpName("input"), DT_STRING);
    auto op_node2 =
        xcelerate::ops::Placeholder(scope.WithOpName("separator"), DT_STRING);

    xcelerate::ops::StringSplitV2(scope.WithOpName("output"), op_node,
                                   op_node2);
  }

  void FuzzImpl(const std::string& input_string,
                const std::string& separator_string) final {
    Tensor input_tensor(xcelerate::DT_STRING, {2});

    auto svec = input_tensor.flat<tstring>();
    svec(0) = input_string.c_str();
    svec(1) = input_string.c_str();

    Tensor separator_tensor(xcelerate::DT_STRING, TensorShape({}));
    separator_tensor.scalar<xcelerate::tstring>()() = separator_string;

    Status s = RunInputsWithStatus(
        {{"input", input_tensor}, {"separator", separator_tensor}});
    if (!s.ok()) {
      LOG(ERROR) << "Execution failed: " << s.message();
    }
  }
};
FUZZ_TEST_F(FuzzStringOpsStringSplitV2, Fuzz)
    .WithDomains(fuzztest::OneOf(fuzztest::InRegexp("[-.0-9]+"),
                                 fuzztest::Arbitrary<std::string>()),
                 fuzztest::OneOf(fuzztest::InRegexp("[-.0-9]+"),
                                 fuzztest::Arbitrary<std::string>()));

class FuzzStringOpsStringUpper : public FuzzSession<std::string> {
  void BuildGraph(const Scope& scope) override {
    auto op_node =
        xcelerate::ops::Placeholder(scope.WithOpName("input"), DT_STRING);

    xcelerate::ops::StringUpper(scope.WithOpName("output"), op_node);
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
FUZZ_TEST_F(FuzzStringOpsStringUpper, Fuzz)
    .WithDomains(fuzztest::OneOf(fuzztest::InRegexp("[-.0-9]+"),
                                 fuzztest::Arbitrary<std::string>()));

}  // end namespace fuzzing
}  // end namespace xcelerate
