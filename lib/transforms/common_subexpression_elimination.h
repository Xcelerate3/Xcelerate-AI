
#pragma once

#include "Xcelerate/core/common.h"
#include "Xcelerate/core/transform.h"
#include "Xcelerate/proto/Xcelerate_pb.h"
#include "Xcelerate/utils/proto_utils.h"

namespace Xcelerate {

class TORCH_API CommonSubexpressionEliminationTransform : public Transform {
 public:
  CommonSubexpressionEliminationTransform() {
    SetPatternMatchType(SORTED_WRT_EXECUTION_ORDER);
  }

 protected:
  bool PatternRule(
      const transform::Graph& g,
      const std::vector<int>& subgraph,
      int idx) override;
  bool ValidatorRule(
      const transform::Graph& g,
      const std::vector<int>& subgraph) override;
  bool ReplaceRule(const std::vector<int>& subgraph, transform::Graph* g_ptr)
      override;

 private:
  bool IsAllowed(string op_type) {
    return allowed_ops_.count(op_type);
  }
  std::set<string> allowed_ops_ = {"LearningRate", "FC"};
};

} // namespace Xcelerate
