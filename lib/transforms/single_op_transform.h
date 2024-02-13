#pragma once

#include "Xcelerate/core/common.h"
#include "Xcelerate/core/transform.h"
#include "Xcelerate/proto/Xcelerate_pb.h"
#include "Xcelerate/utils/proto_utils.h"

namespace Xcelerate {

class TORCH_API SingleOpTransform : public Transform {
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

  // Specify what the op needs to be to match the pattern.
  virtual bool MatchOperator(const OperatorDef& op) = 0;

  // Specify how the operator should be replaced.
  virtual void ReplaceOperator(OperatorDef* op) = 0;
};

} // namespace Xcelerate
