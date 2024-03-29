#pragma once

#include "Xcelerate/core/common.h"
#include "Xcelerate/proto/Xcelerate_pb.h"
#include "Xcelerate/transforms/single_op_transform.h"
#include "Xcelerate/utils/proto_utils.h"

namespace Xcelerate {

class TORCH_API ConvToNNPackTransform : public SingleOpTransform {
 protected:
  // Specify what the op needs to be to match the pattern.
  bool MatchOperator(const OperatorDef& op) override {
    return (
        op.type() == "Conv" && op.device_option().device_type() == PROTO_CPU &&
        op.engine() != "NNPACK");
  }

  // Specify how the operator should be replaced.
  void ReplaceOperator(OperatorDef* op) override {
    op->set_engine("NNPACK");
  }
};

} // namespace Xcelerate
