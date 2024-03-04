#pragma once

#include "xcelerate/predictor/predictor.h"
#include "xcelerate/proto/xcelerate_pb.h"

#include <memory>
#include <string>
#include <vector>

namespace xcelerate {
namespace onnx {
class TORCH_API xcelerateBackendRep {
 public:
  void Run(
      const xcelerate::Predictor::TensorList& inputs,
      xcelerate::Predictor::TensorList* outputs);
  void RunMap(
      const xcelerate::Predictor::TensorMap& inputs,
      xcelerate::Predictor::TensorList* outputs);

  xcelerate::NetDef& init_net() {
    return init_net_;
  }
  xcelerate::NetDef& pred_net() {
    return pred_net_;
  }
  std::vector<std::string>& uninitialized_inputs() {
    return uninitialized_inputs_;
  }

  const xcelerate::NetDef& init_net() const {
    return init_net_;
  }
  const xcelerate::NetDef& pred_net() const {
    return pred_net_;
  }
  const std::vector<std::string>& uninitialized_inputs() const {
    return uninitialized_inputs_;
  }

 private:
  void CheckInit();

  xcelerate::NetDef init_net_;
  xcelerate::NetDef pred_net_;
  std::vector<std::string> uninitialized_inputs_;
  std::unique_ptr<xcelerate::Predictor> predictor_{nullptr};
};
} // namespace onnx
} // namespace xcelerate
