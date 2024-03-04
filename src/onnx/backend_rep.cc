#include "xcelerate/onnx/backend_rep.h"
#include "xcelerate/core/common.h"

#include <iostream>

namespace xcelerate {
namespace onnx {

void xcelerateBackendRep::CheckInit() {
  if (!predictor_) {
    predictor_ = std::make_unique<xcelerate::Predictor>(
        makePredictorConfig(init_net_, pred_net_));
    init_net_.Clear();
    pred_net_.Clear();
  }
}

void xcelerateBackendRep::Run(
    const xcelerate::Predictor::TensorList& inputs,
    xcelerate::Predictor::TensorList* outputs) {
  CheckInit();
  (*predictor_)(inputs, outputs);
}

void xcelerateBackendRep::RunMap(
    const xcelerate::Predictor::TensorMap& inputs,
    xcelerate::Predictor::TensorList* outputs) {
  CheckInit();
  (*predictor_)(inputs, outputs);
}

} // namespace onnx
} // namespace xcelerate
