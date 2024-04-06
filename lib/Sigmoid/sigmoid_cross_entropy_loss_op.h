#ifndef SIGMOID_CROSS_ENTROPY_LOSS_OP_H_
#define SIGMOID_CROSS_ENTROPY_LOSS_OP_H_

#include "xcelerate/core/context.h"
#include "xcelerate/core/logging.h"
#include "xcelerate/core/operator.h"
#include "xcelerate/utils/math.h"

namespace xcelerate {

template <typename T, class Context>
class SigmoidCrossEntropyLossOp final : public Operator<Context> {
 public:
  SigmoidCrossEntropyLossOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        scale_(this->template GetSingleArgument<float>("scale", 1.)),
        normalize_(this->template GetSingleArgument<int>("normalize", 1)) {
    CAFFE_ENFORCE(scale_ >= 0);
    CAFFE_ENFORCE(normalize_ == 0 || normalize_ == 1);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  float scale_;
  int normalize_;
  Tensor losses_{Context::GetDeviceType()};
  Tensor counts_{Context::GetDeviceType()};
  Tensor normalizer_;
};

template <typename T, class Context>
class SigmoidCrossEntropyLossGradientOp final : public Operator<Context> {
 public:
  SigmoidCrossEntropyLossGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        scale_(this->template GetSingleArgument<float>("scale", 1.)),
        normalize_(this->template GetSingleArgument<int>("normalize", 1)) {
    CAFFE_ENFORCE(scale_ >= 0);
    CAFFE_ENFORCE(normalize_ == 0 || normalize_ == 1);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  float scale_;
  int normalize_;
  Tensor counts_{Context::GetDeviceType()};
  Tensor normalizer_;
};

} // namespace xcelerate

#endif // SIGMOID_CROSS_ENTROPY_LOSS_OP_H_
