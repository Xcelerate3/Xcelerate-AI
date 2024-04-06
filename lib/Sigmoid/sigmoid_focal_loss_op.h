#ifndef SIGMOID_FOCAL_LOSS_OP_H_
#define SIGMOID_FOCAL_LOSS_OP_H_

#include "xcelerate/core/context.h"
#include "xcelerate/core/logging.h"
#include "xcelerate/core/operator.h"
#include "xcelerate/utils/math.h"

namespace xcelerate {

template <typename T, class Context>
class SigmoidFocalLossOp final : public Operator<Context> {
 public:
  SigmoidFocalLossOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        scale_(this->template GetSingleArgument<float>("scale", 1.)),
        num_classes_(this->template GetSingleArgument<int>("num_classes", 80)),
        gamma_(this->template GetSingleArgument<float>("gamma", 1.)),
        alpha_(this->template GetSingleArgument<float>("alpha", 0.25)) {
    CAFFE_ENFORCE(scale_ >= 0);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  float scale_;
  int num_classes_;
  float gamma_;
  float alpha_;
  Tensor losses_{Context::GetDeviceType()};
  Tensor counts_{Context::GetDeviceType()};
};

template <typename T, class Context>
class SigmoidFocalLossGradientOp final : public Operator<Context> {
 public:
  SigmoidFocalLossGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        scale_(this->template GetSingleArgument<float>("scale", 1.)),
        num_classes_(this->template GetSingleArgument<int>("num_classes", 80)),
        gamma_(this->template GetSingleArgument<float>("gamma", 1.)),
        alpha_(this->template GetSingleArgument<float>("alpha", 0.25)) {
    CAFFE_ENFORCE(scale_ >= 0);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  float scale_;
  int num_classes_;
  float gamma_;
  float alpha_;
  Tensor counts_{Context::GetDeviceType()};
  Tensor weights_{Context::GetDeviceType()}; // unignored weights
};

} // namespace xcelerate

#endif // SIGMOID_FOCAL_LOSS_OP_H_
