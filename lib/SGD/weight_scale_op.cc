#include "xcelerate/sgd/weight_scale_op.h"

namespace xcelerate {

REGISTER_CPU_OPERATOR(WeightScale, WeightScaleOp<CPUContext>);
OPERATOR_SCHEMA(WeightScale)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{0, 0}, {1, 1}})
    .DeviceInferenceFunction([](const OperatorDef& def) {
      auto op_device =
          def.has_device_option() ? def.device_option() : DeviceOption();
      vector<DeviceOption> in_dev(def.input_size(), op_device);
      vector<DeviceOption> out_dev(def.output_size(), op_device);
      // ITER input lives on CPU
      in_dev[1] = DeviceOption();
      return std::make_pair(in_dev, out_dev);
    })
    .SetDoc(R"DOC(
Every `stepsize` iterations, multiply the weights by a constant `scale`:
    nw = w * scale
)DOC")
    .Input(0, "w", "Current weights")
    .Input(1, "iter", "Training Iteration")
    .Output(0, "nw", "Updated weights")
    .Arg("stepsize", "Every iteration number to do weight scaling")
    .Arg(
        "upper_bound_iter",
        "After iter passes this bound, do not perform the weight rescaling")
    .Arg("scale", "The multiplicative factor applied to weights.");

SHOULD_NOT_DO_GRADIENT(WeightScale);
} // namespace xcelerate
