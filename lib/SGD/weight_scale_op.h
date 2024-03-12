#pragma once

#include "xcelerate/core/operator.h"

#include <stdlib.h>
#include <time.h>

namespace xcelerate {

template <typename T, class Context>
void weight_scale_update(
    int N,
    const T* w,
    const T scale,
    int64_t iter,
    int64_t stepsize,
    int64_t update_upper_bound,
    T* nw,
    Context* context) {
  const auto w_size = N * sizeof(float);
  if (iter % stepsize != 0 || iter >= update_upper_bound) {
    memcpy(nw, w, w_size);
    return;
  }
  // perform the weight scaling
  xcelerate::math::Scale<T, T, Context>(N, scale, w, nw, context);
}

template <class Context>
class WeightScaleOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  WeightScaleOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        stepsize_(OperatorBase::GetSingleArgument<int64_t>(
            "stepsize",
            std::numeric_limits<int64_t>::max())),
        update_upper_bound_(OperatorBase::GetSingleArgument<int64_t>(
            "upper_bound_iter",
            std::numeric_limits<int64_t>::max())),
        scale_(this->template GetSingleArgument<float>("scale", 1.0f)) {}

  bool RunOnDevice() override {
    Output(OUTPUT_WEIGHTS)->ResizeLike(Input(WEIGHTS));
    return DispatchHelper<TensorTypes<float>>::call(this, Input(WEIGHTS));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto iter =
        OperatorBase::Input<Tensor>(ITER, CPU).template data<int64_t>()[0] + 1;

    weight_scale_update<T, Context>(
        Input(WEIGHTS).size(),
        Input(WEIGHTS).template data<T>(),
        scale_,
        iter,
        stepsize_,
        update_upper_bound_,
        Output(OUTPUT_WEIGHTS)->template mutable_data<T>(),
        &context_);
    return true;
  }

 protected:
  int64_t stepsize_;
  int64_t update_upper_bound_;
  float scale_;
  INPUT_TAGS(WEIGHTS, ITER);
  OUTPUT_TAGS(OUTPUT_WEIGHTS);
};

} // namespace xcelerate
