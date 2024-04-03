#ifndef Xcelerate_IMAGE_TRANSFORM_GPU_H_
#define Xcelerate_IMAGE_TRANSFORM_GPU_H_

#include "Xcelerate/core/context.h"

namespace Xcelerate {

template <typename T_IN, typename T_OUT, class Context>
bool TransformOnGPU(
    Tensor& X,
    Tensor* Y,
    Tensor& mean,
    Tensor& std,
    Context* context);

}  // namespace Xcelerate

#endif
