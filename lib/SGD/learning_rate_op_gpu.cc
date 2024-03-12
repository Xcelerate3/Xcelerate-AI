#include "xcelerate/core/context_gpu.h"
#include "xcelerate/sgd/learning_rate_op.h"

namespace xcelerate {
REGISTER_CUDA_OPERATOR(LearningRate, LearningRateOp<float, CUDAContext>);
}  // namespace xcelerate
