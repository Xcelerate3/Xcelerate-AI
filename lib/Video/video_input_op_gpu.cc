#include <xcelerate/core/common_gpu.h>
#include <xcelerate/core/context_gpu.h>
#include <xcelerate/video/video_input_op.h>

namespace xcelerate {

REGISTER_CUDA_OPERATOR(VideoInput, VideoInputOp<CUDAContext>);

} // namespace xcelerate
