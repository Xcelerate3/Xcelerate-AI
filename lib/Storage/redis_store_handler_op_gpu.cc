#include "xcelerate/distributed/redis_store_handler_op.h"

#if !defined(USE_ROCM)
#include <xcelerate/core/context_gpu.h>
#else
#include <xcelerate/core/hip/context_gpu.h>
#endif

namespace xcelerate {

#if !defined(USE_ROCM)
REGISTER_CUDA_OPERATOR(
    RedisStoreHandlerCreate,
    RedisStoreHandlerCreateOp<CUDAContext>);
#else
REGISTER_HIP_OPERATOR(
    RedisStoreHandlerCreate,
    RedisStoreHandlerCreateOp<HIPContext>);
#endif

} // namespace xcelerate
