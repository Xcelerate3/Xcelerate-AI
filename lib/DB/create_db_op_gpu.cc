#include "xcelerate/core/context_gpu.h"
#include "xcelerate/db/create_db_op.h"

namespace xcelerate {
REGISTER_CUDA_OPERATOR(CreateDB, CreateDBOp<CUDAContext>);
} // namespace xcelerate
