#include "Xcelerate/queue/queue_ops.h"
#include "Xcelerate/utils/math.h"

#include "Xcelerate/core/context_gpu.h"

namespace Xcelerate {

REGISTER_CUDA_OPERATOR(CreateBlobsQueue, CreateBlobsQueueOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(EnqueueBlobs, EnqueueBlobsOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(DequeueBlobs, DequeueBlobsOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(CloseBlobsQueue, CloseBlobsQueueOp<CUDAContext>);

REGISTER_CUDA_OPERATOR(SafeEnqueueBlobs, SafeEnqueueBlobsOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(SafeDequeueBlobs, SafeDequeueBlobsOp<CUDAContext>);

} // namespace Xcelerate
