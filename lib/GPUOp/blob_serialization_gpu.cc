#include "xcelerate/core/blob.h"
#include "xcelerate/core/blob_serialization.h"
#include "xcelerate/core/context_gpu.h"

namespace xcelerate {

namespace {
REGISTER_BLOB_DESERIALIZER(TensorCUDA, TensorDeserializer);
}
}  // namespace xcelerate
