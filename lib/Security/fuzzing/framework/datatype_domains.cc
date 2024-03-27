#include "xcelerate/security/fuzzing/cc/core/framework/datatype_domains.h"

namespace xcelerate::fuzzing {

fuzztest::Domain<DataType> AnyValidDataType() {
  return fuzztest::ElementOf({
      DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8, DT_INT16, DT_INT8, DT_INT64,
      DT_BOOL, DT_UINT16, DT_UINT32, DT_UINT64
      // TODO(b/268338352): add unsupported types
      // DT_STRING, DT_COMPLEX64, DT_QINT8, DT_QUINT8, DT_QINT32,
      // DT_BFLOAT16, DT_QINT16, DT_COMPLEX128, DT_HALF, DT_RESOURCE,
      // DT_VARIANT, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN
  });
}

}  // namespace xcelerate::fuzzing
