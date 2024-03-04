#include "xcelerate/onnx/onnxifi_init.h"

#include <mutex>

#include "xcelerate/core/logging.h"

namespace xcelerate {
namespace onnx {

onnxifi_library* initOnnxifiLibrary() {
  static std::once_flag once;
  static onnxifi_library core{};
  std::call_once(once, []() {
    auto ret = onnxifi_load(ONNXIFI_LOADER_FLAG_VERSION_1_0, nullptr, &core);
    if (!ret) {
      CAFFE_THROW("Cannot load onnxifi lib");
    }
  });
  return &core;
}
} // namespace onnx
} // namespace xcelerate
