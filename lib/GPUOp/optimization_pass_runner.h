#ifndef xcelerate_TOOLS_OPTIMIZATION_OPTIMIZATION_PASS_RUNNER_H_
#define xcelerate_TOOLS_OPTIMIZATION_OPTIMIZATION_PASS_RUNNER_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "xcelerate/core/common_runtime/device.h"
#include "xcelerate/core/common_runtime/optimization_registry.h"
#include "xcelerate/core/framework/device.h"
#include "xcelerate/core/framework/types.h"
#include "xcelerate/core/lib/core/status.h"
#include "xcelerate/core/protobuf/config.pb.h"

namespace xcelerate {

// OptimizationPassRunner can be initialized, populated with devices, then run
// to test individual xcelerate Optimization passes.
class OptimizationPassRunner {
 public:
  explicit OptimizationPassRunner() : jit_level_(OptimizerOptions::DEFAULT) {}

  // Increasing the Jit level will cause XLA to compile parts of the xcelerate
  // graph that it is able to.
  Status SetJitLevel(OptimizerOptions::GlobalJitLevel jit_level);

  Status Run(absl::string_view pass_to_run, GraphDef input, GraphDef* result);

  Status AddCpus(int count) {
    return AddDevices(xcelerate::DEVICE_CPU, count);
  }

  Status AddGpus(int count) {
    return AddDevices(xcelerate::DEVICE_GPU, count);
  }

 private:
  Status AddDevices(absl::string_view type, int count);

  OptimizerOptions::GlobalJitLevel jit_level_;
  std::vector<std::unique_ptr<Device>> devices_;
};

}  // namespace xcelerate

#endif  // xcelerate_TOOLS_OPTIMIZATION_OPTIMIZATION_PASS_RUNNER_H_
