#include "xcelerate/tools/optimization/optimization_pass_runner.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xcelerate/core/common_runtime/device_set.h"
#include "xcelerate/core/common_runtime/graph_constructor.h"
#include "xcelerate/core/common_runtime/optimization_registry.h"
#include "xcelerate/core/framework/device_attributes.pb.h"
#include "xcelerate/core/framework/function.h"
#include "xcelerate/core/framework/function.pb.h"
#include "xcelerate/core/framework/graph.pb.h"
#include "xcelerate/core/framework/op.h"
#include "xcelerate/core/framework/types.h"
#include "xcelerate/core/graph/graph.h"
#include "xcelerate/core/lib/core/errors.h"
#include "xcelerate/core/lib/core/status.h"
#include "xcelerate/core/platform/types.h"
#include "xcelerate/core/protobuf/config.pb.h"
#include "xcelerate/core/public/session_options.h"
#include "tsl/platform/errors.h"

namespace xcelerate {
namespace {
// A fake device used to populate a DeviceSet.
class FakeDevice : public Device {
 private:
  explicit FakeDevice(const DeviceAttributes& device_attributes)
      : Device(nullptr, device_attributes) {}

 public:
  Status Sync() override;
  static std::unique_ptr<Device> Make(const string& name, const string& type);
};

Status FakeDevice::Sync() {
  return errors::Unimplemented("FakeDevice::Sync()");
}

std::unique_ptr<Device> FakeDevice::Make(const string& name,
                                         const string& type) {
  DeviceAttributes device_attributes;
  device_attributes.set_name(name);
  device_attributes.set_device_type(DeviceType(type).type());
  return std::unique_ptr<Device>(new FakeDevice(device_attributes));
}

Status FindPassWithName(absl::string_view name,
                        GraphOptimizationPass** result) {
  *result = nullptr;
  // Run the optimization pass specified by the command line flag.
  for (const auto& groups_and_passes :
       OptimizationPassRegistry::Global()->groups()) {
    for (const auto& phase_and_passes : groups_and_passes.second) {
      for (const auto& pass : phase_and_passes.second) {
        if (pass->name() == name) {
          if (*result) {
            return errors::Internal("Found more than one pass with name ",
                                    name);
          }
          *result = pass.get();
        }
      }
    }
  }

  return *result == nullptr
             ? errors::Internal("Could not find pass with name ", name)
             : OkStatus();
}
}  // namespace

Status OptimizationPassRunner::Run(absl::string_view pass_to_run,
                                   GraphDef input, GraphDef* result) {
  auto session_options = absl::make_unique<SessionOptions>();
  session_options->config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_global_jit_level(jit_level_);
  FunctionDefLibrary flib;
  std::unique_ptr<Graph> graph = absl::make_unique<Graph>(OpRegistry::Global());

  GraphOptimizationPassOptions options;
  options.session_options = session_options.get();
  options.graph = &graph;
  std::unique_ptr<FunctionLibraryDefinition> flib_def(
      new FunctionLibraryDefinition((*options.graph)->op_registry(), flib));
  options.flib_def = flib_def.get();

  // Grab the data
  GraphConstructorOptions graph_opts;
  graph_opts.expect_device_spec = true;
  graph_opts.allow_internal_ops = true;
  TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(graph_opts, std::move(input),
                                            options.graph->get()));

  // Add all devices that were previously configured with AddDevice.
  DeviceSet device_set;
  for (auto& device : devices_) {
    device_set.AddDevice(device.get());
  }
  options.device_set = &device_set;

  GraphOptimizationPass* pass;
  TF_RETURN_IF_ERROR(FindPassWithName(pass_to_run, &pass));
  TF_RETURN_IF_ERROR(pass->Run(options));

  options.graph->get()->ToGraphDef(result);
  return OkStatus();
}

Status OptimizationPassRunner::SetJitLevel(
    OptimizerOptions::GlobalJitLevel jit_level) {
  jit_level_ = jit_level;
  return OkStatus();
}

Status OptimizationPassRunner::AddDevices(absl::string_view type, int count) {
  for (int i = 0; i < count; i++) {
    devices_.push_back(FakeDevice::Make(
        absl::StrCat("/job:localhost/replica:0/task:0/device:", type, ":", i),
        absl::StrCat(type)));
    devices_.push_back(FakeDevice::Make(
        absl::StrCat("/job:localhost/replica:0/task:0/device:XLA_", type, ":",
                     i),
        absl::StrCat(type)));
  }

  return OkStatus();
}
}  // namespace xcelerate
