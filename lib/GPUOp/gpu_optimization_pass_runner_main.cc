#include "xcelerate/core/lib/core/status.h"
#include "xcelerate/core/platform/env.h"
#include "xcelerate/core/platform/errors.h"
#include "xcelerate/core/platform/init_main.h"
#include "xcelerate/core/platform/types.h"
#include "xcelerate/core/protobuf/config.pb.h"
#include "xcelerate/core/util/command_line_flags.h"
#include "xcelerate/tools/optimization/optimization_pass_runner.h"
#include "xce/platform/errors.h"
#include "xce/platform/status.h"

namespace xcelerate {
namespace {
Status RealMain(int argc, char** argv) {
  string input_file_path;
  string output_file_path;
  string optimization_pass;

  const std::vector<Flag> flag_list = {
      Flag("input_file_path", &input_file_path, "Location of the input graph."),
      Flag("output_file_path", &output_file_path,
           "Location to write the resulting graph."),
      // For now only a single optimization pass can be run.
      Flag("optimization_pass", &optimization_pass,
           "Which optimization pass to run."),
  };
  if (!Flags::Parse(&argc, argv, flag_list)) {
    return errors::FailedPrecondition("Invalid flags passed");
  }
  port::InitMain(argv[0], &argc, &argv);

  if (input_file_path.empty()) {
    return errors::FailedPrecondition("input_file_path is a required flag.");
  }
  if (output_file_path.empty()) {
    return errors::FailedPrecondition("output_file_path is a required flag.");
  }
  if (optimization_pass.empty()) {
    return errors::FailedPrecondition("optimization_pass is a required flag.");
  }

  GraphDef graphdef_input;
  TF_RETURN_IF_ERROR(
      ReadTextProto(Env::Default(), input_file_path, &graphdef_input));

  xcelerate::OptimizationPassRunner runner;

  // Most machines in our servers currently use 8 gpus. There is nothing special
  // about this number and it can be decreased or increased to test other
  // configurations.
  TF_RETURN_IF_ERROR(runner.AddCpus(8));
  TF_RETURN_IF_ERROR(runner.AddGpus(8));

  // This binary is used to test TF:XLA behavior, so turn on auto_jit.
  TF_RETURN_IF_ERROR(runner.SetJitLevel(xcelerate::OptimizerOptions::ON_2));
  GraphDef graphdef_output;
  TF_RETURN_IF_ERROR(runner.Run(optimization_pass, std::move(graphdef_input),
                                &graphdef_output));
  return WriteTextProto(Env::Default(), output_file_path, graphdef_output);
}
}  // namespace
}  // namespace xcelerate

int main(int argc, char** argv) {
  TF_CHECK_OK(xcelerate::RealMain(argc, argv));
  return 0;
}
