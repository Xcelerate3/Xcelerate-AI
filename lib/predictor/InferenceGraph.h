#pragma once

#include "Xcelerate/core/workspace.h"

namespace Xcelerate {

struct InferenceGraph {
  std::unique_ptr<NetDef> predict_init_net_def;
  // shared_ptr allows to share NetDef with its operators on each of the threads
  // without memory replication. Note that predict_init_net_def_ could be stored
  // by value as its operators are discarded immidiatly after use (via
  // RunNetOnce)
  std::shared_ptr<NetDef> predict_net_def;

  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::vector<std::string> parameter_names;

  bool predictor_net_ssa_rewritten{false};
};
} // namespace Xcelerate
