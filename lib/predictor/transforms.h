#pragma once

#include "Xcelerate/core/net.h"
#include "Xcelerate/core/workspace.h"
#include "Xcelerate/predictor/InferenceGraph.h"

namespace Xcelerate {

void RemoveOpsByType(InferenceGraph& graph_, const std::string& op_type);

} // namespace Xcelerate
