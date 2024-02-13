#include "Xcelerate/transforms/single_op_transform.h"

#include "Xcelerate/core/common.h"
#include "Xcelerate/core/logging.h"
#include "Xcelerate/core/net.h"
#include "Xcelerate/proto/Xcelerate_pb.h"

namespace Xcelerate {

using transform::Graph;

bool SingleOpTransform::PatternRule(
    const Graph& g,
    const std::vector<int>& subgraph,
    int idx) {
  if (subgraph.size() == 0) {
    return MatchOperator(g.node(idx).op);
  }
  return false;
}

bool SingleOpTransform::ValidatorRule(
    const Graph& /*g*/,
    const std::vector<int>& subgraph) {
  if (subgraph.size() == 1) {
    return true;
  }
  return false;
}

bool SingleOpTransform::ReplaceRule(
    const std::vector<int>& subgraph,
    Graph* g_ptr) {
  CHECK(g_ptr);
  auto& g = *g_ptr;
  ReplaceOperator(&(g.node(subgraph[0]).op));
  return true;
}

} // namespace Xcelerate
