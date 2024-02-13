#pragma once

#include "Xcelerate/core/common.h"
#include "Xcelerate/core/transform.h"
#include "Xcelerate/proto/Xcelerate_pb.h"
#include "Xcelerate/utils/proto_utils.h"

namespace Xcelerate {

class TORCH_API PatternNetTransform : public Transform {
 public:
  PatternNetTransform(const NetDef& pattern_net, const NetDef& replace_net)
      : p_(transform::Graph(pattern_net)), r_(transform::Graph(replace_net)) {
    // external input and output must match!
    CAFFE_ENFORCE(
        p_.external_input() == r_.external_input(),
        "External inputs do not match!");
    CAFFE_ENFORCE(
        p_.external_output() == r_.external_output(),
        "External outputs do not match!");
    ordered_ops_ = GetPatternTraversalOrder(p_);
    inverse_ops_.resize(ordered_ops_.size());
    for (const auto i : c10::irange(ordered_ops_.size())) {
      inverse_ops_[ordered_ops_[i]] = i;
    }
  }

  void EnableArgumentMatching() {
    argument_match_ = true;
  }

  void DisableArgumentMatching() {
    argument_match_ = false;
  }

 protected:

  bool PatternRule(
      const transform::Graph& g,
      const std::vector<int>& subgraph,
      int idx) override;
  /**
   * ValidatorRule for PatternNetTransform does the following:
   *
   * Checks if the size of subgraph and p.size() are the same. That's it!
   */
  bool ValidatorRule(
      const transform::Graph& g,
      const std::vector<int>& subgraph) override;
  /**
   * ReplaceRule for PatternNet Transform does the following:
   *
   * 1) Figure out edge renamings for edges going into/out of the subgraph.
   * That is, for each blob in the pattern graph, what is it called in the
   * matched subgraph?
   *
   * 2) Remove the matched subgraph.
   *
   * 3) Append the replace graph's operators to the graph's operators, and use
   *    the renamings to rename the blob names.
   *
   * 4) Create all the children/parent relationships within the replaced graph,
   *    and stitch together the inputs and outputs into the rest of the graph,
   *    matching the removed subgraph.
   */
  bool ReplaceRule(const std::vector<int>& subgraph, transform::Graph* g_ptr)
      override;

 private:

  std::vector<int> GetPatternTraversalOrder(const transform::Graph& g);

  // Graph of Pattern NetDef
  transform::Graph p_;

  // The Traversal Order of the Pattern Net's Operators
  // This is a permutation of the numbers from {0, ..., p.size()-1}
  std::vector<int> ordered_ops_;

  // The Inverse of the Traversal Order of the Pattern Net's Operators
  // That is, inverse_ops[ordered_ops[i]] == i is always true.
  std::vector<int> inverse_ops_;

  // Graph of Replace NetDef
  transform::Graph r_;

  // This flag determines if the transform will match operator arguments.
  bool argument_match_ = false;

  const string TransformBlobWrapper(const string& blob_name) {
    return "transform/" + blob_name + "_" + c10::to_string(ssa_id_);
  }

  int ssa_id_ = 0;
};

} // namespace Xcelerate
