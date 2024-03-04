#pragma once

#include "xcelerate/core/common.h"
#include "xcelerate/core/tensor.h"
#include "xcelerate/onnx/helper.h"
#include "xcelerate/proto/xcelerate_pb.h"
#include "onnx/onnx_pb.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace xcelerate {
namespace onnx {

namespace {
using ::ONNX_NAMESPACE::AttributeProto;
using ::ONNX_NAMESPACE::GraphProto;
using ::ONNX_NAMESPACE::ModelProto;
using ::ONNX_NAMESPACE::NodeProto;
using ::ONNX_NAMESPACE::TensorProto;
} // namespace

using ConvertedResult =
    std::pair<std::vector<NodeProto>, std::vector<TensorProto>>;

// Useful utility function
void rewriteSubnet(
    Argument* arg,
    std::map<std::string, std::string> oldname_to_newname);

// Rewrite xcelerate nets into SSA forms. Notice that we will preserve the external
// output names for predict net.
TORCH_API std::unordered_map<std::string, std::string> SsaRewrite(
    xcelerate::NetDef* init_net,
    xcelerate::NetDef* pred_net,
    bool PreserveInPlaceOps = true);

::ONNX_NAMESPACE::TensorProto::DataType xcelerateTypeToOnnxType(
    xcelerate::TensorProto::DataType t);

class TORCH_API OnnxExporter {
  using SpecialOpConverter = ConvertedResult (OnnxExporter::*)(
      const xcelerate::OperatorDef&,
      const std::unordered_map<std::string, xcelerate::TensorShape>&);

 public:
  OnnxExporter(DummyName* dummy = nullptr) {
    if (dummy) {
      dummy_ = std::shared_ptr<DummyName>(dummy, [](DummyName*) {});
    } else {
      dummy_ = std::make_shared<DummyName>();
    }
  }

  ConvertedResult xcelerateOpToOnnxNodes(
      const xcelerate::OperatorDef& def,
      const std::unordered_map<std::string, xcelerate::TensorShape>& shapes);

  void InitOpToTensorProto(const xcelerate::OperatorDef& def, TensorProto* tensor);

 private:
  ConvertedResult CommonxcelerateOpToOnnxNodes(const xcelerate::OperatorDef& def);

  ConvertedResult CreateArgMaxMinOpNodes(
      const xcelerate::OperatorDef& def,
      const std::unordered_map<std::string, xcelerate::TensorShape>& shapes);

  ConvertedResult CreateBinaryElementwiseOpNodes(
      const xcelerate::OperatorDef& def,
      const std::unordered_map<std::string, xcelerate::TensorShape>& shapes);

  ConvertedResult CreateCastNodes(
      const xcelerate::OperatorDef& def,
      const std::unordered_map<std::string, xcelerate::TensorShape>& shapes);

  ConvertedResult CreateElementwiseLinearNodes(
      const xcelerate::OperatorDef& def,
      const std::unordered_map<std::string, xcelerate::TensorShape>& shapes);

  ConvertedResult CreateConvPoolNodes(
      const xcelerate::OperatorDef& def,
      const std::unordered_map<std::string, xcelerate::TensorShape>& shapes);

  ConvertedResult CreateGemmNodes(
      const xcelerate::OperatorDef& def,
      const std::unordered_map<std::string, xcelerate::TensorShape>& shapes);

  ConvertedResult CreateReshapeNodes(
      const xcelerate::OperatorDef& def,
      const std::unordered_map<std::string, xcelerate::TensorShape>& shapes);

  ConvertedResult CreateSliceNodes(
      const xcelerate::OperatorDef& def,
      const std::unordered_map<std::string, xcelerate::TensorShape>& shapes);

  ConvertedResult CreateChannelShuffleNodes(
      const xcelerate::OperatorDef& def,
      const std::unordered_map<std::string, xcelerate::TensorShape>& shapes);

  ConvertedResult CreateReduceMeanNodes(
      const xcelerate::OperatorDef& def,
      const std::unordered_map<std::string, xcelerate::TensorShape>& shapes);

  ConvertedResult CreateConcatNodes(
      const xcelerate::OperatorDef& def,
      const std::unordered_map<std::string, xcelerate::TensorShape>& shapes);

  ConvertedResult CreateMergeDimNodes(
      const xcelerate::OperatorDef& def,
      const std::unordered_map<std::string, xcelerate::TensorShape>& shapes);

  ConvertedResult CreateLrnNodes(
      const xcelerate::OperatorDef& def,
      const std::unordered_map<std::string, xcelerate::TensorShape>& shapes);

  ConvertedResult CreateUpsampleNodes(
      const xcelerate::OperatorDef& def,
      const std::unordered_map<std::string, xcelerate::TensorShape>& shapes);

  // \brief Check block listed arguments where we won't pass down when
  // converting to ONNX node
  bool IsBlockListed(const xcelerate::Argument& arg);

  // \brief Convert xcelerate argument to Onnx attribute
  void CopyxcelerateArgToOnnxAttr(
      AttributeProto* attr,
      const std::string& op_type,
      const xcelerate::Argument& arg);

  // LUT getters
  const std::unordered_map<std::string, std::string>& get_renamed_operators()
      const;
  const std::unordered_map<std::string, std::string>& get_renamed_attrs() const;
  const std::
      unordered_map<std::string, std::unordered_map<std::string, std::string>>&
      get_per_op_renamed_attrs() const;
  const std::unordered_map<std::string, OnnxExporter::SpecialOpConverter>&
  get_special_operators() const;

  // Dummy name generator
  std::shared_ptr<DummyName> dummy_;
};
} // namespace onnx
} // namespace xcelerate
