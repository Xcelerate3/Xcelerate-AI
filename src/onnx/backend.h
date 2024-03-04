#pragma once

#include "xcelerate/onnx/backend_rep.h"
#include "xcelerate/onnx/device.h"
#include "xcelerate/onnx/helper.h"
#include "xcelerate/proto/xcelerate_pb.h"
#include "onnx/onnx_pb.h"

#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>

constexpr int kKnownOpsetVersion = 9;

namespace xcelerate {
namespace onnx {

using ::ONNX_NAMESPACE::AttributeProto;
using ::ONNX_NAMESPACE::GraphProto;
using ::ONNX_NAMESPACE::ModelProto;
using ::ONNX_NAMESPACE::NodeProto;
using ::ONNX_NAMESPACE::TensorProto;
using ::ONNX_NAMESPACE::ValueInfoProto;

using ValueInfoMap = std::unordered_map<std::string, ValueInfoProto>;

class TORCH_API ConversionContext {
 public:
  ConversionContext(const ValueInfoMap& value_infos, int opset_version)
      : value_infos_(value_infos), opset_version_(opset_version) {}
  const ValueInfoMap& value_infos() const {
    return value_infos_;
  }
  int opset_version() const {
    return opset_version_;
  }

 private:
  const ValueInfoMap& value_infos_;
  const int opset_version_;
};

// \brief This struct holds the converted ops after the onnx->c2 conversion.
// Notice that for RNN ops, it may create ops in init_net. Hence we have the
// `init_ops` field.
struct TORCH_API xcelerateOps {
  ::google::protobuf::RepeatedPtrField<xcelerate::OperatorDef> init_ops;
  ::google::protobuf::RepeatedPtrField<xcelerate::OperatorDef> ops;
  ::google::protobuf::RepeatedPtrField<std::string> interface_blobs;
};

// A convenient class to query attributes of a NodeProto. Note that the
// NodeProto can not be modified during the query of OnnxAttributes object
class TORCH_API OnnxAttributes {
 public:
  OnnxAttributes(const NodeProto& node);

  bool HasAttribute(const std::string& key) const {
    return onnx_attrs_.count(key);
  }

  AttributeProto* AddRewrittenAttribute(const std::string& key) {
    auto tmp = rewritten_onnx_attrs_.emplace(key, AttributeProto());
    auto& attr = tmp.first->second;
    attr.set_name(key);
    return &attr;
  }

  ::google::protobuf::RepeatedPtrField<xcelerate::Argument> OnnxAttrToxcelerateArg(
      std::function<std::string(const std::string&)> mapper) const;

  // Get attribute given attribute name, specialied on data type T. Note that
  // the return value is copied
  template <typename T>
  T get(const std::string& key) const;

  template <typename T>
  T get(const std::string& key, const T& default_value) const {
    if (onnx_attrs_.count(key)) {
      return get<T>(key);
    } else {
      return default_value;
    }
  }

  const AttributeProto* remove(const std::string& key) {
    const AttributeProto* result = nullptr;
    auto iter = onnx_attrs_.find(key);
    if (iter != onnx_attrs_.end()) {
      result = iter->second;
      onnx_attrs_.erase(iter);
    }
    return result;
  }

 private:
  std::unordered_map<std::string, const AttributeProto*> onnx_attrs_;
  std::unordered_map<std::string, AttributeProto> rewritten_onnx_attrs_;
};

template <>
int64_t OnnxAttributes::get(const std::string& key) const;
template <>
float OnnxAttributes::get(const std::string& key) const;

template <>
::google::protobuf::RepeatedPtrField<std::string> OnnxAttributes::get(
    const std::string& key) const;

template <>
::google::protobuf::RepeatedField<::google::protobuf::int64>
OnnxAttributes::get(const std::string& key) const;

template <>
::google::protobuf::RepeatedField<float> OnnxAttributes::get(
    const std::string& key) const;

template <>
const TensorProto* OnnxAttributes::get(const std::string& key) const;

// convenient class for onnx node
struct TORCH_API OnnxNode {
  OnnxNode(const NodeProto& node_in) : node(node_in), attributes(node_in) {}

  const NodeProto& node;

  OnnxAttributes attributes;
};

class TORCH_API xcelerateBackend {
 public:
  // Since we still have this Python-C++ hybrid flow, we will need to take the
  // DummyName generator from Python as a pointer. In this case, Python env owns
  // the DummyName object and we don't need to keep track of its life time in
  // C++. Therefore in this case, we use a null dtor to prevent C++ shared_ptr
  // from releasing the object
  xcelerateBackend(DummyName* dummy = nullptr) {
    if (dummy) {
      dummy_ = std::shared_ptr<DummyName>(dummy, [](DummyName*) {});
    } else {
      dummy_ = std::make_shared<DummyName>();
    }
  }

  xcelerateBackendRep* Prepare(
      const std::string& onnx_model_str,
      const std::string& device,
      const std::vector<xcelerateOps>& extras);

  bool SupportOp(const std::string tyep) const;

  xcelerateOps ConvertNode(
      const std::string& node_str,
      const ConversionContext& ctx);

  void BuildTensorFillingOp(
      xcelerate::OperatorDef* c2_op,
      const TensorProto& onnx_tensor,
      const std::string& output_name = "",
      const std::string& shape_name = "");

 private:
  using SpecialOpConverter =
      xcelerateOps (xcelerateBackend::*)(OnnxNode*, const ConversionContext&);

  void OnnxToxcelerate(
      xcelerate::NetDef* init_net,
      xcelerate::NetDef* pred_net,
      const ModelProto& onnx_model,
      const std::string& device,
      int opset_version,
      bool include_initializers,
      const std::vector<xcelerateOps>& extras);

  void CheckOpSchemaArguments(
      const xcelerate::OpSchema& schema,
      const xcelerate::OperatorDef& op);

  xcelerateOps OnnxNodeToxcelerateOps(
      const ModelProto& init_model,
      const ModelProto& pred_model,
      const ConversionContext& ctx,
      OnnxNode* onnx_node);

  std::unordered_set<std::string> AllNamesInGraph(const GraphProto& graph);

  xcelerateOps CommonOnnxNodeToxcelerateOps(
      OnnxNode* onnx_node,
      const ConversionContext& ctx);

  xcelerateOps CreateArgMaxMin(OnnxNode* onnx_node, const ConversionContext& ctx);

  xcelerateOps CreateCast(OnnxNode* onnx_node, const ConversionContext& ctx);

  xcelerateOps CreateConstant(OnnxNode* onnx_node, const ConversionContext& ctx);

  xcelerateOps CreateConstantOfShape(
      OnnxNode* onnx_node,
      const ConversionContext& ctx);

  xcelerateOps CreateConvPoolOpBase(
      OnnxNode* onnx_node,
      const ConversionContext& ctx);

  xcelerateOps CreatePadPool(OnnxNode* onnx_node, const ConversionContext& ctx);

  xcelerateOps CreateReshape(OnnxNode* onnx_node, const ConversionContext& ctx);

  xcelerateOps CreateGather(OnnxNode* onnx_node, const ConversionContext& ctx);

  xcelerateOps CreateGemm(OnnxNode* onnx_node, const ConversionContext& ctx);

  xcelerateOps CreatePad(OnnxNode* onnx_node, const ConversionContext& ctx);

  xcelerateOps CreateConcat(OnnxNode* onnx_node, const ConversionContext& ctx);

  xcelerateOps CreateLogSoftmax(OnnxNode* onnx_node, const ConversionContext& ctx);

  xcelerateOps CreateSlice(OnnxNode* onnx_node, const ConversionContext& ctx);

  std::string PreprocessSliceIndexTensor(
      OnnxNode* onnx_node,
      xcelerateOps& ret,
      std::string indices_tensor,
      std::string axes_tensor,
      std::string rank_tensor,
      std::string zero_tensor,
      std::string one_tensor,
      int default_value);

  xcelerateOps CreateDynamicSlice(
      OnnxNode* onnx_node,
      const ConversionContext& ctx);

  xcelerateOps CreateSplit(OnnxNode* onnx_node, const ConversionContext& ctx);

  xcelerateOps CreateReciprocal(OnnxNode* onnx_node, const ConversionContext& ctx);

  xcelerateOps CreateRandomNormal(
      OnnxNode* onnx_node,
      const ConversionContext& ctx);

  xcelerateOps CreateWhereOp(OnnxNode* onnx_node, const ConversionContext& ctx);

  xcelerateOps CreateNonZeroOp(OnnxNode* onnx_node, const ConversionContext& ctx);

  xcelerateOps CreateMultinomialOp(
      OnnxNode* onnx_node,
      const ConversionContext& ctx);

  xcelerateOps CreateBatchNormalization(
      OnnxNode* onnx_node,
      const ConversionContext& ctx);

  xcelerateOps CreateMatMul(OnnxNode* onnx_node, const ConversionContext& ctx);

  xcelerateOps CreateUpsample(OnnxNode* onnx_node, const ConversionContext& ctx);

  xcelerateOps CreateDropout(OnnxNode* onnx_node, const ConversionContext& ctx);

  xcelerateOps CreateLRN(OnnxNode* onnx_node, const ConversionContext& ctx);

  // LUT related getters
  const std::unordered_map<std::string, std::string>& get_renamed_operators()
      const;
  const std::unordered_set<std::string>& get_rnn_operators() const;
  const std::unordered_map<std::string, int>& get_broken_operators() const;
  const std::unordered_map<std::string, std::string>& get_renamed_attrs() const;
  const std::
      unordered_map<std::string, std::unordered_map<std::string, std::string>>&
      get_per_op_renamed_attrs() const;
  const std::unordered_map<std::string, xcelerateBackend::SpecialOpConverter>&
  get_special_operators() const;

  // Dummy name generator
  std::shared_ptr<DummyName> dummy_;
};

} // namespace onnx
} // namespace xcelerate
