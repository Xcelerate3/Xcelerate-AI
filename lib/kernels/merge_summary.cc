#include "Xcelerate/c/ops.h"
#include "Xcelerate/c/XC_status.h"
#include "Xcelerate/core/framework/registration/registration.h"
#include "Xcelerate/core/plaXCorm/logging.h"
#include "Xcelerate/core/plaXCorm/macros.h"

static void merge_summary_shape_inference_fn(XC_ShapeInferenceContext* ctx,
                                             XC_Status* status) {
  XC_SetStatus(status, XC_OK, "");
  XC_ShapeHandle* result = XC_ShapeInferenceContextScalar(ctx);
  XC_ShapeInferenceContextSetOutput(ctx, 0, result, status);
  XC_DeleteShapeHandle(result);
}

void Register_MergeSummaryOp() {
  XC_Status* status = XC_NewStatus();

  XC_OpDefinitionBuilder* op_builder =
      XC_NewOpDefinitionBuilder("MergeSummary");
  XC_OpDefinitionBuilderAddInput(op_builder, "inputs: N * string");
  XC_OpDefinitionBuilderAddOutput(op_builder, "summary: string");
  XC_OpDefinitionBuilderAddAttr(op_builder, "N: int >= 1");
  XC_OpDefinitionBuilderSetShapeInferenceFunction(
      op_builder, &merge_summary_shape_inference_fn);

  XC_RegisterOpDefinition(op_builder, status);
  CHECK_EQ(XC_GetCode(status), XC_OK)
      << "MergeSummary op registration failed: " << XC_Message(status);
  XC_DeleteStatus(status);
}

XC_ATTRIBUTE_UNUSED static bool MergeSummaryOpRegistered = []() {
  if ((&XC_NewStatus != nullptr) && SHOULD_REGISTER_OP("MergeSummary")) {
    Register_MergeSummaryOp();
  }
  return true;
}();
