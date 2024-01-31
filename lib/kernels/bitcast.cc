#include <sstream>
#include <string>

#include "Xcelerate/c/ops.h"
#include "Xcelerate/core/framework/registration/registration.h"
#include "Xcelerate/core/plaXCorm/logging.h"
#include "Xcelerate/core/plaXCorm/macros.h"

static void ComputeNewShape(XC_ShapeInferenceContext* ctx,
                            XC_ShapeHandle* shape, XC_DataType input_type,
                            XC_DataType output_type, XC_Status* status) {
  size_t input_type_size = XC_DataTypeSize(input_type);
  size_t output_type_size = XC_DataTypeSize(output_type);

  if (input_type_size == 0 || output_type_size == 0) {
    std::ostringstream err;
    err << "Cannot bitcast type " << input_type << " to " << output_type
        << " because one of the type sizes is zero";
    XC_SetStatus(status, XC_INVALID_ARGUMENT, err.str().c_str());
    return;
  }

  XC_SetStatus(status, XC_OK, "");
  if (input_type_size < output_type_size) {
    XC_ShapeInferenceContextWithRankAtLeast(ctx, shape, 1, shape, status);

    if (XC_GetCode(status) == XC_OK) {
      XC_DimensionHandle* last_dim = XC_NewDimensionHandle();
      size_t divisor_val = output_type_size / input_type_size;
      XC_ShapeInferenceContextDim(ctx, shape, -1, last_dim);
      if (!XC_DimensionHandleValueKnown(last_dim) ||
          XC_DimensionHandleValue(last_dim) == divisor_val) {
        XC_ShapeInferenceContextSubshape(ctx, shape, 0, -1, shape, status);
      } else {
        std::ostringstream err;
        err << "Cannot bitcast from " << input_type << " to " << output_type
            << " due to shape. " << XC_DimensionHandleValue(last_dim)
            << " does not match " << divisor_val;
        XC_SetStatus(status, XC_INVALID_ARGUMENT, err.str().c_str());
      }
      XC_DeleteDimensionHandle(last_dim);
    }
  } else if (input_type_size > output_type_size) {
    // Input type size is larger than output type size.
    size_t divisor_val = input_type_size / output_type_size;
    XC_ShapeHandle* extension =
        XC_ShapeInferenceContextVectorFromSize(ctx, divisor_val);
    XC_ShapeInferenceContextConcatenateShapes(ctx, shape, extension, shape,
                                              status);
    XC_DeleteShapeHandle(extension);
  }
}

static void bitcast_shape_inference_fn(XC_ShapeInferenceContext* ctx,
                                       XC_Status* status) {
  XC_ShapeHandle* result = XC_NewShapeHandle();
  XC_ShapeInferenceContextGetInput(ctx, 0, result, status);
  if (XC_GetCode(status) == XC_OK &&
      !XC_ShapeInferenceContextRankKnown(ctx, result)) {
    XC_ShapeInferenceContextSetUnknownShape(ctx, status);
    XC_DeleteShapeHandle(result);
    return;
  }

  // Find the size of the input and output data types.
  XC_DataType input_type;
  XC_DataType output_type;

  if (XC_GetCode(status) == XC_OK) {
    XC_ShapeInferenceContext_GetAttrType(ctx, "T", &input_type, status);
  }

  if (XC_GetCode(status) == XC_OK) {
    XC_ShapeInferenceContext_GetAttrType(ctx, "type", &output_type, status);
  }

  if (XC_GetCode(status) == XC_OK) {
    ComputeNewShape(ctx, result, input_type, output_type, status);
  }

  if (XC_GetCode(status) == XC_OK) {
    XC_ShapeInferenceContextSetOutput(ctx, 0, result, status);
  }
  XC_DeleteShapeHandle(result);
}

void RegisterBitcastOp() {
  XC_Status* status = XC_NewStatus();

  XC_OpDefinitionBuilder* op_builder = XC_NewOpDefinitionBuilder("Bitcast");
  XC_OpDefinitionBuilderAddInput(op_builder, "input: T");
  XC_OpDefinitionBuilderAddOutput(op_builder, "output: type");
  XC_OpDefinitionBuilderAddAttr(
      op_builder,
      "T: {bfloat16, half, float, double, int64, int32, uint8, uint16, "
      "uint32, uint64, int8, int16, complex64, complex128, qint8, quint8, "
      "qint16, quint16, qint32}");
  XC_OpDefinitionBuilderAddAttr(
      op_builder,
      "type: {bfloat16, half, float, double, int64, int32, uint8, uint16, "
      "uint32, uint64, int8, int16, complex64, complex128, qint8, quint8, "
      "qint16, quint16, qint32}");
  XC_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                  &bitcast_shape_inference_fn);

  XC_RegisterOpDefinition(op_builder, status);
  CHECK_EQ(XC_GetCode(status), XC_OK)
      << "Bitcast op registration failed: " << XC_Message(status);
  XC_DeleteStatus(status);
}

XC_ATTRIBUTE_UNUSED static bool IsBitcastOpRegistered = []() {
  if ((&XC_NewStatus != nullptr) && SHOULD_REGISTER_OP("Bitcast")) {
    RegisterBitcastOp();
  }
  return true;
}();
