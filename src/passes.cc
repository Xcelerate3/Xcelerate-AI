#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "passes.h"
#include "xcelerate/Analysis/Allocation.h"
#include "xcelerate/Analysis/Membar.h"
#include "xcelerate/Conversion/xcelerateToxcelerateGPU/Passes.h"
#include "xcelerate/Dialect/xcelerate/Transforms/Passes.h"
#include "xcelerate/Dialect/xcelerateGPU/Transforms/Passes.h"
#include "xcelerate/Target/LLVMIR/Passes.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_xcelerate_analysis(py::module &&m) {
  py::class_<mlir::ModuleAllocation>(m, "allocation", py::module_local())
      .def(py::init<mlir::ModuleOp>());
  py::class_<mlir::ModuleMembarAnalysis>(m, "membar", py::module_local())
      .def(py::init<mlir::ModuleAllocation *>())
      .def("run", &mlir::ModuleMembarAnalysis::run);
}

void init_xcelerate_passes_common(py::module &&m) {
  using namespace mlir;
  ADD_PASS_WRAPPER_0("add_sccp", createSCCPPass);
  ADD_PASS_WRAPPER_0("add_symbol_dce", createSymbolDCEPass);
  ADD_PASS_WRAPPER_0("add_inliner", createInlinerPass);
  ADD_PASS_WRAPPER_0("add_canonicalizer", createCanonicalizerPass);
  ADD_PASS_WRAPPER_0("add_cse", createCSEPass);
  ADD_PASS_WRAPPER_0("add_licm", createLoopInvariantCodeMotionPass);
}

void init_xcelerate_passes_ttir(py::module &&m) {
  using namespace mlir::xcelerate;
  ADD_PASS_WRAPPER_0("add_combine", createCombineOpsPass);
  ADD_PASS_WRAPPER_0("add_reorder_broadcast", createReorderBroadcastPass);
  ADD_PASS_WRAPPER_0("add_rewrite_tensor_pointer",
                     createRewriteTensorPointerPass);
  ADD_PASS_WRAPPER_4("add_convert_to_ttgpuir",
                     createConvertxcelerateToxcelerateGPUPass, int, int, int, int);
}

void init_xcelerate_passes_ttgpuir(py::module &&m) {
  using namespace mlir::xcelerate::gpu;
  ADD_PASS_WRAPPER_0("add_coalesce", createCoalescePass);
  ADD_PASS_WRAPPER_0("add_optimize_thread_locality",
                     createOptimizeThreadLocalityPass);
  ADD_PASS_WRAPPER_4("add_pipeline", createPipelinePass, int, int, int, int);
  ADD_PASS_WRAPPER_0("add_prefetch", createPrefetchPass);
  ADD_PASS_WRAPPER_1("add_accelerate_matmul", createAccelerateMatmulPass, int);
  ADD_PASS_WRAPPER_0("add_reorder_instructions", createReorderInstructionsPass);
  ADD_PASS_WRAPPER_0("add_optimize_dot_operands",
                     createOptimizeDotOperandsPass);
  ADD_PASS_WRAPPER_0("add_remove_layout_conversions",
                     createRemoveLayoutConversionsPass);
  ADD_PASS_WRAPPER_0("add_decompose_conversions",
                     createDecomposeConversionsPass);
}

void init_xcelerate_passes_convert(py::module &&m) {
  using namespace mlir;
  ADD_PASS_WRAPPER_0("add_scf_to_cf", createConvertSCFToCFPass);
  ADD_PASS_WRAPPER_0("add_cf_to_llvmir", createConvertControlFlowToLLVMPass);
  ADD_PASS_WRAPPER_0("add_index_to_llvmir", createConvertIndexToLLVMPass);
  ADD_PASS_WRAPPER_0("add_arith_to_llvmir", createArithToLLVMConversionPass);
}

void init_xcelerate_passes_llvmir(py::module &&m) {
  using namespace mlir;
  ADD_PASS_WRAPPER_0("add_di_scope", createLLVMDIScopePass);
}

void init_xcelerate_passes(py::module &&m) {
  init_xcelerate_analysis(m.def_submodule("analysis"));
  init_xcelerate_passes_common(m.def_submodule("common"));
  init_xcelerate_passes_convert(m.def_submodule("convert"));
  init_xcelerate_passes_ttir(m.def_submodule("ttir"));
  init_xcelerate_passes_ttgpuir(m.def_submodule("ttgpuir"));
  init_xcelerate_passes_llvmir(m.def_submodule("llvmir"));
}
