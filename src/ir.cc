#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Transforms/Passes.h"
#include "xcelerate/Analysis/Allocation.h"
#include "xcelerate/Dialect/xcelerate/IR/Dialect.h"
#include "xcelerate/Dialect/xcelerate/IR/Types.h"
#include "xcelerate/Tools/Sys/GetEnv.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// A custom op builder that keeps track of the last location
class xcelerateOpBuilder {
public:
  xcelerateOpBuilder(mlir::MLIRContext *context) {
    builder = std::make_unique<mlir::OpBuilder>(context);
    lastLoc = std::make_unique<mlir::Location>(builder->getUnknownLoc());
  }

  mlir::OpBuilder &getBuilder() { return *builder; }

  bool isLineInfoEnabled() { return lineInfoEnabled; }

  void setLastLoc(mlir::Location loc) {
    if (lineInfoEnabled)
      lastLoc = std::make_unique<mlir::Location>(loc);
  }

  void setLastLoc(const std::string &fileName, int line, int column) {
    auto context = builder->getContext();
    setLastLoc(mlir::FileLineColLoc::get(context, fileName, line, column));
  }

  mlir::Location getLastLoc() {
    assert(lastLoc);
    return *lastLoc;
  }

  void setInsertionPointToStart(mlir::Block &block) {
    if (!block.empty())
      setLastLoc(block.begin()->getLoc());
    else
      setLastLoc(builder->getUnknownLoc());
    builder->setInsertionPointToStart(&block);
  }

  void setInsertionPointToEnd(mlir::Block &block) {
    if (!block.empty())
      setLastLoc(block.back().getLoc());
    else
      setLastLoc(builder->getUnknownLoc());
    builder->setInsertionPointToEnd(&block);
  }

  void setInsertionPointAfter(mlir::Operation &op) {
    setLastLoc(op.getLoc());
    builder->setInsertionPointAfter(&op);
  }

  void restoreInsertionPoint(mlir::OpBuilder::InsertPoint pt) {
    if (pt.isSet() && pt.getPoint() != pt.getBlock()->end())
      setLastLoc(pt.getPoint()->getLoc());
    else
      setLastLoc(builder->getUnknownLoc());
    builder->restoreInsertionPoint(pt);
  }

  template <typename OpTy, typename... Args> OpTy create(Args &&...args) {
    auto loc = getLastLoc();
    return builder->create<OpTy>(loc, std::forward<Args>(args)...);
  }

  // Overload to create or fold a single result operation.
  template <typename OpTy, typename... Args>
  std::enable_if_t<OpTy::template hasTrait<mlir::OpTrait::OneResult>(),
                   mlir::Value>
  createOrFold(Args &&...args) {
    auto loc = getLastLoc();
    return builder->createOrFold<OpTy>(loc, std::forward<Args>(args)...);
  }

  // Overload to create or fold a zero result operation.
  template <typename OpTy, typename... Args>
  std::enable_if_t<OpTy::template hasTrait<mlir::OpTrait::ZeroResults>(), OpTy>
  createOrFold(Args &&...args) {
    auto loc = getLastLoc();
    return builder->createOrFold<OpTy>(loc, std::forward<Args>(args)...);
  }

private:
  std::unique_ptr<mlir::OpBuilder> builder;
  std::unique_ptr<mlir::Location> lastLoc;
  bool lineInfoEnabled = !xcelerate::tools::getBoolEnv("xcelerate_DISABLE_LINE_INFO");
};

static std::string locationToString(mlir::Location loc) {
  std::string str;
  llvm::raw_string_ostream os(str);
  loc.print(os);
  os.flush(); // Make sure all the content is dumped into the 'str' string
  return str;
}

static void outputWarning(mlir::Location loc, const std::string &msg) {
  std::string locStr = locationToString(loc);

  PyErr_WarnEx(PyExc_UserWarning, (locStr + ": " + msg).c_str(),
               /*stack_level=*/2);
}

/*****************************************************************************/
/* Python bindings for xcelerate::ir                                            */
/*****************************************************************************/

void init_xcelerate_ir(py::module &&m) {
  using ret = py::return_value_policy;
  using namespace pybind11::literals;

  py::enum_<mlir::xcelerate::PaddingOption>(m, "PADDING_OPTION",
                                         py::module_local())
      .value("PAD_ZERO", mlir::xcelerate::PaddingOption::PAD_ZERO)
      .value("PAD_NAN", mlir::xcelerate::PaddingOption::PAD_NAN)
      .export_values();

  py::enum_<mlir::xcelerate::CacheModifier>(m, "CACHE_MODIFIER",
                                         py::module_local())
      .value("NONE", mlir::xcelerate::CacheModifier::NONE)
      .value("CA", mlir::xcelerate::CacheModifier::CA)
      .value("CG", mlir::xcelerate::CacheModifier::CG)
      .value("WB", mlir::xcelerate::CacheModifier::WB)
      .value("CS", mlir::xcelerate::CacheModifier::CS)
      .value("WT", mlir::xcelerate::CacheModifier::WT)
      .export_values();

  py::enum_<mlir::xcelerate::MemSemantic>(m, "MEM_SEMANTIC", py::module_local())
      .value("ACQUIRE_RELEASE", mlir::xcelerate::MemSemantic::ACQUIRE_RELEASE)
      .value("ACQUIRE", mlir::xcelerate::MemSemantic::ACQUIRE)
      .value("RELEASE", mlir::xcelerate::MemSemantic::RELEASE)
      .value("RELAXED", mlir::xcelerate::MemSemantic::RELAXED)
      .export_values();

  py::enum_<mlir::xcelerate::MemSyncScope>(m, "MEM_SYNC_SCOPE", py::module_local())
      .value("GPU", mlir::xcelerate::MemSyncScope::GPU)
      .value("CTA", mlir::xcelerate::MemSyncScope::CTA)
      .value("SYSTEM", mlir::xcelerate::MemSyncScope::SYSTEM)
      .export_values();

  py::enum_<mlir::xcelerate::EvictionPolicy>(m, "EVICTION_POLICY",
                                          py::module_local())
      .value("NORMAL", mlir::xcelerate::EvictionPolicy::NORMAL)
      .value("EVICT_FIRST", mlir::xcelerate::EvictionPolicy::EVICT_FIRST)
      .value("EVICT_LAST", mlir::xcelerate::EvictionPolicy::EVICT_LAST)
      .export_values();

  py::enum_<mlir::xcelerate::RMWOp>(m, "ATOMIC_OP", py::module_local())
      .value("ADD", mlir::xcelerate::RMWOp::ADD)
      .value("FADD", mlir::xcelerate::RMWOp::FADD)
      .value("AND", mlir::xcelerate::RMWOp::AND)
      .value("OR", mlir::xcelerate::RMWOp::OR)
      .value("XOR", mlir::xcelerate::RMWOp::XOR)
      .value("XCHG", mlir::xcelerate::RMWOp::XCHG)
      .value("MAX", mlir::xcelerate::RMWOp::MAX)
      .value("MIN", mlir::xcelerate::RMWOp::MIN)
      .value("UMIN", mlir::xcelerate::RMWOp::UMIN)
      .value("UMAX", mlir::xcelerate::RMWOp::UMAX);

  py::enum_<mlir::xcelerate::RoundingMode>(m, "ROUNDING_MODE", py::module_local())
      .value("RTZ", mlir::xcelerate::RoundingMode::RTZ)
      .value("RTNE", mlir::xcelerate::RoundingMode::RTNE);

  py::enum_<mlir::xcelerate::PropagateNan>(m, "PROPAGATE_NAN", py::module_local())
      .value("NONE", mlir::xcelerate::PropagateNan::NONE)
      .value("ALL", mlir::xcelerate::PropagateNan::ALL);

  py::class_<mlir::MLIRContext>(m, "context", py::module_local())
      .def(py::init<>());

  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<
        mlir::xcelerate::xcelerateDialect, mlir::xcelerate::gpu::xcelerateGPUDialect,
        mlir::math::MathDialect, mlir::arith::ArithDialect,
        mlir::index::IndexDialect, mlir::scf::SCFDialect, mlir::gpu::GPUDialect,
        mlir::cf::ControlFlowDialect, mlir::LLVM::LLVMDialect>();
    mlir::registerBuiltinDialectTranslation(registry);
    mlir::registerLLVMDialectTranslation(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  py::class_<mlir::Type>(m, "type", py::module_local())
      .def("is_integer", &mlir::Type::isInteger)
      .def("is_fp16", &mlir::Type::isF16)
      .def("__str__", [](mlir::Type &self) {
        std::string str;
        llvm::raw_string_ostream os(str);
        self.print(os);
        return os.str();
      });

  py::class_<mlir::FunctionType>(m, "function_type", py::module_local())
      .def("param_types", [](mlir::FunctionType &self) {
        return std::vector<mlir::Type>(self.getInputs().begin(),
                                       self.getInputs().end());
      });

  py::class_<mlir::Location>(m, "location", py::module_local())
      .def("__str__", [](mlir::Location &self) {
        std::string str;
        llvm::raw_string_ostream os(str);
        self.print(os);
        return os.str();
      });

  py::class_<mlir::Value>(m, "value", py::module_local())
      .def("set_attr",
           [](mlir::Value &self, std::string &name,
              mlir::Attribute &attr) -> void {
             if (mlir::Operation *definingOp = self.getDefiningOp())
               definingOp->setAttr(name, attr);
             else {
               auto arg = self.cast<mlir::BlockArgument>();
               int id = arg.getArgNumber();
               std::string attrName = name + "_arg" + std::to_string(id);
               mlir::Block *owner = arg.getOwner();
               if (owner->isEntryBlock() &&
                   !mlir::isa<mlir::xcelerate::FuncOp>(owner->getParentOp())) {
                 owner->getParentOp()->setAttr(attrName, attr);
               }
             }
           })
      .def("get_context", &mlir::Value::getContext)
      .def("replace_all_uses_with",
           [](mlir::Value &self, mlir::Value &newValue) {
             self.replaceAllUsesWith(newValue);
           })
      .def("get_type", &mlir::Value::getType);

  py::class_<mlir::BlockArgument, mlir::Value>(m, "block_argument",
                                               py::module_local());

  py::class_<mlir::Region>(m, "region", py::module_local())
      .def("get_parent_region", &mlir::Region::getParentRegion, ret::reference)
      .def("size", [](mlir::Region &self) { return self.getBlocks().size(); })
      .def("empty", &mlir::Region::empty);

  py::class_<mlir::Block>(m, "block", py::module_local())
      .def("arg",
           [](mlir::Block &self, int index) -> mlir::BlockArgument {
             return self.getArgument(index);
           })
      .def("add_argument",
           [](mlir::Block &self, mlir::Type ty) {
             auto loc = mlir::UnknownLoc::get(ty.getContext());
             self.addArgument(ty, loc);
           })
      .def("get_num_arguments", &mlir::Block::getNumArguments)
      .def("dump", &mlir::Block::dump)
      .def("move_before", &mlir::Block::moveBefore)
      .def("insert_before", &mlir::Block::insertBefore)
      .def("get_parent", &mlir::Block::getParent, ret::reference)
      .def("merge_block_before",
           [](mlir::Block &self, mlir::Block &dst) {
             // ref: RewriterBase::mergeBlocks()
             if (self.getNumArguments() != 0)
               throw std::runtime_error(
                   "This block has arguments, don't merge");
             dst.getOperations().splice(dst.begin(), self.getOperations());
             self.dropAllUses();
             self.erase();
           })
      .def("replace_use_in_block_with",
           [](mlir::Block &self, mlir::Value &v, mlir::Value &newVal) {
             v.replaceUsesWithIf(newVal, [&](mlir::OpOperand &operand) {
               mlir::Operation *user = operand.getOwner();
               mlir::Block *currentBlock = user->getBlock();
               while (currentBlock) {
                 if (currentBlock == &self)
                   return true;
                 // Move up one level
                 currentBlock =
                     currentBlock->getParent()->getParentOp()->getBlock();
               }
               return false;
             });
           })
      .def("__str__",
           [](mlir::Block &self) {
             std::string str;
             llvm::raw_string_ostream os(str);
             self.print(os);
             return str;
           })
      .def("has_terminator",
           [](mlir::Block &self) {
             return !self.empty() &&
                    self.back().hasTrait<mlir::OpTrait::IsTerminator>();
           })
      .def("has_return",
           [](mlir::Block &self) {
             return !self.empty() &&
                    self.back().hasTrait<mlir::OpTrait::ReturnLike>();
           })
      .def("erase", [](mlir::Block &self) { self.erase(); });

  py::class_<mlir::Attribute>(m, "attribute", py::module_local());
  py::class_<mlir::IntegerAttr, mlir::Attribute>(m, "integer_attr",
                                                 py::module_local());
  py::class_<mlir::BoolAttr, mlir::Attribute>(m, "bool_attr",
                                              py::module_local());

  // Ops
  py::class_<mlir::OpState>(m, "OpState", py::module_local())
      .def("set_attr",
           [](mlir::OpState &self, std::string &name,
              mlir::Attribute &attr) -> void { self->setAttr(name, attr); })
      .def(
          "get_num_results",
          [](mlir::OpState &self) -> unsigned { return self->getNumResults(); })
      .def("get_result",
           [](mlir::OpState &self, unsigned idx) -> mlir::Value {
             return self->getResult(idx);
           })
      .def(
          "get_region",
          [](mlir::OpState &self, unsigned idx) -> mlir::Region & {
            return self->getRegion(idx);
          },
          ret::reference)
      .def(
          "get_body",
          [](mlir::scf::ForOp &self, unsigned idx) -> mlir::Block * {
            return self.getBody(idx);
          },
          ret::reference)
      .def("dump", [](mlir::OpState &self) { self->dump(); })
      .def("__str__",
           [](mlir::OpState &self) -> std::string {
             std::string str;
             llvm::raw_string_ostream os(str);
             auto printingFlags = mlir::OpPrintingFlags();
             printingFlags.enableDebugInfo();
             self->print(os, printingFlags);
             return str;
           })
      .def("append_operand",
           [](mlir::OpState &self, mlir::Value &val) {
             self->insertOperands(self->getNumOperands(), val);
           })
      .def("verify", [](mlir::OpState &self) -> bool {
        return mlir::succeeded(mlir::verify(self.getOperation()));
      });
  // scf Ops
  py::class_<mlir::scf::ForOp, mlir::OpState>(m, "ForOp", py::module_local())
      .def("get_induction_var", &mlir::scf::ForOp::getInductionVar);

  py::class_<mlir::scf::IfOp, mlir::OpState>(m, "IfOp", py::module_local())
      .def("get_then_block", &mlir::scf::IfOp::thenBlock, ret::reference)
      .def("get_else_block", &mlir::scf::IfOp::elseBlock, ret::reference)
      .def("get_then_yield", &mlir::scf::IfOp::thenYield)
      .def("get_else_yield", &mlir::scf::IfOp::elseYield);
  py::class_<mlir::scf::YieldOp, mlir::OpState>(m, "YieldOp",
                                                py::module_local());
  py::class_<mlir::scf::WhileOp, mlir::OpState>(m, "WhileOp",
                                                py::module_local())
      .def("get_before", &mlir::scf::WhileOp::getBefore, ret::reference)
      .def("get_after", &mlir::scf::WhileOp::getAfter, ret::reference);
  py::class_<mlir::scf::ConditionOp, mlir::OpState>(m, "ConditionOp",
                                                    py::module_local());

  // dynamic_attr is used to transfer ownership of the MLIR context to the
  // module
  py::class_<mlir::ModuleOp, mlir::OpState>(m, "module", py::module_local(),
                                            py::dynamic_attr())
      .def("dump", &mlir::ModuleOp::dump)
      .def("str",
           [](mlir::ModuleOp &self) -> std::string {
             std::string str;
             llvm::raw_string_ostream os(str);
             auto printingFlags = mlir::OpPrintingFlags();
             printingFlags.enableDebugInfo();
             self.print(os, printingFlags);
             return str;
           })
      .def("push_back",
           [](mlir::ModuleOp &self, mlir::xcelerate::FuncOp &funcOp) -> void {
             self.push_back(funcOp);
           })
      .def("has_function",
           [](mlir::ModuleOp &self, std::string &funcName) -> bool {
             if (self.lookupSymbol(funcName))
               return true;
             return false;
           })
      .def("get_function",
           [](mlir::ModuleOp &self,
              std::string &funcName) -> mlir::xcelerate::FuncOp {
             return self.lookupSymbol<mlir::xcelerate::FuncOp>(funcName);
           })
      .def("get_int_attr",
           [](mlir::ModuleOp &self, std::string name) -> py::object {
             auto ret = self->getAttrOfType<mlir::IntegerAttr>(name);
             if (!ret)
               return py::none();
             return py::int_(ret.getInt());
           });

  m.def("make_attr",
        [](const std::vector<int> &values, mlir::MLIRContext &context) {
          return mlir::DenseIntElementsAttr::get(
                     mlir::RankedTensorType::get(
                         {static_cast<int64_t>(values.size())},
                         mlir::IntegerType::get(&context, 32)),
                     values)
              .cast<mlir::Attribute>();
        });

  m.def(
      "parse_mlir_module",
      [](const std::string &inputFilename, mlir::MLIRContext &context) {
        // parse module
        mlir::OwningOpRef<mlir::ModuleOp> module =
            mlir::parseSourceFile<mlir::ModuleOp>(inputFilename, &context);
        if (!module)
          throw std::runtime_error("Parse MLIR file failed.");
        // locations are incompatible with ptx < 7.5 !
        module->walk([](mlir::Operation *op) {
          op->setLoc(mlir::UnknownLoc::get(op->getContext()));
        });

        return module->clone();
      },
      ret::take_ownership);

  py::class_<mlir::xcelerate::FuncOp, mlir::OpState>(m, "function",
                                                  py::module_local())
      // .def_property_readonly("attrs", &ir::function::attrs)
      // .def("add_attr", &ir::function::add_attr);
      .def("args",
           [](mlir::xcelerate::FuncOp &self, unsigned idx) -> mlir::BlockArgument {
             return self.getArgument(idx);
           })
      .def(
          "add_entry_block",
          [](mlir::xcelerate::FuncOp &self) -> mlir::Block * {
            return self.addEntryBlock();
          },
          ret::reference)
      .def(
          "set_arg_attr",
          [](mlir::xcelerate::FuncOp &self, int arg_no, const std::string &name,
             int val) {
            // set arg attributes "name" to value "val"
            auto attrTy = mlir::IntegerType::get(self.getContext(), 32);
            self.setArgAttr(arg_no, name, mlir::IntegerAttr::get(attrTy, val));
          },
          ret::reference)
      //  .def("has_attr", &mlir::::FuncOp::hasAttr)
      .def("finalize",
           [](mlir::xcelerate::FuncOp &self) -> void {
             // Remove dead code
             // 1. Unreachable code after return
             self.walk([&](mlir::Block *block) {
               mlir::Operation *retOp = nullptr;
               // It's better to not use walk here because we only want to
               // check operations in the current block
               for (auto &op : block->getOperations()) {
                 if (mlir::isa<mlir::xcelerate::ReturnOp>(op))
                   if (retOp == nullptr) {
                     retOp = &op;
                     break;
                   }
               }
               if (retOp && retOp != &block->back()) {
                 auto pos = retOp->getIterator();
                 pos++;
                 auto *newBlock = block->splitBlock(pos);
                 newBlock->erase();
               }
             });
             // 2. Check if the result of tl.advance is used
             self.walk([&](mlir::Operation *op) {
               if (mlir::isa<mlir::xcelerate::AdvanceOp>(op) &&
                   op->getResult(0).use_empty())
                 outputWarning(op->getLoc(), "The result of tl.advance is not "
                                             "being used. Note that tl.advance "
                                             "does not have any side effects. "
                                             "To move the block pointer, you "
                                             "need to assign the result of "
                                             "tl.advance to a variable.");
             });
           })
      .def_property_readonly("type", &mlir::xcelerate::FuncOp::getFunctionType)
      .def("reset_type", &mlir::xcelerate::FuncOp::setType);

  py::class_<mlir::OpBuilder::InsertPoint>(m, "InsertPoint",
                                           py::module_local());

  py::class_<xcelerateOpBuilder>(m, "builder", py::module_local(),
                              py::dynamic_attr())
      .def(py::init<mlir::MLIRContext *>())
      // getters
      .def("create_module",
           [](xcelerateOpBuilder &self) -> mlir::ModuleOp {
             return self.create<mlir::ModuleOp>();
           })
      // insertion block/point
      .def("set_insertion_point_to_start",
           [](xcelerateOpBuilder &self, mlir::Block &block) -> void {
             self.setInsertionPointToStart(block);
           })
      .def("set_insertion_point_to_end",
           [](xcelerateOpBuilder &self, mlir::Block &block) {
             self.setInsertionPointToEnd(block);
           })
      .def("set_insertion_point_after",
           [](xcelerateOpBuilder &self, mlir::Operation &op) {
             self.setInsertionPointAfter(op);
           })
      .def(
          "get_insertion_block",
          [](xcelerateOpBuilder &self) -> mlir::Block * {
            return self.getBuilder().getInsertionBlock();
          },
          ret::reference)
      .def("get_insertion_point",
           [](xcelerateOpBuilder &self) {
             return self.getBuilder().saveInsertionPoint();
           })
      .def("restore_insertion_point",
           [](xcelerateOpBuilder &self, mlir::OpBuilder::InsertPoint pt) {
             self.restoreInsertionPoint(pt);
           })
      // Attr
      .def("get_bool_attr",
           [](xcelerateOpBuilder &self, bool value) {
             return self.getBuilder().getBoolAttr(value);
           })
      .def("get_int32_attr",
           [](xcelerateOpBuilder &self, int32_t value) {
             return self.getBuilder().getI32IntegerAttr(value);
           })
      // Use arith.ConstantOp to create constants
      // Constants
      .def("get_int1",
           [](xcelerateOpBuilder &self, bool v) -> mlir::Value {
             return mlir::Value(self.create<mlir::arith::ConstantIntOp>(
                 v, self.getBuilder().getI1Type()));
           })
      .def("get_int8",
           [](xcelerateOpBuilder &self, int64_t v) -> mlir::Value {
             return mlir::Value(self.create<mlir::arith::ConstantIntOp>(
                 v, self.getBuilder().getI8Type()));
           })
      .def("get_int16",
           [](xcelerateOpBuilder &self, int64_t v) -> mlir::Value {
             return mlir::Value(self.create<mlir::arith::ConstantIntOp>(
                 v, self.getBuilder().getI16Type()));
           })
      .def("get_int32",
           [](xcelerateOpBuilder &self, int64_t v) -> mlir::Value {
             return mlir::Value(self.create<mlir::arith::ConstantIntOp>(
                 v, self.getBuilder().getI32Type()));
           })
      .def("get_int64",
           [](xcelerateOpBuilder &self, int64_t v) -> mlir::Value {
             return mlir::Value(self.create<mlir::arith::ConstantIntOp>(
                 v, self.getBuilder().getI64Type()));
           })
      .def("get_uint8",
           [](xcelerateOpBuilder &self, uint64_t v) -> mlir::Value {
             return mlir::Value(self.create<mlir::arith::ConstantIntOp>(
                 v, self.getBuilder().getI8Type()));
           })
      .def("get_uint16",
           [](xcelerateOpBuilder &self, uint64_t v) -> mlir::Value {
             return mlir::Value(self.create<mlir::arith::ConstantIntOp>(
                 v, self.getBuilder().getI16Type()));
           })
      .def("get_uint32",
           [](xcelerateOpBuilder &self, uint64_t v) -> mlir::Value {
             return mlir::Value(self.create<mlir::arith::ConstantIntOp>(
                 v, self.getBuilder().getI32Type()));
           })
      .def("get_uint64",
           [](xcelerateOpBuilder &self, uint64_t v) -> mlir::Value {
             return mlir::Value(self.create<mlir::arith::ConstantIntOp>(
                 v, self.getBuilder().getI64Type()));
           })
      .def("get_bf16",
           [](xcelerateOpBuilder &self, float v) -> mlir::Value {
             auto type = self.getBuilder().getBF16Type();
             return self.create<mlir::arith::ConstantFloatOp>(
                 mlir::APFloat(type.getFloatSemantics(), std::to_string(v)),
                 type);
           })
      .def("get_fp16",
           [](xcelerateOpBuilder &self, float v) -> mlir::Value {
             return self.create<mlir::arith::ConstantOp>(
                 self.getBuilder().getF16FloatAttr(v));
           })
      .def("get_fp32",
           [](xcelerateOpBuilder &self, float v) -> mlir::Value {
             return self.create<mlir::arith::ConstantOp>(
                 self.getBuilder().getF32FloatAttr(v));
           })
      .def("get_fp64",
           [](xcelerateOpBuilder &self, double v) -> mlir::Value {
             return self.create<mlir::arith::ConstantOp>(
                 self.getBuilder().getF64FloatAttr(v));
           })
      .def("get_null_value",
           [](xcelerateOpBuilder &self, mlir::Type type) -> mlir::Value {
             if (auto floatTy = type.dyn_cast<mlir::FloatType>())
               return self.create<mlir::arith::ConstantFloatOp>(
                   mlir::APFloat(floatTy.getFloatSemantics(), 0), floatTy);
             else if (auto intTy = type.dyn_cast<mlir::IntegerType>())
               return self.create<mlir::arith::ConstantIntOp>(0, intTy);
             else
               throw std::runtime_error("Not implemented");
           })
      .def("get_all_ones_value",
           [](xcelerateOpBuilder &self, mlir::Type type) -> mlir::Value {
             uint64_t val = 0xFFFFFFFFFFFFFFFF;
             if (auto intTy = type.dyn_cast<mlir::IntegerType>())
               return self.create<mlir::arith::ConstantIntOp>(val, intTy);
             else
               throw std::runtime_error("Not implemented");
           })

      // Types
      .def("get_void_ty",
           [](xcelerateOpBuilder &self) -> mlir::Type {
             return self.getBuilder().getNoneType();
           })
      .def("get_int1_ty",
           [](xcelerateOpBuilder &self) -> mlir::Type {
             return self.getBuilder().getI1Type();
           }) // or ret::copy?
      .def("get_int8_ty",
           [](xcelerateOpBuilder &self) -> mlir::Type {
             return self.getBuilder().getI8Type();
           })
      .def("get_int16_ty",
           [](xcelerateOpBuilder &self) -> mlir::Type {
             return self.getBuilder().getType<mlir::IntegerType>(16);
           })
      .def("get_int32_ty",
           [](xcelerateOpBuilder &self) -> mlir::Type {
             return self.getBuilder().getI32Type();
           })
      .def("get_int64_ty",
           [](xcelerateOpBuilder &self) -> mlir::Type {
             return self.getBuilder().getI64Type();
           })
      .def("get_fp8e4nv_ty",
           [](xcelerateOpBuilder &self) -> mlir::Type {
             return self.getBuilder().getType<mlir::Float8E4M3FNUZType>();
           })
      .def("get_fp8e4b15_ty",
           [](xcelerateOpBuilder &self) -> mlir::Type {
             // TODO: upstream FP8E4B15 into MLIR, or find a way to externally
             // have a float-like type compatible with float only native ops
             return self.getBuilder().getType<mlir::Float8E4M3B11FNUZType>();
           })
      .def("get_fp8e4b15x4_ty",
           [](xcelerateOpBuilder &self) -> mlir::Type {
             // TODO: upstream FP8E4B15 into MLIR, or find a way to externally
             // have a float-like type compatible with float only native ops
             return self.getBuilder().getType<mlir::Float8E4M3FNType>();
           })
      .def("get_fp8e5_ty",
           [](xcelerateOpBuilder &self) -> mlir::Type {
             return self.getBuilder().getType<mlir::Float8E5M2Type>();
           })
      .def("get_half_ty",
           [](xcelerateOpBuilder &self) -> mlir::Type {
             return self.getBuilder().getF16Type();
           })
      .def("get_bf16_ty",
           [](xcelerateOpBuilder &self) -> mlir::Type {
             return self.getBuilder().getBF16Type();
           })
      .def("get_float_ty",
           [](xcelerateOpBuilder &self) -> mlir::Type {
             return self.getBuilder().getF32Type();
           })
      .def("get_double_ty",
           [](xcelerateOpBuilder &self) -> mlir::Type {
             return self.getBuilder().getF64Type();
           })
      .def("get_ptr_ty",
           [](xcelerateOpBuilder &self, mlir::Type &type,
              int addrSpace) -> mlir::Type {
             return mlir::xcelerate::PointerType::get(type, addrSpace);
           })
      .def("get_block_ty",
           [](xcelerateOpBuilder &self, mlir::Type &elementType,
              std::vector<int64_t> &shape) -> mlir::Type {
             return mlir::RankedTensorType::get(shape, elementType);
           })
      .def("get_function_ty",
           [](xcelerateOpBuilder &self, std::vector<mlir::Type> inTypes,
              std::vector<mlir::Type> outTypes) -> mlir::Type {
             return self.getBuilder().getFunctionType(inTypes, outTypes);
           })
      // locs
      .def("set_loc", [](xcelerateOpBuilder &self,
                         mlir::Location loc) { self.setLastLoc(loc); })
      .def("set_loc",
           [](xcelerateOpBuilder &self, const std::string &fileName, int line,
              int column) { self.setLastLoc(fileName, line, column); })
      .def("get_loc",
           [](xcelerateOpBuilder &self) -> mlir::Location {
             return self.getLastLoc();
           })

      // Ops
      .def("get_or_insert_function",
           [](xcelerateOpBuilder &self, mlir::ModuleOp &module,
              std::string &funcName, mlir::Type &funcType,
              std::string &visibility, bool noinline) -> mlir::xcelerate::FuncOp {
             if (mlir::Operation *funcOperation = module.lookupSymbol(funcName))
               return llvm::dyn_cast<mlir::xcelerate::FuncOp>(funcOperation);
             if (auto funcTy = funcType.dyn_cast<mlir::FunctionType>()) {
               llvm::SmallVector<mlir::NamedAttribute> attrs = {
                   mlir::NamedAttribute(
                       self.getBuilder().getStringAttr("sym_visibility"),
                       self.getBuilder().getStringAttr(visibility)),
                   mlir::NamedAttribute(
                       self.getBuilder().getStringAttr("noinline"),
                       self.getBuilder().getBoolAttr(noinline))};
               return self.create<mlir::xcelerate::FuncOp>(funcName, funcTy,
                                                        attrs);
             }
             throw std::runtime_error("invalid function type");
           })
      .def(
          "create_block",
          [](xcelerateOpBuilder &self) -> mlir::Block * {
            mlir::Region *parent = self.getBuilder().getBlock()->getParent();
            return self.getBuilder().createBlock(parent);
          },
          ret::reference)
      .def(
          "create_block_with_parent",
          [](xcelerateOpBuilder &self, mlir::Region &parent,
             std::vector<mlir::Type> &argTypes) -> mlir::Block * {
            // TODO: update arg loc
            auto loc = self.getBuilder().getUnknownLoc();
            llvm::SmallVector<mlir::Location, 8> argLocs(argTypes.size(), loc);
            return self.getBuilder().createBlock(&parent, {}, argTypes,
                                                 argLocs);
          },
          ret::reference)
      .def(
          "new_block",
          [](xcelerateOpBuilder &self) -> mlir::Block * {
            return new mlir::Block();
          },
          ret::reference)
      // Function
      .def("ret",
           [](xcelerateOpBuilder &self,
              std::vector<mlir::Value> &vals) -> mlir::OpState {
             return self.create<mlir::xcelerate::ReturnOp>(vals);
           })
      .def("call",
           [](xcelerateOpBuilder &self, mlir::xcelerate::FuncOp &func,
              std::vector<mlir::Value> &args) -> mlir::OpState {
             return self.create<mlir::xcelerate::CallOp>(func, args);
           })
      // Unstructured control flow
      .def("create_cond_branch",
           [](xcelerateOpBuilder &self, mlir::Value condition,
              mlir::Block *trueDest, mlir::Block *falseDest) -> mlir::OpState {
             return self.create<mlir::cf::CondBranchOp>(condition, trueDest,
                                                        falseDest);
           })
      .def("create_branch",
           [](xcelerateOpBuilder &self, mlir::Block *dest,
              std::vector<mlir::Value> &args) -> mlir::OpState {
             return self.create<mlir::cf::BranchOp>(dest, args);
           })
      // Structured control flow
      .def("create_for_op",
           [](xcelerateOpBuilder &self, mlir::Value &lb, mlir::Value &ub,
              mlir::Value &step,
              std::vector<mlir::Value> &initArgs) -> mlir::scf::ForOp {
             return self.create<mlir::scf::ForOp>(lb, ub, step, initArgs);
           })
      .def("create_if_op",
           [](xcelerateOpBuilder &self, std::vector<mlir::Type> &retTypes,
              mlir::Value &condition, bool withElse) -> mlir::scf::IfOp {
             return self.create<mlir::scf::IfOp>(retTypes, condition, withElse);
           })
      .def("create_yield_op",
           [](xcelerateOpBuilder &self,
              std::vector<mlir::Value> &yields) -> mlir::scf::YieldOp {
             return self.create<mlir::scf::YieldOp>(yields);
           })
      .def("create_while_op",
           [](xcelerateOpBuilder &self, std::vector<mlir::Type> &retTypes,
              std::vector<mlir::Value> &initArgs) -> mlir::scf::WhileOp {
             return self.create<mlir::scf::WhileOp>(retTypes, initArgs);
           })
      .def("create_condition_op",
           [](xcelerateOpBuilder &self, mlir::Value &cond,
              std::vector<mlir::Value> &args) -> mlir::scf::ConditionOp {
             return self.create<mlir::scf::ConditionOp>(cond, args);
           })

      // miscellaneous
      .def("create_make_range",
           [](xcelerateOpBuilder &self, int start, int end) -> mlir::Value {
             auto retType = mlir::RankedTensorType::get(
                 {end - start}, self.getBuilder().getI32Type());
             return self.create<mlir::xcelerate::MakeRangeOp>(retType, start, end);
           })

      // Cast instructions
      // Conversions for custom FP types (FP8 and non-standard rounding modes)
      .def("create_fp_to_fp",
           [](xcelerateOpBuilder &self, mlir::Value &src, mlir::Type &dstType,
              std::optional<::mlir::xcelerate::RoundingMode> roundingMode)
               -> mlir::Value {
             if (roundingMode.has_value())
               return self.create<mlir::xcelerate::FpToFpOp>(
                   dstType, src,
                   mlir::xcelerate::RoundingModeAttr::get(
                       self.getBuilder().getContext(), roundingMode.value()));
             else
               return self.create<mlir::xcelerate::FpToFpOp>(dstType, src);
           })
      // Conversions for standard LLVM builtin types
      .def("create_bitcast",
           [](xcelerateOpBuilder &self, mlir::Value &src,
              mlir::Type &dstType) -> mlir::Value {
             return self.create<mlir::xcelerate::BitcastOp>(dstType, src);
           })
      .def("create_si_to_fp",
           [](xcelerateOpBuilder &self, mlir::Value &src,
              mlir::Type &dstType) -> mlir::Value {
             return self.create<mlir::arith::SIToFPOp>(dstType, src);
           })
      .def("create_ui_to_fp",
           [](xcelerateOpBuilder &self, mlir::Value &src,
              mlir::Type &dstType) -> mlir::Value {
             return self.create<mlir::arith::UIToFPOp>(dstType, src);
           })
      .def("create_fp_to_si",
           [](xcelerateOpBuilder &self, mlir::Value &src,
              mlir::Type &dstType) -> mlir::Value {
             return self.create<mlir::arith::FPToSIOp>(dstType, src);
           })
      .def("create_fp_to_ui",
           [](xcelerateOpBuilder &self, mlir::Value &src,
              mlir::Type &dstType) -> mlir::Value {
             return self.create<mlir::arith::FPToUIOp>(dstType, src);
           })
      .def("create_fp_ext",
           [](xcelerateOpBuilder &self, mlir::Value &src,
              mlir::Type &dstType) -> mlir::Value {
             return self.create<mlir::arith::ExtFOp>(dstType, src);
           })
      .def("create_fp_trunc",
           [](xcelerateOpBuilder &self, mlir::Value &src,
              mlir::Type &dstType) -> mlir::Value {
             return self.create<mlir::arith::TruncFOp>(dstType, src);
           })
      .def("create_int_cast",
           [](xcelerateOpBuilder &self, mlir::Value &src, mlir::Type &dstType,
              bool isSigned) -> mlir::Value {
             // get element type if necessary
             mlir::Type srcType = src.getType();
             auto srcTensorType = srcType.dyn_cast<mlir::RankedTensorType>();
             auto dstTensorType = dstType.dyn_cast<mlir::RankedTensorType>();
             mlir::Type srcEltType = srcType;
             mlir::Type dstEltType = dstType;
             if (dstTensorType && srcTensorType) {
               dstEltType = dstTensorType.getElementType();
               srcEltType = srcTensorType.getElementType();
             }
             unsigned srcWidth = srcEltType.getIntOrFloatBitWidth();
             unsigned dstWidth = dstEltType.getIntOrFloatBitWidth();
             if (srcWidth == dstWidth)
               return self.create<mlir::arith::BitcastOp>(dstType, src);
             else if (srcWidth > dstWidth)
               return self.create<mlir::arith::TruncIOp>(dstType, src);
             else if (isSigned)
               return self.create<mlir::arith::ExtSIOp>(dstType, src);
             else
               return self.create<mlir::arith::ExtUIOp>(dstType, src);
           })
      .def("create_to_index",
           [](xcelerateOpBuilder &self, mlir::Value &input) -> mlir::Value {
             return self.create<mlir::arith::IndexCastOp>(
                 self.getBuilder().getIndexType(), input);
           })
      .def("create_index_to_si",
           [](xcelerateOpBuilder &self, mlir::Value &input) -> mlir::Value {
             return self.create<mlir::arith::IndexCastOp>(
                 self.getBuilder().getI64Type(), input);
           })
      .def("create_fmul",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::MulFOp>(lhs, rhs);
           })
      .def("create_fdiv",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::DivFOp>(lhs, rhs);
           })
      .def("create_frem",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::RemFOp>(lhs, rhs);
           })
      .def("create_fadd",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::AddFOp>(lhs, rhs);
           })
      .def("create_fsub",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::SubFOp>(lhs, rhs);
           })
      .def("create_mul",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::MulIOp>(lhs, rhs);
           })
      .def("create_sdiv",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::DivSIOp>(lhs, rhs);
           })
      .def("create_udiv",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::DivUIOp>(lhs, rhs);
           })
      .def("create_srem",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::RemSIOp>(lhs, rhs);
           })
      .def("create_urem",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::RemUIOp>(lhs, rhs);
           })
      .def("create_add",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::AddIOp>(lhs, rhs);
           })
      .def("create_sub",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return mlir::Value(self.create<mlir::arith::SubIOp>(lhs, rhs));
           })
      .def("create_shl",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return mlir::Value(self.create<mlir::arith::ShLIOp>(lhs, rhs));
           })
      .def("create_lshr",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return mlir::Value(self.create<mlir::arith::ShRUIOp>(lhs, rhs));
           })
      .def("create_ashr",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return mlir::Value(self.create<mlir::arith::ShRSIOp>(lhs, rhs));
           })
      .def("create_minsi",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return mlir::Value(self.create<mlir::arith::MinSIOp>(lhs, rhs));
           })
      .def("create_minui",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return mlir::Value(self.create<mlir::arith::MinUIOp>(lhs, rhs));
           })
      // minimumf follows the torch.minimum convention and returns NaN if either
      // operand is NaN
      .def("create_minimumf",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return mlir::Value(self.create<mlir::arith::MinimumFOp>(lhs, rhs));
           })
      // minnumf follows the torch.fmin convention and returns the non-NaN
      // operand
      .def("create_minnumf",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return mlir::Value(self.create<mlir::arith::MinNumFOp>(lhs, rhs));
           })
      .def("create_maxsi",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return mlir::Value(self.create<mlir::arith::MaxSIOp>(lhs, rhs));
           })
      .def("create_maxui",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return mlir::Value(self.create<mlir::arith::MaxUIOp>(lhs, rhs));
           })
      // maximumf follows the torch.maximum convention and returns NaN if either
      // operand is NaN
      .def("create_maximumf",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return mlir::Value(self.create<mlir::arith::MaximumFOp>(lhs, rhs));
           })
      // maxnumf follows the torch.fmax convention and returns the non-NaN
      // operand
      .def("create_maxnumf",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return mlir::Value(self.create<mlir::arith::MaxNumFOp>(lhs, rhs));
           })
      .def("create_clampf",
           [](xcelerateOpBuilder &self, mlir::Value &input, mlir::Value &min,
              mlir::Value &max,
              mlir::xcelerate::PropagateNan propagateNan) -> mlir::Value {
             return mlir::Value(self.create<mlir::xcelerate::ClampFOp>(
                 input, min, max, propagateNan));
           })
      // AddPtr (similar to GEP)
      .def("create_addptr",
           [](xcelerateOpBuilder &self, mlir::Value &ptr,
              mlir::Value &offset) -> mlir::Value {
             return self.create<mlir::xcelerate::AddPtrOp>(ptr.getType(), ptr,
                                                        offset);
           })
      // Comparison (int)
      .def("create_icmpSLE",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::CmpIOp>(
                 mlir::arith::CmpIPredicate::sle, lhs, rhs);
           })
      .def("create_icmpSLT",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::CmpIOp>(
                 mlir::arith::CmpIPredicate::slt, lhs, rhs);
           })
      .def("create_icmpSGE",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::CmpIOp>(
                 mlir::arith::CmpIPredicate::sge, lhs, rhs);
           })
      .def("create_icmpSGT",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::CmpIOp>(
                 mlir::arith::CmpIPredicate::sgt, lhs, rhs);
           })
      .def("create_icmpULE",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::CmpIOp>(
                 mlir::arith::CmpIPredicate::ule, lhs, rhs);
           })
      .def("create_icmpULT",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::CmpIOp>(
                 mlir::arith::CmpIPredicate::ult, lhs, rhs);
           })
      .def("create_icmpUGE",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::CmpIOp>(
                 mlir::arith::CmpIPredicate::uge, lhs, rhs);
           })
      .def("create_icmpUGT",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::CmpIOp>(
                 mlir::arith::CmpIPredicate::ugt, lhs, rhs);
           })
      .def("create_icmpEQ",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::CmpIOp>(
                 mlir::arith::CmpIPredicate::eq, lhs, rhs);
           })
      .def("create_icmpNE",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::CmpIOp>(
                 mlir::arith::CmpIPredicate::ne, lhs, rhs);
           })
      // Comparison (float)
      .def("create_fcmpOLT",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::CmpFOp>(
                 mlir::arith::CmpFPredicate::OLT, lhs, rhs);
           })
      .def("create_fcmpOGT",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::CmpFOp>(
                 mlir::arith::CmpFPredicate::OGT, lhs, rhs);
           })
      .def("create_fcmpOLE",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::CmpFOp>(
                 mlir::arith::CmpFPredicate::OLE, lhs, rhs);
           })
      .def("create_fcmpOGE",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::CmpFOp>(
                 mlir::arith::CmpFPredicate::OGE, lhs, rhs);
           })
      .def("create_fcmpOEQ",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::CmpFOp>(
                 mlir::arith::CmpFPredicate::OEQ, lhs, rhs);
           })
      .def("create_fcmpONE",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::CmpFOp>(
                 mlir::arith::CmpFPredicate::ONE, lhs, rhs);
           })
      .def("create_fcmpULT",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::CmpFOp>(
                 mlir::arith::CmpFPredicate::ULT, lhs, rhs);
           })
      .def("create_fcmpUGT",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::CmpFOp>(
                 mlir::arith::CmpFPredicate::UGT, lhs, rhs);
           })
      .def("create_fcmpULE",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::CmpFOp>(
                 mlir::arith::CmpFPredicate::ULE, lhs, rhs);
           })
      .def("create_fcmpUGE",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::CmpFOp>(
                 mlir::arith::CmpFPredicate::UGE, lhs, rhs);
           })
      .def("create_fcmpUEQ",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::CmpFOp>(
                 mlir::arith::CmpFPredicate::UEQ, lhs, rhs);
           })
      .def("create_fcmpUNE",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::CmpFOp>(
                 mlir::arith::CmpFPredicate::UNE, lhs, rhs);
           })
      // // Logical
      .def("create_and",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::AndIOp>(lhs, rhs);
           })
      .def("create_xor",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::XOrIOp>(lhs, rhs);
           })
      .def("create_or",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::OrIOp>(lhs, rhs);
           })
      // Input/Output
      .def("create_load",
           [](xcelerateOpBuilder &self, mlir::Value &ptrs,
              mlir::xcelerate::CacheModifier cacheModifier,
              mlir::xcelerate::EvictionPolicy evictionPolicy,
              bool isVolatile) -> mlir::Value {
             return self.create<mlir::xcelerate::LoadOp>(
                 ptrs, cacheModifier, evictionPolicy, isVolatile);
           })
      .def("create_store",
           [](xcelerateOpBuilder &self, mlir::Value &ptrs, mlir::Value &value,
              mlir::xcelerate::CacheModifier cacheModifier,
              mlir::xcelerate::EvictionPolicy evictionPolicy) -> void {
             self.create<mlir::xcelerate::StoreOp>(ptrs, value, cacheModifier,
                                                evictionPolicy);
           })
      .def("create_tensor_pointer_load",
           [](xcelerateOpBuilder &self, mlir::Value &ptr,
              std::vector<int32_t> &boundaryCheck,
              std::optional<mlir::xcelerate::PaddingOption> paddingOption,
              mlir::xcelerate::CacheModifier cacheModifier,
              mlir::xcelerate::EvictionPolicy evictionPolicy,
              bool isVolatile) -> mlir::Value {
             return self.create<mlir::xcelerate::LoadOp>(
                 ptr, boundaryCheck, paddingOption, cacheModifier,
                 evictionPolicy, isVolatile);
           })
      .def("create_tensor_pointer_store",
           [](xcelerateOpBuilder &self, mlir::Value &ptr, mlir::Value &val,
              std::vector<int32_t> &boundaryCheck,
              mlir::xcelerate::CacheModifier cacheModifier,
              mlir::xcelerate::EvictionPolicy evictionPolicy) -> void {
             self.create<mlir::xcelerate::StoreOp>(ptr, val, boundaryCheck,
                                                cacheModifier, evictionPolicy);
           })
      .def("create_masked_load",
           [](xcelerateOpBuilder &self, mlir::Value &ptrs, mlir::Value &mask,
              std::optional<mlir::Value> &other,
              mlir::xcelerate::CacheModifier cacheModifier,
              mlir::xcelerate::EvictionPolicy evictionPolicy,
              bool isVolatile) -> mlir::Value {
             return self.create<mlir::xcelerate::LoadOp>(
                 ptrs, mask, other.value_or(mlir::Value()), cacheModifier,
                 evictionPolicy, isVolatile);
           })
      .def("create_masked_store",
           [](xcelerateOpBuilder &self, mlir::Value &ptrs, mlir::Value &val,
              mlir::Value &mask, mlir::xcelerate::CacheModifier cacheModifier,
              mlir::xcelerate::EvictionPolicy evictionPolicy) -> void {
             self.create<mlir::xcelerate::StoreOp>(ptrs, val, mask, cacheModifier,
                                                evictionPolicy);
           })
      .def("create_reshape",
           [](xcelerateOpBuilder &self, mlir::Value &arg,
              std::vector<int64_t> &shape, bool allowReorder) -> mlir::Value {
             auto argType =
                 arg.getType().cast<mlir::RankedTensorType>().getElementType();
             return self.create<mlir::xcelerate::ReshapeOp>(
                 mlir::RankedTensorType::get(shape, argType), arg,
                 allowReorder);
           })
      .def(
          "create_expand_dims",
          [](xcelerateOpBuilder &self, mlir::Value &arg, int axis) -> mlir::Value {
            auto argType = arg.getType().dyn_cast<mlir::RankedTensorType>();
            auto argEltType = argType.getElementType();
            std::vector<int64_t> retShape = argType.getShape();
            retShape.insert(retShape.begin() + axis, 1);
            return self.create<mlir::xcelerate::ExpandDimsOp>(
                mlir::RankedTensorType::get(retShape, argEltType), arg, axis);
          })
      .def("create_cat",
           [](xcelerateOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto lhsType = lhs.getType().dyn_cast<mlir::RankedTensorType>();
             auto rhsType = rhs.getType().dyn_cast<mlir::RankedTensorType>();
             if (!(lhsType.getShape().size() == 1 &&
                   rhsType.getShape().size() == 1))
               throw std::runtime_error(
                   "shape not supported by cat. Expecting rank-1 inputs");
             std::vector<int64_t> shape{lhsType.getShape()[0] +
                                        rhsType.getShape()[0]};
             return self.create<mlir::xcelerate::CatOp>(
                 mlir::RankedTensorType::get(shape, lhsType.getElementType()),
                 lhs, rhs);
           })
      .def("create_interleave",
           [](xcelerateOpBuilder &self, mlir::Value &a,
              mlir::Value &b) -> mlir::Value {
             auto aTy = a.getType().cast<mlir::RankedTensorType>();
             llvm::SmallVector<int64_t> shape(aTy.getShape().begin(),
                                              aTy.getShape().end());
             shape[shape.size() - 1] *= 2;
             return self.create<mlir::xcelerate::ExperimentalInterleaveOp>(
                 mlir::RankedTensorType::get(shape, aTy.getElementType()), a,
                 b);
           })
      .def("create_trans",
           [](xcelerateOpBuilder &self, mlir::Value &arg) -> mlir::Value {
             auto argType = arg.getType().dyn_cast<mlir::RankedTensorType>();
             auto argEltType = argType.getElementType();
             std::vector<int64_t> retShape = argType.getShape();
             std::reverse(retShape.begin(), retShape.end());
             return self.create<mlir::xcelerate::TransOp>(
                 mlir::RankedTensorType::get(retShape, argEltType), arg);
           })
      .def("create_broadcast",
           [](xcelerateOpBuilder &self, mlir::Value &arg,
              std::vector<int64_t> &shape) -> mlir::Value {
             if (auto argType =
                     arg.getType().dyn_cast<mlir::RankedTensorType>())
               return self.createOrFold<mlir::xcelerate::BroadcastOp>(
                   mlir::RankedTensorType::get(shape, argType.getElementType()),
                   arg);
             throw std::runtime_error(
                 "arg is not of RankedTensorType, use create_splat");
           })
      .def("create_splat",
           [](xcelerateOpBuilder &self, mlir::Value &arg,
              std::vector<int64_t> &shape) -> mlir::Value {
             auto argType = arg.getType();
             auto ret = self.createOrFold<mlir::xcelerate::SplatOp>(
                 mlir::RankedTensorType::get(shape, argType), arg);
             return ret;
           })
      // // atomic
      .def("create_atomic_cas",
           [](xcelerateOpBuilder &self, mlir::Value &ptr, mlir::Value &cmp,
              mlir::Value &val, mlir::xcelerate::MemSemantic sem,
              mlir::xcelerate::MemSyncScope scope) -> mlir::Value {
             mlir::Type dstType;
             if (auto srcTensorType =
                     ptr.getType().dyn_cast<mlir::RankedTensorType>()) {
               mlir::Type dstElemType = srcTensorType.getElementType()
                                            .cast<mlir::xcelerate::PointerType>()
                                            .getPointeeType();
               dstType = mlir::RankedTensorType::get(srcTensorType.getShape(),
                                                     dstElemType);
             } else {
               auto ptrType = mlir::getElementTypeOrSelf(ptr)
                                  .cast<mlir::xcelerate::PointerType>();
               dstType = ptrType.getPointeeType();
             }
             return self.create<mlir::xcelerate::AtomicCASOp>(dstType, ptr, cmp,
                                                           val, sem, scope);
           })
      .def("create_atomic_rmw",
           [](xcelerateOpBuilder &self, mlir::xcelerate::RMWOp rmwOp,
              mlir::Value &ptr, mlir::Value &val, mlir::Value &mask,
              mlir::xcelerate::MemSemantic sem,
              mlir::xcelerate::MemSyncScope scope) -> mlir::Value {
             mlir::Type dstType;
             if (auto srcTensorType =
                     ptr.getType().dyn_cast<mlir::RankedTensorType>()) {
               mlir::Type dstElemType = srcTensorType.getElementType()
                                            .cast<mlir::xcelerate::PointerType>()
                                            .getPointeeType();
               dstType = mlir::RankedTensorType::get(srcTensorType.getShape(),
                                                     dstElemType);
             } else {
               auto ptrType = mlir::getElementTypeOrSelf(ptr)
                                  .cast<mlir::xcelerate::PointerType>();
               dstType = ptrType.getPointeeType();
             }
             return self.create<mlir::xcelerate::AtomicRMWOp>(
                 dstType, rmwOp, ptr, val, mask, sem, scope);
           })
      // External
      .def("create_extern_elementwise",
           [](xcelerateOpBuilder &self, const std::string &libName,
              const std::string &libPath, const std::string &symbol,
              std::vector<mlir::Value> &argList, mlir::Type retType,
              bool isPure) -> mlir::Value {
             return self.create<mlir::xcelerate::ExternElementwiseOp>(
                 retType, argList, libName, libPath, symbol, isPure);
           })
      // Built-in instruction
      .def("create_get_program_id",
           [](xcelerateOpBuilder &self, int axis) -> mlir::Value {
             if (axis < 0 || axis > 3)
               throw std::runtime_error("program_id must be in [0,3]");
             return self.create<mlir::xcelerate::GetProgramIdOp>(
                 self.getBuilder().getI32Type(),
                 mlir::xcelerate::ProgramIDDimAttr::get(
                     self.getBuilder().getContext(),
                     mlir::xcelerate::ProgramIDDim(axis)));
           })
      .def("create_get_num_programs",
           [](xcelerateOpBuilder &self, int axis) -> mlir::Value {
             return self.create<mlir::xcelerate::GetNumProgramsOp>(
                 self.getBuilder().getI32Type(),
                 self.getBuilder().getI32IntegerAttr(axis));
           })
      .def("create_dot",
           [](xcelerateOpBuilder &self, mlir::Value &a, mlir::Value &b,
              mlir::Value &c, bool allowTF32,
              int maxNumImpreciseAcc) -> mlir::Value {
             return self.create<mlir::xcelerate::DotOp>(
                 c.getType(), a, b, c, allowTF32, maxNumImpreciseAcc);
           })
      .def("create_exp",
           [](xcelerateOpBuilder &self, mlir::Value &val) -> mlir::Value {
             return self.create<mlir::math::ExpOp>(val);
           })
      .def("create_cos",
           [](xcelerateOpBuilder &self, mlir::Value &val) -> mlir::Value {
             return self.create<mlir::math::CosOp>(val);
           })
      .def("create_sin",
           [](xcelerateOpBuilder &self, mlir::Value &val) -> mlir::Value {
             return self.create<mlir::math::SinOp>(val);
           })
      .def("create_log",
           [](xcelerateOpBuilder &self, mlir::Value &val) -> mlir::Value {
             return self.create<mlir::math::LogOp>(val);
           })
      .def("create_sqrt",
           [](xcelerateOpBuilder &self, mlir::Value &val) -> mlir::Value {
             return self.create<mlir::math::SqrtOp>(val);
           })
      .def("create_fabs",
           [](xcelerateOpBuilder &self, mlir::Value &val) -> mlir::Value {
             return self.create<mlir::math::AbsFOp>(val);
           })
      .def("create_iabs",
           [](xcelerateOpBuilder &self, mlir::Value &val) -> mlir::Value {
             return self.create<mlir::math::AbsIOp>(val);
           })
      .def("create_reduce",
           [](xcelerateOpBuilder &self, std::vector<mlir::Value> operands,
              int axis) -> mlir::OpState {
             return self.create<mlir::xcelerate::ReduceOp>(operands, axis);
           })
      .def("create_reduce_ret",
           [](xcelerateOpBuilder &self, py::args args) -> mlir::OpState {
             llvm::SmallVector<mlir::Value> return_values;
             for (const auto &arg : args) {
               return_values.push_back(py::cast<mlir::Value>(arg));
             }
             return self.create<mlir::xcelerate::ReduceReturnOp>(return_values);
           })
      .def("create_scan",
           [](xcelerateOpBuilder &self, std::vector<mlir::Value> operands,
              int axis) -> mlir::OpState {
             return self.create<mlir::xcelerate::ScanOp>(operands, axis);
           })
      .def("create_scan_ret",
           [](xcelerateOpBuilder &self, py::args args) -> mlir::OpState {
             llvm::SmallVector<mlir::Value> return_values;
             for (const auto &arg : args) {
               return_values.push_back(py::cast<mlir::Value>(arg));
             }
             return self.create<mlir::xcelerate::ScanReturnOp>(return_values);
           })
      .def("create_ptr_to_int",
           [](xcelerateOpBuilder &self, mlir::Value &val,
              mlir::Type &type) -> mlir::Value {
             return self.create<mlir::xcelerate::PtrToIntOp>(type, val);
           })
      .def("create_int_to_ptr",
           [](xcelerateOpBuilder &self, mlir::Value &val,
              mlir::Type &type) -> mlir::Value {
             return self.create<mlir::xcelerate::IntToPtrOp>(type, val);
           })
      .def("create_select",
           [](xcelerateOpBuilder &self, mlir::Value &condition,
              mlir::Value &trueValue, mlir::Value &falseValue) -> mlir::Value {
             return self.create<mlir::arith::SelectOp>(condition, trueValue,
                                                       falseValue);
           })
      .def("create_inline_asm",
           [](xcelerateOpBuilder &self, const std::string &inlineAsm,
              const std::string &constraints,
              const std::vector<mlir::Value> &values,
              const std::vector<mlir::Type> &types, bool isPure,
              int pack) -> mlir::OpState {
             return self.create<mlir::xcelerate::ElementwiseInlineAsmOp>(
                 types, inlineAsm, constraints, isPure, pack, values);
           })
      .def("create_print",
           [](xcelerateOpBuilder &self, const std::string &prefix,
              const std::vector<mlir::Value> &values) -> void {
             self.create<mlir::xcelerate::PrintOp>(
                 mlir::StringAttr::get(self.getBuilder().getContext(),
                                       llvm::StringRef(prefix)),
                 values);
           })
      .def("create_assert",
           [](xcelerateOpBuilder &self, mlir::Value &condition,
              const std::string &message, const std::string &fileName,
              const std::string &funcName, unsigned lineNo) -> void {
             auto messageAttr = mlir::StringAttr::get(
                 self.getBuilder().getContext(), llvm::StringRef(message));
             auto fileNameAttr = mlir::StringAttr::get(
                 self.getBuilder().getContext(), llvm::StringRef(fileName));
             auto funcNameAttr = mlir::StringAttr::get(
                 self.getBuilder().getContext(), llvm::StringRef(funcName));
             auto lineNoAttr = self.getBuilder().getI32IntegerAttr(lineNo);
             self.create<mlir::xcelerate::AssertOp>(condition, messageAttr,
                                                 fileNameAttr, funcNameAttr,
                                                 lineNoAttr);
           })
      // Undef
      .def("create_undef",
           [](xcelerateOpBuilder &self, mlir::Type &type) -> mlir::Value {
             return self.create<::mlir::LLVM::UndefOp>(type);
           })
      .def("create_histogram",
           [](xcelerateOpBuilder &self, mlir::Value operand,
              int numBins) -> mlir::OpState {
             return self.create<mlir::xcelerate::HistogramOp>(
                 mlir::RankedTensorType::get(
                     {static_cast<int64_t>(numBins)},
                     mlir::IntegerType::get(operand.getContext(), 32)),
                 operand);
           })
      // Force GPU barrier
      .def("create_barrier",
           [](xcelerateOpBuilder &self) { self.create<mlir::gpu::BarrierOp>(); })
      // Make a block pointer (tensor pointer in xcelerate IR)
      .def("create_make_block_ptr",
           [](xcelerateOpBuilder &self, mlir::Value &base,
              std::vector<mlir::Value> &shape,
              std::vector<mlir::Value> &strides,
              std::vector<mlir::Value> &offsets,
              std::vector<int32_t> &tensorShape,
              std::vector<int32_t> &order) -> mlir::Value {
             return self.create<mlir::xcelerate::MakeTensorPtrOp>(
                 base, shape, strides, offsets, tensorShape, order);
           })
      // Advance a block pointer
      .def("create_advance",
           [](xcelerateOpBuilder &self, mlir::Value &ptr,
              std::vector<mlir::Value> &offsets) -> mlir::Value {
             return self.create<mlir::xcelerate::AdvanceOp>(ptr.getType(), ptr,
                                                         offsets);
           });

  py::class_<mlir::PassManager>(m, "pass_manager", py::module_local())
      .def(py::init<mlir::MLIRContext *>())
      .def("enable_debug",
           [](mlir::PassManager &self) {
             auto *context = self.getContext();
             context->printOpOnDiagnostic(true);
             context->printStackTraceOnDiagnostic(true);
             context->disableMultithreading();
             context->getDiagEngine().registerHandler(
                 [](mlir::Diagnostic &diag) {
                   llvm::outs() << diag << "\n";
                   return mlir::success();
                 });

             if (!::xcelerate::tools::getBoolEnv("MLIR_ENABLE_DUMP"))
               return;
             auto printingFlags = mlir::OpPrintingFlags();
             printingFlags.elideLargeElementsAttrs(16);
             printingFlags.enableDebugInfo();
             auto print_always = [](mlir::Pass *, mlir::Operation *) {
               return true;
             };
             self.enableIRPrinting(
                 /*shouldPrintBeforePass=*/print_always,
                 /*shouldPrintAfterPass=*/print_always,
                 /*printModuleScope=*/true,
                 /*printAfterOnlyOnChange=*/false,
                 /*printAfterOnlyOnFailure*/ true, llvm::dbgs(), printingFlags);
           })
      .def("run", [](mlir::PassManager &self, mlir::ModuleOp &mod) {
        // TODO: maybe dump module to file and print error for better
        // diagnostics
        auto reproducerPath = ::xcelerate::tools::getenv("xcelerate_REPRODUCER_PATH");
        if (!reproducerPath.empty()) {
          auto anchorName = self.getOpAnchorName();
          auto passes = self.getPasses();
          mlir::Operation *op = mod.getOperation();
          mlir::makeReproducer(anchorName, passes, op, reproducerPath);
        }

        if (mlir::failed(self.run(mod.getOperation())))
          throw std::runtime_error("PassManager::run failed");
      });
}

void init_xcelerate_env_vars(py::module &m) {
  m.def("get_env_vars", []() -> std::map<std::string, bool> {
    std::map<std::string, bool> envVars;
    for (const auto &envVar : xcelerate::ENV_VARS) {
      envVars[envVar] = xcelerate::tools::getBoolEnv(envVar);
    }
    return envVars;
  });
}
