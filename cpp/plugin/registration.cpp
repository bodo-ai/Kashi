#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "iree/compiler/PluginAPI/Client.h"
#include "iree/compiler/PluginAPI/PluginEntryPoint.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"

namespace {
using namespace mlir;
using namespace mlir::iree_compiler;

struct VerifySegments final
    : PassWrapper<VerifySegments, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifySegments)
  /// Identify the verifier in the production start-phase pipeline.
  StringRef getArgument() const override { return "kashi-verify-segments"; }
  /// Describe the verifier in MLIR's registered pass list.
  StringRef getDescription() const override {
    return "Verify Kashi segment entry points";
  }
  /// Ensure partitioned functions can be called as compiler entry points.
  void runOnOperation() override {
    for (auto function : getOperation().getOps<func::FuncOp>()) {
      if (!function.getSymName().starts_with("segment_") ||
          !function.isPublic() || function.isExternal()) {
        function.emitError("expected a public Kashi segment definition");
        signalPassFailure();
      }
    }
  }
};

class ProbeOp final : public Op<ProbeOp> {
public:
  using Op::Op;
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ProbeOp)
  /// Name the temporary operation used to verify typed dialect registration.
  static StringRef getOperationName() { return "kashi.probe"; }
  /// The probe has no attributes and carries no compilation state.
  static ArrayRef<StringRef> getAttributeNames() { return {}; }
};

class KashiDialect final : public Dialect {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(KashiDialect)
  /// Register the probe operation in this invocation's MLIR context.
  explicit KashiDialect(MLIRContext *context)
      : Dialect(getDialectNamespace(), context, TypeID::get<KashiDialect>()) {
    addOperations<ProbeOp>();
  }
  /// Return the private namespace used for checkpoint probe operations.
  static StringRef getDialectNamespace() { return "kashi"; }
};

template <typename Derived, typename OpType>
struct Probe : PassWrapper<Derived, OperationPass<ModuleOp>> {
private:
  friend Derived;
  Probe() = default;

public:
  /// Stop at the first typed operation expected at this checkpoint.
  void runOnOperation() override {
    auto module = this->getOperation();
    if (!module.walk([](OpType) { return WalkResult::interrupt(); })
             .wasInterrupted()) {
      module.emitError("missing typed operation at Kashi checkpoint: ")
          << OpType::getOperationName();
      this->signalPassFailure();
    }
  }
};

struct ProbeTorch final : Probe<ProbeTorch, torch::Torch::AtenReluOp> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ProbeTorch)
  /// Identify the typed Torch probe run before input lowering.
  StringRef getArgument() const override { return "kashi-probe-torch"; }
  /// Check Torch IR and exercise the Kashi dialect without changing the module.
  void runOnOperation() override {
    Probe::runOnOperation();
    auto module = getOperation();
    if (!module.getContext()->getLoadedDialect<KashiDialect>()) {
      module.emitError("Kashi plugin did not register its internal dialect");
      signalPassFailure();
      return;
    }
    OpBuilder builder = OpBuilder::atBlockBegin(module.getBody());
    auto *op = builder.create(
        OperationState(module.getLoc(), ProbeOp::getOperationName()));
    if (!isa<ProbeOp>(op)) {
      op->emitError("Kashi probe operation was not registered");
      signalPassFailure();
    }
    op->erase();
  }
};

struct ProbeFlow final : Probe<ProbeFlow, IREE::Flow::DispatchOp> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ProbeFlow)
  /// Identify the dispatch probe run before Stream lowering.
  StringRef getArgument() const override { return "kashi-probe-flow"; }
};

// Give independent executables the same local function name before linking.
// Their original Flow export names and dispatch references remain different.
struct CollideFlowFunctions final
    : PassWrapper<CollideFlowFunctions, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CollideFlowFunctions)
  /// Identify the test transformation that runs before the Flow snapshot.
  StringRef getArgument() const override {
    return "kashi-collide-flow-functions";
  }
  /// Introduce a name collision across executable-local symbol tables.
  void runOnOperation() override {
    size_t renamed = 0;
    for (auto executable : getOperation().getOps<IREE::Flow::ExecutableOp>()) {
      SymbolTable inner(executable.getInnerModule());
      auto exports = executable.getOps<IREE::Flow::ExecutableExportOp>();
      if (!llvm::hasSingleElement(exports))
        continue;
      auto exportOp = *exports.begin();
      auto *function = inner.lookup(exportOp.getFunctionRef());
      if (!function || failed(inner.rename(function, "shared_kernel"))) {
        exportOp.emitError("cannot set up the colliding Flow export fixture");
        signalPassFailure();
        return;
      }
      exportOp.setFunctionRefAttr(
          FlatSymbolRefAttr::get(&getContext(), "shared_kernel"));
      ++renamed;
    }
    if (renamed < 2) {
      getOperation().emitError(
          "expected at least two independent Flow executables");
      signalPassFailure();
    }
  }
};

struct ProbeConfigurations final
    : Probe<ProbeConfigurations, IREE::HAL::ExecutableVariantOp> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ProbeConfigurations)
  /// Identify the HAL probe run after executable configuration.
  StringRef getArgument() const override {
    return "kashi-probe-configurations";
  }
};

struct ProbeNvvm final : Probe<ProbeNvvm, NVVM::ThreadIdXOp> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ProbeNvvm)
  /// Identify the nested probe run after NVVM lowering.
  StringRef getArgument() const override { return "kashi-probe-nvvm"; }
};

struct ProbeVm final : Probe<ProbeVm, IREE::VM::ModuleOp> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ProbeVm)
  /// Identify the VM probe run before serialization.
  StringRef getArgument() const override { return "kashi-probe-vm"; }
};

struct FailFlow final : PassWrapper<FailFlow, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FailFlow)
  /// Identify the failure injection used only by the Flow recovery test.
  StringRef getArgument() const override { return "kashi-fail-flow-probe"; }
  /// Fail deliberately to test checkpoint diagnostics and session recovery.
  void runOnOperation() override {
    getOperation().emitError("Kashi Flow probe failure");
    signalPassFailure();
  }
};

struct KashiSession : PluginSession<KashiSession> {
  /// Register the production verifier and the implementation-owned test passes.
  static void registerPasses() {
    PassRegistration<VerifySegments>();
    PassRegistration<ProbeTorch>();
    PassRegistration<ProbeFlow>();
    PassRegistration<CollideFlowFunctions>();
    PassRegistration<ProbeConfigurations>();
    PassRegistration<ProbeNvvm>();
    PassRegistration<ProbeVm>();
    PassRegistration<FailFlow>();
  }
  /// Make the internal dialect available in sessions using this plugin.
  void onRegisterDialects(DialectRegistry &registry) override {
    registry.insert<KashiDialect>();
  }
  /// Load the dialect before any checkpoint pass constructs its operations.
  LogicalResult onActivate() override {
    context->loadDialect<KashiDialect>();
    return success();
  }
};

/// Connect the statically linked entry point to IREE's plugin registry.
bool registerKashiPlugin(PluginRegistrar *registrar) {
  registrar->registerPlugin<KashiSession>("kashi");
  return true;
}
} // namespace

IREE_DEFINE_COMPILER_PLUGIN(kashi, registerKashiPlugin, "0.1")
