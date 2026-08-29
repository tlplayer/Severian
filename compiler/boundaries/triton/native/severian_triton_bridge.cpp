#include "severian_triton_bridge.h"

#include "TritonAMDGPUToLLVM/Passes.h"
#include "TritonNVIDIAGPUToLLVM/Passes.h"
#include "NVGPUToLLVM/Passes.h"
#include "lld/Common/Driver.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

LLD_HAS_DRIVER(elf)

namespace {

using mlir::ModuleOp;

struct OwnedKernel {
  std::string entryPoint;
  std::vector<uint8_t> code;
  std::string diagnostics;
};

struct CompileFailure {
  sev_triton_status status;
  std::string message;
};

static std::once_flag initializeOnce;
static std::mutex llvmCodegenMutex;

static sev_triton_bytes bytes(const std::string &value) {
  return {reinterpret_cast<const uint8_t *>(value.data()), value.size()};
}

static sev_triton_bytes bytes(const std::vector<uint8_t> &value) {
  return {value.data(), value.size()};
}

static std::string stringFrom(sev_triton_bytes value) {
  if (value.len == 0)
    return {};
  return std::string(reinterpret_cast<const char *>(value.data), value.len);
}

static void initializeCompiler() {
  std::call_once(initializeOnce, [] {
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmPrinters();
    llvm::InitializeAllAsmParsers();
    mlir::registerAllPasses();
    mlir::triton::registerTritonPasses();
    mlir::triton::gpu::registerTritonGPUPasses();
    mlir::triton::registerConvertTritonToTritonGPUPass();
    mlir::triton::gpu::registerTritonGPUToLLVMPasses();
    // Backend pipelines are assembled through typed pass factories below.
    // Do not register every command-line AMD pass here: the pinned donor has
    // one transform declared in both `mlir` and `mlir::triton::amdgpu`, and
    // its aggregate textual registry references the wrong namespace.  That
    // registry is unnecessary for parsing TTIR or running our typed pipeline.
    mlir::triton::registerTritonAMDGPUToLLVMPasses();
    mlir::triton::registerTritonNVIDIAGPUToLLVMPasses();
    mlir::triton::registerNVGPUToLLVMPasses();
  });
}

static void registerDialects(mlir::DialectRegistry &registry) {
  mlir::registerAllDialects(registry);
  mlir::func::registerInlinerExtension(registry);
  mlir::LLVM::registerInlinerInterface(registry);
  registry.insert<mlir::triton::TritonDialect,
                  mlir::triton::gpu::TritonGPUDialect>();
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerNVVMDialectTranslation(registry);
  mlir::registerROCDLDialectTranslation(registry);
}

static bool run(mlir::PassManager &pm, ModuleOp module, std::string &diagnostic,
                const char *stage) {
  if (mlir::failed(pm.run(module))) {
    diagnostic.append("native Triton pass stage failed: ");
    diagnostic.append(stage);
    diagnostic.push_back('\n');
    return false;
  }
  return true;
}

static bool lowerTTIR(ModuleOp module, const sev_triton_compile_options &opt,
                      std::string &diagnostic) {
  mlir::PassManager pm(module.getContext());
  pm.enableVerifier(true);
  pm.addPass(mlir::createInlinerPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::triton::createTritonCombineOps());
  pm.addPass(mlir::triton::createTritonReorderBroadcast());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::triton::createTritonLoopInvariantCodeMotion());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::triton::createTritonLoopUnroll());
  if (!run(pm, module, diagnostic, "ttir"))
    return false;

  const std::string backend =
      opt.target == SEV_TRITON_AMD_GPU
          ? "hip:" + stringFrom(opt.architecture)
          : "cuda:" + stringFrom(opt.architecture);
  mlir::PassManager convert(module.getContext());
  convert.enableVerifier(true);
  mlir::triton::ConvertTritonToTritonGPUOptions convertOptions;
  convertOptions.target = backend;
  convertOptions.numWarps = static_cast<int>(opt.num_warps);
  convertOptions.threadsPerWarp = static_cast<int>(opt.warp_size);
  convertOptions.numCTAs = static_cast<int>(opt.num_ctas);
  convert.addPass(
      mlir::triton::createConvertTritonToTritonGPU(convertOptions));
  return run(convert, module, diagnostic, "ttir-to-ttgir");
}

static bool optimizeTTGIR(ModuleOp module,
                          const sev_triton_compile_options &opt,
                          std::string &diagnostic) {
  using namespace mlir::triton::gpu;
  mlir::PassManager pm(module.getContext());
  pm.enableVerifier(true);
  pm.addPass(createTritonGPUCoalesce());
  TritonGPUF32DotTCOptions f32DotOptions;
  f32DotOptions.emuTF32 = false;
  pm.addPass(createTritonGPUF32DotTC(f32DotOptions));
  pm.addPass(createTritonGPURemoveLayoutConversions());
  pm.addPass(createTritonGPUOptimizeThreadLocality());
  // The generic accelerator dispatches on ttg.target. Rank and batch remain
  // properties of the tt.dot IR, never pass or symbol identities.
  pm.addPass(createTritonGPUAccelerateMatmul());
  pm.addPass(createTritonGPURemoveLayoutConversions());
  TritonGPUOptimizeDotOperandsOptions dotOperandOptions;
  dotOperandOptions.hoistLayoutConversion = false;
  pm.addPass(createTritonGPUOptimizeDotOperands(dotOperandOptions));
  pm.addPass(createTritonGPUFuseNestedLoops());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::triton::createTritonLoopInvariantCodeMotion());
  pm.addPass(createTritonGPUScheduleLoops());
  TritonGPUPipelineOptions pipelineOptions;
  pipelineOptions.numStages = static_cast<int>(opt.num_stages);
  pipelineOptions.dumpIntermediateSteps = false;
  pm.addPass(createTritonGPUPipeline(pipelineOptions));
  pm.addPass(createTritonGPURemoveLayoutConversions());
  pm.addPass(createTritonGPUReduceDataDuplication());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());
  return run(pm, module, diagnostic, "ttgir-optimize");
}

static bool lowerToLLVMDialect(ModuleOp module,
                               const sev_triton_compile_options &opt,
                               int capability, int ptxVersion,
                               std::string &diagnostic) {
  mlir::PassManager pm(module.getContext());
  pm.enableVerifier(true);
  pm.addPass(mlir::triton::gpu::createTritonGPUAllocateWarpGroups());
  pm.addPass(mlir::createSCFToControlFlowPass());
  pm.addPass(mlir::createConvertIndexToLLVMPass());
  if (opt.target == SEV_TRITON_AMD_GPU) {
    const std::string arch = stringFrom(opt.architecture);
    pm.addPass(mlir::triton::createAllocateAMDGPUSharedMemoryPass(arch));
    pm.addPass(
        mlir::triton::gpu::createTritonGPUGlobalScratchAllocationPass());
    pm.addPass(mlir::triton::createConvertTritonAMDGPUToLLVMPass(arch, true));
    pm.addPass(
        mlir::triton::AMD::createTritonAMDGPUConvertWarpSpecializeToLLVMPass(
            arch));
  } else {
    pm.addPass(
        mlir::triton::createAllocateSharedMemoryNvPass(capability, ptxVersion));
    pm.addPass(mlir::triton::createConvertTritonGPUToLLVMPass(
        capability, ptxVersion, false));
    pm.addPass(mlir::triton::createConvertWarpSpecializeToLLVM());
    pm.addPass(mlir::triton::createConvertNVGPUToLLVM());
  }
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createConvertControlFlowToLLVMPass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  if (opt.target == SEV_TRITON_NVIDIA_GPU)
    pm.addPass(mlir::createConvertNVVMToLLVMPass());
  if (opt.target == SEV_TRITON_AMD_GPU)
    pm.addPass(mlir::triton::createConvertBuiltinFuncToLLVMPass(
        stringFrom(opt.architecture), true));
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  pm.addPass(mlir::createSymbolDCEPass());
  return run(pm, module, diagnostic, "ttgir-to-llvm-dialect");
}

static int parseNvidiaCapability(const std::string &architecture) {
  std::string digits = architecture;
  if (digits.rfind("sm_", 0) == 0)
    digits.erase(0, 3);
  else if (digits.rfind("sm", 0) == 0)
    digits.erase(0, 2);
  if (digits.empty() ||
      !std::all_of(digits.begin(), digits.end(),
                   [](char c) { return c >= '0' && c <= '9'; }))
    return -1;
  int value = 0;
  for (char c : digits)
    value = value * 10 + (c - '0');
  return value;
}

static std::optional<std::string>
emitLLVM(llvm::Module &module, const std::string &triple,
         const std::string &processor, const std::string &features,
         llvm::CodeGenFileType fileType, std::string &diagnostic) {
  std::lock_guard<std::mutex> lock(llvmCodegenMutex);
  const llvm::Triple targetTriple(triple);
  std::string lookupError;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(targetTriple, lookupError);
  if (!target) {
    diagnostic += "LLVM target lookup failed: " + lookupError + "\n";
    return std::nullopt;
  }
  llvm::TargetOptions targetOptions;
  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      targetTriple, processor, features, targetOptions, llvm::Reloc::PIC_,
      std::nullopt, llvm::CodeGenOptLevel::Aggressive));
  if (!machine) {
    diagnostic += "LLVM could not construct a target machine\n";
    return std::nullopt;
  }
  module.setTargetTriple(targetTriple);
  module.setDataLayout(machine->createDataLayout());
  llvm::SmallVector<char, 0> artifact;
  llvm::raw_svector_ostream stream(artifact);
  llvm::legacy::PassManager passes;
  if (machine->addPassesToEmitFile(passes, stream, nullptr, fileType)) {
    diagnostic += "LLVM target cannot emit the requested artifact\n";
    return std::nullopt;
  }
  passes.run(module);
  return std::string(artifact.begin(), artifact.end());
}

static std::optional<std::string>
prepareAmdKernel(llvm::Module &module,
                 const sev_triton_compile_options &options,
                 std::string &diagnostic) {
  module.setTargetTriple(llvm::Triple("amdgcn-amd-amdhsa"));

  // These are the target-finalization steps performed by Triton's AMD donor
  // immediately after MLIR-to-LLVM translation.  They are deliberately data
  // driven by launch options; no dtype or rank participates in kernel identity.
  llvm::Type *i32 = llvm::Type::getInt32Ty(module.getContext());
  if (!module.getNamedGlobal("__oclc_ABI_version")) {
    auto *abi = new llvm::GlobalVariable(
        module, i32, true, llvm::GlobalValue::LinkOnceODRLinkage,
        llvm::ConstantInt::get(i32, 500), "__oclc_ABI_version", nullptr,
        llvm::GlobalValue::ThreadLocalMode::NotThreadLocal, 4);
    abi->setVisibility(llvm::GlobalValue::ProtectedVisibility);
    abi->setAlignment(llvm::MaybeAlign(4));
    abi->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Local);
  }
  module.addModuleFlag(llvm::Module::Error, "amdhsa_code_object_version", 500);

  llvm::Function *kernel = nullptr;
  for (llvm::Function &function : module) {
    if (!function.isDeclaration() && function.hasExternalLinkage()) {
      if (kernel) {
        diagnostic += "AMD LLVM module contains more than one external kernel\n";
        return std::nullopt;
      }
      kernel = &function;
    }
  }
  if (!kernel) {
    diagnostic += "AMD LLVM module contains no external kernel\n";
    return std::nullopt;
  }

  kernel->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
  kernel->addFnAttr("amdgpu-cluster-dims",
                    std::to_string(options.num_ctas) + ",1,1");
  kernel->addFnAttr(
      "amdgpu-flat-work-group-size",
      "1," + std::to_string(options.num_warps * options.warp_size));
  kernel->addFnAttr("uniform-work-group-size", "true");
  kernel->addFnAttr("denormal-fp-math-f32", "ieee");
  for (llvm::Argument &argument : kernel->args()) {
    if (!argument.hasByRefAttr() && !argument.hasNestAttr())
      argument.addAttr(llvm::Attribute::InReg);
  }
  return kernel->getName().str();
}

static std::optional<std::vector<uint8_t>>
linkHsaco(const std::string &object, std::string &diagnostic) {
  llvm::SmallString<128> inputPath;
  llvm::SmallString<128> outputPath;
  int inputFd = -1;
  int outputFd = -1;
  if (auto error = llvm::sys::fs::createTemporaryFile("severian", "o", inputFd,
                                                       inputPath)) {
    diagnostic += "could not create temporary AMD object: " + error.message();
    return std::nullopt;
  }
  if (auto error = llvm::sys::fs::createTemporaryFile(
          "severian", "hsaco", outputFd, outputPath)) {
    llvm::sys::fs::remove(inputPath);
    diagnostic += "could not create temporary HSACO: " + error.message();
    return std::nullopt;
  }
  {
    llvm::raw_fd_ostream input(inputFd, true);
    input.write(object.data(), object.size());
  }
  llvm::sys::fs::closeFile(outputFd);
  const std::array<const char *, 8> args = {
      "ld.lld", "--threads=1", "-shared", inputPath.c_str(), "-o",
      outputPath.c_str(), "--no-undefined", nullptr};
  std::vector<const char *> actual(args.begin(), args.end() - 1);
  std::string linkerError;
  llvm::raw_string_ostream linkerErrors(linkerError);
  auto result = lld::lldMain(actual, llvm::outs(), linkerErrors,
                             {{lld::Gnu, &lld::elf::link}});
  linkerErrors.flush();
  llvm::sys::fs::remove(inputPath);
  if (result.retCode != 0 || !result.canRunAgain) {
    llvm::sys::fs::remove(outputPath);
    diagnostic += "LLD failed to link HSACO: " + linkerError;
    return std::nullopt;
  }
  auto buffer = llvm::MemoryBuffer::getFile(outputPath);
  llvm::sys::fs::remove(outputPath);
  if (!buffer) {
    diagnostic += "could not read linked HSACO: " +
                  buffer.getError().message();
    return std::nullopt;
  }
  llvm::StringRef contents = buffer.get()->getBuffer();
  return std::vector<uint8_t>(contents.bytes_begin(), contents.bytes_end());
}

static uint64_t sharedMemory(ModuleOp module) {
  if (auto value = module->getAttrOfType<mlir::IntegerAttr>("ttg.shared"))
    return value.getValue().getZExtValue();
  return 0;
}

struct ScratchMemory {
  uint64_t globalBytesPerProgram = 0;
  uint64_t globalAlignment = 1;
  uint64_t profileBytesPerProgram = 0;
  uint64_t profileAlignment = 1;
};

static uint64_t moduleInteger(ModuleOp module, llvm::StringRef name,
                              uint64_t fallback) {
  if (auto value = module->getAttrOfType<mlir::IntegerAttr>(name))
    return value.getValue().getZExtValue();
  return fallback;
}

static ScratchMemory scratchMemory(ModuleOp module) {
  return {
      moduleInteger(module, "ttg.global_scratch_memory_size", 0),
      moduleInteger(module, "ttg.global_scratch_memory_alignment", 1),
      moduleInteger(module, "ttg.profile_scratch_memory_size", 0),
      moduleInteger(module, "ttg.profile_scratch_memory_alignment", 1),
  };
}

static uint64_t nodeElements(const sev_triton_compile_request &request,
                             uint32_t nodeId) {
  const auto &region = *request.region;
  for (size_t i = 0; i < request.specialization->shape_count; ++i) {
    const auto &shape = request.specialization->shapes[i];
    if (shape.node_id != nodeId)
      continue;
    uint64_t count = 1;
    for (size_t axis = 0; axis < shape.dimensions.len; ++axis) {
      if (shape.dimensions.data[axis] != 0 &&
          count > std::numeric_limits<uint64_t>::max() /
                      shape.dimensions.data[axis])
        return 1;
      count *= shape.dimensions.data[axis];
    }
    return count;
  }
  for (size_t i = 0; i < region.node_count; ++i) {
    const auto &node = region.nodes[i];
    if (node.id != nodeId || node.rank != SEV_TRITON_RANKED)
      continue;
    uint64_t count = 1;
    for (size_t axis = 0; axis < node.dimensions.len; ++axis) {
      if (node.dimensions.data[axis] < 0)
        return 1;
      const uint64_t extent = static_cast<uint64_t>(node.dimensions.data[axis]);
      if (extent != 0 && count > std::numeric_limits<uint64_t>::max() / extent)
        return 1;
      count *= extent;
    }
    return count;
  }
  return 1;
}

static uint64_t gridX(const sev_triton_compile_request &request) {
  if (request.region->outputs.len == 0)
    return 1;
  const uint64_t elements =
      nodeElements(request, request.region->outputs.data[0]);
  return std::max<uint64_t>(1, (elements + 255) / 256);
}

static void publish(sev_triton_compiled_kernel *output, OwnedKernel *owner,
                    sev_triton_kernel_format format,
                    const sev_triton_compile_options *options,
                    uint64_t grid, uint64_t shared,
                    ScratchMemory scratch = {}) {
  output->abi_version = SEV_TRITON_ABI_VERSION;
  output->format = format;
  output->entry_point = bytes(owner->entryPoint);
  output->code = bytes(owner->code);
  output->diagnostics = bytes(owner->diagnostics);
  output->launch = {};
  output->launch.grid_x = grid;
  output->launch.grid_y = 1;
  output->launch.grid_z = 1;
  output->launch.num_warps = options ? options->num_warps : 0;
  output->launch.warp_size = options ? options->warp_size : 0;
  output->launch.num_ctas = options ? options->num_ctas : 0;
  output->launch.shared_memory_bytes = shared;
  output->launch.global_scratch_bytes_per_program =
      scratch.globalBytesPerProgram;
  output->launch.global_scratch_alignment = scratch.globalAlignment;
  output->launch.profile_scratch_bytes_per_program =
      scratch.profileBytesPerProgram;
  output->launch.profile_scratch_alignment = scratch.profileAlignment;
  output->owner = owner;
}

static sev_triton_status fail(sev_triton_compiled_kernel *output,
                              sev_triton_status status, std::string message,
                              const sev_triton_compile_options *options = nullptr) {
  auto owner = std::make_unique<OwnedKernel>();
  owner->diagnostics = std::move(message);
  publish(output, owner.get(), SEV_TRITON_LLVM_IR, options, 0, 0);
  output->owner = owner.release();
  return status;
}

} // namespace

extern "C" sev_triton_status
sev_triton_compile(const sev_triton_compile_request *request,
                   sev_triton_compiled_kernel *output) {
  if (!output)
    return SEV_TRITON_INVALID_ARGUMENT;
  std::memset(output, 0, sizeof(*output));
  output->abi_version = SEV_TRITON_ABI_VERSION;
  if (!request)
    return fail(output, SEV_TRITON_INVALID_ARGUMENT, "request is null");
  if (request->abi_version != SEV_TRITON_ABI_VERSION)
    return fail(output, SEV_TRITON_INVALID_ARGUMENT,
                "Triton bridge ABI mismatch: expected 6");
  if (!request->region || !request->specialization || !request->options)
    return fail(output, SEV_TRITON_INVALID_ARGUMENT,
                "region, specialization, and options are required");
  const auto &options = *request->options;
  if (options.target != SEV_TRITON_AMD_GPU &&
      options.target != SEV_TRITON_NVIDIA_GPU)
    return fail(output, SEV_TRITON_UNSUPPORTED_TARGET,
                "only AMD and NVIDIA targets are supported", &options);
  if (!request->ttir.data || request->ttir.len == 0)
    return fail(output, SEV_TRITON_INVALID_ARGUMENT, "TTIR is empty", &options);
  if (!options.architecture.data || options.architecture.len == 0)
    return fail(output, SEV_TRITON_INVALID_ARGUMENT,
                "GPU architecture is empty", &options);

  initializeCompiler();
  mlir::DialectRegistry registry;
  registerDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  std::string diagnostics;
  mlir::ScopedDiagnosticHandler handler(&context, [&](mlir::Diagnostic &diag) {
    llvm::raw_string_ostream stream(diagnostics);
    diag.print(stream);
    stream << '\n';
    stream.flush();
    return mlir::success();
  });
  mlir::ParserConfig parserConfig(&context);
  mlir::OwningOpRef<ModuleOp> module =
      mlir::parseSourceString<ModuleOp>(
          llvm::StringRef(reinterpret_cast<const char *>(request->ttir.data),
                          request->ttir.len),
          parserConfig, "severian.ttir");
  if (!module)
    return fail(output, SEV_TRITON_PARSE_FAILURE, std::move(diagnostics),
                &options);

  if (!lowerTTIR(*module, options, diagnostics) ||
      !optimizeTTGIR(*module, options, diagnostics))
    return fail(output, SEV_TRITON_PASS_FAILURE, std::move(diagnostics),
                &options);

  int capability = 80;
  int ptxVersion = 84;
  if (options.target == SEV_TRITON_NVIDIA_GPU) {
    capability = parseNvidiaCapability(stringFrom(options.architecture));
    if (capability < 0)
      return fail(output, SEV_TRITON_INVALID_ARGUMENT,
                  "NVIDIA architecture must look like sm_80 or 80", &options);
  }
  if (!lowerToLLVMDialect(*module, options, capability, ptxVersion,
                          diagnostics))
    return fail(output, SEV_TRITON_PASS_FAILURE, std::move(diagnostics),
                &options);

  const uint64_t shared = sharedMemory(*module);
  const ScratchMemory scratch = scratchMemory(*module);
  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(*module, llvmContext);
  if (!llvmModule)
    return fail(output, SEV_TRITON_CODEGEN_FAILURE,
                diagnostics + "MLIR-to-LLVM translation failed\n", &options);

  auto owner = std::make_unique<OwnedKernel>();
  owner->entryPoint =
      "severian_region_" + std::to_string(request->region->region_id);
  sev_triton_kernel_format format;
  if (options.target == SEV_TRITON_NVIDIA_GPU) {
    if (options.emit != SEV_TRITON_PTX && options.emit != SEV_TRITON_LLVM_IR)
      return fail(output, SEV_TRITON_CODEGEN_FAILURE,
                  "native NVIDIA bridge emits PTX; CUBIN requires an optional "
                  "ptxas packaging step",
                  &options);
    auto ptx = emitLLVM(*llvmModule, "nvptx64-nvidia-cuda",
                        "sm_" + std::to_string(capability), "+ptx84",
                        llvm::CodeGenFileType::AssemblyFile, diagnostics);
    if (!ptx)
      return fail(output, SEV_TRITON_CODEGEN_FAILURE, std::move(diagnostics),
                  &options);
    owner->code.assign(ptx->begin(), ptx->end());
    format = SEV_TRITON_PTX;
  } else {
    const std::string arch = stringFrom(options.architecture);
    auto entryPoint = prepareAmdKernel(*llvmModule, options, diagnostics);
    if (!entryPoint)
      return fail(output, SEV_TRITON_CODEGEN_FAILURE,
                  std::move(diagnostics), &options);
    owner->entryPoint = std::move(*entryPoint);
    if (options.emit != SEV_TRITON_HSACO &&
        options.emit != SEV_TRITON_AMDGCN &&
        options.emit != SEV_TRITON_LLVM_IR)
      return fail(output, SEV_TRITON_CODEGEN_FAILURE,
                  "AMD bridge supports AMDGCN and HSACO output", &options);
    const bool wantObject = options.emit == SEV_TRITON_HSACO;
    auto generated = emitLLVM(
        *llvmModule, "amdgcn-amd-amdhsa", arch, "",
        wantObject ? llvm::CodeGenFileType::ObjectFile
                   : llvm::CodeGenFileType::AssemblyFile,
        diagnostics);
    if (!generated)
      return fail(output, SEV_TRITON_CODEGEN_FAILURE, std::move(diagnostics),
                  &options);
    if (wantObject) {
      auto hsaco = linkHsaco(*generated, diagnostics);
      if (!hsaco)
        return fail(output, SEV_TRITON_CODEGEN_FAILURE,
                    std::move(diagnostics), &options);
      owner->code = std::move(*hsaco);
      format = SEV_TRITON_HSACO;
    } else {
      owner->code.assign(generated->begin(), generated->end());
      format = SEV_TRITON_AMDGCN;
    }
  }
  owner->diagnostics = std::move(diagnostics);
  publish(output, owner.get(), format, &options, gridX(*request), shared,
          scratch);
  output->owner = owner.release();
  return SEV_TRITON_OK;
}

extern "C" void
sev_triton_destroy_kernel(sev_triton_compiled_kernel *kernel) {
  if (!kernel)
    return;
  delete static_cast<OwnedKernel *>(kernel->owner);
  std::memset(kernel, 0, sizeof(*kernel));
}
