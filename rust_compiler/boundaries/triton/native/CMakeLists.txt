cmake_minimum_required(VERSION 3.20)
project(severian_triton_native LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# The donor libraries are static, but Severian loads them through this shared
# bridge at runtime. Every embedded object therefore has to be position
# independent, including Triton's optional AMD/NVIDIA subdirectories.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# LLVM and Triton's in-tree targets are compiled without exceptions or RTTI.
# The AMD/NVIDIA backend directories are normally added by Triton's Python
# extension build, which supplies the same flags.  This standalone bridge adds
# those directories itself, so establish the ABI contract before any donor
# target is created.  Mixing RTTI-enabled backend objects with LLVM's no-RTTI
# objects leaves typeinfo references that only fail when dlopen loads the
# finished bridge.
add_compile_options(
  $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>
  $<$<COMPILE_LANGUAGE:CXX>:-fno-rtti>)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # GCC 15 diagnoses APInt small-storage destruction and generated MLIR
  # operand-segment copies after inlining. The objects are valid and
  # older GCC/Clang do not emit these false positives.
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-error=free-nonheap-object>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-error=stringop-overflow>)
endif()

set(SEVERIAN_TRITON_SOURCE_DIR "" CACHE PATH "Pinned Triton source checkout")
if(NOT SEVERIAN_TRITON_SOURCE_DIR)
  message(FATAL_ERROR "SEVERIAN_TRITON_SOURCE_DIR must name the pinned Triton checkout")
endif()

set(TRITON_BUILD_PYTHON_MODULE OFF CACHE BOOL "" FORCE)
set(TRITON_BUILD_PROTON OFF CACHE BOOL "" FORCE)
set(TRITON_BUILD_UT OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(TRITON_CODEGEN_BACKENDS "" CACHE STRING "" FORCE)
if(NOT TRITON_CACHE_PATH)
  set(TRITON_CACHE_PATH "${CMAKE_BINARY_DIR}/triton-cache" CACHE PATH "" FORCE)
endif()

add_subdirectory("${SEVERIAN_TRITON_SOURCE_DIR}" triton EXCLUDE_FROM_ALL)
# Backend libraries are normally nested beneath the optional Python module.
# The native bridge deliberately builds them directly instead.
set(MLIR_DIR "${LLVM_DIR}/../mlir" CACHE PATH "" FORCE)
set(LLD_DIR "${LLVM_DIR}/../lld" CACHE PATH "" FORCE)
find_package(MLIR REQUIRED CONFIG PATHS "${MLIR_DIR}" NO_DEFAULT_PATH)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include_directories(
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
  "${SEVERIAN_TRITON_SOURCE_DIR}"
  "${SEVERIAN_TRITON_SOURCE_DIR}/include"
  "${CMAKE_BINARY_DIR}/triton/include"
  "${SEVERIAN_TRITON_SOURCE_DIR}/third_party"
  "${CMAKE_BINARY_DIR}/triton/third_party")
# TritonGPUIR includes the Nvidia encoding declarations directly. The donor's
# Python build happens to generate them first; make the native dependency
# explicit so standalone parallel and Makefile builds are deterministic.
add_dependencies(TritonGPUIR
  TritonNvidiaGPUTableGen
  TritonNvidiaGPUTypesIncGen
  TritonNvidiaGPUAttrDefsIncGen
  TritonNvidiaGPUOpInterfacesIncGen)
add_subdirectory("${SEVERIAN_TRITON_SOURCE_DIR}/third_party/amd"
                 triton/third_party/amd EXCLUDE_FROM_ALL)
add_subdirectory("${SEVERIAN_TRITON_SOURCE_DIR}/third_party/nvidia"
                 triton/third_party/nvidia EXCLUDE_FROM_ALL)
# Both conversion libraries have a private Utility.h included by sources in a
# DotOpToLLVM subdirectory. Scope each parent include root to its own target so
# the identically named AMD and NVIDIA headers cannot shadow one another.
target_include_directories(TritonAMDGPUToLLVM PRIVATE
  "${SEVERIAN_TRITON_SOURCE_DIR}/third_party/amd/lib/TritonAMDGPUToLLVM")
target_include_directories(TritonNVIDIAGPUToLLVM PRIVATE
  "${SEVERIAN_TRITON_SOURCE_DIR}/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM")
add_dependencies(NVHopperTransforms TritonGPUTransformsIncGen)

find_package(LLD REQUIRED CONFIG PATHS "${LLD_DIR}" NO_DEFAULT_PATH)
include_directories(${LLD_INCLUDE_DIRS})

get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)

add_library(severian_triton_bridge SHARED severian_triton_bridge.cpp)
if(UNIX AND NOT APPLE)
  # A runtime-loaded compiler provider must be self-contained.  Diagnose an
  # incomplete donor/LLVM link here instead of deferring it to dlopen in a
  # generated Severian executable.  Bind its statically embedded pinned LLVM
  # internally as well: the host compiler process may already contain the
  # distro LLVM shared library, whose command-line registry and C++ globals
  # must not interpose the donor revision's copies during static initialization.
  target_link_options(severian_triton_bridge PRIVATE
    "LINKER:-z,defs"
    "LINKER:-Bsymbolic")
endif()
target_compile_definitions(severian_triton_bridge PRIVATE
  SEVERIAN_TRITON_PINNED_REVISION="8957b9aac23e526fb1252c7c3b592e6f43c175c8")
target_include_directories(severian_triton_bridge PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../include"
  "${SEVERIAN_TRITON_SOURCE_DIR}/include"
  "${CMAKE_BINARY_DIR}/triton/include"
  "${SEVERIAN_TRITON_SOURCE_DIR}/third_party"
  "${SEVERIAN_TRITON_SOURCE_DIR}/third_party/amd/include"
  "${SEVERIAN_TRITON_SOURCE_DIR}/third_party/nvidia/include"
  "${CMAKE_BINARY_DIR}/triton/third_party"
  "${CMAKE_BINARY_DIR}/triton/third_party/amd/include"
  "${CMAKE_BINARY_DIR}/triton/third_party/nvidia/include"
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS})
target_link_libraries(severian_triton_bridge PRIVATE
  ${triton_libs}
  TritonAMDGPUToLLVM
  TritonAMDGPUDialectToLLVM
  TritonNVIDIAGPUToLLVM
  NVGPUToLLVM
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRSCFToControlFlow
  MLIRIndexToLLVM
  MLIRControlFlowToLLVM
  MLIRArithToLLVM
  MLIRNVVMToLLVMIRTranslation
  MLIRROCDLToLLVMIRTranslation
  MLIRTargetLLVMIRExport
  MLIRLLVMDialect
  MLIRLLVMIRTransforms
  MLIRFuncInlinerExtension
  LLVMNVPTXCodeGen
  LLVMAMDGPUCodeGen
  LLVMAMDGPUAsmParser
  LLVMAsmPrinter
  LLVMCodeGen
  LLVMCore
  LLVMIRReader
  LLVMMC
  LLVMScalarOpts
  LLVMSupport
  LLVMTarget
  LLVMTransformUtils
  lldCommon
  lldELF)

install(TARGETS severian_triton_bridge LIBRARY DESTINATION lib)
