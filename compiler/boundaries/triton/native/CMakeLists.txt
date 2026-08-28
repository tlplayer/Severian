cmake_minimum_required(VERSION 3.20)
project(severian_triton_native LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # GCC 15 diagnoses an APInt small-storage destructor path after inlining;
  # the object is valid and older GCC/Clang do not emit this false positive.
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wno-error=free-nonheap-object>)
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
add_dependencies(NVHopperTransforms TritonGPUTransformsIncGen)

find_package(LLD REQUIRED CONFIG PATHS "${LLD_DIR}" NO_DEFAULT_PATH)
include_directories(${LLD_INCLUDE_DIRS})

get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)

add_library(severian_triton_bridge SHARED severian_triton_bridge.cpp)
target_compile_definitions(severian_triton_bridge PRIVATE
  SEVERIAN_TRITON_PINNED_REVISION="8957b9aac23e526fb1252c7c3b592e6f43c175c8")
target_include_directories(severian_triton_bridge PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../include"
  "${SEVERIAN_TRITON_SOURCE_DIR}/include"
  "${CMAKE_BINARY_DIR}/triton/include"
  "${SEVERIAN_TRITON_SOURCE_DIR}/third_party"
  "${CMAKE_BINARY_DIR}/triton/third_party"
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS})
target_link_libraries(severian_triton_bridge PRIVATE
  ${triton_libs}
  TritonAMDGPUToLLVM
  TritonAMDGPUTransforms
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
