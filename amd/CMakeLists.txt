include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
add_compile_definitions(USE_ROCM=1)
add_subdirectory(include)
add_subdirectory(lib)
add_triton_plugin(TritonAMD ${CMAKE_CURRENT_SOURCE_DIR}/python/triton_amd.cc LINK_LIBS TritonAMDGPUToLLVM TritonAMDGPUTransforms)