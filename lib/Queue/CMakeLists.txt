# ---[ GPU files
# ------[ general GPU
file(GLOB tmp *_gpu.cc)
set(Xcelerate_GPU_SRCS ${Xcelerate_GPU_SRCS} ${tmp})
# ------[ CUDA sources
file(GLOB tmp *.cu)
set(Xcelerate_GPU_SRCS ${Xcelerate_GPU_SRCS} ${tmp})
# exclude test files
file(GLOB tmp *_test.cc)
exclude(Xcelerate_GPU_SRCS "${Xcelerate_GPU_SRCS}" ${tmp})

# ------[ general HIP
file(GLOB tmp hip/*.cc)
set(Xcelerate_HIP_SRCS ${Xcelerate_HIP_SRCS} ${tmp})
# ---[ HIP files.
file(GLOB tmp hip/*.hip)
set(Xcelerate_HIP_SRCS ${Xcelerate_HIP_SRCS} ${tmp})
# exclude test files
file(GLOB tmp hip/*_test.cc)
exclude(Xcelerate_HIP_SRCS "${Xcelerate_HIP_SRCS}" ${tmp})

# ---[ CPU files.
file(GLOB tmp *.cc)
set(Xcelerate_CPU_SRCS ${Xcelerate_CPU_SRCS} ${tmp})
# exclude test, gpu and hip files
file(GLOB tmp *_test.cc)
exclude(Xcelerate_CPU_SRCS "${Xcelerate_CPU_SRCS}" ${tmp})
exclude(Xcelerate_CPU_SRCS "${Xcelerate_CPU_SRCS}" ${Xcelerate_GPU_SRCS} ${Xcelerate_HIP_SRCS})

# ---[ GPU test files
file(GLOB tmp *_gpu_test.cc)
set(Xcelerate_GPU_TEST_SRCS ${Xcelerate_GPU_TEST_SRCS} ${tmp})

# ---[ HI test files
file(GLOB tmp hip/*_test.cc)
set(Xcelerate_HIP_TEST_SRCS ${Xcelerate_HIP_TEST_SRCS} ${tmp})

# ---[ CPU test files
file(GLOB tmp *_test.cc)
set(Xcelerate_CPU_TEST_SRCS ${Xcelerate_CPU_TEST_SRCS} ${tmp})
exclude(Xcelerate_CPU_TEST_SRCS "${Xcelerate_CPU_TEST_SRCS}" ${Xcelerate_GPU_TEST_SRCS} ${Xcelerate_HIP_TEST_SRCS})

# ---[ Send the lists to the parent scope.
set(Xcelerate_CPU_SRCS ${Xcelerate_CPU_SRCS} PARENT_SCOPE)
set(Xcelerate_GPU_SRCS ${Xcelerate_GPU_SRCS} PARENT_SCOPE)
set(Xcelerate_HIP_SRCS ${Xcelerate_HIP_SRCS} PARENT_SCOPE)
set(Xcelerate_CPU_TEST_SRCS ${Xcelerate_CPU_TEST_SRCS} PARENT_SCOPE)
set(Xcelerate_GPU_TEST_SRCS ${Xcelerate_GPU_TEST_SRCS} PARENT_SCOPE)
set(Xcelerate_HIP_TEST_SRCS ${Xcelerate_HIP_TEST_SRCS} PARENT_SCOPE)
