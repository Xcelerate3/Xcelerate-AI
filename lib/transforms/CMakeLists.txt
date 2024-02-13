# ---[ Get non-tests
file(GLOB tmp *.cc)
set(Xcelerate_CPU_SRCS ${Xcelerate_CPU_SRCS} ${tmp})

# exclude test files
file(GLOB tmp *_test.cc)
exclude(Xcelerate_CPU_SRCS "${Xcelerate_CPU_SRCS}" ${tmp})

# ---[ Get tests
file(GLOB tmp *_test.cc)
set(Xcelerate_CPU_TEST_SRCS ${Xcelerate_CPU_TEST_SRCS} ${tmp})

# ---[ Send the lists to the parent scope.
set(Xcelerate_CPU_SRCS ${Xcelerate_CPU_SRCS} PARENT_SCOPE)
set(Xcelerate_CPU_TEST_SRCS ${Xcelerate_CPU_TEST_SRCS} PARENT_SCOPE)
