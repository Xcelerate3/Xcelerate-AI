set(Xcelerate_PREDICTOR_CPU_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/predictor.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/predictor_utils.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/predictor_config.cc"
)
set(Xcelerate_PREDICTOR_CPU_TEST_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/predictor_test.cc")

# Common files that are always going to be included.
list(APPEND Xcelerate_CPU_SRCS ${Xcelerate_PREDICTOR_CPU_SRC})
list(APPEND Xcelerate_CPU_TEST_SRCS ${Xcelerate_PREDICTOR_CPU_TEST_SRC})

set(Xcelerate_CPU_SRCS ${Xcelerate_CPU_SRCS} PARENT_SCOPE)
set(Xcelerate_CPU_TEST_SRCS ${Xcelerate_CPU_TEST_SRCS} PARENT_SCOPE)
