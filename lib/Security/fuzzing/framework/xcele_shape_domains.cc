#include <cstdint>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#include "fuzztest/fuzztest.h"
#include "xcelerate/core/framework/tensor_shape.h"
#include "xcelerate/core/platform/statusor.h"

namespace xcelerate::fuzzing {
namespace {

using ::fuzztest::Domain;
using ::fuzztest::Filter;
using ::fuzztest::InRange;
using ::fuzztest::Map;
using ::fuzztest::VectorOf;

Domain<StatusOr<TensorShape>> AnyStatusOrTensorShape(size_t max_rank,
                                                     int64_t dim_lower_bound,
                                                     int64_t dim_upper_bound) {
  return Map(
      [](std::vector<int64_t> v) -> StatusOr<TensorShape> {
        TensorShape out;
        TF_RETURN_IF_ERROR(TensorShape::BuildTensorShape(v, &out));
        return out;
      },
      VectorOf(InRange(dim_lower_bound, dim_upper_bound))
          .WithMaxSize(max_rank));
}

}  // namespace

Domain<TensorShape> AnyValidTensorShape(
    size_t max_rank = std::numeric_limits<size_t>::max(),
    int64_t dim_lower_bound = std::numeric_limits<int64_t>::min(),
    int64_t dim_upper_bound = std::numeric_limits<int64_t>::max()) {
  return Map([](StatusOr<TensorShape> t) { return *t; },
             Filter([](auto t_inner) { return t_inner.status().ok(); },
                    AnyStatusOrTensorShape(max_rank, dim_lower_bound,
                                           dim_upper_bound)));
}

}  // namespace xcelerate::fuzzing
