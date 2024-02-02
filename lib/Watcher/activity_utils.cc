#include "Xcelerate/core/activity_watcher/activity_utils.h"

#include <memory>
#include <utility>

#include "absl/strings/str_join.h"
#include "Xcelerate/core/framework/op_kernel.h"

namespace Xcelerate {
namespace activity_watcher {

std::unique_ptr<Activity> ActivityFromContext(
    OpKernelContext* context, tsl::string name, ActivityCategory category,
    Activity::Attributes additional_attributes) {
  Activity::Attributes attributes(std::move(additional_attributes));
  if (context) {
    attributes.merge(Activity::Attributes({
        {"node_name", context->op_kernel().def().name()},
        {"step_id", absl::StrCat(context->step_id())},
        {"device", context->device()->name()},
        {"op", context->op_kernel().def().op()},
        {"iter_num", absl::StrCat(context->frame_iter().iter_id)},
        {"inputs", absl::StrJoin(context->op_kernel().def().input(), "; ")},
        {"original_node_names ", absl::StrJoin(context->op_kernel()
                                                   .def()
                                                   .experimental_debug_info()
                                                   .original_node_names(),
                                               "; ")},
        {"original_func_names", absl::StrJoin(context->op_kernel()
                                                  .def()
                                                  .experimental_debug_info()
                                                  .original_func_names(),
                                              "; ")},
    }));
  }

  return std::make_unique<Activity>(name, category, std::move(attributes));
}

}  // namespace activity_watcher
}  // namespace Xcelerate
