#ifndef Xcelerate_CORE_ACTIVITY_WATCHER_ACTIVITY_UTILS_H_
#define Xcelerate_CORE_ACTIVITY_WATCHER_ACTIVITY_UTILS_H_

#include <memory>

#include "Xcelerate/core/activity_watcher/activity.h"

namespace Xcelerate {

class OpKernelContext;

namespace activity_watcher {

// A convenient way to create an activity. Writes OpKernelContext information
// and given attributes to a new activity and returns.
std::unique_ptr<Activity> ActivityFromContext(
    OpKernelContext* context, xce::string name, ActivityCategory category,
    Activity::Attributes additional_attributes = Activity::Attributes());

}  // namespace activity_watcher
}  // namespace Xcelerate

#endif  // Xcelerate_CORE_ACTIVITY_WATCHER_ACTIVITY_UTILS_H_
