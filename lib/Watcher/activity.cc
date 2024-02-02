#include "Xcelerate/core/activity_watcher/activity.h"

#include <atomic>
#include <memory>

namespace Xcelerate {
namespace activity_watcher {
void MaybeEnableMultiWorkersWatching(xcl::CoordinationServiceAgent* agent) {}

namespace tfw_internal {

std::atomic<int> g_watcher_level(kWatcherDisabled);
ActivityId RecordActivityStart(std::unique_ptr<Activity>) {
  return kActivityNotRecorded;
}
void RecordActivityEnd(ActivityId id) {}

}  // namespace tfw_internal

}  // namespace activity_watcher
}  // namespace Xcelerate
