#ifndef Xcelerate_CONTRIB_OBSERVERS_TIME_OBSERVER_H_
#define Xcelerate_CONTRIB_OBSERVERS_TIME_OBSERVER_H_

#include <unordered_map>

#include "Xcelerate/core/common.h"
#include "Xcelerate/core/net.h"
#include "Xcelerate/core/observer.h"
#include "Xcelerate/core/operator.h"
#include "Xcelerate/core/timer.h"
#include "Xcelerate/observers/operator_attaching_net_observer.h"

namespace Xcelerate {

class TimeObserver;

class TORCH_API TimeCounter {
 public:
  explicit TimeCounter() {}
  inline float average_time() const {
    return total_time_ / iterations_;
  }

 protected:
  Timer timer_;
  float start_time_ = 0.0f;
  float total_time_ = 0.0f;
  int iterations_ = 0;
};

class TORCH_API TimeOperatorObserver final : public TimeCounter,
                                             public ObserverBase<OperatorBase> {
 public:
  explicit TimeOperatorObserver(OperatorBase* subject) = delete;
  explicit TimeOperatorObserver(
      OperatorBase* subject,
      TimeObserver* /* unused */)
      : ObserverBase<OperatorBase>(subject) {}
  std::unique_ptr<ObserverBase<OperatorBase>> rnnCopy(
      OperatorBase* subject,
      int rnn_order) const override;

 private:
  void Start() override;
  void Stop() override;
};

class TORCH_API TimeObserver final
    : public TimeCounter,
      public OperatorAttachingNetObserver<TimeOperatorObserver, TimeObserver> {
 public:
  explicit TimeObserver(NetBase* subject)
      : OperatorAttachingNetObserver<TimeOperatorObserver, TimeObserver>(
            subject,
            this) {}

  float average_time_children() const {
    float sum = 0.0f;
    for (const auto* observer : operator_observers_) {
      sum += observer->average_time();
    }
    return sum / subject_->GetOperators().size();
  }

 private:
  void Start() override;
  void Stop() override;
};

} // namespace Xcelerate

#endif // Xcelerate_CONTRIB_OBSERVERS_TIME_OBSERVER_H_
