#pragma once

#include <unordered_map>

#include "Xcelerate/core/common.h"
#include "Xcelerate/core/event.h"
#include "Xcelerate/core/net.h"
#include "Xcelerate/core/observer.h"
#include "Xcelerate/core/operator.h"
#include "Xcelerate/core/timer.h"
#include "Xcelerate/observers/operator_attaching_net_observer.h"

namespace Xcelerate {

/**
 * This observer displays a description of each operator executed in a network.
 * This includes input and tensors (name, size, type), arguments, and execution
 * time. This can be used to analyze different performance characteristics.
 * NOTE: Currently this observer only supports synchronized computation
 **/

class ProfileObserver;
class ProfileCounter {
 public:
  explicit ProfileCounter() {}

 protected:
  Timer timer_;
  float start_time_ = 0.0f;
  float run_time_ = 0.0f;
};

class TORCH_API ProfileOperatorObserver final
    : public ProfileCounter,
      public ObserverBase<OperatorBase> {
 public:
  explicit ProfileOperatorObserver(OperatorBase* subject) = delete;
  explicit ProfileOperatorObserver(
      OperatorBase* subject,
      ProfileObserver* netObserver)
      : ObserverBase<OperatorBase>(subject), netObserver_(netObserver) {
    if (subject) {
      net_position_ = subject->net_position();
    }
  }
  explicit ProfileOperatorObserver(
      OperatorBase* subject,
      ProfileObserver* netObserver,
      int net_position,
      int rnn_order)
      : ProfileOperatorObserver(subject, netObserver) {
    net_position_ = net_position;
    rnn_order_ = rnn_order;
  }

  std::unique_ptr<ObserverBase<OperatorBase>> rnnCopy(
      OperatorBase* subject,
      int rnn_order) const override;

  void Dump() const;

  virtual std::string getId() const {
    std::stringstream ss;
    ss << net_position_;
    if (rnn_order_ != OperatorBase::kNoNetPositionSet) {
      ss << "-" << rnn_order_;
    }
    return ss.str();
  }

 protected:
  ProfileObserver* netObserver_;
  int net_position_; // Needed because this is not visible in RNN Executor
  int rnn_order_ = OperatorBase::kNoNetPositionSet;

 private:
  void Start() override;
  void Stop() override;
};

class TORCH_API ProfileObserver final : public OperatorAttachingNetObserver<
                                            ProfileOperatorObserver,
                                            ProfileObserver> {
 public:
  explicit ProfileObserver(NetBase* subject)
      : OperatorAttachingNetObserver<ProfileOperatorObserver, ProfileObserver>(
            subject,
            this) {}

  void Start() override{};
  void Stop() override{};

 private:
  vector<const ProfileOperatorObserver*> operator_observers_;
};

} // namespace Xcelerate
