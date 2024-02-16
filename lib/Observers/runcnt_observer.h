#pragma once

#include "Xcelerate/core/net.h"
#include "Xcelerate/core/observer.h"
#include "Xcelerate/core/operator.h"
#include "Xcelerate/observers/operator_attaching_net_observer.h"

namespace Xcelerate {

class RunCountNetObserver;

class TORCH_API RunCountOperatorObserver final
    : public ObserverBase<OperatorBase> {
 public:
  explicit RunCountOperatorObserver(OperatorBase* op) = delete;
  RunCountOperatorObserver(OperatorBase* op, RunCountNetObserver* netObserver);
  ~RunCountOperatorObserver() {}
  std::unique_ptr<ObserverBase<OperatorBase>> rnnCopy(
      OperatorBase* subject,
      int rnn_order) const override;

 private:
  void Start() override;
  void Stop() override;

 private:
  RunCountNetObserver* netObserver_;
};

class TORCH_API RunCountNetObserver final : public OperatorAttachingNetObserver<
                                                RunCountOperatorObserver,
                                                RunCountNetObserver> {
 public:
  explicit RunCountNetObserver(NetBase* subject_)
      : OperatorAttachingNetObserver<
            RunCountOperatorObserver,
            RunCountNetObserver>(subject_, this),
        cnt_(0) {}
  ~RunCountNetObserver() {}

  std::string debugInfo() override;

  friend class RunCountOperatorObserver;

 private:
  void Start() override;
  void Stop() override;

 protected:
  std::atomic<int> cnt_;
};

} // namespace Xcelerate
