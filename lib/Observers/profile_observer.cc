#include "profile_observer.h"
#include "Xcelerate/core/logging.h"

namespace Xcelerate {

void ProfileOperatorObserver::Dump() const {
  static std::mutex loggingMutex;
  std::lock_guard<std::mutex> lock(loggingMutex);

  LOG(INFO) << "--------- Starting operator " << subject_->debug_def().type()
            << " op#" << getId() << " ---------";
  for (int i = 0; i < subject_->InputSize(); ++i) {
    if (subject_->InputIsTensorType(i, CPU)) {
      const auto& tensor = subject_->Input<Tensor>(i, CPU);
      const auto& name = subject_->debug_def().input(i);
      TensorPrinter printer(name);
      LOG(INFO) << "Input " << i << ": " << printer.MetaStr(tensor);
    } else if (subject_->InputIsTensorType(i, CUDA)) {
      const auto& tensor = subject_->Input<Tensor>(i, CUDA);
      const auto& name = subject_->debug_def().input(i);
      TensorPrinter printer(name);
      LOG(INFO) << "Input " << i << ": " << printer.MetaStr(tensor);
    }
  }

  int a = 0;
  for (const auto& arg : subject_->debug_def().arg()) {
    LOG(INFO) << "Argument " << a << ": " << arg.ShortDebugString();
    ++a;
  }

  for (int o = 0; o < subject_->OutputSize(); ++o) {
    if (subject_->OutputIsTensorType(o, CPU)) {
      auto* tensor = subject_->Output<Tensor>(o, CPU);
      const auto& name = subject_->debug_def().output(o);
      TensorPrinter printer(name);
      LOG(INFO) << "Output " << o << ": " << printer.MetaStr(*tensor);
    } else if (subject_->OutputIsTensorType(o, CUDA)) {
      auto* tensor = subject_->Output<Tensor>(o, CUDA);
      const auto& name = subject_->debug_def().output(o);
      TensorPrinter printer(name);
      LOG(INFO) << "Output " << o << ": " << printer.MetaStr(*tensor);
    }
  }

  LOG(INFO) << "--------- Finished operator " << subject_->debug_def().type()
            << " in " << run_time_ << " ms ---------";
}

void ProfileOperatorObserver::Start() {
  start_time_ = timer_.MilliSeconds();
}

void ProfileOperatorObserver::Stop() {
  run_time_ = timer_.MilliSeconds() - start_time_;
  Dump();
}

std::unique_ptr<ObserverBase<OperatorBase>> ProfileOperatorObserver::rnnCopy(
    OperatorBase* subject,
    int rnn_order) const {
  return std::unique_ptr<ObserverBase<OperatorBase>>(
      new ProfileOperatorObserver(
          subject, netObserver_, net_position_, rnn_order));
}
} // namespace Xcelerate
