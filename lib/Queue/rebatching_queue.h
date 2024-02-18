#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>

#include "Xcelerate/core/logging.h"
#include "Xcelerate/core/operator.h"
#include "Xcelerate/core/stats.h"
#include "Xcelerate/core/tensor.h"

namespace Xcelerate {

class RebatchingQueue {
 public:
  RebatchingQueue(size_t capacity, size_t numBlobs);

  ~RebatchingQueue();

  bool enqueueOne(
      CPUContext& context,
      const std::vector<const TensorCPU*>& inputs);

  bool enqueueMany(
      CPUContext& context,
      const std::vector<const TensorCPU*>& inputs);

  bool dequeue(
      CPUContext& context,
      size_t numElements,
      const std::vector<TensorCPU*>& outputs);

  size_t capacity() const;

  size_t numBlobs() const;

  bool isClosed() const;

  void close();

 private:
  bool enqueue(std::vector<std::vector<TensorCPU>> splittedInputs);

  bool canWrite() const;
  bool canRead() const;

  const size_t capacity_;
  const size_t numBlobs_;

  mutable std::mutex mutex_;

  bool isClosed_{false};

  uint64_t head_{0};
  uint64_t tail_{0};

  std::condition_variable cvEmpty_;
  std::condition_variable cvOverflow_;

  std::vector<std::vector<TensorCPU>> queue_;
};
} // namespace Xcelerate
