#pragma once

#include <c10/macros/Export.h>
#include <c10/util/flat_hash_map.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace c10 {

/*
 * Given a sequence of allocations in a thread, AllocationPlan records
 * 1. size of each allocation
 * 2. Lifetime of each allocation.
 * 3. allocation offsets: Memory offset for each allocation in a single blob of
 * memory
 * 4. Total size of a blob of memory required to satisfy all the allocations.
 */
class C10_API AllocationPlan {
 private:
  // Records size of each allocation by their sequential allocation ids.
  std::vector<uint64_t> allocation_sizes;

  std::vector<uint64_t> allocation_lifetimes;
  // Maps an allocation to some offset in a blob of memory.
  std::vector<uint64_t> allocation_offsets;
  uint64_t total_size{0};
  void clear();
  friend class AllocationPlanner;
  friend class CPUProfilingAllocator;
};

class C10_API AllocationPlanner {
 private:
  AllocationPlan* allocation_plan_{nullptr};
  // Maps allocated ptr to its allocation id.
  // This is used when freeing the memory to look up the allocation id
  // in order to establish the lifetime of a particular allocation.
  ska::flat_hash_map<const void*, uint64_t> allocation_ptr_to_id_;
  uint64_t allocation_id_{0};
  bool validation_mode_{false};

  bool validate_allocation(const uint64_t size, const void* ptr);
  bool validate_free(const void* ptr);

 public:
  bool validation_success{true};

  AllocationPlanner() = delete;
  AllocationPlanner(AllocationPlan* plan, bool validate = false)
      : allocation_plan_(plan), validation_mode_(validate) {}
  void record_allocation(const uint64_t size, const void* ptr);
  void record_free(const void* ptr);
  void formulate_plan();
  void clear();
};

// NOT THREAD SAFE profiling allocator.
class C10_API CPUProfilingAllocator {
 private:
  const AllocationPlan* plan_{nullptr};
  uint64_t allocation_id_{0};
  uint64_t current_size_{0};
  void* blob_{nullptr};
  ska::flat_hash_map<const void*, uint64_t> allocation_ptr_to_id_;

 public:
  ~CPUProfilingAllocator();
  void set_plan(const AllocationPlan* plan);
  void unset_plan();
  void* allocate(const size_t bytes);
  void free(void* const ptr);
};

/*
 * Usage: Profile allocations made by one run of the model.
 * AllocationPlan plan;
 * {
 *   WithProfileAllocationGuard profile_guard(&plan);
 *   module.forward(...);
 * }
 * plan now contains allocation plan.
 */
class C10_API WithProfileAllocationsGuard {
 public:
  WithProfileAllocationsGuard(AllocationPlan* plan);
  ~WithProfileAllocationsGuard();

 private:
  std::unique_ptr<AllocationPlanner> planner_;
};

/*
 * Usage: Validate allocation plan made with WithProfileAllocationGuard
 * bool plan_validation_success, success = true;
 * for (some number of representative inputs)
 * {
 *   WithValidateAllocationPlanGuard(&plan, &plan_validation_success);
 *   module.forward(...);
 *   success = success && plan_validation_success;
 * }
 * success == true means allocations are according to plan
 * else for some inputs allocation pattern changed.
 */
class C10_API WithValidateAllocationPlanGuard {
 public:
  WithValidateAllocationPlanGuard(AllocationPlan* plan, bool* success);
  ~WithValidateAllocationPlanGuard();

 private:
  std::unique_ptr<AllocationPlanner> planner_;
  bool* success_;
};

AllocationPlanner* GetThreadLocalAllocationPlanner();

/*
 * Usage: Allocate tensors accordingly to allocation plan
 * First make allocation plan.
 *  See WithProfileAllocationsGuard usage.
 * Second validate allocation plan.
 *  See WithValidateAllocationPlanGuard usage.
 * CPUProfilingAllocator profiling_allocator;
 * {
 *   WithProfilingAllocatorGuard allocator_guard(&profiling_allocator, &plan);
 *   module.forward(...);
 * }
 */
class C10_API WithProfilingAllocatorGuard {
 public:
  WithProfilingAllocatorGuard(
      CPUProfilingAllocator* allocator,
      const AllocationPlan* plan);
  ~WithProfilingAllocatorGuard();
};

CPUProfilingAllocator* GetThreadLocalProfilingAllocator();

} // namespace c10