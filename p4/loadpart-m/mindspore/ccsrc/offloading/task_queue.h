#ifndef MINDSPORE_OFFLOADING_TASK_QUEUE_H
#define MINDSPORE_OFFLOADING_TASK_QUEUE_H

#include <vector>
#include <unordered_map>
#include <memory>
#include <string>
#include <queue>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <set>
#include <thread>
#include "utils.h"

namespace mindspore {

namespace offloading {

using TaskCallBack = std::function<void()>;

struct Task {
  size_t stage_id;
  // inputs should be sorted according to the order of entry points
  std::vector<NameToQTensorMap> inputs;
  // len(session_ids) == len(inputs)
  std::vector<uint64_t> session_ids;
  // sum(plan->bsz_list) == len(inputs)
  std::shared_ptr<ExecutionPlan> plan;
  // exec time and queueing time
  std::array<int64_t, 3> q_times;
  std::array<int64_t, 3> e_times;
  std::vector<int64_t> req_q_times;
  double factor;
};
using TaskPtr = std::unique_ptr<Task>;

class TaskQueue {
  public:
    TaskQueue() = default;
    ~TaskQueue() = default;
    bool IsEmpty();
    void PushTask(TaskPtr&& task);
    TaskPtr PopTask();
  private:
    size_t stage_id_;
    std::deque<TaskPtr> queue_;
    TaskCallBack task_callback_;
    std::mutex mu_;  // Lock only when the queue changes to avoid deadlock caused by lock in complex scenarios.
    std::condition_variable cv_;
};

}
}

#endif