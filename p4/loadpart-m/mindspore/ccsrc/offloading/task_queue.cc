#include "task_queue.h"

namespace mindspore {

namespace offloading {

bool TaskQueue::IsEmpty() {
  std::lock_guard<std::mutex> lock{mu_};
  return queue_.empty();
}

void TaskQueue::PushTask(TaskPtr&& task) {
  {
    std::lock_guard<std::mutex> lock{mu_};
    queue_.push_back(std::move(task));
  }
  cv_.notify_all();
}

TaskPtr TaskQueue::PopTask() {
  std::unique_lock<std::mutex> lock{mu_};
  if (queue_.empty()) {
    cv_.wait(lock, [this]() { return !queue_.empty(); });
  }

  auto ret = std::move(queue_.front());
  queue_.pop_front();
  return ret;
}

}
}