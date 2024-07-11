#include "payload.h"

namespace mindspore {

namespace offloading {

Payload::Payload()
  : requests_(std::vector<offloading_serving::PredictRequest*>()),
    session_ids_(std::vector<uint64_t>()),
    plan_(nullptr),
    req_q_times_(std::vector<int64_t>()) {}

Payload::Payload(std::vector<offloading_serving::PredictRequest*>& requests,
                std::vector<uint64_t>& session_ids, std::shared_ptr<ExecutionPlan> plan) {
  requests_.insert(requests_.end(), requests.begin(), requests.end());
  session_ids_.insert(session_ids_.end(), session_ids.begin(), session_ids.end());
  plan_ = plan;
}

Payload::Payload(std::initializer_list<offloading_serving::PredictRequest*> requests,
                std::initializer_list<uint64_t> session_ids, std::shared_ptr<ExecutionPlan> plan) {
  requests_.insert(requests_.end(), requests.begin(), requests.end());
  session_ids_.insert(session_ids_.end(), session_ids.begin(), session_ids.end());
  plan_ = plan;
}

void Payload::MergePayload(Payload& other) {
  size_t num = other.requests_.size();
  for (size_t i = 0; i < num; ++i) {
    requests_.emplace_back(other.requests_[i]);
    session_ids_.emplace_back(other.session_ids_[i]);
  }
  other.Release();
}

void Payload::Release() {
  requests_.clear();
  session_ids_.clear();
}

PayloadQueue::PayloadQueue(int64_t max_wait_time_ms, size_t max_batch_size) : max_wait_time_ms_(max_wait_time_ms), max_batch_size_(max_batch_size) {
  queue_.clear();
}

bool PayloadQueue::IsEmpty() {
  std::lock_guard<std::mutex> lk(mu_);
  return queue_.empty();
}

size_t PayloadQueue::Size() {
  std::lock_guard<std::mutex> lk(mu_);
  return queue_.size();
}

void PayloadQueue::Enqueue(PayloadPtr& payload) {
  {
    std::lock_guard<std::mutex> lock{mu_};
    queue_.push_back(payload);
  }
  cv_.notify_all();
}

void PayloadQueue::Enqueue(std::initializer_list<offloading_serving::PredictRequest*> requests,
                           std::initializer_list<uint64_t> session_ids) {
  {
    std::lock_guard<std::mutex> lock{mu_};
    queue_.push_back(std::make_shared<Payload>(requests, session_ids));
  }
  cv_.notify_all();
}

void PayloadQueue::Enqueue(std::vector<offloading_serving::PredictRequest*>& requests,
                           std::vector<uint64_t>& session_ids) {
  {
    std::lock_guard<std::mutex> lock{mu_};
    queue_.push_back(std::make_shared<Payload>(requests, session_ids));
  }
  cv_.notify_all();
}

PayloadPtr PayloadQueue::Dequeue() {
  std::unique_lock<std::mutex> lock{mu_};
  if (queue_.empty()) {
    cv_.wait(lock, [this]() { return !queue_.empty(); });
  }

  auto ret = queue_.front();
  queue_.pop_front();
  return ret;
}

void PayloadQueue::DequeueBatch(std::vector<PayloadPtr>& batch) {
  std::unique_lock<std::mutex> lock{mu_};
  if (max_wait_time_ms_ > 0) {
    if (queue_.size() < max_batch_size_) {
        cv_.wait_for(lock, std::chrono::milliseconds(max_wait_time_ms_), [this]() { return queue_.size() >= max_batch_size_; });
    }
  } else {
    if (queue_.empty()) {
        cv_.wait(lock, [this]() { return !queue_.empty(); });
    }
  }

  for (size_t i = 0; !queue_.empty() && i < max_batch_size_; ++i) {
    batch.emplace_back(queue_.front());
    queue_.pop_front();
  }
}

}
}