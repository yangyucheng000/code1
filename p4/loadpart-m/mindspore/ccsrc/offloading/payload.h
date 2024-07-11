#ifndef MINDSPORE_OFFLOADING_PAYLOAD_H
#define MINDSPORE_OFFLOADING_PAYLOAD_H

#include "utils.h"

using offloading_serving::PredictRequest;
using offloading_serving::PredictReply;

namespace mindspore {

namespace offloading {

struct Payload {
  Payload();
  Payload(std::initializer_list<offloading_serving::PredictRequest*> requests,
          std::initializer_list<uint64_t> session_ids, std::shared_ptr<ExecutionPlan> plan = nullptr);
  Payload(std::vector<offloading_serving::PredictRequest*>& requests,
          std::vector<uint64_t>& session_ids, std::shared_ptr<ExecutionPlan> plan = nullptr);
  void MergePayload(Payload& other);
  void Release();
  
  std::vector<offloading_serving::PredictRequest*> requests_;
  std::vector<uint64_t> session_ids_;
  std::shared_ptr<ExecutionPlan> plan_;
  int64_t q_time_;
  int64_t e_time_;
  std::vector<int64_t> req_q_times_;
};

using PayloadPtr = std::shared_ptr<Payload>;

struct PayloadQueue {
  PayloadQueue(int64_t max_wait_time_ms, size_t max_batch_size);
  bool IsEmpty();
  size_t Size();
  void Enqueue(PayloadPtr& payload);
  void Enqueue(std::initializer_list<offloading_serving::PredictRequest*> requests,
               std::initializer_list<uint64_t> session_ids);
  void Enqueue(std::vector<offloading_serving::PredictRequest*>& requests,
               std::vector<uint64_t>& session_ids);
  PayloadPtr Dequeue();
  void DequeueBatch(std::vector<PayloadPtr>& batch);

  std::deque<PayloadPtr> queue_;
  std::mutex mu_;
  std::condition_variable cv_;

  int64_t max_wait_time_ms_;
  size_t max_batch_size_;
};

}
}

#endif